"""Entrenamiento de la ResNet18 (celdas 10.1 a 10.5 y 13.2 del notebook) como script.

Misma lógica que el notebook: limpieza con modelo/preprocesamiento.py, partición 70/15/15 (semilla 42),
ResNet18 con layer4 + fc entrenables, AdamW lr 1e-4 con decaimiento coseno, lote 16, pesos por clase,
30 épocas, mejor época según F1 macro de validación y 3 semillas (42, 7, 123).

IMPORTANTE (CLAUDE.md): las cifras del informe salen de la corrida del notebook en Colab del 16-nov
(notebooks/Grupo3_limones_drive_Colab_ejecutado_16nov.ipynb). Este script existe para que un tercero pueda
reproducir el proceso. Por defecto NO sobrescribe modelo/: exporta en <salida>/modelo.

Uso:
  python -m src.entrenar --fotos D:\\...\\etiquetado_final_agregarfotos --salida resultados_src
  python -m src.entrenar --fotos ... --solo-nuevas          (Tarea G: solo fotos nuevas)
  python -m src.entrenar --fotos ... --epocas 2 --semillas 42  (prueba rápida)
"""
import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import time

import numpy as np
import pandas as pd

from src.comun import CLASES, EPOCAS, LOTE, LR, NOMBRE, RAIZ, SEMILLAS, TAM, UMBRAL_H_FIJO, DESV, MEDIA, fijar_semilla, ms


def commit_actual():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=True, cwd=RAIZ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "desconocido"


def entrenar_una(IMG, variante, seed, epocas, TR, VA, TE, y, device, mlflow=None, tags=None):
    """Celda 10.2: devuelve (net, probabilidades en prueba, historial, métricas)."""
    import torch
    import torch.nn as nn
    from sklearn.metrics import f1_score
    from src.metricas import metricas
    from src.modelos.resnet import DS, aug, crear_resnet, predecir

    fijar_semilla(seed)
    imgs = IMG[variante]
    net = crear_resnet(device)
    peso = torch.tensor(len(TR) / (len(CLASES) * np.bincount(y[TR], minlength=3).clip(1)), dtype=torch.float32)
    crit = nn.CrossEntropyLoss(weight=peso.to(device))
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epocas)
    dl = torch.utils.data.DataLoader(DS(imgs, TR, aug, y), batch_size=LOTE, shuffle=True,
                                     generator=torch.Generator().manual_seed(seed))
    hist, mejor_f1, mejor_estado = [], -1, None
    corrida = mlflow.start_run(run_name=f"resnet18_{variante}_s{seed}") if mlflow else None
    try:
        if mlflow:
            mlflow.log_params(dict(variante=variante, seed=seed, epocas=epocas, lr=LR, batch=LOTE,
                                   congeladas="conv1-layer3", tam=TAM))
            mlflow.set_tags(tags or {})
        for ep in range(epocas):
            net.train(); tot = 0
            for xb, yb in dl:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(); loss = crit(net(xb), yb); loss.backward(); opt.step(); tot += loss.item() * len(xb)
            sch.step()
            pv = predecir(net, imgs, VA, y, device)
            with torch.no_grad():   # log-probabilidades como logits: softmax(log p) = p
                lv = crit(torch.log(torch.tensor(pv) + 1e-9).to(device), torch.tensor(y[VA]).to(device)).item()
            f1v = f1_score(y[VA], pv.argmax(1), average="macro")
            f1t = f1_score(y[TR], predecir(net, imgs, TR, y, device).argmax(1), average="macro")
            hist.append(dict(epoca=ep + 1, perdida_train=tot / len(TR), perdida_val=lv, f1_train=f1t, f1_val=f1v))
            if mlflow:
                mlflow.log_metrics({k: v for k, v in hist[-1].items() if k != "epoca"}, step=ep + 1)
            if f1v > mejor_f1:                                       # mejor época según validación
                mejor_f1, mejor_estado = f1v, {k: v.detach().clone() for k, v in net.state_dict().items()}
            print(f"   semilla {seed} época {ep + 1}/{epocas}: pérdida {hist[-1]['perdida_train']:.4f} | F1 val {f1v:.3f}")
        net.load_state_dict(mejor_estado)
        t0 = time.perf_counter(); pt = predecir(net, imgs, TE, y, device); lat = (time.perf_counter() - t0) / len(TE) * 1000
        m = metricas(y[TE], pt.argmax(1)); m["latencia_ms"] = lat
        if mlflow:
            mlflow.log_metrics({f"test_{k}": v for k, v in m.items()})
    finally:
        if corrida:
            mlflow.end_run()
    return net, pt, pd.DataFrame(hist), m


def exportar(net, destino, info_extra, ruta_pre):
    """Celda 13.2: TorchScript + modelo_info.json + preprocesamiento.py."""
    import copy
    import torch
    os.makedirs(destino, exist_ok=True)
    net_cpu = copy.deepcopy(net).cpu().eval()
    ts = torch.jit.trace(net_cpu, torch.zeros(1, 3, TAM, TAM))
    ruta_pt = os.path.join(destino, "resnet18_limon.pt")
    ts.save(ruta_pt)
    shutil.copy2(ruta_pre, os.path.join(destino, "preprocesamiento.py"))
    info = dict(modelo="ResNet18 (torchvision), layer4 + fc entrenadas", formato="TorchScript", archivo="resnet18_limon.pt",
                clases=CLASES, nombres=[NOMBRE[c] for c in CLASES], entrada=f"tensor 1x3x{TAM}x{TAM}, RGB normalizado",
                media=MEDIA, desv=DESV, salida="logits (aplicar softmax)",
                preprocesamiento="cargar (EXIF, lado 600) -> segmentar (Otsu en S) -> corregir (iluminación y balance de blancos) "
                                 f"-> recortar {TAM}x{TAM} -> BGR a RGB -> normalizar con media/desv de ImageNet",
                umbral_h=UMBRAL_H_FIJO, seleccion="mejor F1 macro de validación", **info_extra,
                fecha=time.strftime("%Y-%m-%d %H:%M"), sha256=hashlib.sha256(open(ruta_pt, "rb").read()).hexdigest())
    json.dump(info, open(os.path.join(destino, "modelo_info.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fotos", required=True, help="carpeta etiquetado_final_agregarfotos")
    ap.add_argument("--avance1", default=None, help="copia del parcial etiquetado_avance1 (marca fotos antiguas)")
    ap.add_argument("--salida", default="resultados_src")
    ap.add_argument("--variante", default="corregida", choices=["cruda", "corregida", "clahe"])
    ap.add_argument("--semillas", default=",".join(map(str, SEMILLAS)))
    ap.add_argument("--epocas", type=int, default=EPOCAS)
    ap.add_argument("--solo-nuevas", action="store_true", help="entrenar solo con fotos que no están en el parcial")
    ap.add_argument("--mlflow-db", default="", help="ruta de mlflow.db (por defecto <salida>/mlflow.db)")
    ap.add_argument("--integrante", default="", help="tag de MLflow: quién corre el experimento")
    ap.add_argument("--no-exportar", action="store_true")
    a = ap.parse_args()

    import mlflow
    import torch
    from src.datos.inventario import dataset
    from src.datos.limpieza import limpiar
    from src.datos.particion import indices, particionar
    from src.evaluar import tabla_metas, tabla_por_clase

    t_inicio = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tablas = os.path.join(a.salida, "tablas")
    os.makedirs(tablas, exist_ok=True)
    semillas = [int(s) for s in a.semillas.split(",")]
    print(f"Dispositivo: {device} | torch {torch.__version__} | Python {platform.python_version()} | "
          f"variante {a.variante} | semillas {semillas} | épocas {a.epocas} | solo nuevas: {a.solo_nuevas}")

    _, df = dataset(a.fotos, a.avance1, a.solo_nuevas)
    df = particionar(df)
    TR, VA, TE = indices(df)
    y = df.y.values
    print(f"train {len(TR)} | val {len(VA)} | test {len(TE)}")
    IMG, _ = limpiar(df, variantes=(a.variante,))

    db = os.path.abspath(a.mlflow_db or os.path.join(a.salida, "mlflow.db"))
    mlflow.set_tracking_uri("sqlite:///" + db.replace("\\", "/"))
    mlflow.set_experiment("limon_madurez_src" + ("_solo_nuevas" if a.solo_nuevas else ""))
    tags = {"integrante": a.integrante or "no indicado", "commit": commit_actual(), "dispositivo": device,
            "solo_nuevas": str(a.solo_nuevas), "origen": "src/entrenar.py"}

    res, probs, nets = [], [], []
    for s in semillas:
        net, pt, hist, m = entrenar_una(IMG, a.variante, s, a.epocas, TR, VA, TE, y, device, mlflow, tags)
        res.append(m); probs.append(pt); nets.append(net)
        hist.to_csv(os.path.join(tablas, f"historial_{a.variante}_s{s}.csv"), index=False)
        print(f"semilla {s}: " + ", ".join(f"{k} {v:.3f}" for k, v in m.items()))

    R = pd.DataFrame(res)
    resumen = pd.DataFrame([{"Modelo": f"ResNet18 ({a.variante})", **{k: ms(R[k]) for k in
                             ["exactitud", "f1_macro", "recall_maduro", "precision_maduro", "fp_maduro"]},
                             "n_train": len(TR), "n_test": len(TE), "semillas": len(semillas), "solo_nuevas": a.solo_nuevas}])
    t_clase, F = tabla_por_clase(y[TE], probs)
    t_metas = tabla_metas(res, F)
    R.assign(semilla=semillas).to_csv(os.path.join(tablas, "metricas_por_semilla.csv"), index=False)
    resumen.to_csv(os.path.join(tablas, "resumen.csv"), index=False, encoding="utf-8-sig")
    t_clase.to_csv(os.path.join(tablas, "metricas_por_clase.csv"), index=False, encoding="utf-8-sig")
    t_metas.to_csv(os.path.join(tablas, "tabla_metas.csv"), index=False, encoding="utf-8-sig")
    print("\n" + resumen.to_string(index=False))
    print(t_clase.to_string(index=False))
    print(t_metas.drop(columns="Justificación").to_string(index=False))

    if not a.no_exportar:
        info = exportar(nets[0], os.path.join(a.salida, "modelo"),
                        dict(semilla=semillas[0], epocas=a.epocas, n_train=int(len(TR)), n_val=int(len(VA)),
                             n_test=int(len(TE)), metricas_prueba={k: round(float(v), 4) for k, v in res[0].items()}),
                        os.path.join(RAIZ, "modelo", "preprocesamiento.py"))
        print(f"\nModelo exportado en {a.salida}/modelo (sha256 {info['sha256'][:16]}...). modelo/ del repositorio NO se tocó.")
    print(f"Tiempo total: {(time.time() - t_inicio) / 60:.1f} min | MLflow: {db}")


if __name__ == "__main__":
    main()
