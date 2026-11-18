"""Perfil del entrenamiento (sección 5.5, opción 2 de la plantilla: medición del cuello de botella).

No reentrena el modelo final. Corre pocas épocas (por defecto 2) de la misma ResNet18 y, por época, mide:
  - tiempo de carga de lotes: lo que el bucle espera al DataLoader (lectura + aumento de datos + armado del lote);
  - tiempo de cómputo: forward + backward + paso del optimizador (con sincronización de CUDA si hay GPU);
  - uso de CPU del proceso (y de sus procesos hijos) durante la época, si psutil está instalado.
Se prueba con num_workers = 0 y 2.

Conclusión automática: si la carga de lotes es más de la mitad del tiempo, el entrenamiento está limitado por E/S
(preparación de datos); si no, por cómputo. Las cifras de escalamiento al final son ESTIMACIONES a partir de lo medido.

Uso (Colab con GPU o PC con CPU):
  python -m src.perfilar_entrenamiento --fotos /ruta/etiquetado_final_agregarfotos --salida resultados_perfil
"""
import argparse
import os
import platform
import time

import numpy as np
import pandas as pd

from src.comun import CLASES, LOTE, LR, fijar_semilla


def sincronizar(device):
    import torch
    if device == "cuda":
        torch.cuda.synchronize()


def perfilar(IMG, TR, y, device, num_workers, epocas, seed=42):
    import torch
    import torch.nn as nn
    from src.modelos.resnet import DS, aug, crear_resnet
    try:
        import psutil
        proc = psutil.Process()
    except ImportError:
        psutil = proc = None

    fijar_semilla(seed)
    net = crear_resnet(device)
    peso = torch.tensor(len(TR) / (len(CLASES) * np.bincount(y[TR], minlength=3).clip(1)), dtype=torch.float32)
    crit = nn.CrossEntropyLoss(weight=peso.to(device))
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=LR, weight_decay=1e-4)
    dl = torch.utils.data.DataLoader(DS(IMG, TR, aug, y), batch_size=LOTE, shuffle=True, num_workers=num_workers,
                                     generator=torch.Generator().manual_seed(seed),
                                     persistent_workers=num_workers > 0)
    filas = []
    for ep in range(epocas):
        net.train()
        t_carga = t_comp = 0.0
        n = 0
        if proc:
            proc.cpu_percent(None)
            for h in proc.children(recursive=True):
                h.cpu_percent(None)
        t_ep = time.perf_counter()
        t0 = time.perf_counter()
        for xb, yb in dl:
            t1 = time.perf_counter()
            t_carga += t1 - t0
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss = crit(net(xb), yb); loss.backward(); opt.step()
            sincronizar(device)
            t0 = time.perf_counter()
            t_comp += t0 - t1
            n += len(xb)
        total = time.perf_counter() - t_ep
        cpu = None
        if proc:
            cpu = proc.cpu_percent(None) + sum(h.cpu_percent(None) for h in proc.children(recursive=True))
        filas.append(dict(num_workers=num_workers, epoca=ep + 1, fotos=n, tiempo_epoca_s=total, carga_lotes_s=t_carga,
                          computo_s=t_comp, pct_carga=100 * t_carga / total, fotos_por_s=n / total,
                          cpu_pct=cpu, nucleos=os.cpu_count()))
        print(f"   num_workers={num_workers} época {ep + 1}: {total:.2f} s | carga {t_carga:.2f} s "
              f"({100 * t_carga / total:.1f} %) | cómputo {t_comp:.2f} s | {n / total:.1f} fotos/s"
              + (f" | CPU {cpu:.0f} %" if cpu is not None else ""))
    return filas


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fotos", required=True)
    ap.add_argument("--salida", default="resultados_perfil")
    ap.add_argument("--epocas", type=int, default=2)
    ap.add_argument("--workers", default="0,2")
    a = ap.parse_args()

    import torch
    from src.datos.inventario import dataset
    from src.datos.limpieza import limpiar
    from src.datos.particion import indices, particionar

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hw = torch.cuda.get_device_name(0) if device == "cuda" else platform.processor() or platform.machine()
    os.makedirs(a.salida, exist_ok=True)
    lineas = [f"Fecha: {time.strftime('%Y-%m-%d %H:%M:%S')}",
              f"Dispositivo: {device} ({hw}) | os.cpu_count() = {os.cpu_count()} | torch {torch.__version__} | "
              f"Python {platform.python_version()}",
              f"Lote {LOTE}, lr {LR}, épocas medidas por configuración: {a.epocas}"]
    print("\n".join(lineas))
    _, df = dataset(a.fotos)
    df = particionar(df)
    TR, _, _ = indices(df)
    y = df.y.values
    t0 = time.perf_counter()
    IMG, _ = limpiar(df, variantes=("corregida",))
    t_limpieza = time.perf_counter() - t0
    lineas.append(f"Limpieza previa (una vez, fuera del bucle; igual que el notebook): {t_limpieza:.1f} s para {len(df)} fotos "
                  f"({1000 * t_limpieza / len(df):.0f} ms/foto)")
    print(lineas[-1])

    filas = []
    for w in [int(x) for x in a.workers.split(",")]:
        filas += perfilar(IMG["corregida"], TR, y, device, w, a.epocas)
    t = pd.DataFrame(filas)
    t.round(3).to_csv(os.path.join(a.salida, "perfil_entrenamiento.csv"), index=False)
    # la primera época incluye el arranque (pesos, procesos de carga): se resume con las demás si hay
    res = t[t.epoca > 1] if (t.epoca > 1).any() else t
    res = res.groupby("num_workers")[["tiempo_epoca_s", "carga_lotes_s", "computo_s", "pct_carga", "fotos_por_s", "cpu_pct"]].mean()
    lineas += ["", "Promedio por época (sin la primera si hay más de una):", res.round(2).to_string(), ""]
    w_mejor = res.tiempo_epoca_s.idxmin()
    pct = res.loc[w_mejor, "pct_carga"]
    limite = "E/S (preparación de lotes)" if pct > 50 else "CÓMPUTO (forward + backward)"
    lineas.append(f"Conclusión: con num_workers={w_mejor} la carga de lotes es el {pct:.1f} % del tiempo de la época -> "
                  f"el entrenamiento está limitado por {limite}.")
    t_ep = res.loc[w_mejor, "tiempo_epoca_s"]
    lineas += ["",
               "Paralelismo que correspondería: de DATOS (DistributedDataParallel), no de modelo: la ResNet18 (11.2 M parámetros)",
               "cabe completa en la memoria de una GPU o de la CPU.",
               "Ajuste de la tasa de aprendizaje al crecer el lote efectivo (regla lineal con calentamiento, Goyal et al., 2017):",
               f"  lr = {LR} x N, con lote efectivo = {LOTE} x N (N = número de réplicas).",
               "Estimación (no medida) del tiempo por época con N réplicas, suponiendo reparto perfecto del cómputo:"]
    for N in (1, 2, 4):
        comp = res.loc[w_mejor, "computo_s"] / N
        carga = res.loc[w_mejor, "carga_lotes_s"] / (N if pct <= 50 else 1)
        lineas.append(f"  N={N}: lote efectivo {LOTE * N}, lr {LR * N:.0e}, ~{comp + carga:.2f} s por época (medido con N=1: {t_ep:.2f} s)")
    lineas.append("Costo: horas de cómputo = épocas x tiempo por época x semillas / 3600; multiplicar por el precio por hora "
                  "del equipo usado (PENDIENTE: precio).")
    texto = "\n".join(lineas)
    print("\n" + "\n".join(lineas[3:]))
    open(os.path.join(a.salida, "perfil_entrenamiento.txt"), "w", encoding="utf-8").write(texto + "\n")
    print(f"\nGuardado en {a.salida}/perfil_entrenamiento.csv y perfil_entrenamiento.txt")


if __name__ == "__main__":       # necesario en Windows con num_workers > 0
    main()
