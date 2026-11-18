"""Evaluación (celdas 10.4, 10.5 y 12.1 del notebook): métricas, métricas por clase, metas y matriz de confusión.

Como script, evalúa un modelo exportado (TorchScript) sobre la partición de prueba del dataset
(misma partición 70/15/15, semilla 42, que el notebook):

  python -m src.evaluar --fotos D:\\...\\etiquetado_final_agregarfotos --modelo modelo --salida resultados_src

Las funciones también las usa src/entrenar.py.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from src.comun import CLASES, METAS, NOMBRE, ms
from src.metricas import metricas


def tabla_por_clase(y_test, probs_por_semilla):
    P, Rc, F = (np.array([precision_recall_fscore_support(y_test, pt.argmax(1), labels=[0, 1, 2], zero_division=0)[k]
                          for pt in probs_por_semilla]) for k in range(3))
    n_te = np.bincount(y_test, minlength=3)
    t = pd.DataFrame([{"Clase": NOMBRE[c], "N prueba": int(n_te[k]), "Precisión": ms(P[:, k]),
                       "Recall": ms(Rc[:, k]), "F1": ms(F[:, k])} for k, c in enumerate(CLASES)])
    return t, F


def tabla_metas(res, F):
    """res: lista de dicts de metricas() (una por semilla); F: matriz semillas x clases de F1."""
    R = pd.DataFrame(res)
    valores = {k: R[k].values for k in ["exactitud", "f1_macro", "recall_maduro", "precision_maduro", "fp_maduro"]}
    valores.update(f1_verde=F[:, 0], f1_pinton=F[:, 1], f1_maduro=F[:, 2])
    filas = []
    for clave, nom, op, meta, porque in METAS:
        v = valores[clave].mean()
        ok = v >= meta if op == ">=" else v <= meta
        filas.append({"Métrica": nom, "Meta": f"{op} {meta:.2f}", "Valor (media ± desv.)": ms(valores[clave]),
                      "¿Cumple?": "Sí" if ok else "No", "Justificación": porque})
    return pd.DataFrame(filas)


def matriz_confusion(y_test, pred, ruta_png=None):
    cm = confusion_matrix(y_test, pred, labels=[0, 1, 2])
    if ruta_png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay
        cmn = cm / cm.sum(1, keepdims=True).clip(1)
        ConfusionMatrixDisplay(cmn, display_labels=[NOMBRE[c] for c in CLASES]).plot(cmap="Greens", values_format=".2f")
        plt.title("Matriz de confusión normalizada (prueba)")
        plt.tight_layout(); plt.savefig(ruta_png, dpi=150); plt.close()
    return cm


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fotos", required=True)
    ap.add_argument("--avance1", default=None, help="copia del parcial (solo para marcar fotos antiguas)")
    ap.add_argument("--modelo", default="modelo")
    ap.add_argument("--salida", default="resultados_src")
    ap.add_argument("--solo-nuevas", action="store_true")
    a = ap.parse_args()

    import torch
    from src.datos.inventario import dataset
    from src.datos.limpieza import limpiar
    from src.datos.particion import indices, particionar
    from src.modelos.resnet import predecir

    os.makedirs(os.path.join(a.salida, "tablas"), exist_ok=True)
    os.makedirs(os.path.join(a.salida, "figuras"), exist_ok=True)
    _, df = dataset(a.fotos, a.avance1, a.solo_nuevas)
    df = particionar(df)
    TR, VA, TE = indices(df)
    y = df.y.values
    IMG, _ = limpiar(df, variantes=("corregida",))
    info = json.load(open(os.path.join(a.modelo, "modelo_info.json"), encoding="utf-8"))
    net = torch.jit.load(os.path.join(a.modelo, info.get("archivo", "resnet18_limon.pt")), map_location="cpu").eval()
    t0 = time.perf_counter()
    pt = predecir(net, IMG["corregida"], TE, y, "cpu")
    lat = (time.perf_counter() - t0) / len(TE) * 1000
    m = metricas(y[TE], pt.argmax(1))
    print(f"train {len(TR)} | val {len(VA)} | test {len(TE)}")
    print("Métricas en prueba:", ", ".join(f"{k} {v:.4f}" for k, v in m.items()), f"| latencia {lat:.1f} ms/foto (CPU)")
    t_clase, F = tabla_por_clase(y[TE], [pt])
    t_metas = tabla_metas([m], F)
    print(t_clase.to_string(index=False)); print(t_metas.drop(columns="Justificación").to_string(index=False))
    cm = matriz_confusion(y[TE], pt.argmax(1), os.path.join(a.salida, "figuras", "matriz_confusion_evaluar.png"))
    print("Matriz de confusión (filas = real, columnas = predicho):\n", cm)
    t_clase.to_csv(os.path.join(a.salida, "tablas", "evaluar_por_clase.csv"), index=False, encoding="utf-8-sig")
    t_metas.to_csv(os.path.join(a.salida, "tablas", "evaluar_metas.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame([{**m, "latencia_ms_cpu": lat, "n_test": len(TE)}]).to_csv(
        os.path.join(a.salida, "tablas", "evaluar_metricas.csv"), index=False)
    if "metricas_prueba" in info:
        print("Referencia en modelo_info.json:", info["metricas_prueba"],
              "(puede variar si el dataset cambió o por la diferencia CPU/GPU)")


if __name__ == "__main__":
    main()
