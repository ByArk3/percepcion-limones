"""Inventario, duplicados y dataset final (celdas 2.1 local, 3.1, 3.2 y 3.3 del notebook).

Lee la carpeta etiquetado_final_agregarfotos (Integrante/clase_integrante/foto.jpg). Solo se usan las subcarpetas
que empiezan con verde_, pinton_ o maduro_. Quita duplicados exactos (MD5) y casi duplicados (dHash, distancia <= 10)
y las fotos externas de 1056 x 1056. Marca como "antigua" la foto cuyo MD5 está en la copia del parcial.
"""
import hashlib
import os
import re

import cv2
import numpy as np
import pandas as pd

from src.comun import CLASES, preprocesamiento

PATRON_NOMBRE = re.compile(r"^([vpm])_([A-Za-z0-9]+)_(\d{3})\.jpe?g$", re.I)   # ej. v_A26_601.jpg
EXT_IMG = (".jpg", ".jpeg", ".png", ".webp", ".jfif", ".bmp", ".heic", ".heif")


def md5(ruta):
    with open(ruta, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def dhash(bgr, n=16):
    g = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (n + 1, n), interpolation=cv2.INTER_AREA)
    return (g[:, 1:] > g[:, :-1]).flatten()


def _subcarpetas(p):
    return sorted(d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d)))


def _es_clase(nombre):
    """verde_ark, pinton_franco... (carpetas del Drive) o verde, pinton, maduro (como las deja Colab en /content/dataset)."""
    n = nombre.lower()
    return any(n == c or n.startswith(c + "_") for c in CLASES)


def listar(raiz, carpeta_avance1=None):
    """Una fila por foto con integrante, clase, ruta y si es antigua (del parcial)."""
    huellas = set()
    if carpeta_avance1 and os.path.isdir(carpeta_avance1):
        huellas = {md5(os.path.join(r, f)) for r, _, fs in os.walk(carpeta_avance1) for f in fs
                   if f.lower().endswith(EXT_IMG)}
    filas = []
    for integ in _subcarpetas(raiz):
        if "avance" in integ.lower():
            continue
        for sub in _subcarpetas(os.path.join(raiz, integ)):
            if not _es_clase(sub):
                continue
            clase = next(c for c in CLASES if sub.lower().startswith(c))
            carpeta = os.path.join(raiz, integ, sub)
            for nombre in sorted(os.listdir(carpeta)):
                ruta = os.path.join(carpeta, nombre)
                if os.path.isfile(ruta) and nombre.lower().endswith(EXT_IMG):
                    filas.append(dict(ruta=ruta, integrante=integ, clase=clase, nombre=nombre,
                                      antigua=md5(ruta) in huellas))
    if not filas:
        raise SystemExit(f"No encontré fotos en {raiz} (se esperan carpetas Integrante/verde_*, pinton_*, maduro_*).")
    return pd.DataFrame(filas)


def dataset(raiz, carpeta_avance1=None, solo_nuevas=False, usar_externas=False, imprimir=print):
    """Devuelve (df_todo, df): todas las fotos con sus marcas y el dataset final para el análisis."""
    pre = preprocesamiento()
    origen = listar(raiz, carpeta_avance1)
    filas = []
    for o in origen.itertuples():
        try:
            bgr, (w, h) = pre.cargar(o.ruta)
        except Exception as e:                                    # archivo dañado: se avisa y no se usa
            imprimir(f"AVISO: no se pudo abrir {o.integrante}/{o.nombre} ({type(e).__name__}); no se usa.")
            continue
        m = PATRON_NOMBRE.match(o.nombre)
        filas.append(dict(archivo=f"{o.integrante}/{o.nombre}", clase=o.clase, ruta=o.ruta, integrante=o.integrante,
                          celular=m.group(2).upper() if m else "sin código", antigua=bool(o.antigua),
                          ancho=w, alto=h, externa=(w, h) == (1056, 1056),
                          nitidez=cv2.Laplacian(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var(),
                          md5=md5(o.ruta), _dh=dhash(bgr)))
    df_todo = pd.DataFrame(filas)
    imprimir(f"{len(df_todo)} fotos encontradas (antiguas {int(df_todo.antigua.sum())}, nuevas {int((~df_todo.antigua).sum())})")

    # duplicados exactos (MD5) y casi duplicados (dHash <= 10 de 256 bits) - celda 3.2
    df_todo["duplicado_de"] = ""
    vistos = {}
    H = np.stack(df_todo["_dh"].values)
    for i, r in df_todo.iterrows():
        if r.md5 in vistos:
            j = vistos[r.md5]
            df_todo.at[i, "duplicado_de"] = df_todo.at[j, "clase"] + "/" + df_todo.at[j, "archivo"]
            continue
        vistos[r.md5] = i
        previos = [j for j in vistos.values() if j < i]
        if previos:
            d = (H[previos] != H[i]).sum(1)
            if d.min() <= 10:
                j = previos[int(d.argmin())]
                df_todo.at[i, "duplicado_de"] = df_todo.at[j, "clase"] + "/" + df_todo.at[j, "archivo"] + f" (casi, d={d.min()})"
    imprimir(f"Duplicados: {(df_todo.duplicado_de != '').sum()}")

    # borrosas y externas; dataset final - celda 3.3
    df_todo["borrosa"] = df_todo.nitidez < 0.25 * df_todo.nitidez.median()
    filtro = (df_todo.duplicado_de == "") & (usar_externas | ~df_todo.externa)
    if solo_nuevas:
        filtro &= ~df_todo.antigua
    df = df_todo[filtro].drop(columns="_dh").reset_index(drop=True)
    df["y"] = df.clase.map({c: i for i, c in enumerate(CLASES)})
    vacias = [c for c in CLASES if (df.clase == c).sum() == 0]
    if vacias:
        raise SystemExit(f"No hay fotos en: {vacias}.")
    imprimir(f"{len(df)} fotos para el análisis" + (" (solo nuevas)" if solo_nuevas else ""))
    return df_todo.drop(columns="_dh"), df
