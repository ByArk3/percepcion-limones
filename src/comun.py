"""Parámetros y utilidades comunes, copiados del notebook (celdas 2.2 y 2.3). No cambiar sin acordarlo (CLAUDE.md, sección 5)."""
import os
import random
import sys

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASES = ["verde", "pinton", "maduro"]                       # orden de las clases (0, 1, 2)
NOMBRE = {"verde": "Verde", "pinton": "Pintón", "maduro": "Maduro"}
SEMILLAS = [42, 7, 123]
EPOCAS = 30
LR = 1e-4
TAM = 224
LOTE = 16
UMBRAL_H_FIJO = 27                                           # H < 27 (OpenCV 0-179) es amarillo. FIJO.
MEDIA, DESV = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
METAS = [   # (clave, nombre, comparación, valor, de dónde sale) - celda 2.2
    ("exactitud",        "Exactitud",                   ">=", 0.85, "Igual o mejor que un inspector humano (~80-85 %, See, 2015)"),
    ("f1_macro",         "F1 macro",                    ">=", 0.85, "Similar a lo publicado con ~200 fotos (melón, 230 fotos: F1 0.87)"),
    ("recall_maduro",    "Recall Maduro",               ">=", 0.90, "Clase crítica: un maduro que pasa como fresco (NTP 011.006, >30 % amarillo)"),
    ("precision_maduro", "Precisión Maduro",            ">=", 0.90, "Pocos limones buenos enviados a maduro"),
    ("fp_maduro",        "Falsos positivos de Maduro",  "<=", 0.05, "Fruta verde o pintona descartada por error"),
    ("f1_verde",         "F1 Verde",                    ">=", 0.85, "Clase con más fotos y más fácil"),
    ("f1_pinton",        "F1 Pintón",                   ">=", 0.75, "Clase intermedia: la frontera es difusa incluso para personas"),
]


def fijar_semilla(s):
    import torch
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def preprocesamiento(dir_modelo=os.path.join(RAIZ, "modelo")):
    """Las funciones cargar, segmentar, corregir y recortar son las de modelo/preprocesamiento.py
    (exportadas por la celda 13.2 del notebook), así entrenamiento, Spark y API usan el mismo código."""
    try:                                                         # fotos HEIC de celular (igual que el notebook)
        from pillow_heif import register_heif_opener
        register_heif_opener()
    except ImportError:
        pass
    if dir_modelo not in sys.path:
        sys.path.insert(0, dir_modelo)
    import preprocesamiento as pre
    return pre


def ms(v):
    """media ± desv. estándar muestral, como en el notebook (celda 10.4)."""
    v = np.asarray(v, dtype=float)
    return f"{v.mean():.3f} ± {v.std(ddof=1):.3f}" if len(v) > 1 else f"{v.mean():.3f}"
