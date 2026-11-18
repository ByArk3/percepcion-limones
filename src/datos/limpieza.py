"""Limpieza de las fotos (celdas 4.1 y 4.2 del notebook).

Se arman tres versiones de cada foto para la comparación de entradas (sección 5.3):
  cruda     recorte del limón sin corregir
  corregida iluminación + balance de blancos (la que usa el modelo final)
  clahe     corregida + CLAHE en el canal L
"""
import cv2

from src.comun import preprocesamiento


def clahe_L(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def limpiar(df, variantes=("cruda", "corregida", "clahe"), imprimir=print):
    """Devuelve (IMG, MASK): IMG[variante] es una lista de imágenes BGR 224x224 en el orden de df."""
    pre = preprocesamiento()
    IMG = {v: [] for v in variantes}
    MASK, AREA = [], []
    for r in df.itertuples():
        bgr, _ = pre.cargar(r.ruta)
        m = pre.segmentar(bgr)
        AREA.append(m.mean() / 255)
        c_cor, mk = pre.recortar(pre.corregir(bgr, m), m)
        if "cruda" in IMG:
            IMG["cruda"].append(pre.recortar(bgr, m)[0])
        if "corregida" in IMG:
            IMG["corregida"].append(c_cor)
        if "clahe" in IMG:
            IMG["clahe"].append(clahe_L(c_cor))
        MASK.append(mk)
    raras = [a for a, area in zip(df.archivo, AREA) if area < 0.03 or area > 0.6]
    if raras:
        imprimir(f"Revisar segmentación (área fuera de 3 % a 60 %): {raras}")
    return IMG, MASK
