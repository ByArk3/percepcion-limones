"""Carga del modelo UNA sola vez y clasificación de fotos.

Usa la misma clase Clasificador que el pipeline de Spark (pipeline/inferencia.py, copiada a la imagen
al construirla) y el mismo modelo/preprocesamiento.py. Así la API y Spark dan las mismas probabilidades.
"""
import hashlib
import os
import time

from app.inferencia import Clasificador

DIR_MODELO = os.environ.get("DIR_MODELO", "/modelo")
VERSION_MODELO = os.environ.get("VERSION_MODELO", "modelo-v1")


class Servicio:
    def __init__(self):
        t0 = time.perf_counter()
        self.clf = Clasificador(DIR_MODELO, hilos=int(os.environ.get("HILOS_TORCH", "0")) or None)
        ruta_pt = os.path.join(DIR_MODELO, self.clf.info.get("archivo", "resnet18_limon.pt"))
        self.sha256 = hashlib.sha256(open(ruta_pt, "rb").read()).hexdigest()
        self.sha256_info = self.clf.info.get("sha256")
        self.version = VERSION_MODELO
        self.clases = self.clf.clases
        self.carga_s = time.perf_counter() - t0

    def clasificar(self, lista_bytes):
        """Devuelve una lista de dicts con clase, probabilidades y latencia por foto (ms)."""
        t0 = time.perf_counter()
        prob = self.clf.predecir(lista_bytes)
        ms = 1000 * (time.perf_counter() - t0) / max(len(lista_bytes), 1)
        salida = []
        for fila in prob:
            if any(p != p for p in fila):                 # NaN: la foto no se pudo leer
                salida.append(dict(clase="error", probabilidades=None, latencia_ms=round(ms, 2)))
            else:
                salida.append(dict(clase=self.clases[int(fila.argmax())],
                                   probabilidades={c: round(float(p), 6) for c, p in zip(self.clases, fila)},
                                   latencia_ms=round(ms, 2)))
        return salida
