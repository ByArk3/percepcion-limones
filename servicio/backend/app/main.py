"""API de inferencia (sección 8.1) · madurez del limón sutil · Grupo 3.

Endpoints:
  GET  /health              estado, versión y sha256 del modelo
  GET  /modelo              datos del modelo desplegado (de modelo_info.json, sin cifras escritas a mano)
  POST /predict             una foto  -> clase, probabilidades, versión, latencia (se guarda en PostgreSQL)
  POST /predict_lote        un .zip   -> tabla de resultados y conteo por clase (se guarda en PostgreSQL)
  GET  /historial           últimas predicciones guardadas
  POST /lotes/sincronizar   copia a PostgreSQL los resultados en Parquet de los lotes de Spark
  GET  /lotes               conteo por lote, celular y predicción de los lotes de Spark

El modelo se carga UNA sola vez al arrancar.
"""
import hashlib
import io
import os
import time
import uuid
import zipfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile

from app import base_datos as bd
from app.modelo import Servicio

EXT_FOTO = (".jpg", ".jpeg", ".png")
MAX_FOTOS_ZIP = int(os.environ.get("MAX_FOTOS_ZIP", "300"))
MAX_MB = int(os.environ.get("MAX_MB", "200"))
DIR_PARQUET_SPARK = os.environ.get("DIR_PARQUET_SPARK", "/datos/resultados_spark")
LOTE_INFERENCIA = 16

estado = {}


@asynccontextmanager
async def ciclo_de_vida(app):
    estado["servicio"] = Servicio()                       # una sola carga del modelo
    bd.crear_tablas()
    s = estado["servicio"]
    print(f"Modelo {s.version} cargado en {s.carga_s:.2f} s | sha256 {s.sha256[:16]}... "
          f"({'coincide' if s.sha256 == s.sha256_info else 'NO coincide'} con modelo_info.json)", flush=True)
    yield


app = FastAPI(title="Madurez del limón sutil · Grupo 3", version="1.0", lifespan=ciclo_de_vida)


def fila_bd(origen, id_envio, archivo, datos, r, s):
    p = r["probabilidades"] or {}
    return dict(origen=origen, id_envio=id_envio, archivo=archivo, md5=hashlib.md5(datos).hexdigest(),
                clase=r["clase"], prob_verde=p.get("verde"), prob_pinton=p.get("pinton"),
                prob_maduro=p.get("maduro"), latencia_ms=r["latencia_ms"], version_modelo=s.version,
                sha256_modelo=s.sha256)


async def leer(archivo: UploadFile):
    datos = await archivo.read()
    if len(datos) > MAX_MB * 1024 * 1024:
        raise HTTPException(413, f"El archivo pasa de {MAX_MB} MB.")
    if not datos:
        raise HTTPException(400, "El archivo está vacío.")
    return datos


@app.get("/health")
def health():
    s = estado["servicio"]
    return {"estado": "ok", "version_modelo": s.version, "sha256": s.sha256,
            "sha256_coincide_con_info": s.sha256 == s.sha256_info, "clases": s.clases}


@app.get("/modelo")
def modelo():
    """Lo que dice modelo/modelo_info.json del modelo cargado (métricas de prueba de la semilla 42)."""
    s = estado["servicio"]
    info = s.clf.info
    claves = ("modelo", "formato", "clases", "nombres", "entrada", "preprocesamiento", "umbral_h", "semilla",
              "seleccion", "epocas", "n_train", "n_val", "n_test", "metricas_prueba", "fecha")
    return {**{k: info.get(k) for k in claves}, "version_modelo": s.version, "sha256": s.sha256}


@app.post("/predict")
async def predict(archivo: UploadFile = File(...)):
    s = estado["servicio"]
    datos = await leer(archivo)
    r = s.clasificar([datos])[0]
    if r["clase"] == "error":
        raise HTTPException(422, "No se pudo leer la imagen.")
    bd.guardar([fila_bd("foto", uuid.uuid4().hex[:12], archivo.filename or "foto", datos, r, s)])
    return {"clase": r["clase"], "probabilidades": r["probabilidades"], "version_modelo": s.version,
            "sha256": s.sha256, "latencia_ms": r["latencia_ms"]}


@app.post("/predict_lote")
async def predict_lote(archivo: UploadFile = File(...)):
    s = estado["servicio"]
    datos = await leer(archivo)
    try:
        z = zipfile.ZipFile(io.BytesIO(datos))
    except zipfile.BadZipFile:
        raise HTTPException(400, "El archivo no es un .zip válido.")
    nombres = sorted(n for n in z.namelist()
                     if n.lower().endswith(EXT_FOTO) and not os.path.basename(n).startswith(".")
                     and "__MACOSX" not in n)
    if not nombres:
        raise HTTPException(400, "El .zip no tiene fotos .jpg, .jpeg o .png.")
    if len(nombres) > MAX_FOTOS_ZIP:
        raise HTTPException(413, f"El .zip tiene {len(nombres)} fotos; el máximo es {MAX_FOTOS_ZIP}.")
    id_envio = uuid.uuid4().hex[:12]
    t0 = time.perf_counter()
    resultados, filas = [], []
    for i in range(0, len(nombres), LOTE_INFERENCIA):
        parte = nombres[i:i + LOTE_INFERENCIA]
        bytes_parte = [z.read(n) for n in parte]
        for n, b, r in zip(parte, bytes_parte, s.clasificar(bytes_parte)):
            resultados.append(dict(archivo=n, clase=r["clase"], probabilidades=r["probabilidades"]))
            filas.append(fila_bd("zip", id_envio, n, b, r, s))
    bd.guardar(filas)
    conteo = {c: sum(r["clase"] == c for r in resultados) for c in s.clases + ["error"]}
    return {"id_envio": id_envio, "fotos": len(resultados), "conteo": conteo, "resultados": resultados,
            "tiempo_total_s": round(time.perf_counter() - t0, 3), "version_modelo": s.version, "sha256": s.sha256}


@app.get("/historial")
def historial(limite: int = 50):
    return bd.historial(max(1, min(limite, 500)))


@app.post("/lotes/sincronizar")
def lotes_sincronizar():
    if not os.path.isdir(DIR_PARQUET_SPARK):
        raise HTTPException(404, f"No hay resultados de Spark en {DIR_PARQUET_SPARK}. Corre primero el job batch.")
    cargados = bd.sincronizar_spark(DIR_PARQUET_SPARK)
    if not cargados:
        raise HTTPException(404, "No se encontró ninguna carpeta lote=... con Parquet.")
    return {"cargados": cargados}


@app.get("/lotes")
def lotes():
    return bd.lotes_spark()
