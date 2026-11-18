"""Funciones comunes del pipeline: sesión de Spark, clasificador, UDF y bitácora.

Las usan job_batch.py y medir_escalabilidad.py. El preprocesamiento NO se reescribe:
se importa de modelo/preprocesamiento.py (el archivo que exporta la celda 13.2 del notebook).
"""
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from urllib.parse import unquote, urlparse

import numpy as np
import pandas as pd

# torch.jit.load/trace avisan que serán reemplazados; el modelo exportado es TorchScript y funciona igual
warnings.filterwarnings("ignore", message=r".*torch\.jit.*deprecated")

ESQUEMA_SALIDA = ("path string, lote string, celular string, clase string, md5 string, prediccion string, "
                  "prob_verde double, prob_pinton double, prob_maduro double, "
                  "particion int, carga_modelo_ms double")


# ---------------------------------------------------------------- bitácora
class Bitacora:
    """Imprime en consola y guarda lo mismo en un archivo de texto (para subirlo al repositorio)."""

    def __init__(self, ruta):
        os.makedirs(os.path.dirname(ruta) or ".", exist_ok=True)
        self.f = open(ruta, "w", encoding="utf-8")

    def __call__(self, *x):
        texto = " ".join(str(v) for v in x)
        print(texto, flush=True)
        self.f.write(texto + "\n")
        self.f.flush()

    def titulo(self, texto):
        self("\n" + "=" * 78 + "\n" + texto + "\n" + "=" * 78)


# ---------------------------------------------------------------- entorno
def cpus_docker():
    """Límite de CPU del contenedor (cgroup v2), o None si no hay límite."""
    try:
        cuota, periodo = open("/sys/fs/cgroup/cpu.max").read().split()
        return None if cuota == "max" else round(int(cuota) / int(periodo), 2)
    except (OSError, ValueError):
        return None


def memoria_gb():
    try:
        for linea in open("/proc/meminfo"):
            if linea.startswith("MemTotal"):
                return round(int(linea.split()[1]) / 1e6, 2)
    except OSError:
        pass
    return None


def describir_entorno(log, dir_modelo):
    """Escribe fecha, versiones, núcleos y huella del modelo. Devuelve un dict con lo principal."""
    import cv2, PIL, pyarrow, pyspark, torch, torchvision
    try:
        java = subprocess.run(["java", "-version"], capture_output=True, text=True).stderr.splitlines()
        java = next((l for l in java if "version" in l), "?")
    except OSError:
        java = "no encontrado"
    try:
        import mlflow
        v_mlflow = mlflow.__version__
    except ImportError:
        v_mlflow = "no instalado"
    afinidad = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    ent = dict(fecha=time.strftime("%Y-%m-%d %H:%M:%S"), sistema=platform.platform(), python=platform.python_version(),
               java=java.strip(), pyspark=pyspark.__version__, torch=torch.__version__,
               torchvision=torchvision.__version__, numpy=np.__version__, pandas=pd.__version__,
               opencv=cv2.__version__, pillow=PIL.__version__, pyarrow=pyarrow.__version__, mlflow=v_mlflow,
               os_cpu_count=os.cpu_count(), cpus_afinidad=afinidad, cpus_limite_docker=cpus_docker(),
               memoria_gb=memoria_gb())
    log.titulo("Entorno")
    for k, v in ent.items():
        log(f"{k:20s}: {v}")
    info = leer_info(dir_modelo)
    ruta_pt = os.path.join(dir_modelo, info.get("archivo", "resnet18_limon.pt"))
    sha = hashlib.sha256(open(ruta_pt, "rb").read()).hexdigest()
    ent.update(modelo=ruta_pt, sha256_modelo=sha, sha256_info=info.get("sha256"), epocas=info.get("epocas"))
    log(f"{'modelo':20s}: {ruta_pt} ({os.path.getsize(ruta_pt) / 1e6:.1f} MB)")
    log(f"{'sha256 del .pt':20s}: {sha}")
    log(f"{'sha256 en info':20s}: {info.get('sha256')}  -> "
        + ("coinciden" if sha == info.get("sha256") else "NO COINCIDEN (revisar que sea el modelo correcto)"))
    log(f"{'epocas (info)':20s}: {info.get('epocas')}")
    log(f"{'clases (info)':20s}: {info.get('clases')}")
    return ent


def leer_info(dir_modelo):
    return json.load(open(os.path.join(dir_modelo, "modelo_info.json"), encoding="utf-8"))


# ---------------------------------------------------------------- Spark
def crear_spark(nucleos, memoria="4g", filas_por_lote=16, nombre="limon_batch"):
    from pyspark.sql import SparkSession
    spark = (SparkSession.builder.master(f"local[{nucleos}]").appName(nombre)
             .config("spark.driver.memory", memoria)
             .config("spark.ui.showConsoleProgress", "false")
             # lote, celular y clase quedan como texto (sin esto "2026-11-04" se vuelve fecha)
             .config("spark.sql.sources.partitionColumnTypeInference.enabled", "false")
             # pocas filas por lote de Arrow: cada fila trae la foto completa (~3 MB)
             .config("spark.sql.execution.arrow.maxRecordsPerBatch", str(filas_por_lote))
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def enviar_codigo(spark, dir_modelo):
    """Manda este archivo y preprocesamiento.py a los procesos de Python de Spark."""
    sc = spark.sparkContext
    sc.addPyFile(os.path.abspath(__file__))
    sc.addPyFile(os.path.abspath(os.path.join(dir_modelo, "preprocesamiento.py")))


def leer_lago(spark, lago):
    """Lectura con binaryFile: lote, celular y clase salen de las carpetas clave=valor."""
    return (spark.read.format("binaryFile")
            .option("pathGlobFilter", "*.jpg")
            .option("recursiveFileLookup", "false")
            .load(lago))


def ruta_local(path):
    """file:/repo/datos/lago/... -> /repo/datos/lago/..."""
    return unquote(urlparse(path).path) if path.startswith("file:") else path


# ---------------------------------------------------------------- clasificador
class Clasificador:
    """Carga el modelo TorchScript una vez y clasifica bytes de fotos con el preprocesamiento exportado."""

    def __init__(self, dir_modelo, hilos=1):
        import cv2, torch
        from torchvision import transforms
        t0 = time.perf_counter()
        if hilos:
            torch.set_num_threads(hilos)
            cv2.setNumThreads(hilos)
        if os.path.abspath(dir_modelo) not in sys.path:
            sys.path.insert(0, os.path.abspath(dir_modelo))
        import preprocesamiento                       # modelo/preprocesamiento.py, sin cambios
        self.pre = preprocesamiento
        self.info = leer_info(dir_modelo)
        self.clases = list(self.info["clases"])
        self.net = torch.jit.load(os.path.join(dir_modelo, self.info.get("archivo", "resnet18_limon.pt")),
                                  map_location="cpu").eval()
        # igual que sin_aug de la celda 10.1 del notebook
        self.tf = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),
                                      transforms.Normalize(self.info["media"], self.info["desv"])])
        self.torch = torch
        self.carga_ms = 1000 * (time.perf_counter() - t0)

    def tensor(self, datos):
        bgr, _ = self.pre.cargar(io.BytesIO(datos))
        m = self.pre.segmentar(bgr)
        im, mk = self.pre.recortar(self.pre.corregir(bgr, m), m)
        return self.tf(np.ascontiguousarray(im[:, :, ::-1]))       # BGR -> RGB

    def predecir(self, lista_bytes):
        """Devuelve un arreglo (n, 3) de probabilidades; fila con NaN si la foto no se pudo leer."""
        tensores, ok = [], []
        for b in lista_bytes:
            try:
                tensores.append(self.tensor(b)); ok.append(True)
            except Exception:                                     # foto dañada: no detiene el lote
                ok.append(False)
        prob = np.full((len(lista_bytes), len(self.clases)), np.nan)
        if tensores:
            with self.torch.inference_mode():
                p = self.torch.softmax(self.net(self.torch.stack(tensores)), 1).numpy().astype(np.float64)
            prob[np.array(ok)] = p
        return prob


def tabla_resultado(pdf, prob, datos, clases, particion, carga_ms):
    out = pd.DataFrame({"path": pdf["path"].values,
                        "lote": pdf["lote"].values, "celular": pdf["celular"].values, "clase": pdf["clase"].values})
    out["md5"] = [hashlib.md5(b).hexdigest() for b in datos]
    out["prediccion"] = [clases[i] if not np.isnan(fila).any() else "error"
                         for i, fila in zip(np.nan_to_num(prob).argmax(1), prob)]
    for j, c in enumerate(clases):
        out[f"prob_{c}"] = prob[:, j]
    out["particion"] = particion
    out["carga_modelo_ms"] = carga_ms
    return out


def funcion_particion(dir_modelo, hilos=1, lote=16):
    """Función para mapInPandas. Spark la llama UNA vez por partición: ahí se carga el modelo,
    y luego se procesan todos los lotes de filas de esa partición.
    Si la tabla trae la columna content (binaryFile) usa esos bytes; si no, lee la foto desde path."""
    dir_modelo = os.path.abspath(dir_modelo)

    def inferir(iterador):
        from pyspark import TaskContext
        clf = Clasificador(dir_modelo, hilos)                     # una sola carga por partición
        pid = TaskContext.get().partitionId()
        for pdf in iterador:
            for i in range(0, len(pdf), lote):
                parte = pdf.iloc[i:i + lote]
                if "content" in parte.columns:
                    datos = [bytes(b) for b in parte["content"]]
                else:
                    datos = [open(ruta_local(p), "rb").read() for p in parte["path"]]
                yield tabla_resultado(parte, clf.predecir(datos), datos, clf.clases, pid, clf.carga_ms)

    return inferir


def secuencial(rutas_locales, filas, dir_modelo, hilos=1, lote=1):
    """Bucle de Python puro (sin Spark) con las mismas funciones. Incluye la carga del modelo.
    lote=1: de a una foto (como en la corrida del 16-nov). lote=16: por lotes, igual que la UDF de Spark."""
    t0 = time.perf_counter()
    clf = Clasificador(dir_modelo, hilos)
    partes = []
    filas = filas.reset_index(drop=True)
    for i in range(0, len(rutas_locales), lote):
        datos = [open(r, "rb").read() for r in rutas_locales[i:i + lote]]
        pdf = filas.iloc[i:i + lote]
        partes.append(tabla_resultado(pdf, clf.predecir(datos), datos, clf.clases, -1, clf.carga_ms))
    return pd.concat(partes, ignore_index=True), time.perf_counter() - t0
