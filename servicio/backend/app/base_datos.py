"""Acceso a PostgreSQL: crear tablas, guardar predicciones y cargar los lotes de Spark."""
import glob
import os
import time

import pandas as pd
import psycopg

URL_DB = os.environ.get("URL_DB", "postgresql://limon:limon@db:5432/limon")
INIT_SQL = os.path.join(os.path.dirname(__file__), "init.sql")


def conectar(intentos=30):
    """La base puede tardar unos segundos en aceptar conexiones al levantar todo con docker compose."""
    for i in range(intentos):
        try:
            return psycopg.connect(URL_DB, autocommit=True)
        except psycopg.OperationalError:
            if i == intentos - 1:
                raise
            time.sleep(1)


def crear_tablas():
    with conectar() as con:
        con.execute(open(INIT_SQL, encoding="utf-8").read())


def guardar(filas):
    """filas: lista de dicts con las columnas de la tabla predicciones (sin id ni fecha)."""
    if not filas:
        return
    cols = ["origen", "id_envio", "archivo", "md5", "clase", "prob_verde", "prob_pinton", "prob_maduro",
            "latencia_ms", "version_modelo", "sha256_modelo"]
    with conectar() as con, con.cursor() as cur:
        cur.executemany(f"INSERT INTO predicciones ({', '.join(cols)}) VALUES ({', '.join(['%s'] * len(cols))})",
                        [tuple(f[c] for c in cols) for f in filas])


def historial(limite=50):
    with conectar() as con, con.cursor() as cur:
        cur.execute("""SELECT id, fecha, origen, id_envio, archivo, md5, clase, prob_verde, prob_pinton,
                              prob_maduro, latencia_ms, version_modelo, sha256_modelo
                       FROM predicciones ORDER BY id DESC LIMIT %s""", (limite,))
        nombres = [d.name for d in cur.description]
        filas = [dict(zip(nombres, r)) for r in cur.fetchall()]
        cur.execute("SELECT clase, count(*) FROM predicciones GROUP BY clase")
        conteo = dict(cur.fetchall())
    for f in filas:
        f["fecha"] = f["fecha"].isoformat(timespec="seconds")
    return dict(total=sum(conteo.values()), por_clase=conteo, ultimas=filas)


def sincronizar_spark(dir_parquet):
    """Lee el Parquet que deja job_batch.py (una carpeta lote=AAAA-MM-DD por lote) y lo copia a la base.
    Si el lote ya estaba cargado, se reemplaza (se puede correr varias veces)."""
    cargados = []
    for carpeta in sorted(glob.glob(os.path.join(dir_parquet, "lote=*"))):
        lote = os.path.basename(carpeta).split("=", 1)[1]
        df = pd.read_parquet(carpeta)                          # prediccion sale de las carpetas prediccion=...
        df["prediccion"] = df["prediccion"].astype(str)
        filas = [(lote, r.path, r.celular, r.clase, r.md5, r.prediccion, r.prob_verde, r.prob_pinton, r.prob_maduro)
                 for r in df.itertuples(index=False)]
        with conectar() as con, con.cursor() as cur:
            cur.execute("DELETE FROM predicciones_spark WHERE lote = %s", (lote,))
            cur.executemany("""INSERT INTO predicciones_spark (lote, path, celular, clase_etiqueta, md5, prediccion,
                               prob_verde, prob_pinton, prob_maduro) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                            filas)
        cargados.append(dict(lote=lote, fotos=len(filas)))
    return cargados


def lotes_spark():
    with conectar() as con, con.cursor() as cur:
        cur.execute("""SELECT lote, celular, prediccion, count(*) FROM predicciones_spark
                       GROUP BY lote, celular, prediccion ORDER BY lote DESC, celular, prediccion""")
        detalle = [dict(lote=l, celular=c, prediccion=p, fotos=n) for l, c, p, n in cur.fetchall()]
        cur.execute("""SELECT lote, count(*), max(cargado) FROM predicciones_spark
                       GROUP BY lote ORDER BY lote DESC""")
        resumen = [dict(lote=l, fotos=n, cargado=t.isoformat(timespec="seconds")) for l, n, t in cur.fetchall()]
    return dict(lotes=resumen, detalle=detalle)
