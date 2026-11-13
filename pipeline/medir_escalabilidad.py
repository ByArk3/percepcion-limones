"""Mediciones de escalabilidad del pipeline (sección 7.4 del informe) en UNA sola sesión de Spark.

  a) Speedup   : volumen fijo, repartition(k) con k = 1, 2, 4, 8 + fila "Secuencial (Python puro)".
  b) Volumen   : 100, 500, 1000 (y 5000 con --con-5000) fotos, ms por foto. Criterio H2: variación < 15 %.
  c) Cruce     : bucle secuencial vs Spark por volumen; si Spark no gana, se estima el punto de cruce.
  d) Parquet vs CSV: tamaño en disco y tiempo de leer 1 columna.
  e) Registro en MLflow (parámetros, métricas y artefactos).

Cómo se mide:
  - Siempre con toPandas()/collect() o escritura, nunca con count() (Spark quitaría la UDF del plan).
  - Si no hay suficientes fotos, se repiten las rutas hasta llegar al volumen.
  - La lista de fotos sale de binaryFile; para estas mediciones la UDF lee los bytes desde la ruta,
    así repartition(k) solo baraja rutas y no megabytes de imágenes.
  - Antes de medir se hace un calentamiento que no se cuenta.

Requiere haber corrido antes job_batch.py (usa el lago y el Parquet que deja).
Uso:
    python pipeline/medir_escalabilidad.py --modelo modelo --salida resultados_spark
"""
import argparse
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inferencia as inf                                    # noqa: E402

CRITERIO_H2 = 15.0          # % de variación del tiempo por foto entre 100 y 1000 fotos (lógica acordada)


def repetir(base, n):
    """Tabla de n filas repitiendo las fotos del lago si hace falta."""
    veces = int(np.ceil(n / len(base)))
    return pd.concat([base] * veces, ignore_index=True).iloc[:n].reset_index(drop=True)


def tamano_carpeta(ruta):
    total = 0
    for raiz, _, archivos in os.walk(ruta):
        for f in archivos:
            if not f.startswith((".", "_")):        # sin _SUCCESS ni .crc
                total += os.path.getsize(os.path.join(raiz, f))
    return total


def ajuste_lineal(n, t):
    """t = costo_fijo + costo_por_foto * n (mínimos cuadrados)."""
    b, a = np.polyfit(np.asarray(n, float), np.asarray(t, float), 1)
    return a, b


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lago", default="datos/lago")
    ap.add_argument("--lote", default="2026-11-04")
    ap.add_argument("--modelo", default="modelo")
    ap.add_argument("--salida", default="resultados_spark")
    ap.add_argument("--parquet", default="datos/resultados_spark", help="Parquet que dejó job_batch.py")
    ap.add_argument("--nucleos", type=int, default=os.cpu_count())
    ap.add_argument("--hilos-torch", type=int, default=1)
    ap.add_argument("--lote-inferencia", type=int, default=16)
    ap.add_argument("--memoria", default="4g")
    ap.add_argument("--volumen-speedup", type=int, default=1000)
    ap.add_argument("--particiones", default="1,2,4,8")
    ap.add_argument("--volumenes", default="100,500,1000")
    ap.add_argument("--con-5000", action="store_true", help="agrega el volumen de 5000 fotos (tarda bastante)")
    ap.add_argument("--mlflow", default="", help="URI de MLflow; por defecto sqlite:///<salida>/mlflow.db")
    a = ap.parse_args()

    tablas, figuras = os.path.join(a.salida, "tablas"), os.path.join(a.salida, "figuras")
    os.makedirs(tablas, exist_ok=True); os.makedirs(figuras, exist_ok=True)
    log = inf.Bitacora(os.path.join(a.salida, "log_medir_escalabilidad.txt"))
    log.titulo("Mediciones de escalabilidad (sección 7.4) · Grupo 3")
    log("Comando:", " ".join(sys.argv))
    ent = inf.describir_entorno(log, a.modelo)
    N = a.nucleos
    log(f"{'nucleos Spark':20s}: local[{N}]")
    log(f"{'hilos torch/tarea':20s}: {a.hilos_torch} (también en el bucle secuencial)")

    spark = inf.crear_spark(N, a.memoria, a.lote_inferencia, nombre="limon_escalabilidad")
    inf.enviar_codigo(spark, a.modelo)
    f = inf.funcion_particion(a.modelo, a.hilos_torch, a.lote_inferencia)

    base = (inf.leer_lago(spark, a.lago).filter(f"lote = '{a.lote}'")
            .select("path", "lote", "celular", "clase").toPandas().sort_values("path").reset_index(drop=True))
    log(f"Fotos distintas en el lago: {len(base)}")
    if base.empty:
        log("No hay fotos. Corre primero job_batch.py."); sys.exit(1)

    medidas = {}                                   # (modo, n, k) -> segundos

    def spark_t(n, k):
        if ("spark", n, k) not in medidas:
            sdf = spark.createDataFrame(repetir(base, n)).repartition(k)
            t0 = time.perf_counter()
            res = sdf.mapInPandas(f, inf.ESQUEMA_SALIDA).toPandas()
            medidas[("spark", n, k)] = time.perf_counter() - t0
            assert len(res) == n, f"Spark devolvió {len(res)} filas, se esperaban {n}"
            log(f"  Spark n={n:5d} k={k}: {medidas[('spark', n, k)]:8.2f} s "
                f"({1000 * medidas[('spark', n, k)] / n:6.1f} ms/foto), modelo cargado "
                f"{res.particion.nunique()} veces")
        return medidas[("spark", n, k)]

    def seq_t(n):
        if ("seq", n, 1) not in medidas:
            tab = repetir(base, n)
            _, t = inf.secuencial([inf.ruta_local(p) for p in tab.path], tab, a.modelo, a.hilos_torch)
            medidas[("seq", n, 1)] = t
            log(f"  Secuencial n={n:5d}: {t:8.2f} s ({1000 * t / n:6.1f} ms/foto)")
        return medidas[("seq", n, 1)]

    t0 = time.perf_counter()
    spark.createDataFrame(base.iloc[:2]).mapInPandas(f, inf.ESQUEMA_SALIDA).toPandas()
    log(f"Calentamiento (2 fotos, no se cuenta): {time.perf_counter() - t0:.2f} s")

    # ------------------------------------------------------------ a) speedup
    log.titulo(f"a) Speedup con volumen fijo de {a.volumen_speedup} fotos")
    V = a.volumen_speedup
    ks = [int(x) for x in a.particiones.split(",")]
    t_seq = seq_t(V)
    filas = [dict(configuracion="Secuencial (Python puro)", particiones=1, tiempo_s=t_seq, speedup=1.0,
                  ideal=1, eficiencia=1.0)]
    for k in ks:
        t = spark_t(V, k)
        ideal = min(k, N)
        filas.append(dict(configuracion=f"Spark repartition({k})", particiones=k, tiempo_s=t,
                          speedup=t_seq / t, ideal=ideal, eficiencia=(t_seq / t) / ideal))
    t_sp = pd.DataFrame(filas).round(3)
    t_sp.to_csv(os.path.join(tablas, "spark_speedup.csv"), index=False)
    log(t_sp.to_string(index=False))
    log("Speedup = tiempo secuencial / tiempo Spark. Ideal = mín(k, núcleos). La eficiencia es solo un dato "
        "(el criterio de eficiencia >= 0.8 fue anulado; ver CLAUDE.md, lógica acordada 5).")

    # ------------------------------------------------------------ b) volumen
    vols = [int(x) for x in a.volumenes.split(",")] + ([5000] if a.con_5000 else [])
    log.titulo(f"b) Escalamiento de volumen con repartition({N})")
    filas = []
    for n in vols:
        t = spark_t(n, N)
        filas.append(dict(fotos=n, tiempo_s=t, ms_por_foto=1000 * t / n))
    t_vol = pd.DataFrame(filas)
    ms = t_vol.set_index("fotos").ms_por_foto
    if 100 in ms.index and 1000 in ms.index:
        var = 100 * (ms[1000] - ms[100]) / ms[100]
        cumple_h2 = abs(var) < CRITERIO_H2
        t_vol["variacion_vs_100_pct"] = 100 * (t_vol.ms_por_foto - ms[100]) / ms[100]
        log(f"Variación del tiempo por foto de 100 a 1000 fotos: {var:+.1f} % "
            f"(criterio H2: |variación| < {CRITERIO_H2:.0f} %) -> {'CUMPLE' if cumple_h2 else 'NO CUMPLE'}")
    else:
        var, cumple_h2 = float("nan"), None
        log("No se midieron 100 y 1000 fotos: no se puede evaluar H2.")
    t_vol = t_vol.round(3)
    t_vol.to_csv(os.path.join(tablas, "spark_volumen.csv"), index=False)
    log(t_vol.to_string(index=False))

    # ------------------------------------------------------------ c) punto de cruce
    log.titulo("c) Punto de cruce: secuencial vs Spark")
    filas = []
    for n in vols:
        filas.append(dict(fotos=n, secuencial_s=seq_t(n), spark_s=spark_t(n, N)))
    t_cr = pd.DataFrame(filas)
    t_cr["gana"] = np.where(t_cr.spark_s < t_cr.secuencial_s, "Spark", "Secuencial")
    a_seq, b_seq = ajuste_lineal(t_cr.fotos, t_cr.secuencial_s)
    a_sp, b_sp = ajuste_lineal(t_cr.fotos, t_cr.spark_s)
    log(f"Ajuste lineal secuencial: {a_seq:.2f} s fijos + {1000 * b_seq:.1f} ms/foto")
    log(f"Ajuste lineal Spark     : {a_sp:.2f} s fijos + {1000 * b_sp:.1f} ms/foto")
    if b_sp < b_seq and a_sp > a_seq:
        n_cruce = (a_sp - a_seq) / (b_seq - b_sp)
        texto_cruce = (f"Spark tiene más costo fijo ({a_sp:.2f} s vs {a_seq:.2f} s) pero menos costo por foto; "
                       f"según el ajuste empieza a ganar desde unas {n_cruce:.0f} fotos.")
    elif b_sp < b_seq:
        n_cruce = 0.0
        texto_cruce = "Spark gana en todo el rango medido (menos costo fijo y menos costo por foto)."
    else:
        n_cruce = float("inf")
        texto_cruce = ("Spark no baja el costo por foto frente al bucle secuencial: con estos núcleos no hay "
                       "punto de cruce (el paralelismo no compensa el costo de Spark; ver ley de Amdahl).")
    log(texto_cruce)
    t_cr["n_cruce_estimado"] = n_cruce
    t_cr.round(3).to_csv(os.path.join(tablas, "spark_cruce.csv"), index=False)
    log(t_cr.round(3).to_string(index=False))

    # ------------------------------------------------------------ d) Parquet vs CSV
    log.titulo("d) Parquet vs CSV")
    ruta_pq = os.path.join(a.parquet, f"lote={a.lote}")
    t_pc = None
    if os.path.isdir(ruta_pq):
        ruta_csv = os.path.join(a.parquet + "_csv", f"lote={a.lote}")
        pq = spark.read.parquet(ruta_pq)
        shutil.rmtree(ruta_csv, ignore_errors=True)
        pq.write.mode("overwrite").partitionBy("prediccion").option("header", True).csv(ruta_csv)
        esquema = pq.select([c for c in pq.columns if c != "prediccion"]).schema

        def leer_columna(lector, reps=5):
            ts = []
            for _ in range(reps):
                t0 = time.perf_counter()
                lector().agg({"prob_maduro": "sum"}).collect()
                ts.append(time.perf_counter() - t0)
            return float(np.median(ts))

        t_leer_pq = leer_columna(lambda: spark.read.parquet(ruta_pq).select("prob_maduro"))
        t_leer_csv = leer_columna(lambda: spark.read.schema(esquema).option("header", True)
                                  .csv(ruta_csv).select("prob_maduro"))
        filas_pq = pq.count()   # aquí count() solo cuenta filas del archivo, no se mide tiempo con él
        t_pc = pd.DataFrame([
            dict(formato="Parquet", filas=filas_pq, bytes=tamano_carpeta(ruta_pq), leer_1_columna_s=t_leer_pq),
            dict(formato="CSV", filas=filas_pq, bytes=tamano_carpeta(ruta_csv), leer_1_columna_s=t_leer_csv)])
        t_pc["kb"] = (t_pc.bytes / 1024).round(1)
        t_pc = t_pc.round(4)
        t_pc.to_csv(os.path.join(tablas, "parquet_vs_csv.csv"), index=False)
        log(t_pc.to_string(index=False))
        log("Lectura de 1 columna: mediana de 5 repeticiones de select(prob_maduro).agg(sum).collect().")
    else:
        log(f"No existe {ruta_pq}: corre primero job_batch.py. Parquet vs CSV = PENDIENTE.")

    # ------------------------------------------------------------ figuras
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sp = t_sp[t_sp.configuracion != "Secuencial (Python puro)"]
    kmax = max(ks)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    xs = np.arange(1, kmax + 1)
    ax.plot(xs, np.minimum(xs, N), "--", color="gray", label="ideal = mín(k, núcleos)")
    ax.plot(sp.particiones, sp.speedup, "o-", color="tab:green", label="Spark medido")
    ax.axhline(1, color="tab:red", lw=1, ls=":", label="secuencial (Python puro)")
    ax.axvline(N, color="black", lw=1, alpha=0.6, label=f"núcleos reales = {N}")
    ax.set_xlabel("Particiones (k)"); ax.set_ylabel("Speedup vs secuencial")
    ax.set_title(f"Speedup con {V} fotos"); ax.set_xticks(ks); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(figuras, "spark_speedup.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(t_vol.fotos, t_vol.ms_por_foto, "o-", color="tab:green", label=f"Spark repartition({N})")
    if 100 in ms.index:
        ax.axhspan(ms[100] * (1 - CRITERIO_H2 / 100), ms[100] * (1 + CRITERIO_H2 / 100), color="tab:green",
                   alpha=0.12, label="±15 % del valor con 100 fotos (H2)")
    ax.set_xscale("log"); ax.set_xticks(t_vol.fotos); ax.set_xticklabels(t_vol.fotos)
    ax.set_xlabel("Fotos en el lote"); ax.set_ylabel("ms por foto"); ax.set_title("Escalamiento de volumen")
    ax.set_ylim(bottom=0); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(figuras, "spark_volumen.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    nmax = max(vols) if not np.isfinite(n_cruce) or n_cruce <= max(vols) else n_cruce * 1.2
    xx = np.linspace(0, nmax, 100)
    ax.plot(t_cr.fotos, t_cr.secuencial_s, "o", color="tab:red", label="secuencial (medido)")
    ax.plot(t_cr.fotos, t_cr.spark_s, "s", color="tab:green", label=f"Spark {N} particiones (medido)")
    ax.plot(xx, a_seq + b_seq * xx, "--", color="tab:red", alpha=0.6, label="ajuste lineal")
    ax.plot(xx, a_sp + b_sp * xx, "--", color="tab:green", alpha=0.6)
    if np.isfinite(n_cruce) and n_cruce > 0:
        ax.axvline(n_cruce, color="black", lw=1, ls=":", label=f"cruce estimado ~{n_cruce:.0f} fotos")
    ax.set_xlabel("Fotos en el lote"); ax.set_ylabel("Tiempo total (s)"); ax.set_title("Punto de cruce")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(figuras, "spark_cruce.png"), dpi=150); plt.close(fig)
    log("\nFiguras guardadas en", figuras)

    # ------------------------------------------------------------ e) MLflow
    log.titulo("e) Registro en MLflow")
    uri = a.mlflow or "sqlite:///" + os.path.abspath(os.path.join(a.salida, "mlflow.db"))
    try:
        import mlflow
        mlflow.set_tracking_uri(uri)
        nombre_exp = "spark_pipeline"
        if mlflow.get_experiment_by_name(nombre_exp) is None:
            mlflow.create_experiment(nombre_exp, artifact_location="file://" + os.path.abspath(
                os.path.join(os.path.dirname(uri.replace("sqlite:///", "")) or ".", "mlruns_spark")))
        mlflow.set_experiment(nombre_exp)
        with mlflow.start_run(run_name=f"spark_lote_{a.lote}"):
            mlflow.log_params(dict(nucleos=N, os_cpu_count=ent["os_cpu_count"],
                                   cpus_limite_docker=ent["cpus_limite_docker"], hilos_torch=a.hilos_torch,
                                   lote_inferencia=a.lote_inferencia, volumen_speedup=V, particiones=a.particiones,
                                   volumenes=",".join(map(str, vols)), fotos_distintas=len(base), lote=a.lote,
                                   spark=spark.version, sha256_modelo=ent["sha256_modelo"][:16]))
            mlflow.log_metric("tiempo_secuencial_s", t_seq)
            for r in sp.itertuples():
                mlflow.log_metric("tiempo_spark_s", r.tiempo_s, step=int(r.particiones))
                mlflow.log_metric("speedup", r.speedup, step=int(r.particiones))
                mlflow.log_metric("eficiencia", r.eficiencia, step=int(r.particiones))
            for r in t_vol.itertuples():
                mlflow.log_metric("ms_por_foto", r.ms_por_foto, step=int(r.fotos))
            if np.isfinite(var):
                mlflow.log_metric("variacion_ms_por_foto_pct", var)
            if np.isfinite(n_cruce):
                mlflow.log_metric("n_cruce_estimado", n_cruce)
            mlflow.log_artifacts(tablas, "tablas")
            mlflow.log_artifacts(figuras, "figuras")
        log(f"Corrida registrada en {uri} (experimento '{nombre_exp}').")
    except Exception as e:                       # MLflow no debe tumbar las mediciones
        log(f"No se pudo registrar en MLflow ({uri}): {type(e).__name__}: {str(e)[:300]}")

    spark.stop()
    log.titulo("Resumen")
    log(f"Núcleos: os.cpu_count()={ent['os_cpu_count']}, límite Docker={ent['cpus_limite_docker']}, Spark local[{N}]")
    mejor = sp.loc[sp.speedup.idxmax()]
    log(f"Mejor speedup: {mejor.speedup:.2f}x con {int(mejor.particiones)} particiones (ideal {int(mejor.ideal)})")
    log(f"H2 (variación 100 -> 1000 fotos): {var:+.1f} % -> "
        + ("CUMPLE" if cumple_h2 else "NO CUMPLE" if cumple_h2 is not None else "PENDIENTE"))
    log(texto_cruce)
    log("Archivos para subir: " + a.salida + "/ (tablas, figuras, logs). No subir datos/ ni mlflow.db.")


if __name__ == "__main__":
    main()
