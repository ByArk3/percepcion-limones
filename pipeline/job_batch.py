"""Job batch de Spark: clasifica todas las fotos de un lote y escribe el resultado en Parquet.

Pasos (sección 7 del informe):
  1. Arma el data lake lote=/celular=/clase= (si se pasa --fotos).
  2. Lee el lago con binaryFile y muestra el esquema y el conteo por celular y clase.
  3. Mide el efecto de las particiones: sin ajustar vs coalesce(núcleos) (el modelo se carga una vez por partición).
  4. Inferencia con mapInPandas + ResNet18 TorchScript + preprocesamiento.py del modelo.
  5. Verificación de coherencia contra un bucle secuencial de Python (diferencia máxima < 1e-5).
  6. Escribe Parquet particionado por predicción y hace una consulta de ejemplo (plan con PartitionFilters).

Uso (dentro del contenedor, desde la raíz del repositorio):
    python pipeline/job_batch.py --fotos /datos/fotos --modelo modelo --salida resultados_spark
"""
import argparse
import contextlib
import io
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inferencia as inf                                    # noqa: E402
from preparar_lago import armar_lago, CLASES                # noqa: E402

TOLERANCIA = 1e-5


def plan_de(df):
    """Texto del plan físico (lo que imprime df.explain())."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        df.explain()
    return buf.getvalue()


def esquema_de(df):
    return df._jdf.schema().treeString()


def correr(df, f, etiqueta, log):
    """Ejecuta la inferencia y la mide con collect() (nunca con count())."""
    t0 = time.perf_counter()
    res = df.mapInPandas(f, inf.ESQUEMA_SALIDA).toPandas()
    t = time.perf_counter() - t0
    cargas = res.groupby("particion").carga_modelo_ms.first()
    log(f"{etiqueta}: {df.rdd.getNumPartitions()} particiones, {len(res)} fotos, {t:.2f} s "
        f"({1000 * t / max(len(res), 1):.1f} ms/foto) | modelo cargado {len(cargas)} veces, "
        f"{cargas.sum() / 1000:.2f} s en total de carga")
    return res, t, cargas


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fotos", help="carpeta etiquetado_final_agregarfotos; si se omite, se usa el lago ya armado")
    ap.add_argument("--lago", default="datos/lago")
    ap.add_argument("--lote", default="2026-11-04")
    ap.add_argument("--modelo", default="modelo", help="carpeta con resnet18_limon.pt, modelo_info.json, preprocesamiento.py")
    ap.add_argument("--salida", default="resultados_spark", help="tablas, figuras y bitácora (se suben al repositorio)")
    ap.add_argument("--parquet", default="datos/resultados_spark", help="carpeta del Parquet (no se sube)")
    ap.add_argument("--nucleos", type=int, default=os.cpu_count())
    ap.add_argument("--hilos-torch", type=int, default=1, help="hilos de PyTorch por tarea (1 = el paralelismo lo da Spark)")
    ap.add_argument("--lote-inferencia", type=int, default=16, help="fotos por lote dentro de la UDF")
    ap.add_argument("--memoria", default="4g", help="spark.driver.memory")
    a = ap.parse_args()

    tablas = os.path.join(a.salida, "tablas")
    os.makedirs(tablas, exist_ok=True)
    log = inf.Bitacora(os.path.join(a.salida, "log_job_batch.txt"))
    log.titulo("Job batch de Spark · madurez del limón sutil · Grupo 3")
    log("Comando:", " ".join(sys.argv))
    inf.describir_entorno(log, a.modelo)
    log(f"{'nucleos Spark':20s}: local[{a.nucleos}]")
    log(f"{'hilos torch/tarea':20s}: {a.hilos_torch}")
    log(f"{'lote inferencia':20s}: {a.lote_inferencia}")

    # 1. data lake
    if a.fotos:
        log.titulo("1. Data lake (preparar_lago.py)")
        tab, ign = armar_lago(a.fotos, a.lago, a.lote, imprimir=log)
        tab.groupby(["celular", "clase"]).size().rename("fotos").reset_index().to_csv(
            os.path.join(tablas, "lago_conteo.csv"), index=False)
        ign.to_csv(os.path.join(tablas, "lago_ignoradas.csv"), index=False)

    # 2. lectura con binaryFile
    log.titulo("2. Lectura con binaryFile")
    spark = inf.crear_spark(a.nucleos, a.memoria, a.lote_inferencia)
    inf.enviar_codigo(spark, a.modelo)
    log("Spark", spark.version)
    ruta_lote = os.path.join(a.lago, f"lote={a.lote}")
    df = inf.leer_lago(spark, a.lago).filter(f"lote = '{a.lote}'")
    log("Esquema del DataFrame (printSchema):")
    log(esquema_de(df))
    # solo columnas de partición: Spark no abre las fotos para contar
    conteo = df.groupBy("celular", "clase").count().toPandas()
    tabla_conteo = conteo.pivot(index="celular", columns="clase", values="count").fillna(0).astype(int)
    tabla_conteo = tabla_conteo.reindex(columns=[c for c in CLASES if c in tabla_conteo.columns])
    tabla_conteo["total"] = tabla_conteo.sum(1)
    log("Fotos por celular y clase (leído por Spark, sin decodificar imágenes):")
    log(tabla_conteo.to_string())
    tabla_conteo.to_csv(os.path.join(tablas, "spark_conteo_celular_clase.csv"))
    n_fotos = int(tabla_conteo.total.sum())
    if n_fotos == 0:
        log(f"No hay fotos en {ruta_lote}. Revisar --fotos o --lago.")
        sys.exit(1)

    # 3 y 4. particiones + inferencia
    log.titulo("3-4. Inferencia con mapInPandas (modelo una vez por partición)")
    f = inf.funcion_particion(a.modelo, a.hilos_torch, a.lote_inferencia)
    t0 = time.perf_counter()
    df.limit(1).mapInPandas(f, inf.ESQUEMA_SALIDA).toPandas()      # calentamiento (arranque de Python en Spark)
    log(f"Calentamiento (1 foto, no se cuenta): {time.perf_counter() - t0:.2f} s")
    df_aj = df.coalesce(a.nucleos)
    # se alternan las dos configuraciones para que ninguna salga favorecida por ir primero
    res_def, t_def, c_def = correr(df, f, "Sin ajustar        ", log)
    res_aj, t_aj, c_aj = correr(df_aj, f, f"coalesce({a.nucleos})        ", log)
    res_def, t2, _ = correr(df, f, "Sin ajustar (2.a)  ", log)
    res_aj, t3, _ = correr(df_aj, f, f"coalesce({a.nucleos}) (2.a)  ", log)
    t_def, t_aj = min(t_def, t2), min(t_aj, t3)
    log(f"Se toma el menor de las 2 mediciones: sin ajustar {t_def:.2f} s, coalesce {t_aj:.2f} s")
    part = pd.DataFrame([
        dict(configuracion="sin ajustar (binaryFile por defecto)", particiones=df.rdd.getNumPartitions(),
             cargas_modelo=len(c_def), carga_total_s=round(c_def.sum() / 1000, 3),
             tiempo_s=round(t_def, 3), ms_por_foto=round(1000 * t_def / len(res_def), 1)),
        dict(configuracion=f"coalesce({a.nucleos})", particiones=df_aj.rdd.getNumPartitions(),
             cargas_modelo=len(c_aj), carga_total_s=round(c_aj.sum() / 1000, 3),
             tiempo_s=round(t_aj, 3), ms_por_foto=round(1000 * t_aj / len(res_aj), 1))])
    part.to_csv(os.path.join(tablas, "spark_particiones.csv"), index=False)
    log(part.to_string(index=False))
    if df.rdd.getNumPartitions() <= a.nucleos:
        log(f"Nota: binaryFile ya creó {df.rdd.getNumPartitions()} particiones (<= {a.nucleos} núcleos), "
            "así que coalesce no puede reducirlas más; las dos filas usan las mismas particiones.")
    errores = int((res_aj.prediccion == "error").sum())
    log(f"Fotos que no se pudieron leer: {errores}")

    # 5. coherencia contra el bucle secuencial
    log.titulo("5. Verificación de coherencia (Spark vs bucle secuencial de Python)")
    res_aj = res_aj.sort_values("path").reset_index(drop=True)
    rutas = [inf.ruta_local(p) for p in res_aj.path]
    seq, t_seq = inf.secuencial(rutas, res_aj[["path", "lote", "celular", "clase"]], a.modelo, a.hilos_torch)
    cols = [f"prob_{c}" for c in CLASES]
    validas = (res_aj.prediccion != "error") & (seq.prediccion != "error")
    dif = np.abs(res_aj.loc[validas, cols].values - seq.loc[validas, cols].values)
    dif_max = float(dif.max()) if dif.size else float("nan")
    md5_ok = bool((res_aj.md5.values == seq.md5.values).all())
    pred_ok = int((res_aj.prediccion.values == seq.prediccion.values).sum())
    cumple = bool(dif_max < TOLERANCIA)
    log(f"Fotos comparadas: {int(validas.sum())} | diferencia máxima de probabilidad: {dif_max:.2e} "
        f"(tolerancia {TOLERANCIA:.0e}) -> {'CUMPLE' if cumple else 'NO CUMPLE'}")
    log(f"Misma predicción: {pred_ok}/{len(res_aj)} | MD5 iguales: {'sí' if md5_ok else 'NO'}")
    log(f"Tiempo del bucle secuencial (dato): {t_seq:.2f} s ({1000 * t_seq / len(seq):.1f} ms/foto)")
    pd.DataFrame([dict(fotos=int(validas.sum()), diferencia_maxima=dif_max, tolerancia=TOLERANCIA, cumple=cumple,
                       misma_prediccion=pred_ok, md5_iguales=md5_ok, tiempo_secuencial_s=round(t_seq, 3),
                       tiempo_spark_s=round(t_aj, 3))]).to_csv(os.path.join(tablas, "spark_coherencia.csv"), index=False)

    # 6. escritura en Parquet + consulta de ejemplo
    log.titulo("6. Escritura en Parquet particionado por predicción")
    salida_pq = os.path.join(a.parquet, f"lote={a.lote}")
    # lote no se guarda como columna: ya va en la carpeta lote=...
    pdf = res_aj.drop(columns=["lote", "particion", "carga_modelo_ms"])
    pdf["path"] = [os.path.relpath(r, a.lago) for r in rutas]
    sdf = spark.createDataFrame(pdf)
    t0 = time.perf_counter()
    sdf.write.mode("overwrite").partitionBy("prediccion").parquet(salida_pq)
    log(f"Parquet escrito en {salida_pq} en {time.perf_counter() - t0:.2f} s")
    pdf.assign(lote=a.lote).to_csv(os.path.join(tablas, "spark_predicciones.csv"), index=False)

    log("\nConsulta de ejemplo (consumidor): fotos por celular y predicción, leyendo el Parquet")
    leido = spark.read.parquet(a.parquet)
    consulta = (leido.filter(f"lote = '{a.lote}'").groupBy("celular", "prediccion").count().toPandas()
                .pivot(index="celular", columns="prediccion", values="count").fillna(0).astype(int))
    log(consulta.to_string())
    consulta.to_csv(os.path.join(tablas, "spark_consulta_celular_prediccion.csv"))
    log("Nota: esto NO es una métrica de evaluación (el lote incluye fotos de entrenamiento).")

    maduros = leido.filter(f"lote = '{a.lote}' AND prediccion = 'maduro'")
    plan = plan_de(maduros)
    open(os.path.join(a.salida, "plan_partition_filters.txt"), "w", encoding="utf-8").write(plan)
    log("\nPlan físico de la consulta prediccion = 'maduro' (debe mostrar PartitionFilters):")
    log(plan)
    log("PartitionFilters en el plan:", "sí" if "PartitionFilters: [" in plan and "maduro" in plan else "NO")

    spark.stop()
    log.titulo("Fin del job batch")
    log("Siguiente paso: python pipeline/medir_escalabilidad.py (mediciones de la sección 7.4)")


if __name__ == "__main__":
    main()
