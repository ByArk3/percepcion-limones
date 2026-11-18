"""Registro del modelo en MLflow Model Registry (secciones 8.2 y 8.3 del informe).

Qué hace:
  1. Hace una copia de respaldo del mlflow.db antes de tocarlo.
  2. Agrega tags a las corridas de entrenamiento del notebook (integrante, commit del notebook, entorno)
     que el notebook no guardó (lo pide la sección 8.2). No cambia parámetros ni métricas.
  3. Lee las metas (tabla_metas.csv que deja el notebook) y la latencia p95 del contenedor
     (resultados_servicio/latencia_resumen.txt que deja pruebas/medir_latencia.py).
  4. Crea una corrida "registro_modelo" con el modelo (.pt, modelo_info.json, preprocesamiento.py),
     la tabla de metas, la latencia y las figuras como artefactos.
  5. Registra el modelo como "limon-madurez" (una versión por cada SHA-256 distinto) y le pone el alias:
       produccion  si cumple las 7 metas Y la latencia p95 < 150 ms en el contenedor
       candidato   si no (y escribe qué falló)
  6. Escribe un resumen en resultados_mlflow/registro_modelo.txt.

Las cifras salen de los archivos de entrada; si falta alguno, el criterio queda PENDIENTE y el modelo va como
candidato.

Uso (PowerShell, en la carpeta del repositorio; con la imagen limon-spark:1.0, que ya trae mlflow 3.18):
  docker run --rm -v "${PWD}:/repo" -v "D:\\Proyectos\\percepcion-limones\\resultados_colab_16nov:/colab" -w /repo `
      limon-spark:1.0 python src/registrar_modelo.py --mlflow-db /colab/mlflow.db `
      --tabla-metas /colab/tablas/tabla_metas.csv --figuras /colab/figuras --commit 44b7ddf
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

NOMBRE_MODELO = "limon-madurez"
EXPERIMENTO_ENTRENAMIENTO = "limon_madurez"
EXPERIMENTO_REGISTRO = "limon_registro"
P95_MAX_MS = 150.0                       # criterio de promoción acordado (CLAUDE.md, Tarea D)
FIGURAS_UTILES = ("fig_matriz_confusion.png", "fig_curvas.png", "fig_ejemplos_prediccion.png", "fig_gradcam.png")


def leer_metas(ruta):
    """Devuelve lista de (métrica, meta, valor, cumple) desde tabla_metas.csv del notebook."""
    filas = []
    with open(ruta, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            cumple = r["¿Cumple?"].strip().lower() in ("sí", "si", "yes", "true")
            filas.append((r["Métrica"], r["Meta"], r["Valor (media ± desv.)"], cumple))
    return filas


def leer_latencia(ruta):
    """Devuelve (mediana, p95) de ida y vuelta desde latencia_resumen.txt, o (None, None)."""
    if not ruta or not os.path.isfile(ruta):
        return None, None
    texto = open(ruta, encoding="utf-8").read()
    m = re.search(r"Ida y vuelta \(ms\): mediana ([\d.]+) \| p95 ([\d.]+)", texto)
    return (float(m.group(1)), float(m.group(2))) if m else (None, None)


def commit_de(archivo):
    try:
        return subprocess.run(["git", "log", "-1", "--format=%h", "--", archivo], capture_output=True,
                              text=True, check=True).stdout.strip() or "desconocido"
    except (OSError, subprocess.CalledProcessError):
        return "desconocido"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mlflow-db", required=True, help="mlflow.db de la corrida del notebook (resultados.zip)")
    ap.add_argument("--modelo", default="modelo", help="carpeta con resnet18_limon.pt, modelo_info.json, preprocesamiento.py")
    ap.add_argument("--tabla-metas", required=True, help="tablas/tabla_metas.csv de la misma corrida")
    ap.add_argument("--latencia", default="resultados_servicio/latencia_resumen.txt")
    ap.add_argument("--figuras", default="", help="carpeta figuras/ de la misma corrida (opcional)")
    ap.add_argument("--integrante", default="Franco Marquez", help="quién corrió el entrenamiento")
    ap.add_argument("--entorno", default="Google Colab, GPU Tesla T4")
    ap.add_argument("--notebook", default="notebooks/Grupo3_limones_drive_Colab.ipynb")
    ap.add_argument("--commit", default="", help="commit del notebook usado (si no hay git, p. ej. dentro de Docker)")
    ap.add_argument("--salida", default="resultados_mlflow")
    a = ap.parse_args()

    import mlflow
    from mlflow.tracking import MlflowClient

    db = Path(a.mlflow_db).resolve()
    if not db.is_file():
        sys.exit(f"No existe {db}")
    respaldo = db.with_name(db.stem + "_respaldo_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S") + db.suffix)
    shutil.copy2(db, respaldo)
    os.makedirs(a.salida, exist_ok=True)
    lineas = []

    def log(*x):
        t = " ".join(str(v) for v in x)
        print(t, flush=True)
        lineas.append(t)

    log("Registro del modelo en MLflow ·", dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log(f"mlflow.db: {db}  (respaldo: {respaldo.name})  | mlflow {mlflow.__version__}")
    mlflow.set_tracking_uri("sqlite:///" + db.as_posix())
    cli = MlflowClient()

    # ---------------------------------------------------------------- 1. tags de las corridas del notebook
    exp = cli.get_experiment_by_name(EXPERIMENTO_ENTRENAMIENTO)
    if exp is None:
        sys.exit(f"El mlflow.db no tiene el experimento '{EXPERIMENTO_ENTRENAMIENTO}'.")
    corridas = [r for r in cli.search_runs([exp.experiment_id], order_by=["attributes.start_time ASC"])
                if r.data.tags.get("mlflow.runName", "").startswith("resnet18_")]     # solo entrenamiento
    commit = a.commit or commit_de(a.notebook)
    log(f"\n1. Corridas de entrenamiento en '{EXPERIMENTO_ENTRENAMIENTO}': {len(corridas)}")
    for r in corridas:
        nombre = r.data.tags.get("mlflow.runName", r.info.run_id[:8])
        for k, v in {"integrante": a.integrante, "commit_notebook": commit, "entorno": a.entorno,
                     "notebook": os.path.basename(a.notebook)}.items():
            cli.set_tag(r.info.run_id, k, v)
        m = r.data.metrics
        dur = (r.info.end_time - r.info.start_time) / 1000 if r.info.end_time else float("nan")
        log(f"   {nombre:26s} f1_macro={m.get('test_f1_macro', float('nan')):.4f} "
            f"recall_maduro={m.get('test_recall_maduro', float('nan')):.4f} duración={dur:.1f} s")
    log(f"   Tags agregados: integrante='{a.integrante}', commit_notebook={commit}, entorno='{a.entorno}'")

    # ---------------------------------------------------------------- 2. criterio de promoción
    metas = leer_metas(a.tabla_metas)
    mediana, p95 = leer_latencia(a.latencia)
    fallas = [f"{n} {v} (meta {m})" for n, m, v, ok in metas if not ok]
    log(f"\n2. Metas (de {a.tabla_metas}): {sum(ok for *_, ok in metas)} de {len(metas)} cumplidas")
    for n, m, v, ok in metas:
        log(f"   {'Sí' if ok else 'NO'}  {n:28s} {m:8s} {v}")
    if p95 is None:
        log(f"   Latencia p95 en el contenedor: PENDIENTE (no existe o no se pudo leer {a.latencia})")
        fallas.append("latencia p95 PENDIENTE (falta correr pruebas/medir_latencia.py)")
    else:
        log(f"   Latencia en el contenedor: mediana {mediana} ms, p95 {p95} ms (meta p95 < {P95_MAX_MS:.0f} ms)")
        if p95 >= P95_MAX_MS:
            fallas.append(f"latencia p95 {p95} ms (meta < {P95_MAX_MS:.0f} ms)")
    alias = "produccion" if not fallas else "candidato"
    log(f"   Decisión: alias '{alias}'" + ("" if not fallas else " porque: " + "; ".join(fallas)))

    # ---------------------------------------------------------------- 3. corrida de registro con artefactos
    info = json.load(open(os.path.join(a.modelo, "modelo_info.json"), encoding="utf-8"))
    pt = os.path.join(a.modelo, info.get("archivo", "resnet18_limon.pt"))
    sha = hashlib.sha256(open(pt, "rb").read()).hexdigest()
    if sha != info.get("sha256"):
        sys.exit(f"El SHA-256 de {pt} ({sha}) no coincide con modelo_info.json ({info.get('sha256')}).")
    if cli.get_experiment_by_name(EXPERIMENTO_REGISTRO) is None:
        artefactos = (db.parent / "mlruns_registro").resolve()
        artefactos.mkdir(exist_ok=True)
        cli.create_experiment(EXPERIMENTO_REGISTRO, artifact_location=artefactos.as_uri())
    mlflow.set_experiment(EXPERIMENTO_REGISTRO)
    semilla = str(info.get("semilla"))
    origen = next((r for r in corridas if r.data.tags.get("mlflow.runName") == f"resnet18_corregida_s{semilla}"), None)
    with mlflow.start_run(run_name=f"registro_modelo_{sha[:8]}") as run:
        mlflow.log_params({"sha256": sha, "semilla": semilla, "epocas": info.get("epocas"),
                           "n_train": info.get("n_train"), "n_test": info.get("n_test"),
                           "corrida_entrenamiento": origen.info.run_id if origen else "no encontrada"})
        for k, v in (info.get("metricas_prueba") or {}).items():
            mlflow.log_metric(f"semilla{semilla}_{k}", float(v))
        if p95 is not None:
            mlflow.log_metrics({"latencia_mediana_ms": mediana, "latencia_p95_ms": p95})
        mlflow.set_tags({"integrante": a.integrante, "commit_notebook": commit, "entorno": a.entorno,
                         "alias_decidido": alias})
        for f in ("modelo_info.json", "preprocesamiento.py", os.path.basename(pt)):
            mlflow.log_artifact(os.path.join(a.modelo, f), "modelo")
        mlflow.log_artifact(a.tabla_metas, "evaluacion")
        if p95 is not None:
            mlflow.log_artifact(a.latencia, "evaluacion")
        if a.figuras and os.path.isdir(a.figuras):
            for f in FIGURAS_UTILES:
                if os.path.isfile(os.path.join(a.figuras, f)):
                    mlflow.log_artifact(os.path.join(a.figuras, f), "figuras")
        uri_modelo = run.info.artifact_uri.rstrip("/") + "/modelo"
        id_registro = run.info.run_id

    # ---------------------------------------------------------------- 4. Model Registry
    try:
        cli.get_registered_model(NOMBRE_MODELO)
    except Exception:
        cli.create_registered_model(NOMBRE_MODELO, description=(
            "ResNet18 TorchScript que clasifica la madurez del limón sutil (verde, pintón, maduro). Grupo 3, UPAO."))
    existentes = [v for v in cli.search_model_versions(f"name='{NOMBRE_MODELO}'") if v.tags.get("sha256") == sha]
    if existentes:
        version = existentes[0].version
        log(f"\n3. Ya existía la versión {version} con este SHA-256: se actualizan sus tags y su alias.")
    else:
        mv = cli.create_model_version(NOMBRE_MODELO, source=uri_modelo, run_id=id_registro,
                                      description=f"Semilla {semilla}, {info.get('epocas')} épocas, {a.entorno}.")
        version = mv.version
        log(f"\n3. Registrada la versión {version} de '{NOMBRE_MODELO}'.")
    etiquetas = {"sha256": sha, "metas_cumplidas": f"{sum(ok for *_, ok in metas)}/{len(metas)}",
                 "latencia_p95_ms": "PENDIENTE" if p95 is None else str(p95),
                 "criterio": f"7 metas y p95 < {P95_MAX_MS:.0f} ms en el contenedor",
                 "motivo": "cumple todo" if not fallas else "; ".join(fallas)}
    for k, v in etiquetas.items():
        cli.set_model_version_tag(NOMBRE_MODELO, version, k, v)
    otro = "candidato" if alias == "produccion" else "produccion"
    try:                                                  # un alias apunta a una sola versión
        if cli.get_model_version_by_alias(NOMBRE_MODELO, otro).version == version:
            cli.delete_registered_model_alias(NOMBRE_MODELO, otro)
    except Exception:
        pass
    cli.set_registered_model_alias(NOMBRE_MODELO, alias, version)
    log(f"   Alias '{alias}' -> versión {version} | sha256 {sha[:16]}...")
    log(f"   Fuente del modelo: {uri_modelo}")
    log("\nCómo sabe el servicio qué versión usa: el backend carga modelo/ y devuelve su sha256 en /health y en")
    log("cada /predict; cada fila de la tabla predicciones de PostgreSQL guarda ese sha256. Ese mismo sha256 es")
    log(f"el tag de la versión {version} de '{NOMBRE_MODELO}' en el registro, así que cada predicción se puede")
    log("rastrear hasta la versión registrada.")
    log(f"\nVer en la interfaz: mlflow ui --backend-store-uri sqlite:///{db.as_posix()}  y abrir http://127.0.0.1:5000")
    open(os.path.join(a.salida, "registro_modelo.txt"), "w", encoding="utf-8").write("\n".join(lineas) + "\n")
    print(f"\nResumen guardado en {a.salida}/registro_modelo.txt")


if __name__ == "__main__":
    main()
