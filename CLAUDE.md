# Guía para Claude Code · Proyecto "Madurez del limón sutil" (Grupo 3)

> **Cómo usar este archivo.** Va en la raíz del **repositorio de GitHub** del proyecto como `CLAUDE.md`.
> Claude Code (web) trabaja sobre ese repositorio y lo lee al empezar. Al final hay un prompt para empezar.
> Actualizado: 13-nov-2026 (versión 2: el trabajo se hace por GitHub).

---

## 0. Forma de trabajo (IMPORTANTE)

- **Claude Code trabaja en la web, sobre el repositorio de GitHub. No tiene acceso a la PC de Franco ni al dataset.**
- **Franco** hace `git pull`, construye las imágenes y corre los contenedores en **su Docker Desktop (Windows)**, montando el dataset de su PC. Luego sube al repositorio los resultados (tablas, figuras, log) o los pega en el chat.
- **No habrá zip** con fotos ni modelo. Las fotos (~560 MB) **nunca** se suben a GitHub.

| Qué | Dónde está | Quién |
|---|---|---|
| Código (`pipeline/`, `servicio/`, `src/`, Dockerfiles, README) | Repositorio | Claude Code lo escribe y lo sube |
| `CLAUDE.md`, `notebooks/`, `requirements_local.txt` | Repositorio | Franco los sube una vez |
| `modelo/modelo_info.json` y `modelo/preprocesamiento.py` (pesan poco) | Repositorio, carpeta `modelo/` | Franco |
| `resnet18_limon.pt` (~45 MB, TorchScript) | **Release** del repositorio (archivo adjunto), no en el código | Franco lo sube una vez |
| Fotos | Solo en la PC: `D:\Proyectos\percepcion-limones\dataset_limon_sutil\etiquetado_final_agregarfotos` | — |
| `docker build` / `docker run` | PC de Franco | Franco |
| Resultados reales (tablas, figuras, salida de consola) | Repositorio, carpeta `resultados_spark/` (y luego `resultados_servicio/`) | Franco los sube; Claude Code los revisa |

Reglas que salen de esto:
1. Todo script debe recibir las rutas por argumento o variable de entorno (dataset, modelo, salida). Nada de rutas fijas de Windows dentro del código.
2. Las imágenes Docker **no** llevan el dataset adentro: se monta con `-v`. El modelo se monta o se copia desde `modelo/` (Franco descarga el `.pt` de la Release a esa carpeta antes del `docker build`/`run`).
3. Para probar en la nube sin fotos reales, Claude Code usa **imágenes sintéticas** solo para comprobar que el código corre. **Las cifras del informe salen únicamente de la PC de Franco.**
4. Cada script deja un log de texto en la carpeta de resultados con fecha, versiones, núcleos y todo lo medido, para que Franco lo suba tal cual.
5. Los comandos para Franco se escriben en **PowerShell** y se documentan en el `README.md` de cada carpeta.

### Enlaces del dataset (Google Drive)
- Dataset actual (`etiquetado_final_agregarfotos`): https://drive.google.com/drive/folders/1KqT60QOlIsdWQMDNuu0GVYzsaB4-WwLy
- Copia del parcial (`etiquetado_avance1`): https://drive.google.com/drive/folders/1cdC6sah1bsUlJzsVQ73LsvLIjOQjuPKE
- El acceso debe ser **"Lector"** (estaba como "cualquiera con el enlace puede editar"). Solo sirven como referencia para `data/README.md`; el pipeline no descarga de Drive.

### Respuestas al pedido de archivos de la Tarea A (13-nov)
- Ya no se envía zip: el modelo va por la Release y `modelo/`, y las fotos se quedan en la PC (ver tabla de arriba).
- **Estado del dataset:** en el **Drive** no se ha limpiado (revisado el 04-nov, igual que el 19-oct): siguen las 16 fotos de iPhone de Leoncio en sus carpetas de clase, `v_RN10_001`–`008` duplicados y las fotos de Franco con código `FRANCO`. En la **copia de la PC**: PENDIENTE confirmar si Franco ya movió las de Leoncio a `descartadas_leoncio`. El pipeline debe ignorar carpetas que no empiecen con `verde_`, `pinton_` o `maduro_` y reportar cuántas fotos hay por celular y clase.
- **CPU, memoria y recursos de Docker de la PC:** PENDIENTE (Franco pega la salida de `Get-CimInstance Win32_Processor`, `docker info` y Settings > Resources cuando se le pida). No suponer núcleos: medirlos dentro del contenedor con `os.cpu_count()`.
- **Nombre del lote:** `lote=2026-11-04` es la fecha real de la copia del dataset de la PC, así que se usa ese. Cuando se descargue una copia nueva, se crea un lote nuevo con su fecha (`lote=AAAA-MM-DD`).
- **Modelo correcto:** antes de subirlo, Franco revisa que `modelo_info.json` diga `"epocas": 30`. Si dice otro número, usar la copia `resultados_04nov_30epocas\modelo\`. Comprobar la huella con `sha256` de `modelo_info.json`.

---

## 1. Contexto

- **Curso:** Percepción Computacional (ISIA-111), UPAO, NRC 6013. Docente: Ms. Armando Caballero Alvarado.
- **Grupo 3:** Franco Márquez (coordina, trabaja en esta PC), Joel "Ark", Guinno, Leoncio.
- **Problema:** clasificar la madurez del **limón sutil** (*Citrus aurantifolia*) en **verde, pintón y maduro** a partir de una foto. Solo limón sutil (otras especies fuera de alcance). El sistema **no** detecta defectos.
- **Regla de clases (NTP 011.006 + criterio del equipo):** % de superficie amarilla < 10 % = verde, 10–30 % = pintón, > 30 % = maduro.
- **Entregas:** el parcial (secciones 1–6) ya se entregó (informe 19/20). Falta la **entrega final (semana 14)**: secciones 7–12 + todo lo anterior corregido, repositorio, video y exposición.
- **Plantilla oficial:** `docs/ESTRUCTURA_DEL_PROYECTO_PERCEPCION_COMPUTACIONAL.docx` (léela antes de cada sección).

## 2. Reglas de trabajo (obligatorias)

1. Español simple, como estudiante. Sin emojis en el código ni en los textos del informe.
2. Haz **solo** lo que se pide en cada tarea, lo más simple posible. Si algo no está claro, pregunta antes.
3. **No inventes datos.** Todo número sale de una ejecución real. Si algo no se ha medido, escribe `PENDIENTE`.
4. **No cambies la lógica acordada** (sección 5 de esta guía) sin preguntar.
5. **No reentrenes el modelo fuera del notebook.** Spark y la API usan el modelo ya exportado (en el repositorio: `modelo/resnet18_limon.pt`, descargado de la Release; en la PC venía de `resultados\modelo\`).
6. Antes de modificar el informe Word, muestra el texto en el chat para que Franco lo apruebe.
7. Reporta los resultados tal como salen (aunque no cumplan una meta) y explica por qué. La rúbrica premia la honestidad y castiga cifras infladas.
8. Revisa cada entregable antes de darlo por terminado (ejecutar, abrir el archivo, comparar cifras).
9. No borres ni sobrescribas `resultados/` sin antes copiarlo (la celda 2.2 del notebook borra `resultados/` al iniciar).

## 3. Entorno de la PC

| Cosa | Valor |
|---|---|
| Sistema | Windows, VS Code |
| Carpeta del proyecto | `D:\Proyectos\percepcion-limones` |
| Python | 3.11.9 (kernel del notebook) |
| PyTorch | 2.15.0+cpu, torchvision 0.29.2+cpu (sin GPU NVIDIA) |
| MLflow / OpenCV | 3.18.0 / 5.0.0 |
| Java | **23** instalado (Spark 4.0 necesita 17 o 21: por eso Spark va en Docker) |
| Docker | Docker Desktop funcionando (engine Linux, builder `desktop-linux`) |
| CPU | Intel64 Family 6 Model 60 (2 núcleos lógicos visibles en las pruebas de Colab; verificar con `os.cpu_count()` en la PC) |

## 4. Archivos y carpetas actuales

**En la PC de Franco** (origen de los archivos; el repositorio se clona en una subcarpeta, por ejemplo `D:\Proyectos\percepcion-limones\repo`):

```
D:\Proyectos\percepcion-limones\
├── Grupo3_limones_local.ipynb        notebook principal (versión local, misma lógica que el Colab)
├── requirements_local.txt
├── dataset_limon_sutil\              zip de Drive descomprimido (dataset vigente al 04-nov-2026)
│   ├── etiquetado_final_agregarfotos\
│   │   ├── Ark\      verde_ark\  pinton_ark\  maduro_ark\
│   │   ├── Franco\   verde_franco\ pinton_franco\ maduro_franco\
│   │   ├── Guinno\   verde_guinno\ pinton_guinno\ maduro_guinno\
│   │   └── Leoncio\  (sus 16 fotos son de iPhone: van a descartadas_leoncio\)
│   └── etiquetado_avance1\           copia congelada del parcial (marca fotos antiguas por MD5)
├── resultados\                       salida del notebook (tablas, figuras, mlflow.db, modelo\)
│   └── modelo\
│       ├── resnet18_limon.pt         ResNet18 TorchScript (~45 MB), semilla 42
│       ├── modelo_info.json          clases, media/desv, preprocesamiento, métricas, sha256
│       └── preprocesamiento.py       funciones cargar, segmentar, corregir, recortar (idénticas al notebook)
└── resultados_04nov_30epocas\        (recomendado) copia de la corrida de 30 épocas
```

- **Nombres de fotos:** `CLASE_CELULAR_NUMERO.jpg` con clase `v`/`p`/`m`. Celulares: Guinno `RN10` (001–200), Franco `R14P` (201–400, hoy todavía `FRANCO`), Leoncio `A56` (401–600), Ark `A26` (601–800).
- **Carpetas que no se usan:** cualquier subcarpeta que no empiece con `verde_`, `pinton_` o `maduro_` (por ejemplo `descartadas_leoncio`).
- **Preprocesamiento del modelo (exacto):** `bgr, _ = cargar(ruta_o_BytesIO)` → `m = segmentar(bgr)` → `im, mk = recortar(corregir(bgr, m), m)` (224×224, BGR) → `im[:, :, ::-1]` (RGB) → `ToTensor` → `Normalize(media=[0.485,0.456,0.406], desv=[0.229,0.224,0.225])`. La salida del modelo son logits; aplicar softmax. Orden de clases: `["verde", "pinton", "maduro"]`. `cargar()` acepta también un `io.BytesIO` (sirve para los bytes de Spark y de la API).

## 5. Lógica del proyecto (ACORDADA, no cambiar)

| # | Tema | Decisión |
|---|---|---|
| 1 | Métricas | Exactitud y F1 macro + precisión, recall y F1 **por clase**; media ± desv. de 3 semillas (42, 7, 123) |
| 2 | Metas (3.3) | Exactitud ≥ 0.85 · F1 macro ≥ 0.85 · Recall Maduro ≥ 0.90 · Precisión Maduro ≥ 0.90 · Falsos positivos Maduro ≤ 0.05 · F1 Verde ≥ 0.85 · F1 Pintón ≥ 0.75 |
| 3 | H1 | ResNet18 vs mejor clásico (SVM espectral): F1 macro ≥ 1 − (1 − F1_SVM)/2 y recall Maduro ≥ 0.90. Umbral **provisional** en cada avance; en la final se fija una vez en `H1_UMBRAL_FINAL` (celda 2.2) |
| 4 | Umbral de color | **H < 27 fijo** (OpenCV 0–179). El calibrado solo se muestra como alerta |
| 5 | H2 (Spark) | Criterio de eficiencia ≥ 0.8 **anulado** (hardware limitado); speedup y eficiencia solo como dato. Criterio vigente: variación del tiempo por foto **< 15 %** al pasar de 100 a 1,000 fotos. Declararlo en el informe (original 0.8, medido 0.64 en Colab) |
| 6 | Prueba | Todas las fotos (antiguas + nuevas), 70/15/15 estratificado, semilla 42. Se sella cuando el dataset esté completo |
| 7 | Desempeño desagregado (6.4) | Por celular (A26, RN10, R14P, A56) |
| 8 | Arquitectura | **Batch** (lotes de fotos por recepción/turno) |
| 9 | Objetivo de operación | API en Docker, latencia **mediana < 150 ms** por foto (y p95 reportado), en CPU |
| 10 | Informe 7.4 | Quitar la frase "1.96× en otra máquina con 2 núcleos físicos" (sin evidencia) |

## 6. Estado actual (corrida exploratoria del 04-nov-2026, PC con CPU, 30 épocas)

- Dataset: 213 fotos, 204 tras quitar duplicados (verde 96, pintón 43, maduro 65). Prueba: 31 (14/7/10). **Ojo:** esa corrida todavía incluyó las 16 fotos de iPhone de Leoncio y las de Franco con código `FRANCO`.
- ResNet18: exactitud **0.925 ± 0.019**, F1 macro **0.909 ± 0.025**, recall Maduro **0.867 ± 0.058**, precisión Maduro 0.963, FP Maduro 0.016. Latencia en notebook (CPU): 36.8 ms/foto.
- Baseline HOG+LBP+color+SVM: F1 0.701. SVM espectral (mejor clásico): F1 0.819 → umbral H1 provisional 0.910 → **H1 no se cumple** (F1 0.909 y recall 0.867).
- Metas: 6 de 7 (falla recall Maduro). Wilson recall Maduro semilla 42: 9/10, IC [0.60, 0.98].
- Entradas (semilla 42): sin corrección 0.923 = corregida 0.923; CLAHE 0.846.
- Por celular: A26 tiene 23 de 31 fotos de prueba; los demás 1–4 (no concluyente).
- Comparación CPU (PC) vs GPU (Colab 21-oct, F1 0.879 ± 0.058): la diferencia está dentro de la variación normal; **la corrida final se hace en un solo entorno** y se declara.
- Spark (sección 14 del notebook) **no arrancó** en Windows por Java 23. Además en Windows escribir Parquet exige `winutils.exe`. Decisión: **Spark en Docker**.

---

## 7. Tareas para Claude Code (en este orden)

### TAREA A · Spark en Docker (sección 7 del informe) — URGENTE (checkpoint semana 11)

**Objetivo:** pipeline batch que lee las fotos con `binaryFile`, las clasifica con la ResNet18 exportada cargada **una vez por partición** y escribe los resultados en **Parquet**, más todas las mediciones de la sección 7.4.

Archivos a crear:
```
pipeline/
├── Dockerfile               python:3.11-slim (versión fija) + openjdk-17-jre-headless + dependencias fijas
├── requirements.txt         pyspark==4.0.1, pyarrow, torch==2.15.0 (CPU), torchvision==0.29.2 (CPU), numpy, pandas,
│                            opencv-python-headless, pillow, matplotlib, mlflow==3.18.0 (versiones EXACTAS)
├── preparar_lago.py         arma el data lake clave=valor
├── job_batch.py             el job de inferencia
├── medir_escalabilidad.py   speedup, volumen, punto de cruce, Parquet vs CSV
└── README.md                cómo construir y correr
```
Nota: para instalar PyTorch **CPU** dentro de Linux (Docker) usar `--extra-index-url https://download.pytorch.org/whl/cpu`; así la imagen no baja la versión con CUDA (varios GB). La misma regla vale para `servicio/`.

1. **Data lake** (`preparar_lago.py`): desde `dataset_limon_sutil/etiquetado_final_agregarfotos` (solo subcarpetas `verde_*`, `pinton_*`, `maduro_*`) copiar a
   `datos/lago/lote=2026-11-04/celular=<CODIGO>/clase=<clase>/<archivo>.jpg`. El código sale del nombre (`v_A26_601.jpg` → `A26`); si no coincide con el patrón, `celular=SIN_CODIGO`. Spark convertirá `lote`, `celular` y `clase` en columnas (descubrimiento de particiones).
2. **Lectura:** `spark.read.format("binaryFile").option("pathGlobFilter", "*.jpg").option("recursiveFileLookup", "false").load("datos/lago")`. Mostrar `printSchema()` y el conteo por celular y clase sin decodificar imágenes.
3. **Particiones:** binaryFile con muchos archivos pequeños crea muchas particiones, y el modelo se carga **una vez por partición** (hallazgo medido en la tarea S9: con 19 particiones la carga se repetía 19 veces). Ajustar con `coalesce(NUCLEOS)` antes de la UDF y **medir** la diferencia (sin ajustar vs ajustado). Reportarlo como optimización medida (7.3).
4. **Inferencia:** `pandas_udf` de tipo iterador (`Iterator[pd.Series] -> Iterator[pd.DataFrame]` o `mapInPandas`), que carga `modelo/resnet18_limon.pt` con `torch.jit.load` una sola vez por partición y procesa los lotes. Usar exactamente el preprocesamiento de `modelo/preprocesamiento.py` (importarlo, no reescribirlo). Devolver `prediccion`, `prob_verde`, `prob_pinton`, `prob_maduro`, `md5`. No mezclar `.map()` de RDD con UDF de DataFrame.
5. **Verificación de coherencia:** para todas las fotos, las probabilidades de Spark deben coincidir con las de un bucle secuencial de Python que use las mismas funciones (diferencia máxima < 1e-5). Imprimirlo.
6. **Escritura:** `write.mode("overwrite").partitionBy("prediccion").parquet("datos/resultados_spark/lote=2026-11-04")`. Comparar contra CSV: tamaño en disco y tiempo de leer 1 columna.
7. **Mediciones (7.4)** en la **misma sesión** de Spark (no reiniciar con `local[1]`, `local[2]`: es inestable):
   - **Speedup:** volumen fijo (todas las fotos, o repetidas hasta ~1,000), `repartition(k)` con k = 1, 2, 4, 8 (incluye más particiones que núcleos). Tabla con fila **"Secuencial (Python puro)"** + filas de Spark; columnas tiempo, speedup observado, ideal = mín(k, núcleos), eficiencia. Gráfico con línea ideal y línea vertical de núcleos reales.
   - **Volumen:** 100, 500, 1,000 (y 5,000 si el tiempo lo permite) fotos (repitiendo rutas si no alcanzan), ms por foto. Verificar el criterio de H2: variación < 15 % de 100 a 1,000.
   - **Punto de cruce:** tiempos del bucle secuencial vs Spark por volumen; si Spark no gana en el rango, estimar el cruce y explicarlo (costo fijo, Amdahl). Ese resultado vale aunque Spark pierda.
   - Medir siempre con `agg()`/`collect()` o con la escritura, **nunca con `count()`** (Spark elimina la UDF del plan y el tiempo sale falso).
   - Declarar núcleos reales (`os.cpu_count()` dentro del contenedor y los CPUs asignados a Docker).
8. **MLflow:** registrar la corrida de Spark (parámetros: núcleos, particiones, volumen; métricas: tiempos, speedup; artefactos: tablas y gráficos) en el mismo `resultados/mlflow.db` o en `resultados/mlruns` montado.
9. **Salidas:** `figuras/spark_speedup.png`, `figuras/spark_volumen.png`, `figuras/spark_cruce.png`, `tablas/spark_*.csv`, `tablas/parquet_vs_csv.csv`.
10. **Ejecución en la PC de Franco** (documentar en `pipeline/README.md`, en PowerShell). El repositorio clonado y el dataset se montan por separado, por ejemplo:
    ```powershell
    cd D:\Proyectos\percepcion-limones\repo          # carpeta donde Franco clonó el repositorio
    git pull
    # descargar resnet18_limon.pt de la Release a modelo\ (solo la primera vez)
    docker build -t limon-spark:1.0 -f pipeline/Dockerfile .
    docker run --rm `
      -v "D:\Proyectos\percepcion-limones\dataset_limon_sutil\etiquetado_final_agregarfotos:/datos/fotos:ro" `
      -v "${PWD}:/repo" -w /repo `
      limon-spark:1.0 python pipeline/job_batch.py --fotos /datos/fotos --modelo modelo --salida resultados_spark
    ```
    El dataset se monta **solo lectura** (`:ro`). Todo lo que produzca el job queda en `resultados_spark/` dentro del repositorio para que Franco lo suba con `git add resultados_spark && git commit && git push`.
11. **Diagrama 7.2:** generar un diagrama (Mermaid o PNG) donde **cada caja exista en el código**: fotos de los celulares → data lake `lote=/celular=/clase=` → Spark binaryFile → pandas_udf con ResNet18 (una vez por partición) → Parquet particionado por predicción → consumidor (consulta/tablero simple o API). 

Criterio de terminado: el job corre de punta a punta en Docker, la verificación de coherencia da < 1e-5, y existen las tablas y figuras de speedup, volumen, cruce y Parquet vs CSV.

### TAREA B · Escalamiento del entrenamiento (sección 5.5, obligatoria)

Opción 2 de la plantilla (medición de cuello de botella), sin reentrenar el modelo final:
- Script `src/perfilar_entrenamiento.py` (o celda nueva en el notebook) que, con 2–3 épocas, mida por época: tiempo de carga de lotes (dataloader) vs tiempo de cómputo (forward + backward + optimizador), y uso de CPU. Probar `num_workers` 0 y 2 (en Windows usar `if __name__ == "__main__":`).
- Concluir con evidencia si está limitado por cómputo o por E/S y qué paralelismo correspondería (de datos, no de modelo: ResNet18 cabe en memoria). Incluir ajuste de tasa de aprendizaje al crecer el lote efectivo y estimación de costo.

### TAREA C · Servicio de inferencia en Docker (sección 8.1)

```
servicio/
├── Dockerfile          FROM python:3.11-slim (versión fija), PyTorch CPU, EXPOSE 8000, HEALTHCHECK
├── requirements.txt    versiones EXACTAS (fastapi, uvicorn, torch==2.15.0 CPU, numpy, opencv-python-headless, pillow, python-multipart)
├── app/main.py
└── README.md
```
- `GET /health` → `{"estado": "ok", "version_modelo": ..., "sha256": ...}`.
- `POST /predict` → recibe una foto (multipart) → `{"clase": ..., "probabilidades": {...}, "version_modelo": ..., "latencia_ms": ...}`.
- Cargar el modelo **una vez** al iniciar. Usar `preprocesamiento.py` sin cambios.
- **No** instalar TensorFlow ni nada que no se use (la plantilla lo penaliza).
- `pruebas/medir_latencia.py`: ≥ 200 llamadas al contenedor con fotos reales → **mediana y p95**. Reportar también el tamaño de la imagen Docker.
- Evidencia: un `curl` de ejemplo y su respuesta.
- `pruebas/prueba_fotos_nuevas.py`: 6 fotos que NO estén en el dataset (2 por clase, las da el equipo); imprimir probabilidades con 4 decimales; deben coincidir entre notebook/script y API.

### TAREA D · MLflow: comparación y Model Registry (8.2 y 8.3)

- Comprobar que `resultados/mlflow.db` tiene varias corridas (3 semillas + variantes de entrada). Dejar instrucciones para sacar la **captura comparando corridas** (`mlflow ui --backend-store-uri sqlite:///resultados/mlflow.db`).
- Script `src/registrar_modelo.py`: registrar la ResNet18 ganadora como `limon-madurez`, versión N. **Criterio de promoción:** cumplir las metas de la sección 5 de esta guía **y** latencia p95 < 150 ms en el contenedor. Si cumple → alias `produccion`; si no → alias `candidato` y explicar qué falló (hoy fallaría por recall Maduro 0.867). Documentar cómo el servicio sabe qué versión carga (por ejemplo, `version_modelo` y `sha256` de `modelo_info.json`).

### TAREA E · Huella computacional (9.2)

- Sumar tiempos reales de entrenamiento (notebook) y de inferencia (Spark y API) y estimar energía (kWh) con la potencia del equipo y CO₂ con un factor de emisión de la red eléctrica del Perú **con fuente citada**. Si falta un dato (modelo exacto de CPU, potencia), dejar `PENDIENTE` y preguntar.

### TAREA F · Repositorio GitHub (sección 11.1, estructura obligatoria)

```
percepcion-limones/
├── README.md                 instalación, ejecución, resultados esperados (el profesor lo va a ejecutar)
├── requirements.txt          versiones exactas
├── data/README.md            cómo obtener las fotos (NO subir el dataset)
├── notebooks/                Grupo3_limones_local.ipynb (y el de Colab)
├── src/                      datos/, modelos/, entrenar.py, evaluar.py, perfilar_entrenamiento.py, registrar_modelo.py
├── pipeline/                 job_batch.py, preparar_lago.py, medir_escalabilidad.py, Dockerfile, README.md
├── servicio/                 Dockerfile, app/main.py, requirements.txt
├── modelo/                   modelo_info.json, preprocesamiento.py (el .pt va en la Release)
├── resultados_spark/         tablas, figuras y log reales que sube Franco desde su PC
├── pruebas/                  medir_latencia.py, prueba_fotos_nuevas.py, capturas/
└── figuras/
```
- `.gitignore`: dataset, `resultados*/`, `*.zip`, `*.docx`, `mlflow.db`, `.venv`, `*.pt`.
- El modelo `.pt` (45 MB) **no** va en el repo: subirlo como **Release** de GitHub y documentar la descarga en el README.
- README con reproducibilidad (8.4): semillas, versiones exactas, comandos exactos, fuentes de no determinismo (ejemplo real: CPU vs GPU dieron F1 0.909 vs 0.879 con la misma semilla).
- Probar desde un clon limpio antes de darlo por terminado.

### TAREA G · Pendientes del notebook (antes de la corrida final)

- Celda nueva: comparar entrenando con todas las fotos vs `SOLO_NUEVAS = True` (tabla de 2 filas).
- Al final del proyecto: escribir `H1_UMBRAL_FINAL` en la celda 2.2, sellar la prueba y hacer **una** corrida final en un solo entorno (declararlo en 5.4).

---

## 8. Lo que hace el equipo (no Claude Code)

**Franco, una sola vez, para preparar el repositorio:**
1. Crear el repositorio en GitHub (por ejemplo `percepcion-limones`) y clonarlo en `D:\Proyectos\percepcion-limones\repo`.
2. Copiar al repositorio: `CLAUDE.md`, `notebooks\`, `requirements_local.txt`, y en `modelo\` los archivos `modelo_info.json` y `preprocesamiento.py` (de `resultados\modelo\`, revisando que diga `"epocas": 30`).
3. Crear `.gitignore` con `*.pt`, `dataset_limon_sutil/`, `resultados/`, `resultados_*/` excepto `resultados_spark/` y `resultados_servicio/`, `*.zip`, `*.docx`, `mlflow.db`, `.venv/`.
4. `git add . ; git commit -m "Estructura inicial" ; git push`.
5. En GitHub → Releases → crear la release `modelo-v1` y adjuntar `resnet18_limon.pt`.
6. Conectar el repositorio a Claude Code web y pegar el prompt de la sección 11.

**Cada vez que Claude Code termine una tarea:** `git pull` → seguir los comandos del README de esa carpeta → subir los resultados (`git add resultados_spark ; git commit ; git push`) o pegar la salida en el chat.

**Además:**

- Tomar fotos nuevas: mínimo 50 por clase por integrante, con su propio celular (Franco y Guinno: pintón y maduro; Leoncio: todo con su A56).
- Limpiar Drive: mover las 16 fotos de iPhone de Leoncio a `descartadas_leoncio`, borrar `v_RN10_001`–`008` (duplicados), correr el Colab de fotos para pasar `FRANCO` → `R14P`, decidir `p_A26_640` y revisar `m_A26_636` (maduro con 17 % de amarillo).
- Redactar el informe final (secciones 9 y 10, bitácora y declaración de uso de IA), grabar el video (5–8 min, sistema funcionando) y preparar la exposición (los 4 deben dominar todo).

## 9. No hacer

- No usar Java 23 para Spark ni pelear con winutils en Windows: Spark va en Docker.
- No medir Spark con `count()`.
- No reentrenar fuera del notebook ni cambiar hiperparámetros, semillas, metas o umbrales.
- No inventar cifras ni "mejorar" números; no copiar valores del ejemplo de la plantilla.
- No subir datos ni pesos al repositorio.
- No modificar el informe Word sin aprobación.

## 10. Checklist de evidencias para el informe y el video

| Evidencia | Sección |
|---|---|
| Esquema del DataFrame de binaryFile con columnas `lote`, `celular`, `clase` | 7.3 |
| Medición modelo una vez por partición (sin `coalesce` vs con `coalesce`) | 7.3 |
| Tabla y gráfico de speedup (con fila secuencial y línea de núcleos) | 7.4 |
| Tabla de volumen (ms/foto, variación 100→1,000) | 7.4 / H2 |
| Punto de cruce secuencial vs Spark | 7.4 |
| Parquet vs CSV (tamaño, lectura de 1 columna) y plan con `PartitionFilters` | 7.3 |
| Diagrama del sistema (cada caja en el código) | 7.2 |
| `docker build`/`docker run`, `curl` a `/predict` y `/health`, latencia mediana y p95, tamaño de la imagen | 8.1 |
| Captura de MLflow comparando corridas | 8.2 |
| Modelo registrado, versión, alias y criterio de promoción | 8.3 |
| README ejecutado desde un clon limpio | 8.4 / 11.1 |
| Perfil del entrenamiento (dataloader vs cómputo) | 5.5 |
| Huella computacional | 9.2 |

## 11. Prompt para empezar en Claude Code

```text
Lee CLAUDE.md completo antes de hacer nada (en especial la sección 0: forma de trabajo por GitHub).
Es mi proyecto universitario de Percepción Computacional (Grupo 3, limón sutil).
Reglas: español simple, sin emojis, haz solo lo que te pido, no inventes cifras (usa PENDIENTE), no cambies la lógica acordada,
no reentrenes el modelo y muéstrame los textos del informe antes de cambiarlos.
No tienes acceso a mi PC ni a las fotos: escribe el código en el repo, pruébalo con imágenes sintéticas y déjame
los comandos de PowerShell para que yo construya y corra las imágenes en mi Docker con el dataset real.
Empieza por la TAREA A (Spark en Docker). Primero revisa que en el repo estén modelo/modelo_info.json,
modelo/preprocesamiento.py y la Release con resnet18_limon.pt, y dime qué encontraste y tu plan en 8 líneas antes de escribir código.
```
