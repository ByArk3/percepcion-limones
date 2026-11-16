# Resultados reales para el informe final · Grupo 3

> **Para quien redacte el informe (persona o agente).** Todas las cifras de este archivo salen de ejecuciones
> reales, copiadas tal cual de las bitácoras. No redondear de más, no "mejorar" números, no copiar valores del
> ejemplo de la plantilla. Lo que no se ha medido figura como **PENDIENTE**. Reglas completas en `CLAUDE.md`.
> Actualizado: 16-nov-2026.

## Índice de fuentes

| Qué | Archivo en el repositorio |
|---|---|
| Entrenamiento y evaluación (secciones 4 a 6) | `notebooks/Grupo3_limones_drive_Colab_ejecutado_16nov.ipynb` (con salidas) |
| Modelo desplegado | `modelo/modelo_info.json` y Release `modelo-v1` |
| Spark, bitácoras completas (sección 7) | `resultados_spark/log_job_batch.txt`, `resultados_spark/log_medir_escalabilidad.txt` |
| Spark, tablas | `resultados_spark/tablas/*.csv` |
| Spark, figuras | `resultados_spark/figuras/spark_speedup.png`, `spark_volumen.png`, `spark_cruce.png` |
| Spark, plan con `PartitionFilters` | `resultados_spark/plan_partition_filters.txt` |
| Diagrama del sistema (7.2) | `pipeline/README.md` (sección 7) y `README.md` (arquitectura completa) |
| Servicio (8.1) | `servicio/README.md`; resultados reales en `resultados_servicio/` (**PENDIENTE**) |

---

## 1. Entornos de ejecución (declarar en 4.8, 5.4 y 8.4)

| Etapa | Dónde | Hardware | Software |
|---|---|---|---|
| Entrenamiento y evaluación | Google Colab, 16-nov-2026 17:17 | GPU Tesla T4 | Python 3.13.16, PyTorch 2.14.1+cu130, torchvision 0.29.1+cu130, MLflow 3.18.0, OpenCV 4.14.0 |
| Spark (inferencia por lotes) | PC de Franco, Docker Desktop (WSL2), 16-nov-2026 21:47 | Intel Core i7-4770 @ 3.40 GHz, **4 núcleos físicos / 8 lógicos**; Docker con 8 CPUs y 14.3 GB de RAM | Imagen `limon-spark:1.0`: Python 3.11.18, OpenJDK 17.0.21, PySpark 4.0.1, PyTorch 2.15.0+cpu, torchvision 0.29.2+cpu, NumPy 2.3.5, pandas 2.3.3, OpenCV 5.0.0, Pillow 12.0.0, PyArrow 20.0.0, MLflow 3.18.0 |
| Servicio (API + web + BD) | PC de Franco, Docker Desktop | igual que Spark | **PENDIENTE** de correr en la PC |

- Imagen Docker de Spark: **1.12 GB** comprimida (CONTENT SIZE) y **3.97 GB** en disco (DISK USAGE). Construcción: 694 s.
- El modelo se entrenó en GPU y la inferencia corre en CPU. Un modelo TorchScript entrenado en GPU funciona igual en CPU.

## 2. Modelo desplegado

| Campo | Valor |
|---|---|
| Arquitectura | ResNet18 (torchvision), layer4 + fc entrenadas, TorchScript |
| Semilla / épocas | 42 / 30 |
| Fotos entrenamiento / validación / prueba | 158 / 34 / 35 |
| SHA-256 | `8eec3d12ef05ec3286157fe2f8174f82b3b2d4ca237348b3576658e77dbf62ef` |
| Métricas de prueba (solo semilla 42) | exactitud 0.8857, F1 macro 0.8571, recall Maduro 0.9286, precisión Maduro 0.9286, FP Maduro 0.0476 |

## 3. Resultados del entrenamiento (Colab, 16-nov; secciones 4 a 6)

- Dataset: 236 fotos encontradas, 9 duplicados, **227 para el análisis** (verde 96, pintón 43, maduro 88).
  Partición 70/15/15 estratificada: entrenamiento 67/30/61, validación 14/7/13, prueba 15/6/14.
- **ResNet18 (media ± desv. de 3 semillas):** exactitud 0.895 ± 0.016, F1 macro 0.870 ± 0.023,
  recall Maduro 0.976 ± 0.041, precisión Maduro 0.976 ± 0.041, FP Maduro 0.016 ± 0.027.
- Por clase: F1 Verde 0.892 ± 0.036, F1 Pintón 0.743 ± 0.049, F1 Maduro 0.976 ± 0.041.
- **Metas: 6 de 7.** Falla **F1 Pintón** (0.743 < 0.75).
- Baseline HOG+LBP+color+SVM: F1 0.703. **Mejor clásico (SVM espectral): F1 0.897**, más alto que la ResNet18.
  Umbral provisional H1 = 1 − (1 − 0.897)/2 = 0.949 → **H1 no se cumple**.
- Wilson recall Maduro: semilla 42 13/14 = 0.929 [0.69, 0.99]; semillas 7 y 123 14/14 = 1.000 [0.78, 1.00].
- Por celular (prueba): A26 tiene 21 de 35 fotos; los demás 2 a 8 → no concluyente.
- Comparación de entradas (semilla 42): sin corrección F1 0.828; con corrección 0.857; corrección + CLAHE 0.857.
- **No determinismo (8.4):** con el mismo código y la misma GPU T4, la corrida del 13-nov dio recall Maduro 14/14
  en la semilla 42 y la del 16-nov 13/14; el baseline pasó de F1 0.680 a 0.703. Entre una y otra solo se renombraron
  las fotos de Franco (`IMG_...` → `m_FRANCO_...`). Causas probables: operaciones no deterministas en GPU y orden de
  lectura de archivos.
- Calidad de etiquetas: 47 fotos (79.3 % de coincidencia) no coinciden con el % de amarillo medido con H < 27;
  casi todos los pintones de Ark tienen menos de 10 % de amarillo. El umbral que mejor coincidiría sería H < 31
  (el umbral sigue fijo en 27 por la lógica acordada).

## 4. Pipeline de Spark (sección 7) · PC, 16-nov, lote `2026-11-16`

### 4.1 Data lake y lectura (7.3)

- 236 fotos copiadas al lago (501.7 MB) en `lote=2026-11-16/celular=XXX/clase=YYY/`; 9 archivos ignorados por estar
  fuera de carpetas `verde_*`, `pinton_*`, `maduro_*`.
- Esquema de `binaryFile`: `path`, `modificationTime`, `length`, `content`, `lote`, `celular`, `clase`
  (las 3 últimas salen de las carpetas `clave=valor`).
- Fotos por celular y clase leídas por Spark sin decodificar imágenes:

| Celular | Verde | Pintón | Maduro | Total |
|---|---|---|---|---|
| A26 | 63 | 34 | 51 | 148 |
| FRANCO | 23 | 0 | 23 | 46 |
| LEONCIO | 2 | 10 | 4 | 16 |
| RN10 | 16 | 0 | 10 | 26 |

  Nota: Spark procesa las 236 fotos del lote (incluye los 9 duplicados que el notebook quita) y las fotos de Franco
  todavía con código `FRANCO` (no `R14P`).

### 4.2 Carga del modelo una vez por partición (7.3)

| Configuración | Particiones | Cargas del modelo | Carga total (s) | Tiempo (s) | ms/foto |
|---|---|---|---|---|---|
| Sin ajustar (binaryFile por defecto) | 12 | 12 | 17.780 | 32.406 | 137.3 |
| `coalesce(8)` | 8 | 8 | 21.969 | 32.278 | 136.8 |

- Cada configuración se midió 2 veces, alternadas, y se tomó la menor. La primera corrida sin ajustar dio 45.87 s solo
  por ir primero (arranque); la segunda de coalesce dio 34.21 s → variación entre repeticiones de ~6 %.
- **Conclusión honesta:** `coalesce` reduce las cargas del modelo de 12 a 8, pero con 236 fotos el tiempo total no
  cambia de forma apreciable (32.41 vs 32.28 s, dentro de la variación). Cada carga tarda ~2 s y se hacen en paralelo.

### 4.3 Coherencia (Spark vs bucle secuencial de Python)

- 236 fotos comparadas, diferencia máxima de probabilidad **1.19e-07** (tolerancia 1e-5) → **cumple**.
- Misma predicción en 236/236; MD5 iguales.
- Tiempo del bucle secuencial en el job: 86.50 s (366.5 ms/foto) vs Spark 32.28 s → ~2.7× con 236 fotos.

### 4.4 Escritura y consumidor (7.3)

- Parquet particionado por `prediccion` escrito en 5.33 s.
- El plan de `prediccion = 'maduro'` muestra
  `PartitionFilters: [isnotnull(lote), isnotnull(prediccion), (lote = 2026-11-16), (prediccion = maduro)]`.
- Consulta del consumidor (fotos por celular y predicción; **no es evaluación**, el lote incluye fotos de entrenamiento):

| Celular | Maduro | Pintón | Verde |
|---|---|---|---|
| A26 | 46 | 41 | 61 |
| FRANCO | 22 | 1 | 23 |
| LEONCIO | 2 | 12 | 2 |
| RN10 | 10 | 2 | 14 |

### 4.5 Speedup (7.4) · 1,000 fotos (rutas repetidas), Spark `local[8]`, 1 hilo de PyTorch por tarea

| Configuración | k | Tiempo (s) | Speedup vs secuencial | Ideal = mín(k, 8) | Eficiencia | Speedup vs Spark k=1 * |
|---|---|---|---|---|---|---|
| Secuencial (Python puro) | 1 | 362.991 | 1.000 | 1 | 1.000 | — |
| Spark `repartition(1)` | 1 | 321.394 | 1.129 | 1 | 1.129 | 1.00 |
| Spark `repartition(2)` | 2 | 188.444 | 1.926 | 2 | 0.963 | 1.71 |
| Spark `repartition(4)` | 4 | 125.927 | 2.883 | 4 | 0.721 | 2.55 |
| Spark `repartition(8)` | 8 | 112.586 | 3.224 | 8 | 0.403 | 2.85 |

\* Columna calculada aquí con los tiempos de la tabla (321.394 / tiempo de cada fila); no está en el CSV.

Cómo leerlo:

1. **Spark con 1 partición ganó al secuencial (1.13×).** Hay una diferencia de método: el bucle secuencial
   clasifica de a **1 foto**, y Spark en **lotes de 16** dentro de la UDF (`--lote-inferencia 16`). Parte de esa
   ventaja viene de procesar por lotes, no del paralelismo. Para aislar el paralelismo, usar la columna
   "vs Spark k=1": 1.71× con 2, 2.55× con 4 y 2.85× con 8.
2. **De 4 a 8 particiones casi no mejora** (125.9 → 112.6 s). La PC tiene **4 núcleos físicos**; los 8 lógicos
   son hyperthreading y comparten las unidades de cálculo. Con 8 particiones la eficiencia es 0.403 frente al ideal
   de 8 lógicos, pero 3.224 / 4 = 0.81 frente a los 4 físicos.
3. Lo que queda sin paralelizar (lectura de fotos, carga del modelo en cada partición, planificación de Spark) explica
   que el speedup se aleje del ideal (ley de Amdahl).
4. El criterio de eficiencia ≥ 0.8 está **anulado** (lógica acordada 5): la eficiencia se reporta solo como dato.

### 4.6 Volumen y H2 (7.4) · `repartition(8)`

| Fotos | Tiempo (s) | ms/foto | Variación vs 100 |
|---|---|---|---|
| 100 | 15.708 | 157.1 | 0.0 % |
| 500 | 53.310 | 106.6 | −32.1 % |
| 1000 | 112.586 | 112.6 | −28.3 % |

- **H2 (variación del tiempo por foto de 100 a 1,000 fotos < 15 %): −28.3 % → NO CUMPLE**, evaluado como
  |variación| < 15 %.
- La variación es **negativa**: con más fotos, cada foto sale **más barata**. Con 100 fotos el costo fijo
  (8 cargas del modelo de ~2 s, arranque de tareas) pesa mucho; con 500 o más se reparte. **De 500 a 1,000 la
  variación es +5.6 %** (106.6 → 112.6 ms/foto), dentro del 15 %: desde 500 fotos el tiempo por foto es estable.
- **DECISIÓN PENDIENTE DEL EQUIPO:** el CLAUDE.md dice "variación < 15 %" sin aclarar si es en valor absoluto.
  Con valor absoluto no cumple (−28.3 %); si solo se penaliza que el tiempo por foto **suba**, cumpliría.
  No cambiar el criterio sin acordarlo; en el informe explicar la causa (costo fijo) y el dato de 500 → 1,000.
- No se midió 5,000 fotos (opción `-Con5000`). La plantilla sugiere 5,000 y 10,000: **PENDIENTE** si se quiere.

### 4.7 Punto de cruce (7.4)

| Fotos | Secuencial (s) | Spark k=8 (s) | Gana |
|---|---|---|---|
| 100 | 37.268 | 15.708 | Spark |
| 500 | 162.896 | 53.310 | Spark |
| 1000 | 362.991 | 112.586 | Spark |

- **Spark ganó en todo el rango medido (100 a 1,000 fotos).**
- Ajuste lineal: secuencial −6.14 s + 363.5 ms/foto; Spark 2.89 s + 108.1 ms/foto. El "cruce en ~35 fotos" que da
  el script es una **extrapolación** por debajo del rango medido; además el costo fijo negativo del secuencial no
  tiene sentido físico (es ruido del ajuste: 372.7 ms/foto con 100 y 325.8 con 500). En el informe decir que el
  cruce está por debajo de 100 fotos y no se midió con precisión.

### 4.8 Parquet vs CSV (7.3) · 236 filas de resultados

| Formato | Bytes | KB | Leer 1 columna (s, mediana de 5) |
|---|---|---|---|
| Parquet | 62,752 | 61.3 | 0.2980 |
| CSV | 38,937 | 38.0 | 0.2353 |

- **Con 236 filas, Parquet salió más grande y más lento que CSV.** Cada archivo Parquet trae metadatos (esquema,
  estadísticas, pie de archivo) que pesan más que los datos cuando hay tan pocas filas, y está repartido en 3 carpetas
  (una por predicción). Las ventajas de Parquet (compresión por columna, leer solo una columna, `PartitionFilters`)
  aparecen con volúmenes grandes. Reportarlo así; la plantilla valora más el análisis honesto que un número inflado.

### 4.9 MLflow (Spark)

- La corrida se registró en `resultados_spark/mlflow.db`, experimento `spark_pipeline` (parámetros: núcleos,
  particiones, volúmenes, SHA-256; métricas: tiempos, speedup, eficiencia, ms/foto, variación H2; artefactos: tablas y
  figuras). El `mlflow.db` y `mlruns_spark/` no se suben a GitHub (`.gitignore`).
- Los avisos "Failed to import Git" de MLflow son inofensivos: la imagen no tiene `git` instalado.

### 4.10 Comparación con la corrida de Spark en Colab (sección 14 del notebook, solo referencia)

Colab (2 núcleos, pipeline viejo de descriptores con RDD, medido con `count()`): speedup máximo 1.25 con 500 fotos,
variación 100 → 1,000 de −9.2 %. No es comparable con la tabla de arriba (otro pipeline, otro hardware); sirve solo
para explicar por qué se anuló el criterio de eficiencia.

---

## 5. Servicio (sección 8) · PENDIENTE de correr en la PC

Código listo y probado en la nube con fotos sintéticas (ver `servicio/README.md`). Falta en la PC:

| Evidencia | Cómo | Estado |
|---|---|---|
| Tamaño de las imágenes `limon-backend:1.0`, `limon-frontend:1.0`, `postgres:17.6-alpine` | `docker images` | PENDIENTE |
| `curl` a `/health` y `/predict` con su respuesta | `servicio/README.md`, evidencia 2 | PENDIENTE |
| Latencia mediana y p95 (200 llamadas, en el contenedor) | `pruebas/medir_latencia.py` | PENDIENTE |
| 6 fotos nuevas (2 por clase), API vs script | `pruebas/prueba_fotos_nuevas.py` | PENDIENTE (faltan las fotos del equipo) |
| Capturas de la página web | navegador en http://localhost:8080 | PENDIENTE |
| MLflow: captura comparando corridas, tags, registro `limon-madurez` | Tarea D | PENDIENTE |

Criterio de promoción acordado (Tarea D): cumplir las 7 metas **y** p95 < 150 ms en el contenedor →
alias `produccion`; si no, `candidato`. Hoy ya se sabe que falla F1 Pintón (0.743 < 0.75) → saldría **candidato**.
