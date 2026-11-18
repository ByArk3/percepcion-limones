# Pendientes · Grupo 3 (actualizado 18-nov-2026)

Lo que **ya está hecho** y probado: entrenamiento final (Colab, 16-nov), modelo en `modelo/` y Release `modelo-v1`,
pipeline de Spark con resultados reales (`resultados_spark/`), servicio web + API + PostgreSQL funcionando en la PC,
y todo el código de `src/` probado en la nube (con fotos sintéticas y con el `mlflow.db` real del 16-nov).

Comandos en **PowerShell**, desde `D:\Proyectos\percepcion-limones\repo`, siempre después de `git pull`.
Al terminar cada bloque: `git add <carpeta>`, `git commit -m "..."` y `git push`.

---

## A. Decisiones del equipo (antes de lo demás)

1. **¿Se limpia el dataset y se reentrena?** (fotos de iPhone de Leoncio, `FRANCO` → `R14P`, duplicados, revisar
   `p_A26_640` y `m_A26_636`). Si **sí**: hay que volver a correr el notebook en Colab y TODO lo de abajo cambia.
   Si **no**: la corrida del 16-nov queda como final y la limpieza pendiente se declara como limitación (sección 9).
2. **Si la corrida del 16-nov es la final:** escribir `H1_UMBRAL_FINAL = 0.949` en la celda 2.2 del notebook
   (no hace falta volver a correrlo; se declara en 5.4) y decirlo en el informe.
3. **H2:** ¿"variación < 15 %" es en valor absoluto? Hoy está así y da −28.3 % → no cumple
   (de 500 a 1,000 fotos la variación es +5.6 %). Decidir y explicarlo en el informe sin cambiar los números.

## B. En la PC (servicio levantado con `docker compose up -d`)

### B1. Evidencias de la sección 8.1

```powershell
docker images limon-backend:1.0
docker images limon-frontend:1.0
docker images postgres:17.6-alpine
curl.exe http://localhost:8000/health
curl.exe -F "archivo=@D:\Proyectos\percepcion-limones\dataset_limon_sutil\etiquetado_final_agregarfotos\Ark\maduro_ark\m_A26_601.jpg" http://localhost:8000/predict
```

Copiar la salida de esos comandos en `resultados_servicio\evidencias_8_1.txt` (o mandarla en el chat).

```powershell
docker compose run --rm `
  -v "D:\Proyectos\percepcion-limones\dataset_limon_sutil\etiquetado_final_agregarfotos:/fotos:ro" `
  backend python /pruebas/medir_latencia.py --url http://backend:8000 --fotos /fotos --llamadas 200
```

Deja `resultados_servicio/latencia.csv` y `latencia_resumen.txt` (mediana y p95; meta: mediana < 150 ms).
Anotar cuántos minutos tardó: es el dato de la API para la huella (C3).

### B2. Fotos nuevas (necesita 6 fotos del equipo)

Guardar 6 fotos que **no** estén en el dataset (2 verdes, 2 pintonas, 2 maduras), con nombres `v_...`, `p_...`, `m_...`,
en `D:\Proyectos\percepcion-limones\fotos_nuevas\`:

```powershell
docker compose run --rm `
  -v "D:\Proyectos\percepcion-limones\fotos_nuevas:/fotos_nuevas:ro" `
  backend python /pruebas/prueba_fotos_nuevas.py --url http://backend:8000 --fotos /fotos_nuevas
```

### B3. Capturas para el informe y el video (en `pruebas/capturas/`)

- http://localhost:8080 : resultado de una foto, de un zip, pestaña Historial y pestaña Lotes de Spark
  (botón "Cargar resultados de Spark").
- http://localhost:8000/docs : documentación automática de la API.

Subir: `git add resultados_servicio pruebas/capturas`.

### B4. MLflow: registro del modelo y captura comparando corridas (8.2 y 8.3)

Necesita la carpeta `resultados_colab_16nov` (el `resultados.zip` del 16-nov descomprimido; si no está:
`Expand-Archive "$env:USERPROFILE\Downloads\resultados.zip" D:\Proyectos\percepcion-limones\resultados_colab_16nov`).
Hacerlo **después** de B1, para que tome la latencia p95 real.

```powershell
docker run --rm -v "${PWD}:/repo" -v "D:\Proyectos\percepcion-limones\resultados_colab_16nov:/colab" -w /repo `
  limon-spark:1.0 python src/registrar_modelo.py --mlflow-db /colab/mlflow.db `
  --tabla-metas /colab/tablas/tabla_metas.csv --figuras /colab/figuras --commit 44b7ddf `
  --latencia resultados_servicio/latencia_resumen.txt
```

- Hace una copia de respaldo del `mlflow.db`, agrega tags (integrante, commit, entorno) a las 5 corridas de
  entrenamiento, registra `limon-madurez` versión 1 y le pone el alias. Hoy saldrá **candidato** (F1 Pintón 0.743 < 0.75).
- Deja `resultados_mlflow/registro_modelo.txt` → subirlo.

Abrir la interfaz de MLflow para las capturas:

```powershell
docker run --rm -p 5000:5000 -v "D:\Proyectos\percepcion-limones\resultados_colab_16nov:/colab" `
  limon-spark:1.0 mlflow ui --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:////colab/mlflow.db
```

En http://localhost:5000:
1. Experimento `limon_madurez` → marcar las 5 corridas `resnet18_*` → **Compare** → captura (gráfico de `f1_val`
   por época y tabla de métricas `test_*`). Esto explica por qué se eligió la configuración final (sección 8.2).
2. **Models** → `limon-madurez` → versión 1 → captura con el alias `candidato` y los tags (sección 8.3).
Cerrar con `Ctrl + C`.

### B5. Prueba desde un clon limpio (8.4 / 11.1)

```powershell
cd D:\Proyectos
git clone https://github.com/ByArk3/percepcion-limones.git prueba_clon
cd prueba_clon
Invoke-WebRequest -Uri https://github.com/ByArk3/percepcion-limones/releases/download/modelo-v1/resnet18_limon.pt -OutFile modelo\resnet18_limon.pt
copy .env.ejemplo .env
docker compose -p prueba up -d --build
```

Si la página abre en http://localhost:8080 y analiza una foto, el README funciona. Apagar con
`docker compose -p prueba down` y borrar la carpeta `prueba_clon`. (Si el servicio de `repo` está prendido,
apagarlo antes con `docker compose down` en `repo`, porque usan los mismos puertos).

## C. En Colab (GPU T4)

### C1. Perfil del entrenamiento (sección 5.5, obligatoria)

En una celda nueva al final del notebook ya ejecutado (las fotos quedan en `/content/dataset`):

```python
!git clone https://github.com/ByArk3/percepcion-limones.git /content/repo
%cd /content/repo
!python -m src.perfilar_entrenamiento --fotos /content/dataset --salida /content/resultados_perfil --epocas 3
```

Descargar `/content/resultados_perfil/perfil_entrenamiento.txt` y `.csv`, guardarlos en `resultados_perfil/` del
repositorio y subirlos. (También se puede correr en la PC con CPU; declarar dónde se corrió.)

### C2. Tarea G: todas las fotos vs solo las nuevas

La forma más simple usa lo que el notebook ya tiene:

1. Abrir `notebooks/Grupo3_limones_drive_Colab.ipynb` en Colab (GPU T4).
2. En la celda 2.1 marcar la casilla **SOLO_NUEVAS** (pasa a `True`).
3. **Ejecutar todas**. Al final baja `resultados.zip`: renombrarlo `resultados_solo_nuevas.zip` (no confundirlo con el del 16-nov).
4. Mandar el `.ipynb` con salidas o subir `tablas/tabla_6_2_comparacion.csv` y `tablas/tabla_metas.csv` a
   `resultados_perfil/solo_nuevas/`.

La fila "todas las fotos" es la corrida del 16-nov (F1 macro 0.870 ± 0.023). Con eso se arma la tabla de 2 filas.
Ojo: esta corrida NO reemplaza al modelo final (no subir su `modelo/` ni su `.pt`).

### C3. Huella computacional (sección 9.2)

Con el tiempo de la API de B1 (en minutos):

```powershell
python src/huella.py --api-min <minutos>
```

Sin la API ya da: entrenamiento 0.00321 kWh, Spark 0.04312 kWh, total ≈ 10.4 g de CO2 (factor 224.48 g/kWh,
Our World in Data 2024). **Verificar** en https://ark.intel.com que el TDP del i7-4770 es 84 W y en la hoja de datos
de NVIDIA que el de la T4 es 70 W.

## D. Solo el equipo

- Redactar el informe final (secciones 1 a 12). La base de cifras es `docs/resultados_para_informe.md`.
- Bitácora y declaración de uso de IA (11.4).
- Video de 5 a 8 minutos con el sistema funcionando: Spark (`correr_todo.ps1`), la página web, MLflow.
- Exposición: los 4 deben dominar todo.
