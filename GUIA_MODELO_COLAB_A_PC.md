# Guía paso a paso: modelo de Colab → repositorio → Spark en la PC

Plan acordado: **entrenar en Colab con GPU** y hacer **la inferencia (Spark y API) en la PC con CPU**.
En el informe se declara: "entrenamiento en Google Colab (GPU Tesla T4); inferencia en la PC (CPU, Docker)".

Comandos para **PowerShell** en Windows. Si una ruta es distinta en tu PC, cámbiala.

---

## Paso 1. Conseguir `resultados.zip` de la corrida de Colab

**Opción A (si ya lo tienes):** la celda 15.1 descarga `resultados.zip` sola al terminar la corrida.
Búscalo en `Descargas`. Usar el de la corrida del **16-nov** (la última; SHA-256 `8eec3d12ef05ec32...`), no el del 13-nov.

**Opción B (si no lo tienes): volver a correr Colab**

1. Abrir `notebooks/Grupo3_limones_drive_Colab.ipynb` en Google Colab.
2. Menú **Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU T4 → Guardar**.
3. Menú **Entorno de ejecución → Ejecutar todas**.
4. Cuando lo pida, dar permiso a la cuenta de Google (para leer el Drive).
5. Esperar a que termine. Al final se descarga `resultados.zip`.
6. Guardar el notebook con salidas: **Archivo → Descargar → Descargar .ipynb**.

Ojo: una corrida nueva da **números nuevos**, que reemplazan a los de la corrida anterior.
En ese caso, mándame el `.ipynb` nuevo para actualizar las cifras.

## Paso 2. Descomprimir y revisar el modelo

```powershell
cd D:\Proyectos\percepcion-limones
Expand-Archive -Path "$env:USERPROFILE\Downloads\resultados.zip" -DestinationPath resultados_colab_16nov
dir resultados_colab_16nov\modelo
```

Tienen que aparecer 3 archivos: `resnet18_limon.pt` (~45 MB), `modelo_info.json` y `preprocesamiento.py`.

Revisar que sea el modelo correcto:

```powershell
Select-String -Path resultados_colab_16nov\modelo\modelo_info.json -Pattern '"epocas"|"sha256"|"fecha"'
Get-FileHash resultados_colab_16nov\modelo\resnet18_limon.pt -Algorithm SHA256
```

- `"epocas": 30`
- El SHA-256 de `Get-FileHash` tiene que ser **igual** al `"sha256"` del JSON. Con la corrida del 16-nov empieza con `8EEC3D12EF05EC32` (PowerShell lo muestra en mayúsculas).

Si no coinciden, **no sigas** y avísame.

## Paso 3. Subir `modelo_info.json` y `preprocesamiento.py` al repositorio

```powershell
cd D:\Proyectos\percepcion-limones\repo
git pull
Copy-Item ..\resultados_colab_16nov\modelo\modelo_info.json modelo\
Copy-Item ..\resultados_colab_16nov\modelo\preprocesamiento.py modelo\
Copy-Item ..\resultados_colab_16nov\modelo\resnet18_limon.pt modelo\
git status
```

En `git status` deben salir **solo** `modelo/modelo_info.json` y `modelo/preprocesamiento.py`.
El `.pt` **no** debe salir (el `.gitignore` lo excluye). Si sale, no hagas commit y avísame.

```powershell
git add modelo\modelo_info.json modelo\preprocesamiento.py
git commit -m "Modelo final de Colab (GPU T4, 16-nov): info y preprocesamiento"
git push
```

## Paso 4. Subir `resnet18_limon.pt` como Release de GitHub

1. Entrar a https://github.com/ByArk3/percepcion-limones
2. A la derecha, **Releases → Create a new release** (o "Draft a new release").
3. **Choose a tag:** escribir `modelo-v1` y elegir "Create new tag".
4. **Release title:** `Modelo ResNet18 v1 (Colab GPU T4, 16-nov-2026)`
5. **Descripción** (copiar y completar con tu SHA-256 del paso 2):
   ```
   ResNet18 TorchScript, semilla 42, 30 épocas. Entrenado en Google Colab (GPU Tesla T4) el 16-nov-2026.
   SHA-256: <pegar aquí el SHA-256 completo>
   Va en la carpeta modelo/ junto a modelo_info.json y preprocesamiento.py.
   ```
6. Arrastrar `D:\Proyectos\percepcion-limones\resultados_colab_16nov\modelo\resnet18_limon.pt` a la zona de archivos.
7. Esperar a que suba y dar **Publish release**.

## Paso 5. Construir la imagen de Spark

Docker Desktop tiene que estar abierto.

```powershell
cd D:\Proyectos\percepcion-limones\repo
docker build -t limon-spark:1.0 -f pipeline/Dockerfile .
docker images limon-spark:1.0
```

La primera vez tarda varios minutos (baja Java 17 y PyTorch CPU).
Si falla, copia las últimas 30 líneas del error y mándamelas.

## Paso 6. Correr el job batch

```powershell
docker run --rm `
  -v "D:\Proyectos\percepcion-limones\dataset_limon_sutil\etiquetado_final_agregarfotos:/datos/fotos:ro" `
  -v "${PWD}:/repo" -v limon-datos:/repo/datos -w /repo `
  limon-spark:1.0 python pipeline/job_batch.py --fotos /datos/fotos --modelo modelo --salida resultados_spark
```

Al final debe decir `CUMPLE` en la verificación de coherencia y `PartitionFilters en el plan: sí`.

## Paso 7. Correr las mediciones

Puede tardar entre 15 y 40 minutos. No cierres Docker ni apagues la PC.

```powershell
docker run --rm `
  -v "${PWD}:/repo" -v limon-datos:/repo/datos -w /repo `
  limon-spark:1.0 python pipeline/medir_escalabilidad.py --modelo modelo --salida resultados_spark
```

## Paso 8. Subir resultados y mandarme los datos de la PC

```powershell
git add resultados_spark
git commit -m "Resultados reales de Spark en la PC (lote 2026-11-04)"
git push
docker info --format "CPUs={{.NCPU}} Memoria={{.MemTotal}}"
Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors
```

Pégame en el chat la salida de los dos últimos comandos y avísame que hiciste el push. Yo reviso las tablas y figuras.

---

## Resumen de lo que debe quedar

| Dónde | Qué |
|---|---|
| Repositorio `modelo/` | `modelo_info.json`, `preprocesamiento.py` |
| Release `modelo-v1` | `resnet18_limon.pt` |
| Repositorio `resultados_spark/` | tablas, figuras y bitácoras reales de la PC |
| Solo en la PC | fotos, `resultados_colab_16nov\`, volumen Docker `limon-datos` |
