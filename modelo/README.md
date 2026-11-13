# Carpeta modelo/

Aqui van los archivos del modelo exportado por la celda 13.2 del notebook:

| Archivo | Donde se guarda |
|---|---|
| `modelo_info.json` | En el repositorio (lo sube Franco desde `resultados\modelo\`) |
| `preprocesamiento.py` | En el repositorio (lo sube Franco desde `resultados\modelo\`) |
| `resnet18_limon.pt` | Release `modelo-v1` de GitHub (no va en el codigo, ~45 MB) |

Antes de usarlo, revisar que `modelo_info.json` diga `"epocas": 30` y que el SHA-256 del `.pt` coincida con el campo `sha256`.
