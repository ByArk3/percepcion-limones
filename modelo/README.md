# Carpeta modelo/

Modelo final: ResNet18 TorchScript, semilla 42, 30 épocas. Entrenado en Google Colab (GPU Tesla T4) el 16-nov-2026
(notebook con salidas: `notebooks/Grupo3_limones_drive_Colab_ejecutado_16nov.ipynb`).

| Archivo | Dónde está |
|---|---|
| `modelo_info.json` | En el repositorio (clases, media/desv, métricas de la semilla 42, sha256) |
| `preprocesamiento.py` | En el repositorio (funciones `cargar`, `segmentar`, `corregir`, `recortar` de la celda 13.2) |
| `resnet18_limon.pt` | Release `modelo-v1` de GitHub (no va en el código, 44.9 MB) |

SHA-256 de `resnet18_limon.pt`:
`8eec3d12ef05ec3286157fe2f8174f82b3b2d4ca237348b3576658e77dbf62ef`

Descargar el modelo a esta carpeta (PowerShell, desde la raíz del repositorio):

```powershell
Invoke-WebRequest -Uri https://github.com/ByArk3/percepcion-limones/releases/download/modelo-v1/resnet18_limon.pt -OutFile modelo\resnet18_limon.pt
Get-FileHash modelo\resnet18_limon.pt -Algorithm SHA256
```
