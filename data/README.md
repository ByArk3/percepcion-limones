# Datos: fotos de limón sutil

Las fotos **no** están en este repositorio (unos 500 MB). Están en Google Drive:

| Carpeta | Contenido | Enlace |
|---|---|---|
| `etiquetado_final_agregarfotos` | Dataset vigente (236 fotos al 16-nov-2026) | https://drive.google.com/drive/folders/1KqT60QOlIsdWQMDNuu0GVYzsaB4-WwLy |
| `etiquetado_avance1` | Copia congelada del parcial (solo para marcar qué fotos son antiguas, por MD5) | https://drive.google.com/drive/folders/1cdC6sah1bsUlJzsVQ73LsvLIjOQjuPKE |

## Cómo bajarlas

- **Colab:** el notebook `notebooks/Grupo3_limones_drive_Colab.ipynb` (celda 2.1) las descarga solo desde el enlace.
- **PC:** en Drive, clic derecho en `etiquetado_final_agregarfotos`, Descargar, y descomprimir en
  `D:\Proyectos\percepcion-limones\dataset_limon_sutil\` (fuera de la carpeta del repositorio).

## Estructura esperada

```
etiquetado_final_agregarfotos/
├── Ark/      verde_ark/  pinton_ark/  maduro_ark/
├── Franco/   verde_franco/  maduro_franco/
├── Guinno/   verde_guinno/  maduro_guinno/
└── Leoncio/  verde_leoncio/  pinton_leoncio/  maduro_leoncio/
```

- Solo se usan las subcarpetas que empiezan con `verde_`, `pinton_` o `maduro_` (la clase sale de la carpeta).
  Cualquier otra (por ejemplo `descartadas_leoncio`) y los archivos sueltos (`registro_*.csv`) se ignoran.
- **Nombre de cada foto:** `CLASE_CELULAR_NUMERO.jpg`, con clase `v`, `p` o `m`. Ejemplo: `v_A26_601.jpg`.
  El celular sale del nombre: Guinno `RN10` (001-200), Franco `R14P` (201-400; hoy todavía `FRANCO`),
  Leoncio `A56` (401-600), Ark `A26` (601-800). Si no sigue el patrón, el celular queda como "sin código".
- **Regla de clases** (NTP 011.006 + criterio del equipo), según el % de cáscara amarilla:
  verde < 10 %, pintón 10-30 %, maduro > 30 %.

## Composición al 16-nov-2026

| Integrante (celular) | Verde | Pintón | Maduro | Total |
|---|---|---|---|---|
| Ark (A26) | 63 | 34 | 51 | 148 |
| Franco (FRANCO) | 23 | 0 | 23 | 46 |
| Guinno (RN10) | 16 | 0 | 10 | 26 |
| Leoncio (LEONCIO) | 2 | 10 | 4 | 16 |
| **Total** | 104 | 44 | 88 | **236** |

El notebook quita 9 duplicados (MD5 y dHash) y analiza **227** fotos (verde 96, pintón 43, maduro 88).

## Pendientes de limpieza conocidos (declarar en el informe si no se hacen)

- Las 16 fotos de Leoncio son de iPhone (no de su A56) y siguen en las carpetas de clase.
- `v_RN10_001` a `008` son duplicados de otras fotos de Guinno (el notebook ya los quita).
- `p_LEONCIO_403` y `m_LEONCIO_403` son casi la misma foto con clases distintas.
- Las fotos de Franco tienen el código `FRANCO` en vez de `R14P`.
- Revisar `p_A26_640` y `m_A26_636` (maduro con 17 % de amarillo).

## Privacidad y licencia

Fotos tomadas por los integrantes del Grupo 3 con sus celulares, solo de limones (sin personas). Uso académico
(curso ISIA-111, UPAO). Si se cambian los permisos del Drive, dejar el acceso como "Lector".
