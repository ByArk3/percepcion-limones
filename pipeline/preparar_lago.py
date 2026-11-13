"""Arma el data lake clave=valor a partir de las carpetas de fotos del equipo.

Entrada : carpeta etiquetado_final_agregarfotos (montada en solo lectura).
Salida  : <lago>/lote=<fecha>/celular=<CODIGO>/clase=<clase>/<archivo>.jpg

Solo se usan subcarpetas que empiezan con verde_, pinton_ o maduro_.
El código del celular sale del nombre del archivo (v_A26_601.jpg -> A26);
si el nombre no sigue el patrón, va a celular=SIN_CODIGO.

Uso (dentro del contenedor):
    python pipeline/preparar_lago.py --fotos /datos/fotos --lago datos/lago --lote 2026-11-04
"""
import argparse
import os
import re
import shutil

import pandas as pd

CLASES = ("verde", "pinton", "maduro")
# mismo patrón que la celda 2.1 del notebook
PATRON_NOMBRE = re.compile(r"^([vpm])_([A-Za-z0-9]+)_(\d{3})\.jpe?g$", re.I)
EXT_JPG = (".jpg", ".jpeg")


def clase_de_carpeta(nombre):
    """verde_ark -> verde ; descartadas_leoncio -> None"""
    for c in CLASES:
        if nombre.lower().startswith(c + "_"):
            return c
    return None


def armar_lago(fotos, lago, lote, imprimir=print):
    """Copia las fotos al lago y devuelve (tabla de fotos copiadas, tabla de ignoradas)."""
    destino_lote = os.path.join(lago, f"lote={lote}")
    if os.path.isdir(destino_lote):                 # solo se rehace el lote, no el resto del lago
        shutil.rmtree(destino_lote)
    copiadas, ignoradas = [], []
    for raiz, carpetas, archivos in os.walk(fotos):
        carpetas.sort()
        clase = clase_de_carpeta(os.path.basename(raiz))
        for archivo in sorted(archivos):
            ruta = os.path.join(raiz, archivo)
            rel = os.path.relpath(ruta, fotos)
            if clase is None:
                ignoradas.append(dict(archivo=rel, motivo="carpeta no es verde_/pinton_/maduro_"))
                continue
            base, ext = os.path.splitext(archivo)
            if ext.lower() not in EXT_JPG:
                ignoradas.append(dict(archivo=rel, motivo=f"extension {ext} (solo .jpg/.jpeg)"))
                continue
            m = PATRON_NOMBRE.match(archivo)
            celular = m.group(2).upper() if m else "SIN_CODIGO"
            # la extensión se deja en .jpg minúscula para que el filtro "*.jpg" de Spark la lea
            dest = os.path.join(destino_lote, f"celular={celular}", f"clase={clase}", base + ".jpg")
            if os.path.exists(dest):
                ignoradas.append(dict(archivo=rel, motivo="nombre repetido en el lago"))
                continue
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(ruta, dest)
            copiadas.append(dict(archivo=rel, celular=celular, clase=clase, bytes=os.path.getsize(dest)))
    tab = pd.DataFrame(copiadas, columns=["archivo", "celular", "clase", "bytes"])
    ign = pd.DataFrame(ignoradas, columns=["archivo", "motivo"])
    imprimir(f"Lago: {destino_lote}")
    imprimir(f"Fotos copiadas: {len(tab)} ({tab.bytes.sum() / 1e6:.1f} MB) | ignoradas: {len(ign)}")
    if len(tab):
        imprimir("Fotos por celular y clase (copiadas al lago):")
        conteo = tab.groupby(["celular", "clase"]).size().unstack(fill_value=0)
        imprimir(conteo.reindex(columns=[c for c in CLASES if c in conteo.columns]).to_string())
    if len(ign):
        imprimir("Ignoradas por motivo:")
        imprimir(ign.motivo.value_counts().to_string())
    return tab, ign


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fotos", required=True, help="carpeta etiquetado_final_agregarfotos")
    ap.add_argument("--lago", default="datos/lago")
    ap.add_argument("--lote", default="2026-11-04")
    a = ap.parse_args()
    armar_lago(a.fotos, a.lago, a.lote)
