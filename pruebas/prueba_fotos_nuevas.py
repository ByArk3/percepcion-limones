"""Prueba con fotos NUEVAS (que no están en el dataset): la API debe dar lo mismo que el script directo.

Para cada foto:
  - llama a POST /predict de la API;
  - clasifica la misma foto en este proceso con la misma clase Clasificador y el mismo modelo;
  - imprime las probabilidades con 4 decimales y la diferencia máxima.
Si el nombre empieza con v_, p_ o m_ (o la carpeta con verde_, pinton_, maduro_), muestra la clase esperada.

Uso (PowerShell, con docker compose levantado; 2 fotos por clase en una carpeta):
  docker compose run --rm -v "D:\\...\\fotos_nuevas:/fotos_nuevas:ro" backend `
      python /pruebas/prueba_fotos_nuevas.py --url http://backend:8000 --fotos /fotos_nuevas
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/servicio")                      # dentro de la imagen del backend
from cliente import enviar_foto, listar_fotos       # noqa: E402

ESPERADA = {"v": "verde", "p": "pinton", "m": "maduro"}


def clase_esperada(ruta):
    nombre = os.path.basename(ruta).lower()
    carpeta = os.path.basename(os.path.dirname(ruta)).lower()
    for c in ("verde", "pinton", "maduro"):
        if carpeta.startswith(c + "_") or carpeta == c:
            return c
    return ESPERADA.get(nombre[:1]) if len(nombre) > 1 and nombre[1] == "_" else "?"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://backend:8000")
    ap.add_argument("--fotos", required=True)
    ap.add_argument("--modelo", default="/modelo")
    ap.add_argument("--salida", default="/resultados_servicio")
    a = ap.parse_args()

    from app.inferencia import Clasificador
    clf = Clasificador(a.modelo, hilos=None)
    fotos = listar_fotos(a.fotos)
    if not fotos:
        sys.exit(f"No hay fotos en {a.fotos}")

    lineas = [f"{'foto':28s} {'esperada':9s} {'API':8s} {'script':8s}  "
              + "  ".join(f"{c:>15s}" for c in clf.clases) + "   dif_max"]
    dif_total, aciertos = 0.0, 0
    for ruta in fotos:
        r, _ = enviar_foto(a.url, ruta)
        local = clf.predecir([open(ruta, "rb").read()])[0]
        clase_local = clf.clases[int(local.argmax())]
        dif = max(abs(r["probabilidades"][c] - float(p)) for c, p in zip(clf.clases, local))
        dif_total = max(dif_total, dif)
        esp = clase_esperada(ruta)
        aciertos += esp == r["clase"]
        probs = "  ".join(f"{r['probabilidades'][c]:.4f} / {float(p):.4f}" for c, p in zip(clf.clases, local))
        lineas.append(f"{os.path.basename(ruta)[:28]:28s} {esp:9s} {r['clase']:8s} {clase_local:8s}  {probs}   {dif:.1e}")
    lineas += ["", "Probabilidades: API / script directo (4 decimales).",
               f"Diferencia máxima API vs script: {dif_total:.1e} -> "
               f"{'COINCIDEN' if dif_total < 1e-4 else 'NO COINCIDEN'} (tolerancia 1e-4)",
               f"Clase esperada = API en {aciertos} de {len(fotos)} fotos (con '?' no se sabe la esperada)."]
    print("\n".join(lineas))
    os.makedirs(a.salida, exist_ok=True)
    with open(os.path.join(a.salida, "fotos_nuevas.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lineas) + "\n")
    print(f"\nGuardado en {a.salida}/fotos_nuevas.txt")


if __name__ == "__main__":
    main()
