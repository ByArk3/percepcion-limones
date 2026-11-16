"""Latencia de la API en el contenedor (sección 8.1): N llamadas a POST /predict con fotos reales.

Reporta mediana y p95 de dos tiempos:
  - ida y vuelta (cliente -> backend -> cliente), que es lo que siente el usuario;
  - latencia_ms que informa el backend (solo preprocesamiento + modelo).
Las 5 primeras llamadas son de calentamiento y no se cuentan.

Uso (PowerShell, con docker compose ya levantado):
  docker compose run --rm -v "D:\\...\\etiquetado_final_agregarfotos:/fotos:ro" backend `
      python /pruebas/medir_latencia.py --url http://backend:8000 --fotos /fotos --llamadas 200
"""
import argparse
import os
import platform
import statistics
import sys
import time
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cliente import enviar_foto, listar_fotos, salud  # noqa: E402


def percentil(valores, p):
    v = sorted(valores)
    k = (len(v) - 1) * p / 100
    i, f = int(k), k - int(k)
    return v[i] + (v[min(i + 1, len(v) - 1)] - v[i]) * f


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://backend:8000")
    ap.add_argument("--fotos", required=True)
    ap.add_argument("--llamadas", type=int, default=200)
    ap.add_argument("--calentamiento", type=int, default=5)
    ap.add_argument("--salida", default="/resultados_servicio")
    a = ap.parse_args()

    fotos = listar_fotos(a.fotos)
    if not fotos:
        sys.exit(f"No hay fotos en {a.fotos}")
    s = salud(a.url)
    print(f"Servicio: {s['estado']} | modelo {s['version_modelo']} | sha256 {s['sha256'][:16]}... | "
          f"{len(fotos)} fotos disponibles")

    for i in range(a.calentamiento):
        try:
            enviar_foto(a.url, fotos[i % len(fotos)])
        except urllib.error.HTTPError:
            pass
    filas, errores = [], []
    i = 0
    while len(filas) < a.llamadas and i < a.llamadas * 2:
        ruta = fotos[i % len(fotos)]                    # si hay menos fotos que llamadas, se repiten
        i += 1
        try:
            r, ms = enviar_foto(a.url, ruta)
        except urllib.error.HTTPError as e:            # foto que no se puede leer: se anota y no se cuenta
            errores.append(f"{os.path.relpath(ruta, a.fotos)} (HTTP {e.code})")
            continue
        filas.append((os.path.relpath(ruta, a.fotos), r["clase"], ms, r["latencia_ms"]))
        if len(filas) % 50 == 0:
            print(f"  {len(filas)}/{a.llamadas} llamadas")

    ida = [f[2] for f in filas]
    srv = [f[3] for f in filas]
    resumen = [
        f"Fecha: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"URL: {a.url}/predict | llamadas medidas: {len(filas)} (+{a.calentamiento} de calentamiento) | "
        f"fotos distintas: {len(set(f[0] for f in filas))}",
        f"Modelo: {s['version_modelo']} | sha256 {s['sha256']}",
        f"Cliente: Python {platform.python_version()} | os.cpu_count() = {os.cpu_count()}",
        f"Fotos rechazadas por la API (no se cuentan): {len(set(errores))}"
        + (f" -> {', '.join(sorted(set(errores))[:10])}" if errores else ""),
        "",
        f"Ida y vuelta (ms): mediana {statistics.median(ida):.1f} | p95 {percentil(ida, 95):.1f} | "
        f"min {min(ida):.1f} | max {max(ida):.1f}",
        f"Backend (ms)     : mediana {statistics.median(srv):.1f} | p95 {percentil(srv, 95):.1f} | "
        f"min {min(srv):.1f} | max {max(srv):.1f}",
        "",
        f"Meta (lógica acordada 9): mediana < 150 ms -> "
        f"{'CUMPLE' if statistics.median(ida) < 150 else 'NO CUMPLE'} (ida y vuelta)",
        f"Criterio de promoción (Tarea D): p95 < 150 ms -> {'CUMPLE' if percentil(ida, 95) < 150 else 'NO CUMPLE'}",
    ]
    print("\n".join(resumen))
    os.makedirs(a.salida, exist_ok=True)
    with open(os.path.join(a.salida, "latencia.csv"), "w", encoding="utf-8") as f:
        f.write("foto,clase,ms_ida_y_vuelta,ms_backend\n")
        for ruta, clase, m1, m2 in filas:
            f.write(f"{ruta},{clase},{m1:.2f},{m2:.2f}\n")
    with open(os.path.join(a.salida, "latencia_resumen.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(resumen) + "\n")
    print(f"\nGuardado en {a.salida}/latencia.csv y latencia_resumen.txt")


if __name__ == "__main__":
    main()
