"""Huella computacional (sección 9.2): energía (kWh) y CO2 a partir de tiempos REALES medidos.

energía (kWh) = potencia (W) x tiempo (h) / 1000      CO2 (g) = energía (kWh) x factor (g CO2/kWh)

Supuestos (declararlos en el informe):
  - Potencia = TDP del procesador o de la GPU: es un TOPE (el equipo no siempre está al 100 %) y no incluye el resto
    de la PC (placa, RAM, disco, monitor). Por eso la cifra es una estimación de orden de magnitud.
  - Factor de la red del Perú: 224.48 g CO2/kWh (2024), Our World in Data (datos de Ember),
    https://ourworldindata.org/co2/country/peru  (otras fuentes dan 195-238 g CO2/kWh; se muestra el rango).
  - Colab: no se sabe en qué región está el servidor, así que su CO2 se calcula con el mismo rango solo como
    referencia y se marca así.

Tiempos por defecto (todos medidos, ver docs/resultados_para_informe.md):
  - entrenamiento ResNet18 en Colab (GPU Tesla T4, 70 W): 5 corridas de mlflow.db del 16-nov
    35.598 + 32.340 + 32.281 + 32.725 + 31.968 = 164.912 s
  - Spark en la PC (i7-4770, 84 W): job 5.8 min + mediciones 25.0 min = 30.8 min (consola del 16-nov)
  - API en la PC: PENDIENTE (pasar --api-min con el tiempo de pruebas/medir_latencia.py)

Uso:  python src/huella.py   (o con --api-min 3.5, --spark-min ..., etc.)
"""
import argparse

FACTOR_PERU = 224.48          # g CO2/kWh, Our World in Data 2024
RANGO_PERU = (195.0, 238.0)   # LowCarbonPower (12 meses a may-2026) e Inzonex (2025)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entrenamiento-s", type=float, default=164.912, help="segundos de entrenamiento ResNet18 (Colab)")
    ap.add_argument("--gpu-w", type=float, default=70.0, help="TDP de la GPU Tesla T4 (W)")
    ap.add_argument("--spark-min", type=float, default=30.8, help="minutos de Spark en la PC")
    ap.add_argument("--api-min", type=float, default=None, help="minutos de pruebas de la API (PENDIENTE si no se da)")
    ap.add_argument("--cpu-w", type=float, default=84.0, help="TDP del Intel Core i7-4770 (W)")
    ap.add_argument("--factor", type=float, default=FACTOR_PERU, help="g CO2/kWh de la red")
    a = ap.parse_args()

    filas = [("Entrenamiento ResNet18 (Colab, GPU T4)", a.entrenamiento_s / 3600, a.gpu_w, True),
             ("Spark: job + mediciones (PC, i7-4770)", a.spark_min / 60, a.cpu_w, False)]
    if a.api_min is not None:
        filas.append(("API: pruebas de latencia (PC, i7-4770)", a.api_min / 60, a.cpu_w, False))
    print(f"{'Etapa':42s} {'horas':>8s} {'W':>5s} {'kWh':>9s} {'g CO2':>8s}  rango g CO2")
    tot_kwh = tot_g = 0.0
    for nombre, h, w, colab in filas:
        kwh = w * h / 1000
        g = kwh * a.factor
        tot_kwh += kwh; tot_g += g
        nota = "  (servidor de Colab: región desconocida, solo referencia)" if colab else ""
        print(f"{nombre:42s} {h:8.4f} {w:5.0f} {kwh:9.5f} {g:8.2f}  {kwh * RANGO_PERU[0]:.2f}-{kwh * RANGO_PERU[1]:.2f}{nota}")
    print(f"{'TOTAL':42s} {'':8s} {'':5s} {tot_kwh:9.5f} {tot_g:8.2f}")
    if a.api_min is None:
        print("API: PENDIENTE (falta el tiempo real de pruebas/medir_latencia.py)")
    print(f"Factor usado: {a.factor} g CO2/kWh (Perú 2024, Our World in Data). Potencias = TDP (tope; no incluye el resto de la PC).")


if __name__ == "__main__":
    main()
