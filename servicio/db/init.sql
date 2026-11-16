-- Tablas de la base de datos del servicio (PostgreSQL).
-- La ejecuta el backend al arrancar (CREATE ... IF NOT EXISTS: se puede correr varias veces).

-- Cada foto clasificada por la API (una foto o un zip desde la web o con curl)
CREATE TABLE IF NOT EXISTS predicciones (
    id              SERIAL PRIMARY KEY,
    fecha           TIMESTAMPTZ NOT NULL DEFAULT now(),
    origen          TEXT NOT NULL,              -- 'foto' o 'zip'
    id_envio        TEXT NOT NULL,              -- agrupa las fotos de un mismo zip
    archivo         TEXT NOT NULL,
    md5             TEXT NOT NULL,              -- huella de la foto (la foto NO se guarda)
    clase           TEXT NOT NULL,              -- verde, pinton, maduro o error
    prob_verde      DOUBLE PRECISION,
    prob_pinton     DOUBLE PRECISION,
    prob_maduro     DOUBLE PRECISION,
    latencia_ms     DOUBLE PRECISION,
    version_modelo  TEXT NOT NULL,
    sha256_modelo   TEXT NOT NULL               -- qué modelo exacto hizo la predicción (sección 8.3)
);
CREATE INDEX IF NOT EXISTS idx_predicciones_fecha ON predicciones (fecha DESC);

-- Resultados de los lotes de Spark (se cargan desde el Parquet de pipeline/job_batch.py)
CREATE TABLE IF NOT EXISTS predicciones_spark (
    id              SERIAL PRIMARY KEY,
    lote            TEXT NOT NULL,
    path            TEXT NOT NULL,
    celular         TEXT,
    clase_etiqueta  TEXT,                       -- carpeta de origen (etiqueta del equipo)
    md5             TEXT,
    prediccion      TEXT NOT NULL,
    prob_verde      DOUBLE PRECISION,
    prob_pinton     DOUBLE PRECISION,
    prob_maduro     DOUBLE PRECISION,
    cargado         TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_spark_lote ON predicciones_spark (lote);
