# Corre todo el pipeline de Spark (Tarea A) con un solo comando, en Windows con Docker Desktop.
#
# Uso (desde la carpeta del repositorio):
#   powershell -ExecutionPolicy Bypass -File .\pipeline\correr_todo.ps1
#
# Opciones:
#   -Fotos "D:\...\etiquetado_final_agregarfotos"   carpeta de fotos (por defecto la de la guia)
#   -Lote 2026-11-16                                fecha de la copia del dataset
#   -NucleosFisicos 4                               dato de Windows (i7-4770: 4 fisicos, 8 logicos)
#   -SoloMediciones                                 no repite el job (si ya corrio antes)
#   -Con5000                                        agrega el volumen de 5000 fotos (tarda mas)
#
# Que hace:
#   1. Revisa que Docker este encendido y que exista la imagen limon-spark:1.0 (si no, la construye).
#   2. Revisa que modelo\resnet18_limon.pt exista (si no, lo baja de la Release) y que su SHA-256
#      sea igual al de modelo\modelo_info.json.
#   3. Revisa la carpeta de fotos.
#   4. Corre pipeline/job_batch.py (pasos 1 a 6).
#   5. Corre pipeline/medir_escalabilidad.py (pasos 7 y 8).
#   6. Muestra un resumen y los comandos para subir resultados_spark a GitHub.
# Si algo falla, se detiene y dice que paso.

param(
    [string]$Fotos = "D:\Proyectos\percepcion-limones\dataset_limon_sutil\etiquetado_final_agregarfotos",
    [string]$Lote = "2026-11-16",
    [int]$NucleosFisicos = 4,
    [switch]$SoloMediciones,
    [switch]$Con5000
)

$ErrorActionPreference = "Continue"
$Imagen = "limon-spark:1.0"
$UrlModelo = "https://github.com/ByArk3/percepcion-limones/releases/download/modelo-v1/resnet18_limon.pt"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$inicio = Get-Date

function Paso($texto) {
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor Cyan
    Write-Host $texto -ForegroundColor Cyan
    Write-Host ("=" * 70) -ForegroundColor Cyan
}
function Falla($texto) {
    Write-Host ""
    Write-Host "ERROR: $texto" -ForegroundColor Red
    Write-Host "Se detuvo el script. Manda una captura de este mensaje." -ForegroundColor Red
    exit 1
}
function Ok($texto) { Write-Host "OK  $texto" -ForegroundColor Green }

Write-Host "Repositorio: $Repo"
Write-Host "Fotos      : $Fotos"
Write-Host "Lote       : $Lote"

# ---------------------------------------------------------------- 1. Docker
Paso "1. Docker"
$ncpu = docker info --format "{{.NCPU}}" 2>$null
if ($LASTEXITCODE -ne 0 -or -not $ncpu -or "$ncpu".Trim() -eq "0") {
    Falla "Docker no responde. Abre Docker Desktop, espera a que diga 'Engine running' y vuelve a correr el script."
}
Ok "Docker encendido con $("$ncpu".Trim()) CPUs"

docker image inspect $Imagen *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "No existe la imagen $Imagen. Se construye ahora (la primera vez tarda unos 10 minutos)..."
    docker build -t $Imagen -f pipeline/Dockerfile .
    if ($LASTEXITCODE -ne 0) { Falla "fallo docker build." }
}
Ok "Imagen $Imagen lista"

# ---------------------------------------------------------------- 2. Modelo
Paso "2. Modelo"
$pt = Join-Path $Repo "modelo\resnet18_limon.pt"
$info = Join-Path $Repo "modelo\modelo_info.json"
if (-not (Test-Path $info)) { Falla "no existe modelo\modelo_info.json. Haz git pull." }
if (-not (Test-Path (Join-Path $Repo "modelo\preprocesamiento.py"))) { Falla "no existe modelo\preprocesamiento.py. Haz git pull." }
if (-not (Test-Path $pt)) {
    Write-Host "No esta modelo\resnet18_limon.pt. Se descarga de la Release modelo-v1..."
    Invoke-WebRequest -Uri $UrlModelo -OutFile $pt
    if (-not (Test-Path $pt)) { Falla "no se pudo descargar el modelo." }
}
$shaEsperado = (Get-Content $info -Raw | ConvertFrom-Json).sha256
$shaReal = (Get-FileHash $pt -Algorithm SHA256).Hash.ToLower()
if ($shaReal -ne $shaEsperado) {
    Falla "el SHA-256 del modelo ($shaReal) no es igual al de modelo_info.json ($shaEsperado)."
}
Ok "Modelo correcto (SHA-256 $($shaReal.Substring(0,16))...)"

# ---------------------------------------------------------------- 3. Fotos
if (-not $SoloMediciones) {
    Paso "3. Fotos"
    if (-not (Test-Path $Fotos)) { Falla "no existe la carpeta de fotos: $Fotos" }
    $carpetas = Get-ChildItem $Fotos -Recurse -Directory | Where-Object { $_.Name -match '^(verde|pinton|maduro)_' }
    $nFotos = ($carpetas | ForEach-Object { (Get-ChildItem $_.FullName -File).Count } | Measure-Object -Sum).Sum
    if (-not $nFotos) { Falla "no hay fotos en carpetas verde_*, pinton_* o maduro_* dentro de $Fotos" }
    Ok "$nFotos archivos en $($carpetas.Count) carpetas de clase"
}

# ---------------------------------------------------------------- 4. Job batch
if (-not $SoloMediciones) {
    Paso "4. Job batch (job_batch.py)"
    $t0 = Get-Date
    docker run --rm `
        -v "${Fotos}:/datos/fotos:ro" `
        -v "${Repo}:/repo" -v limon-datos:/repo/datos -w /repo `
        $Imagen python pipeline/job_batch.py --fotos /datos/fotos --modelo modelo --salida resultados_spark --lote $Lote
    if ($LASTEXITCODE -ne 0) { Falla "fallo job_batch.py. Revisa las ultimas lineas de arriba." }
    Ok ("Job terminado en {0:N1} minutos" -f ((Get-Date) - $t0).TotalMinutes)
} else {
    Write-Host ""
    Write-Host "Se salta el job (-SoloMediciones): se usa el lago y el Parquet que ya estan en el volumen limon-datos."
}

# ---------------------------------------------------------------- 5. Mediciones
Paso "5. Mediciones de escalabilidad (medir_escalabilidad.py)"
Write-Host "Esto tarda entre 15 y 40 minutos (mas con -Con5000). No cierres esta ventana."
$extra = @()
if ($Con5000) { $extra += "--con-5000" }
$t0 = Get-Date
docker run --rm `
    -v "${Repo}:/repo" -v limon-datos:/repo/datos -w /repo `
    $Imagen python pipeline/medir_escalabilidad.py --modelo modelo --salida resultados_spark --lote $Lote --nucleos-fisicos $NucleosFisicos @extra
if ($LASTEXITCODE -ne 0) { Falla "fallo medir_escalabilidad.py. Revisa las ultimas lineas de arriba." }
Ok ("Mediciones terminadas en {0:N1} minutos" -f ((Get-Date) - $t0).TotalMinutes)

# ---------------------------------------------------------------- 6. Resumen
Paso "6. Resumen"
$logJob = Join-Path $Repo "resultados_spark\log_job_batch.txt"
$logMed = Join-Path $Repo "resultados_spark\log_medir_escalabilidad.txt"
foreach ($log in @($logJob, $logMed)) {
    if (Test-Path $log) {
        Select-String -Encoding UTF8 -Path $log -Pattern "CUMPLE|PartitionFilters en el plan|Mejor speedup|H2 \(|empieza a ganar|no hay punto de cruce|gana en todo" |
            ForEach-Object { Write-Host ("  " + $_.Line) }
    }
}
Write-Host ""
Write-Host "Archivos en resultados_spark:"
Get-ChildItem (Join-Path $Repo "resultados_spark") -Recurse -File |
    Where-Object { $_.Name -ne "mlflow.db" } |
    ForEach-Object { Write-Host ("  " + $_.FullName.Substring($Repo.Length + 1)) }
Write-Host ""
Write-Host ("Tiempo total: {0:N1} minutos" -f ((Get-Date) - $inicio).TotalMinutes) -ForegroundColor Green
Write-Host ""
Write-Host "Para subir los resultados a GitHub:" -ForegroundColor Yellow
Write-Host "  git add resultados_spark"
Write-Host "  git commit -m `"Resultados reales de Spark en la PC (lote $Lote)`""
Write-Host "  git push"
