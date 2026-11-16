"""Cliente HTTP mínimo (solo librería estándar) para llamar a la API con una foto."""
import json
import mimetypes
import os
import time
import urllib.request
import uuid


def enviar_foto(url_base, ruta, endpoint="/predict", timeout=120):
    """POST multipart con el campo 'archivo'. Devuelve (respuesta_json, ms_ida_y_vuelta)."""
    limite = uuid.uuid4().hex
    nombre = os.path.basename(ruta)
    tipo = mimetypes.guess_type(nombre)[0] or "application/octet-stream"
    cuerpo = (f"--{limite}\r\nContent-Disposition: form-data; name=\"archivo\"; filename=\"{nombre}\"\r\n"
              f"Content-Type: {tipo}\r\n\r\n").encode() + open(ruta, "rb").read() + f"\r\n--{limite}--\r\n".encode()
    req = urllib.request.Request(url_base.rstrip("/") + endpoint, data=cuerpo, method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={limite}"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        datos = json.loads(r.read())
    return datos, 1000 * (time.perf_counter() - t0)


def salud(url_base, timeout=10):
    with urllib.request.urlopen(url_base.rstrip("/") + "/health", timeout=timeout) as r:
        return json.loads(r.read())


def listar_fotos(carpeta):
    fotos = []
    for raiz, _, archivos in os.walk(carpeta):
        for a in sorted(archivos):
            if a.lower().endswith((".jpg", ".jpeg", ".png")) and not a.startswith("."):
                fotos.append(os.path.join(raiz, a))
    return sorted(fotos)
