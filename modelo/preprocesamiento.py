import numpy as np, cv2
from PIL import Image, ImageOps
TAM = 224

def cargar(ruta, lado=600):
    """Abre la foto, corrige la orientación EXIF y la reduce (LANCZOS evita aliasing)."""
    im = ImageOps.exif_transpose(Image.open(ruta)).convert("RGB")
    w, h = im.size
    im.thumbnail((lado, lado), Image.LANCZOS)
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR), (w, h)


def segmentar(bgr):
    """Máscara del limón: Otsu sobre la saturación, mayor componente conexa y relleno de huecos."""
    s = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1], (5, 5), 0)
    _, m = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2: return np.ones(m.shape, np.uint8) * 255
    m = np.uint8(lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])) * 255
    cont, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(m, cont, -1, 255, -1)
    return m


def corregir(bgr, m):
    """Iluminación (flat-field estimado del fondo) y balance de blancos con el fondo como referencia."""
    fondo = cv2.erode(255 - m, np.ones((15, 15), np.uint8)) > 0
    if fondo.sum() < 500: return bgr
    out = bgr.astype(np.float32)
    gris = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    rel = gris.copy(); rel[~fondo] = np.median(gris[fondo])
    ilum = cv2.GaussianBlur(rel, (0, 0), sigmaX=max(bgr.shape) / 8)
    out *= (np.median(gris[fondo]) / np.maximum(ilum, 1))[..., None]
    med = np.median(out[fondo], axis=0)                        # color medio del fondo (B, G, R)
    out *= med.mean() / np.maximum(med, 1)
    return np.clip(out, 0, 255).astype(np.uint8)


def recortar(bgr, m, tam=TAM, margen=0.08):
    """Recorte cuadrado centrado en el limón y redimensionado a tam x tam."""
    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        return cv2.resize(bgr, (tam, tam), interpolation=cv2.INTER_AREA), cv2.resize(m, (tam, tam))
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    lado = max(np.ptp(xs), np.ptp(ys)) * (1 + 2 * margen) / 2
    x0, y0 = int(max(cx - lado, 0)), int(max(cy - lado, 0))
    x1, y1 = int(min(cx + lado, bgr.shape[1])), int(min(cy + lado, bgr.shape[0]))
    f = lambda a: cv2.resize(a[y0:y1, x0:x1], (tam, tam), interpolation=cv2.INTER_AREA)
    return f(bgr), (f(m) > 127).astype(np.uint8) * 255
