"""
unet_service.py — Deteccion de agua superficial (balsas, charcas, embalses) en la
parcela con la U-Net entrenada (unet/entrenar.py).

Descarga de Google Earth Engine la mediana Sentinel-2 del ultimo verano completo
(jun-sep) sobre la parcela mas un margen, con las MISMAS 6 bandas, escala (10 m)
y normalizacion que en el entrenamiento, y devuelve una rejilla de probabilidad
de agua lista para pintarse en Leaflet, mas las masas de agua detectadas.
Si el modelo no existe, devuelve disponible=False y el resto de la app sigue igual.
"""

import json
import math
import sys
from datetime import date
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND / "unet"))

MODEL_PATH = BACKEND / "modelo" / "modelo_unet.pth"
METRICAS_PATH = BACKEND / "modelo" / "metricas_unet.json"
MARGEN_M = 150            # margen alrededor de la parcela
MAX_PIXELES = 200_000     # limite de sampleRectangle de Google Earth Engine
AREA_MIN_PX = 4           # masas de menos de 4 pixeles (~400 m2) se descartan como ruido
GRADOS_POR_METRO = 1 / 111319.49

_cache = {"mtime": None, "modelo": None, "ckpt": None}


def unet_disponible() -> bool:
    return MODEL_PATH.exists()


def cargar_metricas_unet():
    if not METRICAS_PATH.exists():
        return None
    with open(METRICAS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _cargar_modelo():
    import torch
    from model import UNetLigero

    mtime = MODEL_PATH.stat().st_mtime
    if _cache["mtime"] != mtime:
        try:
            ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        except TypeError:  # versiones antiguas de torch
            ckpt = torch.load(MODEL_PATH, map_location="cpu")
        modelo = UNetLigero(in_channels=ckpt["in_channels"], out_channels=1, base_ch=ckpt["base_ch"])
        modelo.load_state_dict(ckpt["model_state_dict"])
        modelo.eval()
        _cache.update(mtime=mtime, modelo=modelo, ckpt=ckpt)
    return _cache["modelo"], _cache["ckpt"]


def _ultimo_verano() -> int:
    hoy = date.today()
    return hoy.year if hoy.month >= 10 else hoy.year - 1


def _bbox(coords, margen_m=MARGEN_M):
    lats = [c[1] for c in coords]
    lons = [c[0] for c in coords]
    lat_c = sum(lats) / len(lats)
    dlat = margen_m / 111320.0
    dlon = margen_m / (111320.0 * math.cos(math.radians(lat_c)))
    return min(lons) - dlon, min(lats) - dlat, max(lons) + dlon, max(lats) + dlat


def _descargar(bbox, bandas, anno):
    import ee

    lon_min, lat_min, lon_max, lat_max = bbox
    escala = 10
    while ((lon_max - lon_min) / (escala * GRADOS_POR_METRO)) * \
          ((lat_max - lat_min) / (escala * GRADOS_POR_METRO)) > MAX_PIXELES:
        escala *= 2
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterDate(f"{anno}-06-01", f"{anno}-09-30")
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
          .median().select(bandas).divide(10000))
    imagen = (ee.Image.cat([s2, ee.Image.pixelLonLat()]).toFloat().unmask(0)
              .reproject(crs="EPSG:4326", scale=escala))
    region = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])
    props = imagen.sampleRectangle(region=region, defaultValue=0).getInfo()["properties"]
    return props, escala


def _dentro_poligono(lat, lon, coords):
    """Pixeles cuyo centro cae dentro del poligono (algoritmo de ray casting)."""
    pts = [tuple(c) for c in coords]
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts = pts[:-1]
    x, y = lon.ravel(), lat.ravel()
    dentro = np.zeros(x.size, dtype=bool)
    j = len(pts) - 1
    for i in range(len(pts)):
        xi, yi = pts[i]
        xj, yj = pts[j]
        cruza = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-15) + xi)
        dentro ^= cruza
        j = i
    return dentro.reshape(lat.shape)


def procesar(props, modelo, ckpt, coords, escala, anno):
    """Aplica la U-Net a la respuesta de sampleRectangle y resume el resultado."""
    import torch
    from scipy import ndimage

    bandas = ckpt["bandas"]
    lat = np.asarray(props["latitude"], dtype=float)
    lon = np.asarray(props["longitude"], dtype=float)
    X = np.stack([np.asarray(props[b], dtype=np.float32) for b in bandas])
    # Orientacion: fila 0 = norte y columna 0 = oeste (lo que espera Leaflet)
    if lat.shape[0] > 1 and lat[0, 0] < lat[-1, 0]:
        lat, lon, X = lat[::-1], lon[::-1], X[:, ::-1]
    if lon.shape[1] > 1 and lon[0, 0] > lon[0, -1]:
        lat, lon, X = lat[:, ::-1], lon[:, ::-1], X[:, :, ::-1]
    alto, ancho = lat.shape

    valido = np.any(X != 0, axis=0)
    media = np.asarray(ckpt["media"], dtype=np.float32)[:, None, None]
    desv = np.asarray(ckpt["desv"], dtype=np.float32)[:, None, None]
    Xn = (X - media) / desv
    Xn = np.pad(Xn, ((0, 0), (0, (-alto) % 8), (0, (-ancho) % 8)), mode="edge")
    with torch.no_grad():
        logits = modelo(torch.from_numpy(np.ascontiguousarray(Xn, dtype=np.float32))[None])
    prob = torch.sigmoid(logits)[0, 0].numpy()[:alto, :ancho]
    prob[~valido] = 0.0

    umbral = float(ckpt.get("umbral", 0.5))
    mascara = prob >= umbral
    etiquetas, n = ndimage.label(mascara, structure=np.ones((3, 3)))
    dlat = abs(lat[1, 0] - lat[0, 0]) if alto > 1 else escala * GRADOS_POR_METRO
    dlon = abs(lon[0, 1] - lon[0, 0]) if ancho > 1 else escala * GRADOS_POR_METRO
    area_px_m2 = (dlat * 111320.0) * (dlon * 111320.0 * math.cos(math.radians(float(lat.mean()))))
    dentro = _dentro_poligono(lat, lon, coords)

    masas = []
    for k in range(1, n + 1):
        m = etiquetas == k
        npx = int(m.sum())
        if npx < AREA_MIN_PX:
            mascara[m] = False
            continue
        masas.append({
            "area_ha": round(float(npx * area_px_m2 / 1e4), 3),
            "lat": round(float(lat[m].mean()), 6),
            "lon": round(float(lon[m].mean()), 6),
            "probabilidad_media": round(float(prob[m].mean()), 2),
            "en_parcela": bool((m & dentro).any()),
        })
    masas.sort(key=lambda d: -d["area_ha"])
    prob_mostrar = np.where(mascara, prob, np.minimum(prob, umbral - 0.01))
    met = ckpt.get("metricas_test", {})
    return {
        "disponible": True,
        "bounds": [[float(lat.min() - dlat / 2), float(lon.min() - dlon / 2)],
                   [float(lat.max() + dlat / 2), float(lon.max() + dlon / 2)]],
        "alto": int(alto), "ancho": int(ancho),
        "probabilidad_pct": np.round(np.clip(prob_mostrar, 0, 1) * 100).astype(int).ravel().tolist(),
        "umbral_pct": int(round(umbral * 100)),
        "resolucion_m": int(escala),
        "periodo": f"junio-septiembre {anno} (Sentinel-2)",
        "area_agua_ha": round(float(mascara.sum() * area_px_m2 / 1e4), 2),
        "area_agua_parcela_ha": round(float((mascara & dentro).sum() * area_px_m2 / 1e4), 2),
        "n_masas": len(masas),
        "masas": masas[:10],
        "pixeles_sin_datos_pct": round(float((~valido).mean() * 100), 1),
        "modelo": {"nombre": "U-Net (CNN)", "iou_test": met.get("iou"), "f1_test": met.get("f1_dice"),
                   "precision_test": met.get("precision"), "recall_test": met.get("recall")},
        "aviso": (f"Parcela grande: analizada a {escala} m por pixel (la red se entreno a 10 m)."
                  if escala > 10 else None),
    }


def detectar_agua_superficial(coords):
    if not unet_disponible():
        return {"disponible": False, "mensaje": "La U-Net no esta entrenada (ejecuta unet/entrenar.py)."}
    modelo, ckpt = _cargar_modelo()
    anno = _ultimo_verano()
    props, escala = _descargar(_bbox(coords), ckpt["bandas"], anno)
    return procesar(props, modelo, ckpt, coords, escala, anno)
