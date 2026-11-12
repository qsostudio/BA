"""
ign_service.py — Cálculo de vaguadas, escorrentías y puntos de acumulación de agua.

Algoritmo:
1. A partir del polígono de la finca, genera una malla de puntos 20x20.
2. Consulta OpenTopoData (SRTM 90m) para obtener la elevación de cada punto.
3. Aplica el algoritmo D8 (eight-direction flow) para calcular dirección y acumulación de flujo.
4. Extrae la red de drenaje (celdas con alta acumulación) como GeoJSON LineStrings,
   diferenciando escorrentía (flujo difuso) de vaguada (cauce concentrado), y
   estimando el caudal de cada tramo mediante el Método Racional.
5. Detecta los puntos de máxima acumulación (mejores sitios para una balsa).
"""

import asyncio
import numpy as np
import httpx
from typing import List, Tuple


# ── Configuración de la malla ────────────────────────────────────────────────
GRID_SIZE   = 20   # malla 20×20 → 400 puntos de elevación
BATCH_SIZE  = 100  # OpenTopoData acepta máx. 100 puntos por petición
STREAM_THR  = 8    # umbral de acumulación para considerar línea de drenaje
VAGUADA_THR = 25   # umbral a partir del cual la escorrentía se considera vaguada
                    # (cauce concentrado, no solo flujo difuso) — heurístico,
                    # ajustable igual que STREAM_THR.
TOP_POINTS  = 7    # número de puntos de acumulación máxima a devolver

# ── Método Racional (estimación de caudal) ───────────────────────────────────
# Q (m3/s) = C · I · A / 360   con A en hectáreas, I en mm/h.
# Ambas constantes son valores de referencia asumidos, NO mediciones reales
# ni datos de suelo/lluvia específicos de la parcela — sirven para dar una
# magnitud orientativa (L/s), no una previsión hidrológica precisa.
COEF_ESCORRENTIA_C   = 0.30   # típico de terreno agrícola/rural, pendiente moderada
INTENSIDAD_LLUVIA_MMH = 20.0  # lluvia de referencia moderada-intensa (valor habitual
                               # de diseño para pequeñas cuencas agrícolas en España)


def _area_celda_m2(lat_grid: np.ndarray, lon_grid: np.ndarray) -> float:
    """Área real (m²) que representa cada celda de la malla, a partir de su
    espaciado en grados convertido a metros en el punto medio de la parcela."""
    lat_medio = float(lat_grid.mean())
    m_por_grado_lat = 111320.0
    m_por_grado_lon = 111320.0 * float(np.cos(np.radians(lat_medio)))
    espaciado_lat_m = abs(float(lat_grid[0] - lat_grid[1])) * m_por_grado_lat
    espaciado_lon_m = abs(float(lon_grid[0] - lon_grid[1])) * m_por_grado_lon
    return float(espaciado_lat_m * espaciado_lon_m)


def _caudal_racional_ls(area_ha: float) -> float:
    """Método Racional: Q = C·I·A/360 (m3/s con A en ha, I en mm/h) → L/s."""
    q_m3s = COEF_ESCORRENTIA_C * INTENSIDAD_LLUVIA_MMH * area_ha / 360.0
    return q_m3s * 1000.0


# ── 1. Obtener elevaciones en batch desde OpenTopoData ───────────────────────
async def _fetch_elevations_batch(
    client: httpx.AsyncClient, points: List[Tuple[float, float]]
) -> List[float]:
    locations = "|".join(f"{lat},{lon}" for lat, lon in points)
    r = await client.get(
        "https://api.opentopodata.org/v1/srtm90m",
        params={"locations": locations},
        timeout=20,
    )
    data = r.json()
    return [res["elevation"] if res["elevation"] is not None else 0.0
            for res in data["results"]]


async def _obtener_malla_elevaciones(
    lat_grid: np.ndarray, lon_grid: np.ndarray
) -> np.ndarray:
    """Consulta OpenTopoData para todos los puntos de la malla en batches."""
    n = GRID_SIZE
    all_points = [(lat_grid[i], lon_grid[j])
                  for i in range(n) for j in range(n)]

    elevaciones = []
    async with httpx.AsyncClient() as client:
        for start in range(0, len(all_points), BATCH_SIZE):
            batch = all_points[start : start + BATCH_SIZE]
            elevs = await _fetch_elevations_batch(client, batch)
            elevaciones.extend(elevs)
            if start + BATCH_SIZE < len(all_points):
                await asyncio.sleep(0.4)   # respetar rate-limit

    return np.array(elevaciones).reshape(n, n)


# ── 2. Algoritmo D8 ──────────────────────────────────────────────────────────
# Vecinos en orden: N, NE, E, SE, S, SW, W, NW
_DI = np.array([-1, -1, 0, 1, 1,  1,  0, -1])
_DJ = np.array([ 0,  1, 1, 1, 0, -1, -1, -1])
_DIST = np.array([1.0, 1.414, 1.0, 1.414, 1.0, 1.414, 1.0, 1.414])


def _flow_direction(dem: np.ndarray) -> np.ndarray:
    """Para cada celda, calcula la dirección de máxima pendiente descendente (0-7).

    Si ninguna vecina está más baja, la celda es una depresión (o un llano) y
    se marca con -1: el agua se queda ahí. Antes se elegía igualmente la vecina
    "menos alta", el agua subía cuesta arriba, se formaban ciclos y una misma
    celda se contaba dos veces (acumulaciones mayores que el total de celdas).
    """
    rows, cols = dem.shape
    direction = np.full((rows, cols), -1, dtype=np.int8)

    for i in range(rows):
        for j in range(cols):
            max_slope = 0.0      # solo se acepta una pendiente estrictamente descendente
            best_d    = -1
            for d in range(8):
                ni, nj = i + _DI[d], j + _DJ[d]
                if 0 <= ni < rows and 0 <= nj < cols:
                    slope = (dem[i, j] - dem[ni, nj]) / _DIST[d]
                    if slope > max_slope:
                        max_slope = slope
                        best_d    = d
            direction[i, j] = best_d

    return direction


def _flow_accumulation(dem: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Acumulación de flujo: número de celdas que drenan hacia cada celda."""
    rows, cols = dem.shape
    accumulation = np.ones((rows, cols), dtype=np.float32)

    # Procesar de mayor a menor elevación (orden topológico)
    order = np.argsort(dem.ravel())[::-1]

    for idx in order:
        i, j = divmod(int(idx), cols)
        d     = int(direction[i, j])
        if d < 0:                       # depresión: el agua no sale de esta celda
            continue
        ni, nj = i + _DI[d], j + _DJ[d]
        if 0 <= ni < rows and 0 <= nj < cols:
            accumulation[ni, nj] += accumulation[i, j]

    return accumulation


# ── 3. Extraer red de drenaje como GeoJSON ───────────────────────────────────
def _extraer_lineas_drenaje(
    accumulation: np.ndarray,
    direction: np.ndarray,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    area_celda_m2: float,
) -> list:
    """
    Traza segmentos de línea entre celdas de alta acumulación.
    Cada segmento se clasifica como "escorrentia" (flujo difuso) o "vaguada"
    (cauce concentrado, acumulación >= VAGUADA_THR), y lleva su caudal
    estimado (L/s) por el Método Racional a partir del área de drenaje real
    que representan sus celdas acumuladas.
    Devuelve lista de Features GeoJSON de tipo LineString.
    """
    rows, cols = accumulation.shape
    features   = []

    stream_mask = accumulation >= STREAM_THR

    for i in range(rows):
        for j in range(cols):
            if not stream_mask[i, j]:
                continue
            d   = int(direction[i, j])
            if d < 0:                   # depresión: no hay tramo que dibujar
                continue
            ni  = i + _DI[d]
            nj  = j + _DJ[d]
            if 0 <= ni < rows and 0 <= nj < cols and stream_mask[ni, nj]:
                acum = float(accumulation[i, j])
                intensidad = min(1.0, acum / (GRID_SIZE * 2))
                area_ha    = (acum * area_celda_m2) / 10000.0
                categoria  = "vaguada" if acum >= VAGUADA_THR else "escorrentia"
                features.append({
                    "type": "Feature",
                    "properties": {
                        "tipo":        "linea_drenaje",
                        "categoria":   categoria,
                        "acumulacion": round(acum, 1),
                        "intensidad":  round(intensidad, 2),
                        "area_ha":     round(area_ha, 3),
                        "caudal_ls":   round(_caudal_racional_ls(area_ha), 2),
                    },
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [
                            [float(lon_grid[j]),  float(lat_grid[i])],
                            [float(lon_grid[nj]), float(lat_grid[ni])],
                        ],
                    },
                })

    return features


def _extraer_puntos_acumulacion(
    accumulation: np.ndarray,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
) -> list:
    """
    Devuelve los TOP_POINTS puntos con mayor acumulación de flujo.
    Son los mejores candidatos para una balsa o una zona de recarga. Indican
    dónde se junta el agua de escorrentía, no dónde hay agua subterránea.
    """
    flat_indices = np.argsort(accumulation.ravel())[::-1]
    rows, cols   = accumulation.shape
    features     = []
    vistos       = set()

    for idx in flat_indices:
        if len(features) >= TOP_POINTS:
            break
        i, j = divmod(int(idx), cols)

        # Evitar puntos demasiado cercanos entre sí (mínimo 2 celdas de distancia)
        demasiado_cerca = any(
            abs(i - vi) <= 2 and abs(j - vj) <= 2 for vi, vj in vistos
        )
        if demasiado_cerca:
            continue

        vistos.add((i, j))
        acum = float(accumulation[i, j])
        rang = min(3, max(1, int(acum / GRID_SIZE)))  # 1, 2 o 3 (tamaño visual)

        features.append({
            "type": "Feature",
            "properties": {
                "tipo":        "punto_acumulacion",
                "acumulacion": round(acum, 1),
                "rango":       rang,
                "etiqueta":    (
                    "💧 Alta acumulación — zona idónea para balsa"
                    if rang == 3 else
                    "💧 Acumulación media — posible zona de recarga"
                    if rang == 2 else
                    "💧 Acumulación baja — monitorizar tras lluvias"
                ),
            },
            "geometry": {
                "type":        "Point",
                "coordinates": [float(lon_grid[j]), float(lat_grid[i])],
            },
        })

    return features


# ── 4. Función principal ─────────────────────────────────────────────────────
async def calcular_vaguadas(coords: list) -> dict:
    """
    Punto de entrada principal.
    Recibe los coords del polígono [[lon, lat], ...] y devuelve GeoJSON.
    """
    # Bounding box
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    lon_min, lon_max = min(lons), max(lons)
    lat_min, lat_max = min(lats), max(lats)

    # Malla de puntos (fila = latitud de norte a sur, col = longitud de oeste a este)
    lat_grid = np.linspace(lat_max, lat_min, GRID_SIZE)
    lon_grid = np.linspace(lon_min, lon_max, GRID_SIZE)

    # Obtener elevaciones
    dem = await _obtener_malla_elevaciones(lat_grid, lon_grid)

    # Suavizar ligeramente el DEM para reducir artefactos
    from scipy.ndimage import uniform_filter
    dem_suave = uniform_filter(dem, size=2)

    # D8
    direction    = _flow_direction(dem_suave)
    accumulation = _flow_accumulation(dem_suave, direction)

    # Área real que representa cada celda (para pasar de "celdas acumuladas"
    # a hectáreas de cuenca de drenaje, y de ahí al caudal estimado).
    area_celda_m2 = _area_celda_m2(lat_grid, lon_grid)

    # Extraer red de drenaje y puntos de acumulación
    lineas  = _extraer_lineas_drenaje(accumulation, direction, lat_grid, lon_grid, area_celda_m2)
    puntos  = _extraer_puntos_acumulacion(accumulation, lat_grid, lon_grid)

    geojson = {
        "type": "FeatureCollection",
        "features": lineas + puntos,
    }

    n_vaguadas = sum(1 for l in lineas if l["properties"]["categoria"] == "vaguada")
    area_max_ha = (float(accumulation.max()) * area_celda_m2) / 10000.0

    return {
        "geojson":         geojson,
        "n_lineas":        len(lineas),
        "n_puntos":        len(puntos),
        "n_vaguadas":      n_vaguadas,
        "bbox":            [lon_min, lat_min, lon_max, lat_max],
        "acum_maxima":     round(float(accumulation.max()), 1),
        "caudal_maximo_ls": round(_caudal_racional_ls(area_max_ha), 2),
        "resolucion":      f"Malla {GRID_SIZE}×{GRID_SIZE} puntos (SRTM 90m)",
    }
