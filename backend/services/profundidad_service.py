"""
profundidad_service.py — Estimación de la profundidad del nivel freático y
del margen de excavación recomendado por debajo de ese nivel.

Enfoque (por este orden de preferencia):
  1) Interpolación ponderada por distancia (IDW) de pozos REALES cercanos,
     consultados en vivo al inventario de puntos de agua del IGME
     (cobertura nacional, datos de perforación desde 1966):
       https://mapas.igme.es/gis/rest/services/BasesDatos/IGME_PuntosAgua/MapServer/0
     Campo de profundidad confirmado en el esquema del servicio: PROF_m.
  2) Si no hay pozos reales en el radio de búsqueda, no se inventa ningún
     número de profundidad — se informa honestamente que no hay datos.

Siempre se informa la distancia al pozo real más cercano usado, para que
la confianza del dato sea explícita y no un número "mágico" sin respaldo.
"""

import math
import httpx

from services.geo_service import obtener_geologia

IGME_PUNTOS_AGUA_URL = (
    "https://mapas.igme.es/gis/rest/services/BasesDatos/IGME_PuntosAgua/MapServer/0/query"
)

RADIO_BUSQUEDA_M   = 20000  # 20 km — radio de búsqueda de pozos reales
MAX_POZOS_IDW      = 8      # nº máximo de pozos usados en la interpolación
POTENCIA_IDW       = 2      # potencia estándar del método IDW

DIST_CONFIANZA_ALTA_KM  = 3    # pozo real a <3 km  -> confianza alta
DIST_CONFIANZA_MEDIA_KM = 10   # pozo real a <10 km -> confianza media; más lejos, baja

# Margen de excavación recomendado por DEBAJO del nivel freático hallado,
# según tipo de acuífero. Son valores de referencia orientativos de la
# práctica agrícola habitual (para absorber el descenso estacional del
# nivel y el abatimiento por bombeo) — no una norma técnica oficial ni un
# cálculo hidráulico del pozo concreto.
MARGEN_POR_TIPO_M = {
    "aluvial":   (8, 15),
    "detritico": (10, 20),
    "karstico":  (15, 30),
    "fisurado":  (15, 30),
    "generico":  (10, 20),
}


async def _buscar_pozos_cercanos(lat: float, lon: float) -> tuple:
    """Consulta en vivo el inventario de puntos de agua del IGME dentro de
    una caja rectangular (envelope) alrededor del punto, filtrando a los
    que tienen profundidad conocida.

    Devuelve (features, debug) — debug lleva el código HTTP y un
    fragmento de la respuesta cruda, para poder ver EXACTAMENTE qué
    contesta el servidor real en vez de suponerlo.

    NOTA: se usa un envelope (xmin,ymin,xmax,ymax) en vez de "punto +
    distance/units", porque ese segundo modo de búsqueda por radio es una
    función más moderna de la API de ArcGIS REST que no todos los
    servidores soportan de forma fiable.
    """
    delta_lat = RADIO_BUSQUEDA_M / 111320.0
    delta_lon = RADIO_BUSQUEDA_M / (111320.0 * math.cos(math.radians(lat)))
    xmin, xmax = lon - delta_lon, lon + delta_lon
    ymin, ymax = lat - delta_lat, lat + delta_lat

    params = {
        "f": "json",
        "geometry": f"{xmin},{ymin},{xmax},{ymax}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "PROF_m,COTA_msnm,Naturaleza,Sistema_Acuifero,Unidad_Hidrogeologica,TOPONIMIA",
        "where": "PROF_m IS NOT NULL AND PROF_m > 0",
        "returnGeometry": "true",
    }
    debug = {"url_llamada": f"{IGME_PUNTOS_AGUA_URL}?" + "&".join(f"{k}={v}" for k, v in params.items())}
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(IGME_PUNTOS_AGUA_URL, params=params)
            debug["http_status"] = r.status_code
            debug["respuesta_cruda"] = r.text[:600]
            data = r.json()
        if isinstance(data, dict) and "error" in data:
            debug["error_arcgis"] = data["error"]
            return [], debug
        features = data.get("features", []) or []
        debug["n_features_encontradas"] = len(features)
        return features, debug
    except Exception as e:
        debug["excepcion"] = f"{type(e).__name__}: {e}"
        return [], debug


def _distancia_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distancia aproximada en metros (equirectangular — válida a la escala
    de una parcela o comarca; no pensada para grandes distancias)."""
    m_lat = 111320.0
    m_lon = 111320.0 * math.cos(math.radians((lat1 + lat2) / 2.0))
    dx = (lon2 - lon1) * m_lon
    dy = (lat2 - lat1) * m_lat
    return math.sqrt(dx * dx + dy * dy)


def _clasificar_tipo_acuifero(geologia: dict) -> str:
    """Clasifica el tipo de acuífero a partir del texto de litología (IGME
    GEODE o, en su defecto, Macrostrat), para elegir el margen de
    excavación recomendado."""
    texto = (geologia.get("lith") or "").lower()
    if any(t in texto for t in ["aluvial", "cuaternario", "arena", "grava", "coluvial"]):
        return "aluvial"
    if any(t in texto for t in ["caliza", "dolomia", "dolomita", "karst", "mármol", "travertino"]):
        return "karstico"
    if any(t in texto for t in ["granito", "gneis", "esquisto", "pizarra", "cuarcita",
                                 "metamórfico", "ígneo", "volcánico", "granodiorita", "filita"]):
        return "fisurado"
    if any(t in texto for t in ["arenisca", "conglomerado", "terciario", "detrítico",
                                 "arcilla", "marga", "limo"]):
        return "detritico"
    return "generico"


async def estimar_profundidad(lat: float, lon: float) -> dict:
    pozos_raw, debug_igme = await _buscar_pozos_cercanos(lat, lon)

    pozos = []
    for f in pozos_raw:
        attrs = f.get("attributes", {}) or {}
        geom  = f.get("geometry", {}) or {}
        prof  = attrs.get("PROF_m")
        px, py = geom.get("x"), geom.get("y")
        if prof is None or px is None or py is None:
            continue
        d = _distancia_m(lat, lon, float(py), float(px))
        pozos.append({
            "profundidad_m": float(prof),
            "distancia_m":   d,
            "toponimia":     attrs.get("TOPONIMIA"),
            "acuifero":      attrs.get("Sistema_Acuifero"),
        })

    pozos.sort(key=lambda p: p["distancia_m"])
    pozos_usados = pozos[:MAX_POZOS_IDW]

    geologia = await obtener_geologia(lat, lon)
    tipo_acuifero = _clasificar_tipo_acuifero(geologia)
    margen_min, margen_max = MARGEN_POR_TIPO_M.get(tipo_acuifero, MARGEN_POR_TIPO_M["generico"])

    if pozos_usados:
        pesos = [1.0 / max(p["distancia_m"], 1.0) ** POTENCIA_IDW for p in pozos_usados]
        suma_pesos = sum(pesos)
        profundidad_est = sum(p["profundidad_m"] * w for p, w in zip(pozos_usados, pesos)) / suma_pesos

        distancia_min_km = pozos_usados[0]["distancia_m"] / 1000.0
        if distancia_min_km <= DIST_CONFIANZA_ALTA_KM:
            confianza = "alta"
        elif distancia_min_km <= DIST_CONFIANZA_MEDIA_KM:
            confianza = "media"
        else:
            confianza = "baja"

        profundidades = [p["profundidad_m"] for p in pozos_usados]
        if len(profundidades) >= 2 and (max(profundidades) - min(profundidades)) > 20:
            margen_max += 10

        metodo = f"Interpolación ponderada por distancia de {len(pozos_usados)} pozo(s) reales (inventario IGME)"
    else:
        profundidad_est   = None
        confianza         = "sin_datos_reales"
        distancia_min_km  = None
        metodo = "Sin pozos reales en 20 km a la redonda - no se ha inventado ningún número"

    profundidad_redondeada = round(profundidad_est, 1) if profundidad_est is not None else None

    return {
        "profundidad_freatico_m": profundidad_redondeada,
        "confianza":              confianza,
        "pozo_mas_cercano_km":    round(distancia_min_km, 2) if distancia_min_km is not None else None,
        "n_pozos_usados":         len(pozos_usados),
        "metodo":                 metodo,
        "tipo_acuifero_estimado": tipo_acuifero,
        "geologia":               geologia.get("name", "Desconocido"),
        "margen_recomendado_min_m": margen_min,
        "margen_recomendado_max_m": margen_max,
        "profundidad_total_min_m": (
            round(profundidad_est + margen_min, 1) if profundidad_est is not None else None
        ),
        "profundidad_total_max_m": (
            round(profundidad_est + margen_max, 1) if profundidad_est is not None else None
        ),
        "pozos_referencia": [
            {
                "profundidad_m": p["profundidad_m"],
                "distancia_km":  round(p["distancia_m"] / 1000.0, 2),
                "toponimia":     p["toponimia"],
            }
            for p in pozos_usados[:5]
        ],
        "debug_igme": debug_igme,
    }
