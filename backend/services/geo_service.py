import math
import httpx

# Vocabulario de clasificación — combina los términos en inglés (Macrostrat)
# con los términos en español del IGME (los mismos ya usados para depurar
# los datos de entrenamiento en corregir_geologia_igme.py /
# fusionar_geologia_excel.py), para que la clasificación funcione igual de
# bien venga la litología de una fuente o de la otra.
ROCAS_BUENAS = [
    # Macrostrat (inglés)
    "limestone", "carbonate", "dolomite", "sandstone", "gravel", "alluvium",
    # IGME (español)
    "caliza", "calcarenita", "dolomia", "dolomita", "mármol", "travertino",
    "arenisca", "conglomerado", "grava", "arena", "aluvial", "cuaternario",
    "terciario", "detrítico", "carbonato", "arcosa", "fango", "limo", "brecha",
    "calcoesquisto", "pizarra calcárea", "arenoso", "areniscas", "terraza", "terrazas",
]
ROCAS_MALAS = [
    # Macrostrat (inglés)
    "granite", "schist", "slate", "gneiss", "basalt",
    # IGME (español)
    "granito", "granodiorita", "diorita", "gabro", "riolita", "andesita",
    "pórfido", "esquisto", "gneis", "filita", "pizarra", "cuarcita", "cuarzo",
    "migmatita", "anfibolita", "serpentinita", "peridotita", "metamórfico",
    "plutónico", "volcánico", "intrusivo",
]
TRAD = {
    "limestone": "Caliza",     "carbonate": "Carbonato", "dolomite": "Dolomita",
    "sandstone": "Arenisca",   "gravel":    "Grava",     "alluvium": "Aluvial",
    "granite":   "Granito",    "schist":    "Esquisto",  "slate":    "Pizarra",
    "gneiss":    "Gneis",      "basalt":    "Basalto",
}


async def obtener_humedad(lat: float, lon: float) -> float:
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude":     lat,
                "longitude":    lon,
                "hourly":       "soil_moisture_0_1cm",
                "forecast_days": 1,
            },
        )
        data = r.json()
        return data["hourly"]["soil_moisture_0_1cm"][0]


async def obtener_elevacion_srtm(lat: float, lon: float) -> float | None:
    """SRTM 90m vía OpenTopoData — cobertura global, resolución basta."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(
                "https://api.opentopodata.org/v1/srtm90m",
                params={"locations": f"{lat},{lon}"},
            )
            data = r.json()
            elev = data["results"][0]["elevation"]
            return float(elev) if elev is not None else None
    except Exception:
        return None


async def obtener_elevacion_ign(lat: float, lon: float) -> float | None:
    """
    MDT05 del IGN — 5m de resolución, LiDAR real del vuelo PNOA. Mucho más
    fino que SRTM (90m) para el territorio español.
    Servicio OGC API Processes oficial del IGN/CNIG, endpoint getElevation:
      https://api-processes.idee.es/processes/getElevation/execution
    Devuelve None si el punto cae fuera de la cobertura (fuera de España)
    o si el servicio no responde, para poder caer a SRTM sin romper nada.
    """
    try:
        geom = (
            '{"type": "FeatureCollection","features": [{"geometry":'
            f'{{"coordinates": [{lon},{lat}],"type":"Point"}},"type": "Feature"}}]}}'
        )
        body = {
            "inputs": {
                "crs": 4326,
                "formato": "geojson",
                "geom": geom,
                "outputFormat": "array",
                "withCoord": True,
            }
        }
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(
                "https://api-processes.idee.es/processes/getElevation/execution",
                json=body,
                headers={"accept": "application/json", "Content-Type": "application/json"},
            )
            data = r.json()
        valores = data.get("values", [])
        if not valores or not valores[0]:
            return None
        # withCoord=true -> [lon, lat, elevacion]; el valor de elevación es el último.
        elevacion = valores[0][-1]
        return float(elevacion) if elevacion is not None else None
    except Exception:
        return None


async def obtener_elevacion(lat: float, lon: float) -> float:
    """
    Elevación de un punto — IGN (MDT05, 5m) primero, SRTM (90m) como red de
    seguridad si el punto cae fuera de España o el servicio del IGN falla.
    Devuelve un único float, para no romper ningún código existente que ya
    llama a esta función esperando un número simple.
    """
    elev_ign = await obtener_elevacion_ign(lat, lon)
    if elev_ign is not None:
        return elev_ign
    elev_srtm = await obtener_elevacion_srtm(lat, lon)
    return elev_srtm if elev_srtm is not None else 0.0


async def obtener_elevacion_comparada(lat: float, lon: float) -> dict:
    """
    Igual que obtener_elevacion(), pero devuelve AMBAS fuentes y la
    diferencia entre ellas — útil como dato informativo: una diferencia
    grande entre el LiDAR del IGN (que "ve" el suelo desnudo) y el SRTM
    (que a veces capta la copa vegetal) puede delatar cubierta vegetal
    densa en ese punto exacto.
    """
    elev_ign  = await obtener_elevacion_ign(lat, lon)
    elev_srtm = await obtener_elevacion_srtm(lat, lon)
    diferencia = (
        round(elev_ign - elev_srtm, 1)
        if (elev_ign is not None and elev_srtm is not None)
        else None
    )
    return {
        "elevacion_ign_m":  round(elev_ign, 1) if elev_ign is not None else None,
        "elevacion_srtm_m": round(elev_srtm, 1) if elev_srtm is not None else None,
        "diferencia_m":     diferencia,
        "fuente_principal": "IGN (MDT05, 5m)" if elev_ign is not None else "SRTM (90m) — fuera de cobertura IGN",
    }


async def obtener_geologia_igme(lat: float, lon: float) -> dict:
    """
    Mapa Geológico Continuo de España (GEODE, IGME) — escala 1:50.000,
    muchísimo más detallado que Macrostrat para el territorio español.
    Servicio ArcGIS REST oficial y gratuito del IGME, capa 8 ("Recintos
    geología"): https://mapas.igme.es/gis/rest/services/Cartografia_Geologica/IGME_Geode_50/MapServer/8

    Usa /query (no /identify) con una caja pequeña alrededor del punto —
    el mismo patrón que demostró ser fiable para el inventario de pozos
    del IGME (profundidad_service.py); /identify dio resultados menos
    consistentes en las pruebas anteriores.

    Pide TODOS los campos (outFields=*) en vez de solo DESC_UNIT: ese
    campo, aunque es el de visualización oficial, no siempre viene
    relleno en cada recinto — cuando está vacío, se usa CODE_COTR (el
    código de clasificación, que sí es el que usa el propio IGME para
    colorear el mapa y por eso está prácticamente siempre presente) como
    respaldo, en vez de rendirse directamente a "Desconocido".
    """
    try:
        url = "https://mapas.igme.es/gis/rest/services/Cartografia_Geologica/IGME_Geode_50/MapServer/8/query"
        delta = 0.001  # caja pequeña (~100m) alrededor del punto, solo para garantizar la intersección
        params = {
            "f":            "json",
            "geometry":     f"{lon - delta},{lat - delta},{lon + delta},{lat + delta}",
            "geometryType": "esriGeometryEnvelope",
            "inSR":         "4326",
            "spatialRel":   "esriSpatialRelIntersects",
            "outFields":    "*",
            "returnGeometry": "false",
        }
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url, params=params)
            data = r.json()
        features = data.get("features", [])
        if not features:
            return {"lith": "", "descrip": "No disponible", "name": "Desconocido"}

        attrs = features[0].get("attributes", {})
        desc_unit = (attrs.get("DESC_UNIT") or "").strip()
        if desc_unit:
            return {"lith": desc_unit, "descrip": desc_unit, "name": desc_unit}

        # DESC_UNIT vacío en este recinto concreto — probamos con el
        # código de clasificación como respaldo (dato real del IGME, no
        # inventado; menos legible pero cierto).
        code = attrs.get("CODE_COTR")
        if code not in (None, ""):
            texto = f"Unidad geológica IGME (código {code})"
            return {"lith": str(code), "descrip": texto, "name": texto}

        return {"lith": "", "descrip": "No disponible", "name": "Desconocido"}
    except Exception:
        return {"lith": "", "descrip": "No disponible", "name": "Desconocido"}


# Tokens que NUNCA son un nombre de unidad geológica real — si una fuente
# los devuelve, es señal de que se ha colado un campo equivocado (p. ej.
# el tipo de geometría de un feature GeoJSON), no un dato real.
_TOKENS_NO_GEOLOGICOS = {
    "polygon", "multipolygon", "point", "multipoint",
    "linestring", "multilinestring", "geometrycollection", "null", "none",
}


def _es_nombre_geologico_valido(valor: str) -> bool:
    return bool(valor) and valor.strip().lower() not in _TOKENS_NO_GEOLOGICOS


async def obtener_geologia(lat: float, lon: float) -> dict:
    # 1) IGME primero — cobertura y detalle muy superiores en España.
    geo = await obtener_geologia_igme(lat, lon)
    if _es_nombre_geologico_valido(geo.get("name", "")):
        return geo

    # 2) Macrostrat como red de seguridad (fuera de España, o huecos
    #    puntuales de cobertura del GEODE).
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(
                f"https://macrostrat.org/api/mobile/map_query",
                params={"lng": lon, "lat": lat, "z": 12},
            )
            data    = r.json()
            geo_list = data.get("success", {}).get("data", {}).get("mapData", [])
            if geo_list:
                candidato = {
                    "lith":   geo_list[0].get("lith", ""),
                    "descrip": geo_list[0].get("descrip", "Sin descripción"),
                    "name":   geo_list[0].get("name", "Desconocido"),
                }
                if _es_nombre_geologico_valido(candidato["name"]):
                    return candidato
    except Exception:
        pass
    return {"lith": "", "descrip": "No disponible", "name": "Desconocido"}


def clasificar_geologia(lith: str):
    lith_lower = lith.lower()
    roca_es    = next((TRAD[k] for k in TRAD if k in lith_lower), lith or "Desconocido")

    if any(r in lith_lower for r in ROCAS_BUENAS):
        return roca_es, 40, "buena"
    elif any(r in lith_lower for r in ROCAS_MALAS):
        return roca_es, 0, "mala"
    else:
        return roca_es, 20, "media"
