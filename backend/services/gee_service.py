import os
import json
import ee


def init_gee() -> bool:
    try:
        key_path = os.getenv("GEE_KEY_PATH", "gee_key.json")
        if not os.path.exists(key_path):
            print(f"[GEE] Fichero de credenciales no encontrado: {key_path}")
            return False

        with open(key_path, "r") as f:
            key_data = json.load(f)

        from google.oauth2 import service_account as google_sa
        scopes = ["https://www.googleapis.com/auth/earthengine"]
        credentials = google_sa.Credentials.from_service_account_info(key_data, scopes=scopes)
        ee.Initialize(credentials)
        return True

    except Exception as e:
        print(f"[GEE] Error al inicializar: {e}")
        return False


def get_ndvi_punto(lat: float, lon: float):
    try:
        point = ee.Geometry.Point([lon, lat])
        s2 = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(point)
            .filterDate("2024-01-01", "2024-12-31")
            .sort("CLOUDY_PIXEL_PERCENTAGE")
            .first()
        )
        ndvi_value = (
            s2.normalizedDifference(["B8", "B4"])
            .rename("NDVI")
            .reduceRegion(reducer=ee.Reducer.mean(), geometry=point, scale=10)
            .get("NDVI")
            .getInfo()
        )
        return ndvi_value
    except Exception as e:
        print(f"[GEE] Error NDVI punto: {e}")
        return None


def analizar_parcela_gee(coords: list) -> dict:
    poligono = ee.Geometry.Polygon(coords)

    # NDVI Sentinel-2
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(poligono)
        .filterDate("2024-01-01", "2024-12-31")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .median()
    )
    ndvi = s2.normalizedDifference(["B8", "B4"]).rename("NDVI")

    # Humedad del suelo ERA5
    era5 = (
        ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")
        .filterBounds(poligono)
        .filterDate("2023-01-01", "2024-12-31")
        .select("volumetric_soil_water_layer_1")
        .mean()
        .rename("HUMEDAD")
    )

    # Elevación y curvatura SRTM
    dem          = ee.Image("USGS/SRTMGL1_003")
    elevacion_img = dem.select("elevation").rename("ELEV")
    curvatura    = elevacion_img.convolve(ee.Kernel.laplacian8()).multiply(-1).rename("CURV")

    # Normalizar 0-1
    def normalizar(imagen, nombre):
        stats = imagen.reduceRegion(
            reducer=ee.Reducer.minMax(),
            geometry=poligono,
            scale=50,
            maxPixels=1e8
        )
        min_val = ee.Number(stats.get(nombre + "_min"))
        max_val = ee.Number(stats.get(nombre + "_max"))
        return imagen.subtract(min_val).divide(max_val.subtract(min_val)).rename(nombre + "_NORM")

    ndvi_norm    = normalizar(ndvi, "NDVI")
    humedad_norm = normalizar(era5, "HUMEDAD")
    elev_inv     = elevacion_img.multiply(-1).rename("ELEV_INV")
    elev_norm    = normalizar(elev_inv, "ELEV_INV")
    curv_norm    = normalizar(curvatura, "CURV")

    # Score ponderado
    score = (
        ndvi_norm.multiply(0.35)
        .add(humedad_norm.multiply(0.30))
        .add(elev_norm.multiply(0.20))
        .add(curv_norm.multiply(0.15))
        .rename("SCORE")
    )
    score_limpio = score.focal_median(radius=2, kernelType="circle", units="pixels")

    # Punto máximo
    max_score = score_limpio.reduceRegion(
        reducer=ee.Reducer.max(), geometry=poligono, scale=50, maxPixels=1e8
    ).get("SCORE").getInfo()

    punto_max  = score_limpio.eq(ee.Image.constant(max_score))
    coords_max = (
        punto_max.reduceToVectors(geometry=poligono, scale=50, maxPixels=1e8)
        .geometry().centroid(maxError=1).coordinates().getInfo()
    )
    lon_max, lat_max = coords_max[0], coords_max[1]

    # Calidad
    n_imagenes = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(poligono)
        .filterDate("2024-01-01", "2024-12-31")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .size().getInfo()
    )

    if n_imagenes >= 10:
        calidad, calidad_txt = "Alta", f"{n_imagenes} imágenes satelitales limpias"
    elif n_imagenes >= 4:
        calidad, calidad_txt = "Media", f"Solo {n_imagenes} imágenes limpias"
    else:
        calidad, calidad_txt = "Baja", f"Muy pocas imágenes ({n_imagenes})"

    # Valores en el punto óptimo
    punto_geom = ee.Geometry.Point([lon_max, lat_max])
    ndvi_val   = ndvi.reduceRegion(reducer=ee.Reducer.mean(), geometry=punto_geom, scale=10).get("NDVI").getInfo()
    hum_val    = era5.reduceRegion(reducer=ee.Reducer.mean(), geometry=punto_geom, scale=100).get("HUMEDAD").getInfo()
    elev_val   = elevacion_img.reduceRegion(reducer=ee.Reducer.mean(), geometry=punto_geom, scale=30).get("ELEV").getInfo()

    # Muestreo de puntos para el mapa de calor. A 20 m (antes 50 m): en una
    # parcela de 2-3 ha a 50 m solo caben unos 10 píxeles y el mapa casi no se ve.
    puntos_muestreados = score_limpio.sample(
        region=poligono, scale=20, numPixels=300, geometries=True
    ).getInfo()

    puntos_a, puntos_b, puntos_c = [], [], []
    puntos_heatmap = []

    for feat in puntos_muestreados["features"]:
        sc = feat["properties"].get("SCORE", 0)
        cp = feat["geometry"]["coordinates"]
        if sc is None:
            continue
        sc = float(sc)
        puntos_heatmap.append([cp[1], cp[0], round(sc, 3)])
        pt = {"lat": cp[1], "lon": cp[0], "score": round(sc, 3)}
        if sc >= 0.70:
            puntos_a.append(pt)
        elif sc >= 0.55:
            puntos_b.append(pt)
        elif sc >= 0.40:
            puntos_c.append(pt)

    return {
        "pozo_lat":       lat_max,
        "pozo_lon":       lon_max,
        "pozo_score":     round(max_score * 100, 1),
        "calidad":        calidad,
        "calidad_txt":    calidad_txt,
        "ndvi":           ndvi_val,
        "humedad":        hum_val,
        "elevacion":      elev_val,
        "puntos_a":       puntos_a,
        "puntos_b":       puntos_b,
        "puntos_c":       puntos_c,
        "puntos_heatmap": puntos_heatmap,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SERIE TEMPORAL NDVI — 4 años mes a mes
# ─────────────────────────────────────────────────────────────────────────────
def get_ndvi_serie(coords: list) -> dict:
    """
    Calcula el NDVI medio mensual de la parcela desde 2021 hasta 2024.
    Devuelve una lista ordenada de {fecha, ndvi, mes, anio}.
    La firma de un acuífero es NDVI alto en julio-agosto cuando todo lo demás se seca.
    """
    poligono = ee.Geometry.Polygon(coords)

    # Colección completa 4 años con NDVI calculado por imagen
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(poligono)
        .filterDate("2021-01-01", "2024-12-31")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
        .map(lambda img: img
             .normalizedDifference(["B8", "B4"]).rename("NDVI")
             .set({
                 "mes":   img.date().get("month"),
                 "anio":  img.date().get("year"),
                 "fecha": img.date().format("YYYY-MM"),
             }))
    )

    # Reducir cada imagen al NDVI medio de la parcela (server-side)
    def reducir(img):
        val = img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=poligono,
            scale=20,
            maxPixels=1e8,
        )
        return ee.Feature(None, {
            "fecha": img.get("fecha"),
            "mes":   img.get("mes"),
            "anio":  img.get("anio"),
            "ndvi":  val.get("NDVI"),
        })

    data = s2.map(reducir).getInfo()

    # Agrupar por mes-año y promediar imágenes del mismo mes
    from collections import defaultdict
    agrupado = defaultdict(list)
    for feat in data["features"]:
        p = feat["properties"]
        if p.get("ndvi") is not None:
            agrupado[p["fecha"]].append(float(p["ndvi"]))

    serie = []
    for fecha, valores in sorted(agrupado.items()):
        partes = fecha.split("-")
        serie.append({
            "fecha": fecha,
            "ndvi":  round(sum(valores) / len(valores), 3),
            "mes":   int(partes[1]),
            "anio":  int(partes[0]),
        })

    # Detectar meses de verano con NDVI alto (firma del acuífero)
    verano_alto = [p for p in serie if p["mes"] in [6, 7, 8] and p["ndvi"] > 0.35]
    tiene_firma = len(verano_alto) >= 3

    return {
        "serie":       serie,
        "tiene_firma": tiene_firma,
        "n_imagenes":  len(data["features"]),
        "meses_verano_alto": len(verano_alto),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ANOMALÍAS TÉRMICAS — Landsat temperatura superficial en verano
# ─────────────────────────────────────────────────────────────────────────────
def get_thermal_anomalias(coords: list) -> dict:
    """
    Usa la banda ST_B10 de Landsat 8/9 para detectar zonas más frescas en verano.
    Las zonas frescas sobre acuíferos son potenciales puntos de captación.
    """
    poligono = ee.Geometry.Polygon(coords)

    # Landsat 9 Collection 2 — temperatura superficial en verano (jun-ago 2022-2024)
    def preparar_lst(img):
        lst_celsius = img.select("ST_B10").multiply(0.00341802).add(149.0).subtract(273.15)
        return lst_celsius.rename("LST").set("system:time_start", img.get("system:time_start"))

    landsat = (
        ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
        .filterBounds(poligono)
        .filterDate("2022-06-01", "2024-08-31")
        .filter(ee.Filter.lt("CLOUD_COVER", 20))
        .filter(ee.Filter.calendarRange(6, 8, "month"))
        .map(preparar_lst)
    )

    # Intentar con Landsat 8 si no hay suficientes imágenes de L9
    n_l9 = landsat.size().getInfo()
    if n_l9 < 3:
        landsat = (
            ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .filterBounds(poligono)
            .filterDate("2021-06-01", "2024-08-31")
            .filter(ee.Filter.lt("CLOUD_COVER", 20))
            .filter(ee.Filter.calendarRange(6, 8, "month"))
            .map(preparar_lst)
        )

    lst_mediana = landsat.median()

    # Estadísticas dentro del polígono
    stats = lst_mediana.reduceRegion(
        reducer=ee.Reducer.minMax().combine(ee.Reducer.mean(), sharedInputs=True),
        geometry=poligono,
        scale=30,
        maxPixels=1e8,
    ).getInfo()

    lst_min  = stats.get("LST_min",  20)
    lst_max  = stats.get("LST_max",  50)
    lst_mean = stats.get("LST_mean", 35)

    if lst_min is None or lst_max is None:
        return {"puntos_frios": [], "temp_media": None, "n_imagenes": 0}

    rango = lst_max - lst_min if lst_max != lst_min else 1

    # Normalizar y muestrear
    lst_norm = lst_mediana.subtract(lst_min).divide(rango).rename("LST_NORM")

    puntos_raw = lst_norm.sample(
        region=poligono,
        scale=30,
        numPixels=250,
        geometries=True,
    ).getInfo()

    puntos_frios = []
    for feat in puntos_raw["features"]:
        val = feat["properties"].get("LST_NORM", 1.0)
        if val is not None and val <= 0.25:   # 25% más fresco
            cp = feat["geometry"]["coordinates"]
            temp_real = round(lst_min + val * rango, 1)
            puntos_frios.append({
                "lat":       cp[1],
                "lon":       cp[0],
                "temp_norm": round(float(val), 3),
                "temp_c":    temp_real,
            })

    return {
        "puntos_frios": puntos_frios,
        "temp_media":   round(float(lst_mean), 1),
        "temp_min":     round(float(lst_min), 1),
        "temp_max":     round(float(lst_max), 1),
        "n_imagenes":   landsat.size().getInfo(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PUNTO ÓPTIMO PARA POZO PROFUNDO (agua subterránea) — distinto del punto
# óptimo de "balsa" (analizar_parcela_gee, que usa NDVI anual + humedad +
# elevación + curvatura: señales de acumulación SUPERFICIAL de agua).
#
# Aquí se usan solo señales que indican agua subterránea PERSISTENTE:
#   - NDVI de verano (jun-ago), no la mediana anual: vegetación que se
#     mantiene verde cuando todo se seca solo se explica por raíces
#     alcanzando agua profunda ("firma de acuífero", misma lógica que
#     get_ndvi_serie, aplicada aquí como mapa en vez de serie temporal).
#   - Anomalía térmica de verano (Landsat): las zonas más frescas de lo
#     normal en verano indican evapotranspiración de vegetación con
#     acceso a agua subterránea.
# El refinamiento con pozos reales del IGME y geología se hace fuera de
# GEE, en pozo_profundo_service.py (son fuentes no satelitales).
# ─────────────────────────────────────────────────────────────────────────────
def candidatos_pozo_profundo_gee(coords: list) -> list:
    poligono = ee.Geometry.Polygon(coords)

    # NDVI de verano (jun-ago, 2022-2024) — no la mediana anual.
    s2_verano = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(poligono)
        .filterDate("2022-06-01", "2024-08-31")
        .filter(ee.Filter.calendarRange(6, 8, "month"))
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
        .median()
    )
    ndvi_verano = s2_verano.normalizedDifference(["B8", "B4"]).rename("NDVI_VERANO")

    # Anomalía térmica de verano — misma fuente que get_thermal_anomalias,
    # aquí como imagen espacial completa (no puntos ya muestreados).
    def preparar_lst(img):
        return (
            img.select("ST_B10").multiply(0.00341802).add(149.0).subtract(273.15)
            .rename("LST")
        )

    landsat = (
        ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
        .filterBounds(poligono)
        .filterDate("2022-06-01", "2024-08-31")
        .filter(ee.Filter.lt("CLOUD_COVER", 20))
        .filter(ee.Filter.calendarRange(6, 8, "month"))
        .map(preparar_lst)
    )
    if landsat.size().getInfo() < 3:
        landsat = (
            ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .filterBounds(poligono)
            .filterDate("2021-06-01", "2024-08-31")
            .filter(ee.Filter.lt("CLOUD_COVER", 20))
            .filter(ee.Filter.calendarRange(6, 8, "month"))
            .map(preparar_lst)
        )
    lst_mediana = landsat.median()
    lst_frio = lst_mediana.multiply(-1).rename("LST_FRIO")  # invertido: más frío = mejor

    def normalizar(imagen, nombre):
        stats = imagen.reduceRegion(
            reducer=ee.Reducer.minMax(), geometry=poligono, scale=30, maxPixels=1e8
        )
        min_val = ee.Number(stats.get(nombre + "_min"))
        max_val = ee.Number(stats.get(nombre + "_max"))
        rango = max_val.subtract(min_val).max(0.0001)  # evita división por 0
        return imagen.subtract(min_val).divide(rango).rename(nombre + "_NORM")

    ndvi_v_norm = normalizar(ndvi_verano, "NDVI_VERANO")
    lst_norm    = normalizar(lst_frio, "LST_FRIO")

    # 60% NDVI de verano persistente + 40% anomalía térmica fría.
    score_pozo = ndvi_v_norm.multiply(0.6).add(lst_norm.multiply(0.4)).rename("SCORE_POZO")
    score_limpio = score_pozo.focal_median(radius=2, kernelType="circle", units="pixels")

    max_score = score_limpio.reduceRegion(
        reducer=ee.Reducer.max(), geometry=poligono, scale=30, maxPixels=1e8
    ).get("SCORE_POZO").getInfo()
    if max_score is None:
        return []

    # Se devuelven varios candidatos (no solo el máximo) para poder
    # refinarlos después con pozos reales y geología.
    puntos = score_limpio.sample(
        region=poligono, scale=30, numPixels=200, geometries=True
    ).getInfo()

    candidatos = []
    for feat in puntos["features"]:
        sc = feat["properties"].get("SCORE_POZO")
        if sc is None:
            continue
        cp = feat["geometry"]["coordinates"]
        candidatos.append({"lon": cp[0], "lat": cp[1], "score_satelite": float(sc)})

    candidatos.sort(key=lambda c: c["score_satelite"], reverse=True)
    return candidatos[:15]
