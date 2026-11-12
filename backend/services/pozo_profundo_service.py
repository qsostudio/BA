"""
pozo_profundo_service.py — Punto óptimo para POZO PROFUNDO (agua
subterránea), distinto del punto óptimo de "balsa" que ya calcula
gee_service.analizar_parcela_gee.

Combina, en dos etapas:
  1) Señal satelital (GEE): NDVI de verano persistente + anomalía
     térmica de verano — candidatos_pozo_profundo_gee() en gee_service.py.
  2) Refinamiento con datos NO satelitales, reutilizando servicios ya
     existentes y probados en esta app:
       - Pozos reales cercanos del inventario del IGME (el mismo que usa
         profundidad_service.py) — si hay pozos reales conocidos cerca
         de un candidato, es la prueba más directa de que ahí hay
         acuífero explotable.
       - Geología del punto (IGME GEODE, el mismo que usa geo_service.py)
         — penaliza candidatos en roca desfavorable aunque el satélite
         los puntúe bien.

Ningún dato se inventa: todo reutiliza fuentes y funciones ya
verificadas en el resto de la aplicación.
"""

from services.gee_service import candidatos_pozo_profundo_gee
from services.profundidad_service import _buscar_pozos_cercanos, _distancia_m
from services.geo_service import obtener_geologia, clasificar_geologia

RADIO_BONUS_M = 5000  # a partir de 5 km de cualquier pozo real, el bonus es 0
BONUS_MAXIMO  = 0.20  # hasta un 20% extra de puntuación por tener un pozo real muy cerca
N_CANDIDATOS     = 5   # puntos idóneos para pozo que se muestran (el primero es la estrella dorada)
SEPARACION_MIN_M = 40  # dos candidatos a menos de 40 m son prácticamente el mismo sitio


def _candidatos_idoneos(candidatos, mejor):
    """El punto elegido (estrella dorada) y, como alternativas, los siguientes
    mejores candidatos separados al menos SEPARACION_MIN_M entre sí."""
    elegidos = [mejor]
    for c in candidatos:
        if len(elegidos) >= N_CANDIDATOS:
            break
        if all(_distancia_m(c["lat"], c["lon"], e["lat"], e["lon"]) >= SEPARACION_MIN_M for e in elegidos):
            elegidos.append(c)
    return [{
        "lat": e["lat"], "lon": e["lon"],
        "score": round(e["score_final"] * 100, 1),
        "score_satelite": round(e["score_satelite"] * 100, 1),
        "distancia_pozo_real_m": e.get("distancia_pozo_real_m"),
        "principal": e is mejor,
    } for e in elegidos]


def _pozos_cercanos_para_informe(pozos_reales, punto, n=15):
    salida = []
    for p in pozos_reales:
        try:
            prof = float(p["profundidad"]) if p["profundidad"] not in (None, "") else None
        except (TypeError, ValueError):
            prof = None
        salida.append({
            "lat": p["lat"], "lon": p["lon"], "profundidad": prof,
            "distancia_m": round(_distancia_m(punto["lat"], punto["lon"], p["lat"], p["lon"]), 1),
        })
    salida.sort(key=lambda q: q["distancia_m"])
    return salida[:n]


async def encontrar_punto_pozo_profundo(coords: list) -> dict:
    candidatos = candidatos_pozo_profundo_gee(coords)
    if not candidatos:
        return {"disponible": False, "mensaje": "No se ha podido calcular el punto óptimo de pozo (sin datos satelitales suficientes)."}

    lats = [c[1] for c in coords]
    lons = [c[0] for c in coords]
    lat_c, lon_c = sum(lats) / len(lats), sum(lons) / len(lons)

    # Pozos reales cercanos — UNA sola consulta para toda la parcela
    # (mismo servicio ya probado que usa "Ver humedad y elevación").
    pozos_raw, _debug = await _buscar_pozos_cercanos(lat_c, lon_c)
    pozos_reales = []
    for f in pozos_raw:
        attrs = f.get("attributes", {}) or {}
        geom  = f.get("geometry", {}) or {}
        prof  = attrs.get("PROF_m")
        px, py = geom.get("x"), geom.get("y")
        if px is None or py is None:
            continue
        pozos_reales.append({"lat": float(py), "lon": float(px), "profundidad": prof})

    # Refinar cada candidato con la cercanía a pozos reales.
    for c in candidatos:
        if pozos_reales:
            dist_min = min(_distancia_m(c["lat"], c["lon"], p["lat"], p["lon"]) for p in pozos_reales)
            c["distancia_pozo_real_m"] = round(dist_min, 1)
            c["bonus_pozo_real"] = round(BONUS_MAXIMO * max(0.0, 1 - min(dist_min, RADIO_BONUS_M) / RADIO_BONUS_M), 4)
        else:
            c["distancia_pozo_real_m"] = None
            c["bonus_pozo_real"] = 0.0
        c["score_final"] = c["score_satelite"] * (1 - BONUS_MAXIMO) + c["bonus_pozo_real"]

    candidatos.sort(key=lambda c: c["score_final"], reverse=True)

    # Geología solo del candidato ganador (una consulta, no 15).
    mejor = candidatos[0]
    geo = await obtener_geologia(mejor["lat"], mejor["lon"])
    roca_es, _, estado_geo = clasificar_geologia(geo.get("lith", ""))

    # Si la geología del primer candidato es claramente mala y hay un
    # segundo candidato con geología mejor y una puntuación satelital
    # razonablemente cercana, se prefiere el segundo — sentido común
    # geológico por encima del número frío.
    if estado_geo == "mala" and len(candidatos) > 1:
        segundo = candidatos[1]
        if segundo["score_final"] >= mejor["score_final"] * 0.85:
            geo2 = await obtener_geologia(segundo["lat"], segundo["lon"])
            roca_es2, _, estado_geo2 = clasificar_geologia(geo2.get("lith", ""))
            if estado_geo2 != "mala":
                mejor, geo, roca_es, estado_geo = segundo, geo2, roca_es2, estado_geo2

    return {
        "disponible": True,
        "pozo_profundo_lat": mejor["lat"],
        "pozo_profundo_lon": mejor["lon"],
        "pozo_profundo_score": round(mejor["score_final"] * 100, 1),
        "score_satelite": round(mejor["score_satelite"] * 100, 1),
        "distancia_pozo_real_m": mejor["distancia_pozo_real_m"],
        "n_pozos_reales_considerados": len(pozos_reales),
        "candidatos_pozo": _candidatos_idoneos(candidatos, mejor),
        "pozos_cercanos": _pozos_cercanos_para_informe(pozos_reales, mejor),
        "geologia": {
            "roca": roca_es,
            "nombre_unidad": geo.get("name", "Desconocido"),
            "estado": estado_geo,
        },
        "metodo": (
            "NDVI de verano persistente (60%) + anomalía térmica de verano (40%), "
            "refinado con la cercanía a pozos reales del inventario del IGME "
            "(hasta +20%) y verificado contra la geología real del punto."
        ),
    }
