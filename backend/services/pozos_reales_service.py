"""
pozos_reales_service.py — Pozos reales del IGME, cargados desde un
fichero LOCAL (procesado de los Excel oficiales "BD Puntos Agua v2.0"
que el usuario ya ha aportado), en vez de consultar un servidor externo
en cada petición. Ventaja: rapidez y fiabilidad garantizadas — no
depende de que el servicio en vivo del IGME esté disponible o responda
bien en cada momento.

Cobertura actual: 8 provincias de Andalucía, 4.498 pozos únicos
(deduplicados por Id real del IGME), de los cuales 3.306 tienen
profundidad real conocida.
"""

import json
from pathlib import Path
from typing import Optional

_RUTA_DATOS = Path(__file__).parent.parent / "data" / "pozos_reales_andalucia.json"
_pozos_cache: Optional[list] = None


def _cargar_pozos() -> list:
    global _pozos_cache
    if _pozos_cache is None:
        try:
            with open(_RUTA_DATOS, "r", encoding="utf-8") as f:
                _pozos_cache = json.load(f)
        except FileNotFoundError:
            _pozos_cache = []
    return _pozos_cache


def pozos_en_area(lat_min: float, lat_max: float, lon_min: float, lon_max: float, limite: int = 800) -> list:
    """Pozos reales dentro de una caja de coordenadas (la vista actual del
    mapa, por ejemplo). Limita el nº devuelto para no sobrecargar el mapa
    si la vista abarca una zona muy amplia."""
    pozos = _cargar_pozos()
    resultado = [
        p for p in pozos
        if lat_min <= p["lat"] <= lat_max and lon_min <= p["lon"] <= lon_max
    ]
    return resultado[:limite]


def total_pozos_cargados() -> int:
    return len(_cargar_pozos())
