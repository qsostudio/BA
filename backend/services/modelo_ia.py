"""
modelo_ia.py (v3) — Profundidad del agua subterranea con intervalo de error.

Modelo: Gradient Boosting (regresion) sobre log(1 + profundidad_m).
Objetivo: profundidad de los pozos del inventario del IGME (0-200 m).

Variables. Todas se calculan EXACTAMENTE igual al entrenar y en la app:
    profundidad_idw_vecinos  media ponderada por distancia (IDW, k=5, potencia 2)
                             de la profundidad de los pozos vecinos
    dist_pozo_km             distancia al pozo con profundidad conocida mas cercano
    elevacion                elevacion SRTM 30 m (Google Earth Engine)
    geologia_score           litologia favorable (1), moderada (0,5) o desfavorable (0)

Validacion: validacion cruzada espacial por bloques de 10 km (GroupKFold),
    anidada para elegir los hiperparametros (GridSearchCV). El IDW y la
    distancia se recalculan en cada pliegue SOLO con pozos de entrenamiento,
    para que ningun pozo de test se filtre como variable (sin fuga de datos).

Intervalo: prediccion conformal de Mondrian por tramos de distancia al pozo
    mas cercano: el intervalo se ensancha cuando no hay pozos cerca. Su
    cobertura se valida pozo a pozo con los pliegues espaciales.

Referencias sin entrenamiento: mediana de los pozos e IDW de los pozos vecinos.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut

MODEL_DIR = Path(__file__).parent.parent / "modelo"
MODEL_PATH = MODEL_DIR / "modelo_agua.pkl"
METRICS_PATH = MODEL_DIR / "metricas.json"

FEATURES = ["profundidad_idw_vecinos", "dist_pozo_km", "elevacion", "geologia_score"]
FEATURE_LABELS = {
    "profundidad_idw_vecinos": "Profundidad IDW de los pozos vecinos (m)",
    "dist_pozo_km": "Distancia al pozo conocido mas cercano (km)",
    "elevacion": "Elevacion del terreno, SRTM (m)",
    "geologia_score": "Geologia favorable",
}
TARGET_COLUMN = "profundidad_m"
PROFUNDIDAD_MAX_M = 200.0
GEOLOGIA_A_SCORE = {"mala": 0.0, "media": 0.5, "buena": 1.0}

CATEGORIAS = [
    (0.0, 3.0, "Muy somero", "verde", "Acuifero aluvial superficial: perforacion rapida y barata"),
    (3.0, 8.0, "Somero", "verde", "Aluvial tipico: coste de perforacion bajo"),
    (8.0, 20.0, "Medio", "amarillo", "Terciario detritico: coste de perforacion moderado"),
    (20.0, 50.0, "Profundo", "amarillo", "Roca consolidada: perforacion cara"),
    (50.0, 99999.0, "Muy profundo", "rojo", "Mas de 50 m: perforacion de alto coste y riesgo"),
]

PARAM_GRID = {
    "n_estimators": [100, 300],
    "max_depth": [2, 3, 4],
    "learning_rate": [0.03, 0.1],
}
IDW_K = 5
IDW_POTENCIA = 2
BLOQUE_KM = 10.0
N_PLIEGUES = 5
CONFIANZA = 0.80
TRAMOS_DISTANCIA_KM = [2.0, 5.0]          # tramos: < 2 km, 2-5 km, >= 5 km
NOMBRES_TRAMOS = ["< 2 km", "2-5 km", ">= 5 km"]
MIN_POR_TRAMO = 40
SEMILLA = 42

REF_MEDIANA = "Mediana de los pozos (referencia)"
REF_IDW = "IDW 5 pozos vecinos (referencia)"
MODELO_GB = "Gradient Boosting"


# ── Utilidades espaciales ────────────────────────────────────────────────────
def _distancia_m_vec(lat0, lon0, lat_arr, lon_arr):
    m_lat = 111320.0
    m_lon = 111320.0 * np.cos(np.radians(lat0))
    return np.sqrt(((lon_arr - lon0) * m_lon) ** 2 + ((lat_arr - lat0) * m_lat) ** 2)


def _vecinos(lat_obj, lon_obj, lat_pool, lon_pool, prof_pool, excluir_propio,
             k=IDW_K, potencia=IDW_POTENCIA):
    """
    Para cada punto objetivo devuelve (idw, distancia al vecino mas cercano en km,
    indices de los k vecinos), usando SOLO el pool de pozos que se le pasa.
    excluir_propio=True: objetivo y pool son el mismo conjunto en el mismo orden
    (cada pozo se excluye a si mismo).
    """
    lat_obj, lon_obj = np.asarray(lat_obj, float), np.asarray(lon_obj, float)
    lat_pool, lon_pool = np.asarray(lat_pool, float), np.asarray(lon_pool, float)
    prof_pool = np.asarray(prof_pool, float)
    n = len(lat_obj)
    k_ef = min(k, len(lat_pool) - (1 if excluir_propio else 0))
    if k_ef < 1:
        raise ValueError("No hay pozos suficientes para calcular el IDW.")
    idw, dmin = np.zeros(n), np.zeros(n)
    indices = np.zeros((n, k_ef), dtype=int)
    for i in range(n):
        d = _distancia_m_vec(lat_obj[i], lon_obj[i], lat_pool, lon_pool)
        if excluir_propio:
            d[i] = np.inf
        v = np.argpartition(d, k_ef - 1)[:k_ef]
        v = v[np.argsort(d[v])]
        w = 1.0 / np.maximum(d[v], 1.0) ** potencia
        idw[i] = np.sum(prof_pool[v] * w) / np.sum(w)
        dmin[i] = d[v[0]] / 1000.0
        indices[i] = v
    return idw, dmin, indices


def _idw_vecinos(lat_obj, lon_obj, lat_pool, lon_pool, prof_pool, excluir_propio,
                 k=IDW_K, potencia=IDW_POTENCIA):
    """Compatibilidad con versiones anteriores: solo el IDW."""
    return _vecinos(lat_obj, lon_obj, lat_pool, lon_pool, prof_pool, excluir_propio, k, potencia)[0]


def bloques_espaciales(lat, lon, km=BLOQUE_KM):
    """Identificador de bloque cuadrado de `km` km de lado (aproximacion local)."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    y = lat * 111.32
    x = lon * 111.32 * np.cos(np.radians(np.mean(lat)))
    return np.array([f"{int(a)}_{int(b)}" for a, b in zip(np.floor(x / km), np.floor(y / km))])


def elevacion_srtm_gee(lats, lons, lote=1000):
    """
    Elevacion SRTM 30 m (USGS/SRTMGL1_003) en cada punto, con Google Earth Engine.
    Es la MISMA funcion al entrenar (entrenar_profundidad.py) y al predecir (main.py).
    Requiere GEE inicializado.
    """
    import ee

    dem = ee.Image("USGS/SRTMGL1_003").select("elevation")
    lats, lons = np.asarray(lats, float), np.asarray(lons, float)
    resultado = np.full(len(lats), np.nan)
    for ini in range(0, len(lats), lote):
        puntos = [ee.Feature(ee.Geometry.Point([float(lo), float(la)]), {"i": int(ini + j)})
                  for j, (la, lo) in enumerate(zip(lats[ini:ini + lote], lons[ini:ini + lote]))]
        info = dem.sampleRegions(collection=ee.FeatureCollection(puntos), properties=["i"],
                                 scale=30, geometries=False).getInfo()
        for f in info.get("features", []):
            p = f["properties"]
            if p.get("elevation") is not None:
                resultado[int(p["i"])] = float(p["elevation"])
    return resultado


# ── Datos ────────────────────────────────────────────────────────────────────
def clasificar_profundidad(profundidad_m: float) -> dict:
    for p_min, p_max, etiqueta, color, descripcion in CATEGORIAS:
        if p_min <= profundidad_m < p_max:
            return {"etiqueta": etiqueta, "color": color, "descripcion": descripcion}
    return {"etiqueta": "Muy profundo", "color": "rojo", "descripcion": "Mas de 50 m"}


def cargar_datos(csv_path: str) -> Tuple[pd.DataFrame, pd.Series, np.ndarray, np.ndarray, np.ndarray]:
    """
    Devuelve (base, y, provincias, lat, lon). `base` tiene las variables que no
    dependen de los vecinos (elevacion, geologia_score); el IDW y la distancia
    se calculan dentro de cada pliegue.
    """
    df = pd.read_csv(csv_path)
    necesarias = ["lat", "lon", TARGET_COLUMN, "elevacion", "geologia_score"]
    faltan = [c for c in necesarias if c not in df.columns]
    if faltan:
        raise ValueError(f"[MODELO] Faltan columnas en {csv_path}: {faltan}. Columnas: {list(df.columns)}")
    n_inicial = len(df)
    df = df.dropna(subset=necesarias)
    df = df[(df[TARGET_COLUMN] > 0) & (df[TARGET_COLUMN] <= PROFUNDIDAD_MAX_M)]
    df = df.drop_duplicates(subset=["lat", "lon", TARGET_COLUMN]).reset_index(drop=True)
    print(f"[MODELO] {len(df)} pozos con profundidad (0-{PROFUNDIDAD_MAX_M:.0f} m) y variables completas, "
          f"de {n_inicial} filas")
    if len(df) < 30:
        raise ValueError("[MODELO] Hacen falta al menos 30 pozos con profundidad para entrenar.")
    prof = df[TARGET_COLUMN]
    print(f"[MODELO] Profundidad: mediana {prof.median():.1f} m | media {prof.mean():.1f} m | "
          f"min {prof.min():.1f} | max {prof.max():.1f}")
    provincias = df["provincia"].astype(str).values if "provincia" in df.columns else None
    base = df[["elevacion", "geologia_score"]].astype(float)
    return base, prof.astype(float), provincias, df["lat"].values.astype(float), df["lon"].values.astype(float)


def _matriz(base: pd.DataFrame, idw, dist_km) -> pd.DataFrame:
    X = pd.DataFrame({
        "profundidad_idw_vecinos": np.asarray(idw, float),
        "dist_pozo_km": np.asarray(dist_km, float),
        "elevacion": base["elevacion"].values.astype(float),
        "geologia_score": base["geologia_score"].values.astype(float),
    })
    return X[FEATURES]


def _gb_base():
    return GradientBoostingRegressor(subsample=0.8, min_samples_leaf=5, random_state=SEMILLA)


def _busqueda(X, y_log, bloques):
    n_grupos = len(set(bloques))
    busqueda = GridSearchCV(_gb_base(), PARAM_GRID, cv=GroupKFold(n_splits=max(2, min(4, n_grupos))),
                            scoring="neg_mean_absolute_error", n_jobs=-1)
    busqueda.fit(X, y_log, groups=bloques)
    return busqueda


# ── Metricas ─────────────────────────────────────────────────────────────────
def _metricas(y, p) -> Dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    err = np.abs(y - p)
    cats = [clasificar_profundidad(v)["etiqueta"] for v in p]
    cats_real = [clasificar_profundidad(v)["etiqueta"] for v in y]
    return {
        "mae_m": round(float(mean_absolute_error(y, p)), 2),
        "mediana_error_m": round(float(np.median(err)), 2),
        "rmse_m": round(float(np.sqrt(mean_squared_error(y, p))), 2),
        "r2": round(float(r2_score(y, p)), 3),
        "r2_log": round(float(r2_score(np.log1p(y), np.log1p(p))), 3),
        "pct_error_menor_5m": round(float((err < 5).mean() * 100), 1),
        "pct_error_menor_10m": round(float((err < 10).mean() * 100), 1),
        "pct_error_rel_menor_25pct": round(float((err / np.maximum(y, 1e-6) < 0.25).mean() * 100), 1),
        "acuerdo_categorias": round(float(np.mean([a == b for a, b in zip(cats, cats_real)])), 3),
    }


def _tramo(dist_km, cortes=TRAMOS_DISTANCIA_KM):
    return np.digitize(np.asarray(dist_km, float), cortes)


def _cuantil_conformal(residuos, confianza):
    residuos = np.asarray(residuos, float)
    n = len(residuos)
    if n == 0:
        return float("nan")
    nivel = min(1.0, np.ceil((n + 1) * confianza) / n)
    return float(np.quantile(residuos, nivel))


def _cuantiles_por_tramo(residuos, tramos, confianza, n_tramos=len(TRAMOS_DISTANCIA_KM) + 1):
    """Cuantil conformal de cada tramo de distancia (Mondrian). Tramos con pocos
    datos usan el cuantil global, y se fuerza que el intervalo no se estreche al
    alejarse de los pozos."""
    q_global = _cuantil_conformal(residuos, confianza)
    q = []
    for t in range(n_tramos):
        r = residuos[tramos == t]
        q.append(_cuantil_conformal(r, confianza) if len(r) >= MIN_POR_TRAMO else q_global)
    for t in range(1, n_tramos):
        q[t] = max(q[t], q[t - 1])
    return np.array(q)


def _intervalo(pred_m, q_log):
    centro = np.log1p(np.asarray(pred_m, float))
    return np.maximum(0.0, np.expm1(centro - q_log)), np.expm1(centro + q_log)


# ── Validacion cruzada espacial ──────────────────────────────────────────────
def _validacion_cruzada(base, y, lat, lon, grupos, cv, buscar_hiperparametros=True, params_fijos=None):
    y_log = np.log1p(y)
    n = len(y)
    pred = {REF_MEDIANA: np.full(n, np.nan), REF_IDW: np.full(n, np.nan), MODELO_GB: np.full(n, np.nan)}
    dist_oof, pliegue = np.full(n, np.nan), np.full(n, -1)
    mejores = []
    for k, (tr, te) in enumerate(cv.split(base, y, groups=grupos)):
        idw_tr, d_tr, _ = _vecinos(lat[tr], lon[tr], lat[tr], lon[tr], y[tr], excluir_propio=True)
        idw_te, d_te, _ = _vecinos(lat[te], lon[te], lat[tr], lon[tr], y[tr], excluir_propio=False)
        X_tr, X_te = _matriz(base.iloc[tr], idw_tr, d_tr), _matriz(base.iloc[te], idw_te, d_te)

        pred[REF_MEDIANA][te] = np.median(y[tr])
        pred[REF_IDW][te] = idw_te
        if buscar_hiperparametros:
            busqueda = _busqueda(X_tr, y_log[tr], bloques_espaciales(lat[tr], lon[tr]))
            modelo = busqueda.best_estimator_
            mejores.append(busqueda.best_params_)
        else:
            modelo = _gb_base().set_params(**(params_fijos or {})).fit(X_tr, y_log[tr])
        pred[MODELO_GB][te] = np.maximum(0.0, np.expm1(modelo.predict(X_te)))
        dist_oof[te], pliegue[te] = d_te, k
        print(f"  [Pliegue {k + 1}] test={len(te)} pozos | MAE GB={mean_absolute_error(y[te], pred[MODELO_GB][te]):.1f} m "
              f"| MAE IDW={mean_absolute_error(y[te], idw_te):.1f} m")
    return pred, dist_oof, pliegue, mejores


def _validar_cobertura(y, pred_gb, dist_oof, pliegue, confianza):
    """Cobertura real del intervalo: para cada pliegue, el intervalo se calibra con
    los residuos de los DEMAS pliegues y se comprueba en este."""
    res = np.abs(np.log1p(y) - np.log1p(pred_gb))
    tramos = _tramo(dist_oof)
    lo, hi = np.zeros(len(y)), np.zeros(len(y))
    for k in np.unique(pliegue):
        cal, te = pliegue != k, pliegue == k
        q = _cuantiles_por_tramo(res[cal], tramos[cal], confianza)
        lo[te], hi[te] = _intervalo(pred_gb[te], q[tramos[te]])
    dentro = (y >= lo) & (y <= hi)
    por_tramo = []
    for t, nombre in enumerate(NOMBRES_TRAMOS):
        m = tramos == t
        por_tramo.append({
            "tramo": nombre, "n_pozos": int(m.sum()),
            "cobertura": round(float(dentro[m].mean()), 3) if m.any() else None,
            "anchura_mediana_m": round(float(np.median(hi[m] - lo[m])), 1) if m.any() else None,
        })
    return {"cobertura": round(float(dentro.mean()), 3),
            "anchura_mediana_m": round(float(np.median(hi - lo)), 1),
            "por_distancia": por_tramo}, lo, hi, dentro


def _interpretar(met_gb, met_idw, met_med, cobertura, confianza) -> Dict:
    mejora_idw = (met_idw["mae_m"] - met_gb["mae_m"]) / max(met_idw["mae_m"], 1e-6) * 100
    mejora_med = (met_med["mae_m"] - met_gb["mae_m"]) / max(met_med["mae_m"], 1e-6) * 100
    if mejora_idw > 5 and met_gb["r2"] > 0.3:
        calidad = "Util"
    elif mejora_med > 5:
        calidad = "Orientativo"
    else:
        calidad = "Limitado"
    if mejora_idw > 0:
        frase_idw = f"reduce el error medio un {mejora_idw:.0f} % respecto a usar solo los pozos vecinos (IDW)"
    else:
        frase_idw = f"no mejora a usar solo los pozos vecinos (IDW): su error medio es un {-mejora_idw:.0f} % mayor"
    return {
        "calidad_modelo": calidad,
        "descripcion": (f"En validacion espacial, el modelo {frase_idw} y un {mejora_med:.0f} % respecto "
                        f"a dar siempre la profundidad mediana."),
        "r2_texto": (f"R2 = {met_gb['r2']:.2f} (en log: {met_gb['r2_log']:.2f}). Error absoluto mediano: "
                     f"{met_gb['mediana_error_m']:.1f} m; medio: {met_gb['mae_m']:.1f} m."),
        "rmse_texto": f"RMSE = {met_gb['rmse_m']:.1f} m.",
        "intervalo_texto": (f"El intervalo del {confianza:.0%} contuvo la profundidad real en el "
                            f"{cobertura['cobertura']:.0%} de los pozos de validacion "
                            f"(anchura mediana {cobertura['anchura_mediana_m']:.0f} m)."),
        "acuerdo_texto": f"Acierta la categoria de profundidad en el {met_gb['acuerdo_categorias']:.0%} de los pozos.",
        "mejora_frente_idw_pct": round(float(mejora_idw), 1),
        "mejora_frente_mediana_pct": round(float(mejora_med), 1),
    }


# ── Entrenamiento ────────────────────────────────────────────────────────────
def entrenar(csv_path: str, confianza: float = CONFIANZA, devolver_detalle: bool = False):
    base, y_serie, provincias, lat, lon = cargar_datos(csv_path)
    y = y_serie.values
    bloques = bloques_espaciales(lat, lon)
    n_pliegues = max(2, min(N_PLIEGUES, len(set(bloques))))

    print(f"\n[MODELO] Validacion cruzada espacial anidada: {n_pliegues} pliegues por bloques de {BLOQUE_KM:.0f} km")
    pred, dist_oof, pliegue, mejores = _validacion_cruzada(
        base, y, lat, lon, bloques, GroupKFold(n_splits=n_pliegues), buscar_hiperparametros=True)
    metricas_modelos = {nombre: _metricas(y, p) for nombre, p in pred.items()}
    met_gb = metricas_modelos[MODELO_GB]

    r2_pl, rmse_pl = [], []
    for k in np.unique(pliegue):
        m = pliegue == k
        if m.sum() > 1:
            r2_pl.append(r2_score(y[m], pred[MODELO_GB][m]))
            rmse_pl.append(np.sqrt(mean_squared_error(y[m], pred[MODELO_GB][m])))

    cobertura, lo_oof, hi_oof, dentro_oof = _validar_cobertura(y, pred[MODELO_GB], dist_oof, pliegue, confianza)

    # Modelo final con TODOS los pozos (IDW sin el propio pozo) y la misma busqueda
    print("\n[MODELO] Ajustando el modelo final con todos los pozos...")
    idw_all, d_all, _ = _vecinos(lat, lon, lat, lon, y, excluir_propio=True)
    X_all = _matriz(base, idw_all, d_all)
    busqueda_final = _busqueda(X_all, np.log1p(y), bloques)
    modelo_final = busqueda_final.best_estimator_

    # Escenario exigente: provincia completa sin pozos (con los hiperparametros finales)
    escenario_provincia = None
    if provincias is not None and len(set(provincias)) >= 3:
        print("\n[MODELO] Escenario exigente: dejando fuera una provincia entera cada vez")
        pred_p, _, _, _ = _validacion_cruzada(base, y, lat, lon, provincias, LeaveOneGroupOut(),
                                              buscar_hiperparametros=False, params_fijos=busqueda_final.best_params_)
        escenario_provincia = {nombre: _metricas(y, p) for nombre, p in pred_p.items()}
        escenario_provincia["n_provincias"] = int(len(set(provincias)))

    residuos = np.abs(np.log1p(y) - np.log1p(pred[MODELO_GB]))
    q_final = _cuantiles_por_tramo(residuos, _tramo(dist_oof), confianza)

    importancias = dict(sorted(
        {FEATURE_LABELS[f]: round(float(v), 4) for f, v in zip(FEATURES, modelo_final.feature_importances_)}.items(),
        key=lambda kv: kv[1], reverse=True))

    paquete = {
        "version": 3,
        "modelo": modelo_final,
        "features": FEATURES,
        "objetivo": "log1p(profundidad_m)",
        "confianza": confianza,
        "tramos_km": TRAMOS_DISTANCIA_KM,
        "q_log": [float(v) for v in q_final],
        "pool": {"lat": lat, "lon": lon, "prof": y},
        "idw_k": IDW_K, "idw_potencia": IDW_POTENCIA,
    }
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(paquete, MODEL_PATH)

    metricas = {
        "version_modelo": 3,
        "n_pozos_entrenamiento": int(len(y)),
        "tipo_modelo": "Gradient Boosting: regresion de la profundidad (m) con intervalo conformal",
        "variables": [FEATURE_LABELS[f] for f in FEATURES],
        "mejores_hiperparametros": busqueda_final.best_params_,
        "hiperparametros_por_pliegue": mejores,
        "metricas_cv": {
            "tipo_validacion": (f"Validacion cruzada espacial anidada: {n_pliegues} pliegues por bloques de "
                                f"{BLOQUE_KM:.0f} km; IDW recalculado en cada pliegue (sin fuga de datos)"),
            "n_bloques_externos": int(n_pliegues),
            "r2_medio": round(float(np.mean(r2_pl)), 3),
            "r2_std": round(float(np.std(r2_pl)), 3),
            "r2_global": met_gb["r2"],
            "rmse_medio": round(float(np.mean(rmse_pl)), 2),
            "rmse_std": round(float(np.std(rmse_pl)), 2),
            "mae_m": met_gb["mae_m"],
            "mediana_error_m": met_gb["mediana_error_m"],
            "pct_error_menor_5m": met_gb["pct_error_menor_5m"],
            "pct_error_menor_10m": met_gb["pct_error_menor_10m"],
            "acuerdo_categorias": met_gb["acuerdo_categorias"],
        },
        "comparacion": metricas_modelos,
        "escenario_provincia_no_vista": escenario_provincia,
        "intervalo": {"confianza": confianza, **cobertura,
                      "q_log_por_tramo": dict(zip(NOMBRES_TRAMOS, [round(float(v), 4) for v in q_final]))},
        "importancia_variables": importancias,
        "interpretacion": _interpretar(met_gb, metricas_modelos[REF_IDW], metricas_modelos[REF_MEDIANA],
                                       cobertura, confianza),
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metricas, f, ensure_ascii=False, indent=2)
    print(f"\n[MODELO] Guardado: {MODEL_PATH} y {METRICS_PATH}")

    if devolver_detalle:
        detalle = {"y": y, "pred": pred, "dist_oof": dist_oof, "pliegue": pliegue,
                   "lo": lo_oof, "hi": hi_oof, "dentro": dentro_oof, "lat": lat, "lon": lon}
        return metricas, detalle
    return metricas


# ── Prediccion ───────────────────────────────────────────────────────────────
_cache = {"mtime": None, "paquete": None}


def _cargar_paquete():
    if not MODEL_PATH.exists():
        return None
    mtime = MODEL_PATH.stat().st_mtime
    if _cache["mtime"] != mtime:
        _cache["paquete"], _cache["mtime"] = joblib.load(MODEL_PATH), mtime
    return _cache["paquete"]


def predecir(features: Dict) -> Dict:
    paquete = _cargar_paquete()
    if paquete is None:
        return {"disponible": False, "mensaje": "Modelo no entrenado."}
    if not isinstance(paquete, dict) or paquete.get("version") != 3:
        return {"disponible": False, "mensaje": "Modelo antiguo: vuelve a entrenar con entrenar_profundidad.py."}

    lat = features.get("lat", features.get("latitud"))
    lon = features.get("lon", features.get("longitud"))
    if lat is None or lon is None:
        return {"disponible": False, "mensaje": "Faltan las coordenadas del punto."}
    elevacion = features.get("elevacion")
    if elevacion is None or (isinstance(elevacion, float) and np.isnan(elevacion)):
        return {"disponible": False, "mensaje": "No se pudo obtener la elevacion del punto."}
    avisos = []
    geologia = features.get("geologia_score")
    if geologia is None:
        geologia = 0.5
        avisos.append("Geologia no disponible: se ha supuesto moderada.")

    pool = paquete["pool"]
    idw, dmin, idx = _vecinos([float(lat)], [float(lon)], pool["lat"], pool["lon"], pool["prof"],
                              excluir_propio=False, k=paquete["idw_k"], potencia=paquete["idw_potencia"])
    X = _matriz(pd.DataFrame({"elevacion": [float(elevacion)], "geologia_score": [float(geologia)]}), idw, dmin)
    centro_log = float(paquete["modelo"].predict(X)[0])
    tramo = int(_tramo([dmin[0]], paquete["tramos_km"])[0])
    q = paquete["q_log"][tramo]
    estimada = max(0.0, float(np.expm1(centro_log)))
    minimo = max(0.0, float(np.expm1(centro_log - q)))
    maximo = float(np.expm1(centro_log + q))

    if dmin[0] >= 25:
        avisos.append("El pozo conocido mas cercano esta a mas de 25 km: zona fuera del area de entrenamiento, "
                      "estimacion poco fiable.")
    elif dmin[0] >= TRAMOS_DISTANCIA_KM[-1]:
        avisos.append(f"No hay pozos del IGME a menos de {TRAMOS_DISTANCIA_KM[-1]:.0f} km: el intervalo es mas ancho.")

    cat = clasificar_profundidad(estimada)
    vecinos = [{
        "lat": float(pool["lat"][j]), "lon": float(pool["lon"][j]),
        "profundidad_m": round(float(pool["prof"][j]), 1),
        "distancia_km": round(float(_distancia_m_vec(float(lat), float(lon), np.array([pool["lat"][j]]),
                                                      np.array([pool["lon"][j]]))[0] / 1000), 2),
    } for j in idx[0]]
    confianza_pct = int(round(paquete["confianza"] * 100))
    return {
        "disponible": True,
        "profundidad_estimada_m": round(estimada, 1),
        "intervalo_min_m": round(minimo, 1),
        "intervalo_max_m": round(maximo, 1),
        "confianza_pct": confianza_pct,
        "distancia_pozo_cercano_km": round(float(dmin[0]), 2),
        "tramo_distancia": NOMBRES_TRAMOS[tramo],
        "profundidad_idw_vecinos_m": round(float(idw[0]), 1),
        "pozos_vecinos": vecinos,
        "variables": {"elevacion_m": round(float(elevacion), 1), "geologia_score": float(geologia)},
        "veredicto": f"{cat['etiqueta']}: ~{estimada:.0f} m (entre {minimo:.0f} y {maximo:.0f} m, {confianza_pct} %)",
        "color": cat["color"],
        "descripcion": cat["descripcion"],
        "avisos": avisos,
    }


def cargar_metricas() -> Optional[Dict]:
    if not METRICS_PATH.exists():
        return None
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def modelo_disponible() -> bool:
    paquete = _cargar_paquete() if MODEL_PATH.exists() else None
    return isinstance(paquete, dict) and paquete.get("version") == 3
