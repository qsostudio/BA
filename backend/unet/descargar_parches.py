"""
descargar_parches.py — Descarga de Google Earth Engine los parches para entrenar
la U-Net de agua superficial (balsas, charcas, embalses).

Entrada de la red (6 bandas, Sentinel-2 L2A armonizado, mediana jun-sep 2021):
    B2 (azul), B3 (verde), B4 (rojo), B8 (NIR), B11 y B12 (SWIR), reflectancia 0-1
Etiqueta REAL (no sintetica): ESA WorldCover v200 (2021), clase 80 "aguas
    permanentes", 10 m. Mismo anno que las imagenes.

Parches de 64 x 64 pixeles a 10 m (640 m de lado), centrados en:
    "agua":   puntos con agua de WorldCover a menos de 3 km de pozos del IGME
              (zonas agricolas del interior; se excluye el mar con SRTM > 5 m)
    "tierra": pozos del IGME al azar (contexto agricola, casi siempre sin agua)

Ejecutar desde la carpeta backend/:
    python unet/descargar_parches.py --max 40     # prueba rapida (~1-2 min), en parches_agua_prueba/
    python unet/descargar_parches.py              # 400 + 400 parches (~10-20 min), en parches_agua/
Es reanudable: si se corta, vuelve a lanzarlo y solo descarga lo que falta.
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

CSV_POZOS = BACKEND / "data" / "pozos_andalucia_entrenamiento_filtrado.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "parches_agua"
BANDAS = ["B2", "B3", "B4", "B8", "B11", "B12"]
ANNO = 2021                 # ESA WorldCover v200 corresponde a 2021
LADO = 64
RADIO = LADO // 2           # el kernel de radio 32 da 65 x 65; se recorta a 64 x 64
ESCALA_M = 10
SEMILLA = 42


def imagen_s2(anno):
    import ee
    return (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(f"{anno}-06-01", f"{anno}-09-30")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .median()
        .select(BANDAS)
        .divide(10000)
    )


def mascara_agua():
    import ee
    return ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").eq(80).rename("AGUA")


def centros_agua(lats, lons, n_total, n_grupos=16, buffer_m=3000):
    """Puntos de agua (WorldCover) en el interior, cerca de los pozos."""
    import ee
    dem = ee.Image("USGS/SRTMGL1_003").select("elevation")
    agua_interior = mascara_agua().And(dem.gt(5)).rename("agua").toByte()
    grupos = np.array_split(np.arange(len(lats)), n_grupos)
    por_grupo = int(np.ceil(n_total / n_grupos))

    def un_grupo(g, indices):
        puntos = [ee.Geometry.Point([float(lons[i]), float(lats[i])]).buffer(buffer_m) for i in indices]
        region = ee.FeatureCollection([ee.Feature(p) for p in puntos]).union(maxError=100).geometry()
        muestra = agua_interior.stratifiedSample(
            numPoints=0, classBand="agua", region=region, scale=ESCALA_M, seed=SEMILLA + g,
            classValues=[1], classPoints=[por_grupo], geometries=True, tileScale=4)
        return [f["geometry"]["coordinates"] for f in muestra.getInfo()["features"]]

    coords = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for fut in as_completed([pool.submit(un_grupo, g, idx) for g, idx in enumerate(grupos) if len(idx)]):
            try:
                coords += fut.result()
            except Exception as e:
                print(f"  [AVISO] Un grupo de puntos de agua fallo: {e}")
    return [(c[1], c[0]) for c in coords][:n_total]


def parsear(info, ids_validos):
    resultado = {}
    for f in info.get("features", []):
        p = f["properties"]
        try:
            bandas = np.stack([np.asarray(p[b], dtype=np.float32) for b in BANDAS])
            agua = np.asarray(p["AGUA"], dtype=np.float32)
        except (KeyError, ValueError):
            continue
        if bandas.shape[1] < LADO or bandas.shape[2] < LADO:
            continue
        i = int(p["id"])
        if i in ids_validos:
            resultado[i] = (np.nan_to_num(bandas[:, :LADO, :LADO]), (agua[:LADO, :LADO] > 0.5).astype(np.uint8))
    return resultado


def descargar_lote(imagen_array, centros, ids, reintentos=3):
    import ee
    fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(lon), float(lat)]), {"id": int(i)})
                               for (lat, lon), i in zip(centros, ids)])
    for intento in range(1, reintentos + 1):
        try:
            info = imagen_array.sampleRegions(collection=fc, properties=["id"], scale=ESCALA_M,
                                              tileScale=4, geometries=False).getInfo()
            return parsear(info, set(int(i) for i in ids))
        except Exception as e:
            if intento == reintentos:
                print(f"  [AVISO] Lote descartado tras {reintentos} intentos: {e}")
                return {}
            time.sleep(5 * intento)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-agua", type=int, default=400)
    parser.add_argument("--n-tierra", type=int, default=400)
    parser.add_argument("--max", type=int, default=None, help="limitar el total (prueba)")
    parser.add_argument("--lote", type=int, default=10)
    parser.add_argument("--hilos", type=int, default=4)
    args = parser.parse_args()
    global OUTPUT_DIR
    if args.max:
        args.n_agua = args.n_tierra = max(1, args.max // 2)
        OUTPUT_DIR = OUTPUT_DIR.parent / "parches_agua_prueba"   # la prueba no se mezcla con la descarga real

    from dotenv import load_dotenv
    load_dotenv(BACKEND / ".env")
    import ee
    from services.gee_service import init_gee
    if not init_gee():
        sys.exit("No se pudo inicializar Google Earth Engine (revisa gee_key.json y el .env).")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    indice_csv = OUTPUT_DIR / "indice.csv"
    if indice_csv.exists():
        indice = pd.read_csv(indice_csv)
        print(f"Reanudando: {len(indice)} centros ya elegidos")
    else:
        pozos = pd.read_csv(CSV_POZOS).dropna(subset=["lat", "lon"])
        pozos = pozos.sample(n=min(len(pozos), 1600), random_state=SEMILLA)
        print("Buscando puntos de agua de ESA WorldCover cerca de los pozos...")
        agua = centros_agua(pozos["lat"].values, pozos["lon"].values, args.n_agua)
        tierra = pozos.sample(n=min(len(pozos), args.n_tierra), random_state=SEMILLA + 1)
        indice = pd.DataFrame(
            [{"lat": la, "lon": lo, "tipo": "agua"} for la, lo in agua]
            + [{"lat": la, "lon": lo, "tipo": "tierra"} for la, lo in zip(tierra["lat"], tierra["lon"])])
        indice.insert(0, "id", np.arange(len(indice)))
        indice.to_csv(indice_csv, index=False)
        print(f"Centros elegidos: {(indice.tipo == 'agua').sum()} de agua y {(indice.tipo == 'tierra').sum()} de tierra")

    pendientes = indice[[not (OUTPUT_DIR / f"parche_{i:04d}.npz").exists() for i in indice["id"]]]
    if pendientes.empty:
        print("No hay parches pendientes.")
        return
    imagen = (ee.Image.cat([imagen_s2(ANNO), mascara_agua()]).toFloat().unmask(0)
              .reproject(crs="EPSG:4326", scale=ESCALA_M))
    imagen_array = imagen.neighborhoodToArray(kernel=ee.Kernel.square(RADIO, "pixels"), defaultValue=0)

    filas = list(pendientes.itertuples(index=False))
    lotes = [filas[i:i + args.lote] for i in range(0, len(filas), args.lote)]
    print(f"Descargando {len(filas)} parches en {len(lotes)} lotes ({args.hilos} a la vez)...")
    inicio, n_ok = time.time(), 0
    with ThreadPoolExecutor(max_workers=args.hilos) as pool:
        futuros = {pool.submit(descargar_lote, imagen_array, [(r.lat, r.lon) for r in l], [r.id for r in l]): l
                   for l in lotes}
        for n, fut in enumerate(as_completed(futuros), start=1):
            por_id = {r.id: r for r in futuros[fut]}
            for i, (img, masc) in fut.result().items():
                r = por_id[i]
                np.savez_compressed(OUTPUT_DIR / f"parche_{i:04d}.npz", imagen=img, mascara=masc,
                                    lat=r.lat, lon=r.lon, tipo=r.tipo)
                n_ok += 1
            print(f"  Lote {n}/{len(lotes)} | parches guardados: {n_ok} | {time.time() - inicio:.0f} s")

    archivos = sorted(OUTPUT_DIR.glob("parche_*.npz"))
    pct = [np.load(a)["mascara"].mean() for a in archivos]
    print(f"\nTotal en disco: {len(archivos)} parches en {OUTPUT_DIR}")
    print(f"Pixeles de agua: {np.mean(pct) * 100:.1f} % del total | parches con algo de agua: "
          f"{np.mean(np.array(pct) > 0) * 100:.0f} %")
    planos = sum(np.load(a)["imagen"][3].std() < 1e-6 for a in archivos)
    if planos:
        print(f"[AVISO] {planos} parches con la banda B8 constante: revisa la proyeccion o las nubes.")


if __name__ == "__main__":
    main()
