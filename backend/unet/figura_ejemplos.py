"""
figura_ejemplos.py — Nueva figura de ejemplos de la U-Net en el conjunto de prueba (sin reentrenar).

La figura que genera entrenar.py (fig_unet_ejemplos.png) muestra los tres recortes de prueba con
más agua y el que menos tiene. Los tres primeros son agua al 100 % (el interior de un embalse) y
ahí aciertan todos los métodos, así que la figura no enseña nada.

Este script:
  1. Carga el modelo ya entrenado (modelo/modelo_unet.pth). No reentrena nada.
  2. Lee los recortes de unet/parches_agua y repite la misma separación espacial que entrenar.py
     (misma semilla), así que el conjunto de prueba es exactamente el mismo.
  3. Comprueba que la matriz de confusión en prueba coincide con la de modelo/metricas_unet.json.
     Si no coincide, se para sin dibujar: querría decir que el modelo o los recortes no son los
     de la memoria.
  4. Elige cuatro recortes de prueba con un criterio fijo (no a ojo):
       - el que mezcla agua y tierra (10-90 % de agua) en el que la U-Net más mejora al mejor
         de los dos índices;
       - un caso típico: el recorte con agua y tierra cuyo IoU de la U-Net es la mediana;
       - un recorte con poca agua (1-10 %), también el de IoU mediano;
       - el peor caso de la U-Net entre los recortes con al menos un 1 % de agua.
  5. Dibuja cada método coloreando los aciertos y los errores, con el IoU de cada recorte.

No cambia el modelo ni ninguna métrica de la memoria.

Uso (desde la carpeta backend):
    python unet/figura_ejemplos.py
Genera modelo/fig_unet_ejemplos_v2.png
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from dataset import cargar_parches  # noqa: E402
from entrenar import (BANDAS, BATCH_SIZE, METRICAS_PATH, MODEL_DIR, MODEL_PATH,  # noqa: E402
                      PARCHES_DIR, indice_normalizado, metricas_pixel, separar_espacialmente)
from model import UNetLigero  # noqa: E402

COLORES = {"tierra": "#f1f1ef", "agua_ok": "#1d4ed8", "agua_perdida": "#f59e0b", "falsa_agua": "#dc2626"}
TOLERANCIA = 1e-4  # diferencia máxima admitida en la matriz de confusión (fracción de píxeles)


def coma(x, dec=2):
    return f"{x:.{dec}f}".replace(".", ",")


def cargar_modelo():
    if not MODEL_PATH.exists():
        sys.exit(f"No encuentro {MODEL_PATH}. Este script usa el modelo ya entrenado.")
    try:
        ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    except TypeError:  # versiones antiguas de PyTorch
        ckpt = torch.load(MODEL_PATH, map_location="cpu")
    if list(ckpt["bandas"]) != BANDAS:
        sys.exit(f"Las bandas del modelo {ckpt['bandas']} no coinciden con {BANDAS}.")
    modelo = UNetLigero(in_channels=ckpt["in_channels"], out_channels=1, base_ch=ckpt["base_ch"])
    modelo.load_state_dict(ckpt["model_state_dict"])
    modelo.eval()
    return modelo, ckpt


def predecir(modelo, X, media, desv):
    """Misma normalización que en el entrenamiento y mismo tamaño de lote."""
    Xn = ((X - media[None, :, None, None]) / desv[None, :, None, None]).astype(np.float32)
    probs = []
    with torch.no_grad():
        for i in range(0, len(Xn), BATCH_SIZE):
            probs.append(torch.sigmoid(modelo(torch.from_numpy(Xn[i:i + BATCH_SIZE]))).numpy()[:, 0])
    return np.concatenate(probs)


def comprobar(pred, real):
    if not METRICAS_PATH.exists():
        sys.exit(f"No encuentro {METRICAS_PATH} para comprobar el resultado.")
    ref = json.loads(METRICAS_PATH.read_text(encoding="utf-8"))["test"]["U-Net (CNN)"]["matriz_confusion"]
    actual = metricas_pixel(pred, real)["matriz_confusion"]
    diferencia = sum(abs(actual[k] - ref[k]) for k in ref)
    print(f"Matriz de confusión en prueba: {actual}")
    print(f"En metricas_unet.json:         {ref}")
    if diferencia == 0:
        print("Comprobación: idéntica. Es el mismo modelo y el mismo conjunto de prueba de la memoria.")
    elif diferencia <= TOLERANCIA * real.size:
        print(f"Comprobación: diferencia de {diferencia} píxeles de {real.size} (redondeo numérico). Se continúa.")
    else:
        sys.exit("[OJO] La matriz de confusión no coincide con metricas_unet.json: el modelo o los "
                 "recortes no son los de la memoria. No se genera la figura.")


def iou_por_recorte(pred, real):
    inter = np.logical_and(pred, real).sum(axis=(1, 2))
    union = np.logical_or(pred, real).sum(axis=(1, 2))
    return np.where(union > 0, inter / np.maximum(union, 1), 1.0)


def elegir(frac, iou_u, iou_n, iou_m):
    usados, filas = set(), []

    def libres(idx):
        return np.array([i for i in idx if i not in usados], dtype=int)

    def agregar(nombre, i):
        if i is not None:
            usados.add(int(i))
            filas.append((nombre, int(i)))

    def mediano(idx):
        idx = libres(idx)
        if len(idx) == 0:
            return None
        return idx[np.argmin(np.abs(iou_u[idx] - np.median(iou_u[idx])))]

    mezcla = np.where((frac >= 0.10) & (frac <= 0.90))[0]
    poca = np.where((frac >= 0.01) & (frac < 0.10))[0]
    con_agua = np.where(frac >= 0.01)[0]
    if len(mezcla):
        mejora = iou_u[mezcla] - np.maximum(iou_n[mezcla], iou_m[mezcla])
        agregar("Mayor mejora de la U-Net", mezcla[np.argmax(mejora)])
    agregar("Caso típico (IoU mediano)", mediano(mezcla))
    agregar("Poca agua (1-10 %)", mediano(poca))
    resto = libres(con_agua)
    if len(resto):
        agregar("Peor caso de la U-Net", resto[np.argmin(iou_u[resto])])
    return filas


def mapa_errores(pred, real):
    m = np.zeros(real.shape, np.uint8)  # 0 = tierra bien clasificada
    m[pred & real] = 1                  # agua detectada
    m[~pred & real] = 2                 # agua no detectada
    m[pred & ~real] = 3                 # tierra marcada como agua
    return m


def dibujar(filas, X_te, real, metodos, frac, salida):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    c_err = ListedColormap([COLORES["tierra"], COLORES["agua_ok"], COLORES["agua_perdida"], COLORES["falsa_agua"]])
    c_etq = ListedColormap([COLORES["tierra"], COLORES["agua_ok"]])
    i_rgb = [BANDAS.index(b) for b in ("B4", "B3", "B2")]
    tope = max(float(np.percentile(X_te[:, i_rgb], 98)), 1e-6)  # mismo ajuste de color para todos

    n = len(filas)
    alto_leyenda = 0.45
    fig, ejes = plt.subplots(n, 5, figsize=(8.2, 1.9 * n + alto_leyenda), squeeze=False)
    cabeceras = ["Sentinel-2 (color real)", "Etiqueta (WorldCover)"] + [m[0] for m in metodos]
    for f, (nombre, i) in enumerate(filas):
        rgb = np.clip(np.moveaxis(X_te[i, i_rgb], 0, -1) / tope, 0, 1)
        ejes[f, 0].imshow(rgb, interpolation="nearest")
        ejes[f, 1].imshow(real[i], cmap=c_etq, vmin=0, vmax=1, interpolation="nearest")
        for c, (titulo, pred, iou) in enumerate(metodos, start=2):
            ejes[f, c].imshow(mapa_errores(pred[i], real[i]), cmap=c_err, vmin=0, vmax=3, interpolation="nearest")
            texto = f"IoU {coma(iou[i])}"
            ejes[f, c].set_title(f"{titulo}\n{texto}" if f == 0 else texto, fontsize=8.5)
        if f == 0:
            ejes[f, 0].set_title(cabeceras[0], fontsize=8.5)
            ejes[f, 1].set_title(cabeceras[1], fontsize=8.5)
        for ax in ejes[f]:
            ax.set_xticks([])
            ax.set_yticks([])
            for borde in ax.spines.values():
                borde.set_color("#9ca3af")
        ejes[f, 0].set_ylabel(f"{nombre}\nagua: {frac[i] * 100:.0f} %", fontsize=8.5)
    leyenda = [Patch(facecolor=COLORES["agua_ok"], label="Agua detectada"),
               Patch(facecolor=COLORES["agua_perdida"], label="Agua no detectada"),
               Patch(facecolor=COLORES["falsa_agua"], label="Tierra marcada como agua"),
               Patch(facecolor=COLORES["tierra"], edgecolor="#9ca3af", label="Tierra")]
    fig.legend(handles=leyenda, loc="lower center", bbox_to_anchor=(0.5, 0), ncol=4, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=[0, alto_leyenda / (1.9 * n + alto_leyenda), 1, 1], h_pad=1.6, w_pad=0.4)
    fig.savefig(salida, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--carpeta", default=str(PARCHES_DIR), help="carpeta de recortes")
    parser.add_argument("--salida", default=str(MODEL_DIR / "fig_unet_ejemplos_v2.png"))
    args = parser.parse_args()

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    modelo, ckpt = cargar_modelo()
    X, Y, lat, lon, _ = cargar_parches(args.carpeta)
    _, _, i_te = separar_espacialmente(lat, lon)
    X_te, real = X[i_te], Y[i_te] >= 0.5
    print(f"Recortes de prueba: {len(i_te)}")

    prob = predecir(modelo, X_te, np.asarray(ckpt["media"], np.float32), np.asarray(ckpt["desv"], np.float32))
    pred_u = prob >= ckpt.get("umbral", 0.5)
    comprobar(pred_u, real)

    pred_n = indice_normalizado(X_te, "B3", "B8") > 0
    pred_m = indice_normalizado(X_te, "B3", "B11") > 0
    iou_u, iou_n, iou_m = (iou_por_recorte(p, real) for p in (pred_u, pred_n, pred_m))
    frac = real.mean(axis=(1, 2))

    filas = elegir(frac, iou_u, iou_n, iou_m)
    if not filas:
        sys.exit("No hay recortes de prueba con agua suficiente para la figura.")
    print("\nRecortes elegidos (IoU de cada recorte):")
    for nombre, i in filas:
        print(f"  {nombre:<28} agua {coma(frac[i] * 100, 1):>5} % | U-Net {coma(iou_u[i])} | "
              f"NDWI {coma(iou_n[i])} | MNDWI {coma(iou_m[i])}")

    metodos = [("U-Net (umbral 0,5)", pred_u, iou_u), ("NDWI > 0", pred_n, iou_n), ("MNDWI > 0", pred_m, iou_m)]
    dibujar(filas, X_te, real, metodos, frac, args.salida)
    print(f"\nFigura guardada en {args.salida}")


if __name__ == "__main__":
    main()
