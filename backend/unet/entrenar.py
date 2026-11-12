"""
entrenar.py — Entrena la U-Net de agua superficial y la evalua frente a los
indices espectrales clasicos (MNDWI y NDWI), sin entrenamiento.

Ejecutar desde la carpeta backend/ (despues de unet/descargar_parches.py):
    python unet/entrenar.py --rapido     # prueba de 3 epocas (~1-2 min)
    python unet/entrenar.py              # completo: max. 40 epocas con parada temprana

Separacion ESPACIAL por bloques de 10 km (ningun bloque aparece en dos conjuntos):
    70 % entrenamiento, 15 % validacion (parada temprana y LR) y 15 % test.
Entrenamiento: perdida BCE + Dice (agua = clase minoritaria), Adam, ReduceLROnPlateau,
    recorte de gradiente, BatchNorm, Dropout2d, aumento de datos (rotaciones y volteos)
    y normalizacion con estadisticos SOLO del conjunto de entrenamiento.

Genera en backend/modelo/:
    modelo_unet.pth           pesos + normalizacion (los usa la app)
    metricas_unet.json        metricas de test (U-Net frente a MNDWI y NDWI)
    fig_unet_curvas.png       curvas de aprendizaje
    fig_unet_ejemplos.png     ejemplos de test: imagen, WorldCover, U-Net, MNDWI
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader

AQUI = Path(__file__).resolve().parent
BACKEND = AQUI.parent
sys.path.insert(0, str(AQUI))
from dataset import ParchesAguaDataset, cargar_parches  # noqa: E402
from model import UNetLigero, contar_parametros  # noqa: E402

PARCHES_DIR = AQUI / "parches_agua"
MODEL_DIR = BACKEND / "modelo"
MODEL_PATH = MODEL_DIR / "modelo_unet.pth"
METRICAS_PATH = MODEL_DIR / "metricas_unet.json"
BANDAS = ["B2", "B3", "B4", "B8", "B11", "B12"]
SEMILLA = 42
BATCH_SIZE = 16
LR = 1e-3
EPOCAS = 40
PACIENCIA = 6
BLOQUE_KM = 10.0
UMBRAL = 0.5
BASE_CH = 16


def fijar_semillas(semilla=SEMILLA):
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)


def bloques_espaciales(lat, lon, km=BLOQUE_KM):
    y = np.asarray(lat) * 111.32
    x = np.asarray(lon) * 111.32 * np.cos(np.radians(np.mean(lat)))
    return np.array([f"{int(a)}_{int(b)}" for a, b in zip(np.floor(x / km), np.floor(y / km))])


def separar_espacialmente(lat, lon):
    grupos = bloques_espaciales(lat, lon)
    indices = np.arange(len(lat))
    resto, test = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEMILLA)
                       .split(indices, groups=grupos))
    i_tr, i_va = next(GroupShuffleSplit(n_splits=1, test_size=0.15 / 0.85, random_state=SEMILLA)
                      .split(resto, groups=grupos[resto]))
    return resto[i_tr], resto[i_va], test


class DiceBCELoss(nn.Module):
    """BCE (pixel a pixel) + Dice (solapamiento), robusta al desequilibrio agua/tierra."""

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, objetivo):
        p = torch.sigmoid(logits)
        interseccion = (p * objetivo).sum(dim=(1, 2, 3))
        suma = p.sum(dim=(1, 2, 3)) + objetivo.sum(dim=(1, 2, 3))
        dice = 1 - ((2 * interseccion + 1) / (suma + 1)).mean()
        return self.bce(logits, objetivo) + dice


def metricas_pixel(pred, real):
    pred, real = np.asarray(pred, bool).ravel(), np.asarray(real, bool).ravel()
    tp = int(np.sum(pred & real))
    fp = int(np.sum(pred & ~real))
    fn = int(np.sum(~pred & real))
    tn = int(np.sum(~pred & ~real))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "iou": round(tp / (tp + fp + fn), 4) if tp + fp + fn else 0.0,
        "f1_dice": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "exactitud": round((tp + tn) / max(1, tp + fp + fn + tn), 4),
        "matriz_confusion": {"vp": tp, "fp": fp, "fn": fn, "vn": tn},
    }


def indice_normalizado(X, banda_a, banda_b):
    a, b = X[:, BANDAS.index(banda_a)], X[:, BANDAS.index(banda_b)]
    return (a - b) / (a + b + 1e-6)


def evaluar(modelo, loader, criterio):
    modelo.eval()
    total, n, probs = 0.0, 0, []
    with torch.no_grad():
        for x, y in loader:
            logits = modelo(x)
            total += criterio(logits, y).item() * len(x)
            n += len(x)
            probs.append(torch.sigmoid(logits).numpy()[:, 0])
    return total / max(n, 1), np.concatenate(probs)


def figuras(historial, X_te, Y_te, prob_te, mndwi_te, salida):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib no instalado: no se generan figuras.")
        return
    fig, ejes = plt.subplots(1, 2, figsize=(11, 4))
    ejes[0].plot(historial["perdida_entrenamiento"], label="Entrenamiento")
    ejes[0].plot(historial["perdida_validacion"], label="Validacion")
    ejes[0].axvline(historial["mejor_epoca"], color="grey", linestyle=":", label="Mejor epoca")
    ejes[0].set_xlabel("Epoca")
    ejes[0].set_ylabel("Perdida BCE + Dice")
    ejes[0].legend()
    ejes[0].set_title("Curvas de aprendizaje")
    ejes[1].plot(historial["iou_validacion"], color="#16a34a")
    ejes[1].axvline(historial["mejor_epoca"], color="grey", linestyle=":")
    ejes[1].set_xlabel("Epoca")
    ejes[1].set_ylabel("IoU agua (validacion)")
    ejes[1].set_title("IoU en validacion")
    fig.tight_layout()
    fig.savefig(salida / "fig_unet_curvas.png", dpi=200)
    plt.close(fig)

    orden = np.argsort(-Y_te.reshape(len(Y_te), -1).mean(axis=1))
    elegidos = list(orden[:3]) + list(orden[-1:])
    fig, ejes = plt.subplots(len(elegidos), 4, figsize=(11, 2.8 * len(elegidos)), squeeze=False)
    titulos = ["Sentinel-2 (color real)", "Agua ESA WorldCover (etiqueta)", "U-Net (probabilidad)", "MNDWI > 0 (referencia)"]
    for fila, i in enumerate(elegidos):
        rgb = np.stack([X_te[i, BANDAS.index(b)] for b in ("B4", "B3", "B2")], axis=-1)
        rgb = np.clip(rgb / max(np.percentile(rgb, 98), 1e-6), 0, 1)
        for col, (img, cmap) in enumerate([(rgb, None), (Y_te[i], "Blues"), (prob_te[i], "Blues"),
                                           (mndwi_te[i] > 0, "Blues")]):
            ejes[fila, col].imshow(img, cmap=cmap, vmin=0, vmax=1)
            ejes[fila, col].axis("off")
            if fila == 0:
                ejes[fila, col].set_title(titulos[col], fontsize=9)
    fig.tight_layout()
    fig.savefig(salida / "fig_unet_ejemplos.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epocas", type=int, default=EPOCAS)
    parser.add_argument("--rapido", action="store_true", help="prueba de 3 epocas")
    parser.add_argument("--carpeta", default=str(PARCHES_DIR), help="carpeta de parches")
    args = parser.parse_args()
    epocas = 3 if args.rapido else args.epocas

    fijar_semillas()
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    X, Y, lat, lon, tipo = cargar_parches(args.carpeta)
    i_tr, i_va, i_te = separar_espacialmente(lat, lon)
    print(f"Parches: {len(X)} ({(tipo == 'agua').sum()} centrados en agua) | "
          f"entrenamiento {len(i_tr)}, validacion {len(i_va)}, test {len(i_te)} (bloques de {BLOQUE_KM:.0f} km)")
    print(f"Pixeles de agua: {Y.mean() * 100:.1f} % del total")

    media = X[i_tr].mean(axis=(0, 2, 3)).astype(np.float32)
    desv = (X[i_tr].std(axis=(0, 2, 3)) + 1e-6).astype(np.float32)
    carga = lambda idx, aum, mezclar: DataLoader(  # noqa: E731
        ParchesAguaDataset(X[idx], Y[idx], media, desv, aumentar=aum), batch_size=BATCH_SIZE, shuffle=mezclar)
    dl_tr, dl_va, dl_te = carga(i_tr, True, True), carga(i_va, False, False), carga(i_te, False, False)

    modelo = UNetLigero(in_channels=len(BANDAS), out_channels=1, base_ch=BASE_CH)
    print(f"U-Net ligera: {contar_parametros(modelo):,} parametros entrenables")
    criterio = DiceBCELoss()
    optimizador = torch.optim.Adam(modelo.parameters(), lr=LR, weight_decay=1e-4)
    planificador = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizador, factor=0.5, patience=3)

    historial = {"perdida_entrenamiento": [], "perdida_validacion": [], "iou_validacion": [], "mejor_epoca": 0}
    mejor_perdida, mejor_estado, sin_mejora = float("inf"), None, 0
    inicio = time.time()
    for epoca in range(epocas):
        modelo.train()
        suma, n = 0.0, 0
        for x, y in dl_tr:
            optimizador.zero_grad()
            perdida = criterio(modelo(x), y)
            perdida.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
            optimizador.step()
            suma += perdida.item() * len(x)
            n += len(x)
        perdida_val, prob_val = evaluar(modelo, dl_va, criterio)
        iou_val = metricas_pixel(prob_val >= UMBRAL, Y[i_va] >= 0.5)["iou"]
        planificador.step(perdida_val)
        historial["perdida_entrenamiento"].append(suma / max(n, 1))
        historial["perdida_validacion"].append(perdida_val)
        historial["iou_validacion"].append(iou_val)
        mejora = perdida_val < mejor_perdida - 1e-4
        if mejora:
            mejor_perdida, sin_mejora = perdida_val, 0
            mejor_estado = {k: v.detach().clone() for k, v in modelo.state_dict().items()}
            historial["mejor_epoca"] = epoca
        else:
            sin_mejora += 1
        print(f"  Epoca {epoca + 1:>2}/{epocas} | perdida entr. {suma / max(n, 1):.4f} | val. {perdida_val:.4f} | "
              f"IoU val. {iou_val:.3f} | lr {optimizador.param_groups[0]['lr']:.0e}"
              f"{' *' if mejora else ''} | {time.time() - inicio:.0f} s")
        if sin_mejora >= PACIENCIA:
            print(f"  Parada temprana: {PACIENCIA} epocas sin mejorar la perdida de validacion")
            break

    modelo.load_state_dict(mejor_estado)
    _, prob_te = evaluar(modelo, dl_te, criterio)
    real_te = Y[i_te] >= 0.5
    mndwi_te = indice_normalizado(X[i_te], "B3", "B11")
    ndwi_te = indice_normalizado(X[i_te], "B3", "B8")
    resultados = {
        "U-Net (CNN)": metricas_pixel(prob_te >= UMBRAL, real_te),
        "MNDWI > 0 (referencia, sin entrenamiento)": metricas_pixel(mndwi_te > 0, real_te),
        "NDWI > 0 (referencia, sin entrenamiento)": metricas_pixel(ndwi_te > 0, real_te),
    }

    MODEL_DIR.mkdir(exist_ok=True)
    metricas = {
        "modelo": "U-Net ligera (segmentacion de agua superficial)",
        "etiquetas": "ESA WorldCover v200 (2021), clase 80 'aguas permanentes'",
        "entrada": "Sentinel-2 L2A armonizado, mediana jun-sep 2021: " + ", ".join(BANDAS),
        "parametros_entrenables": contar_parametros(modelo),
        "n_parches": {"entrenamiento": int(len(i_tr)), "validacion": int(len(i_va)), "test": int(len(i_te))},
        "separacion": f"Espacial por bloques de {BLOQUE_KM:.0f} km (70/15/15)",
        "epocas_entrenadas": len(historial["perdida_validacion"]),
        "mejor_epoca": historial["mejor_epoca"] + 1,
        "umbral": UMBRAL,
        "test": resultados,
        "tiempo_entrenamiento_s": round(time.time() - inicio, 1),
    }
    torch.save({"model_state_dict": modelo.state_dict(), "in_channels": len(BANDAS), "base_ch": BASE_CH,
                "bandas": BANDAS, "media": media.tolist(), "desv": desv.tolist(), "umbral": UMBRAL,
                "escala_m": 10, "metricas_test": resultados["U-Net (CNN)"]}, MODEL_PATH)
    with open(METRICAS_PATH, "w", encoding="utf-8") as f:
        json.dump(metricas, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 78)
    print("RESULTADOS EN TEST (pixeles de bloques no vistos)")
    print("=" * 78)
    print(f"{'Metodo':<45}{'IoU':>7}{'F1':>7}{'Prec.':>7}{'Recall':>8}")
    for nombre, m in resultados.items():
        print(f"{nombre:<45}{m['iou']:>7.3f}{m['f1_dice']:>7.3f}{m['precision']:>7.3f}{m['recall']:>8.3f}")
    print(f"\nMatriz de confusion U-Net: {resultados['U-Net (CNN)']['matriz_confusion']}")
    print(f"Guardado: {MODEL_PATH} y {METRICAS_PATH}")
    figuras(historial, X[i_te], Y[i_te], prob_te, mndwi_te, MODEL_DIR)
    print("Figuras: modelo/fig_unet_curvas.png y modelo/fig_unet_ejemplos.png")


if __name__ == "__main__":
    main()
