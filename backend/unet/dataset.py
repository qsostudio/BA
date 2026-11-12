"""
dataset.py — Parches Sentinel-2 + mascara de agua (ESA WorldCover) para la U-Net.

Cada parche es un .npz con:
    imagen   (6, 64, 64) reflectancia de B2, B3, B4, B8, B11, B12 (0-1)
    mascara  (64, 64)    1 = agua segun ESA WorldCover 2021, 0 = resto
    lat, lon             centro del parche
    tipo                 "agua" o "tierra" (como se eligio el centro)
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

LADO = 64


def cargar_parches(carpeta):
    """Lee todos los parches de la carpeta. Devuelve X (N,6,64,64), Y (N,64,64), lat, lon, tipo."""
    archivos = sorted(Path(carpeta).glob("parche_*.npz"))
    X, Y, lat, lon, tipo = [], [], [], [], []
    for a in archivos:
        d = np.load(a)
        if d["imagen"].shape[1:] != (LADO, LADO):
            continue
        X.append(d["imagen"].astype(np.float32))
        Y.append(d["mascara"].astype(np.float32))
        lat.append(float(d["lat"]))
        lon.append(float(d["lon"]))
        tipo.append(str(d["tipo"]))
    if not X:
        raise FileNotFoundError(f"No hay parches en {carpeta}. Ejecuta antes unet/descargar_parches.py")
    return np.stack(X), np.stack(Y), np.array(lat), np.array(lon), np.array(tipo)


class ParchesAguaDataset(Dataset):
    """Normaliza cada banda con la media y desviacion del conjunto de entrenamiento.
    Con aumentar=True aplica volteos y rotaciones de 90 grados (solo entrenamiento)."""

    def __init__(self, X, Y, media, desv, aumentar=False):
        self.X = ((X - media[None, :, None, None]) / desv[None, :, None, None]).astype(np.float32)
        self.Y = Y.astype(np.float32)
        self.aumentar = aumentar

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x, y = self.X[i], self.Y[i]
        if self.aumentar:
            k = np.random.randint(4)
            x, y = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k, axes=(0, 1))
            if np.random.rand() < 0.5:
                x, y = x[:, :, ::-1], y[:, ::-1]
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))[None]
