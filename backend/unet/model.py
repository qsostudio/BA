"""
model.py — U-Net ligera para segmentar agua superficial (balsas, charcas, embalses)
en imagenes Sentinel-2.

Arquitectura (Ronneberger et al., 2015) reducida para entrenar en CPU:
    codificador: 3 bloques [conv 3x3 + BatchNorm + ReLU] x 2, con max pooling 2x2
    cuello de botella: bloque con 8*base filtros y Dropout2d
    decodificador: convolucion transpuesta 2x2 + concatenacion con el codificador
                   (skip connections) + bloque conv
    salida: conv 1x1 -> 1 canal de logits (la sigmoide se aplica fuera, en la
            perdida BCEWithLogits y al predecir)
La entrada debe tener alto y ancho multiplos de 8 (3 poolings).
"""

import torch
import torch.nn as nn


class BloqueConv(nn.Module):
    def __init__(self, c_in, c_out, dropout=0.0):
        super().__init__()
        capas = [
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            capas.append(nn.Dropout2d(dropout))
        self.capas = nn.Sequential(*capas)

    def forward(self, x):
        return self.capas(x)


class UNetLigero(nn.Module):
    def __init__(self, in_channels=6, out_channels=1, base_ch=16):
        super().__init__()
        b = base_ch
        self.pool = nn.MaxPool2d(2)
        self.enc1 = BloqueConv(in_channels, b)
        self.enc2 = BloqueConv(b, 2 * b)
        self.enc3 = BloqueConv(2 * b, 4 * b)
        self.cuello = BloqueConv(4 * b, 8 * b, dropout=0.2)
        self.up3 = nn.ConvTranspose2d(8 * b, 4 * b, 2, stride=2)
        self.dec3 = BloqueConv(8 * b, 4 * b)
        self.up2 = nn.ConvTranspose2d(4 * b, 2 * b, 2, stride=2)
        self.dec2 = BloqueConv(4 * b, 2 * b)
        self.up1 = nn.ConvTranspose2d(2 * b, b, 2, stride=2)
        self.dec1 = BloqueConv(2 * b, b)
        self.salida = nn.Conv2d(b, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)                       # H
        e2 = self.enc2(self.pool(e1))           # H/2
        e3 = self.enc3(self.pool(e2))           # H/4
        c = self.cuello(self.pool(e3))          # H/8
        d3 = self.dec3(torch.cat([self.up3(c), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.salida(d1)                  # logits (N, 1, H, W)


def contar_parametros(modelo) -> int:
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)
