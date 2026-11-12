"""Prepara el dataset de entrenamiento: excluye pozos con profundidad > 200 m.
Justificación: los pozos muy profundos (hasta 1317 m) son captaciones de
sondeos especiales; dominan el error cuadrático y no representan el objetivo
de la herramienta (pozos agrícolas someros)."""
import pandas as pd

PROF_MAX_M = 200
ENTRADA = "data/pozos_andalucia_enriquecido.csv"
SALIDA = "data/pozos_andalucia_entrenamiento_filtrado.csv"


def filtrar():
    df = pd.read_csv(ENTRADA)
    mantener = df["profundidad_m"].isna() | (df["profundidad_m"] <= PROF_MAX_M)
    out = df[mantener]
    out.to_csv(SALIDA, index=False)
    con_prof = df["profundidad_m"].notna().sum()
    excluidos = (~mantener).sum()
    print(f"[FILTRO] Filas totales: {len(df)} | con profundidad: {con_prof}")
    print(f"[FILTRO] Excluidos (> {PROF_MAX_M} m): {excluidos} ({excluidos/con_prof:.1%})")
    print(f"[FILTRO] Guardado: {SALIDA} ({len(out)} filas)")
    return out


if __name__ == "__main__":
    filtrar()
