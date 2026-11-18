"""Partición estratificada 70/15/15 por imagen, semilla 42 (celda 6.1 del notebook)."""
import numpy as np
from sklearn.model_selection import train_test_split

SEMILLA_PARTICION = 42


def particionar(df, seed=SEMILLA_PARTICION):
    """Agrega la columna 'particion' (train / val / test). Una fila por imagen (un limón por foto)."""
    train, resto = train_test_split(df, test_size=0.30, stratify=df["clase"], random_state=seed)
    val, test = train_test_split(resto, test_size=0.50, stratify=resto["clase"], random_state=seed)
    df = df.copy()
    df.loc[train.index, "particion"] = "train"
    df.loc[val.index, "particion"] = "val"
    df.loc[test.index, "particion"] = "test"
    assert df["ruta"].is_unique                     # cada imagen aparece una sola vez
    return df


def indices(df):
    """(TR, VA, TE): posiciones de cada partición."""
    return tuple(np.where(df.particion == p)[0] for p in ("train", "val", "test"))
