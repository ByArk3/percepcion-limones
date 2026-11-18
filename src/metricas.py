"""Métricas del proyecto (celda 9.1 del notebook)."""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def metricas(yt, yp):
    yt, yp = np.asarray(yt), np.asarray(yp)
    no_mad = yt != 2                                   # limones verdes o pintones
    return dict(exactitud=accuracy_score(yt, yp), f1_macro=f1_score(yt, yp, labels=[0, 1, 2], average="macro", zero_division=0),
                recall_maduro=recall_score(yt, yp, labels=[2], average="macro", zero_division=0),
                precision_maduro=precision_score(yt, yp, labels=[2], average="macro", zero_division=0),
                fp_maduro=float((yp[no_mad] == 2).mean()) if no_mad.any() else 0.0)
