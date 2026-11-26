"""Recalcula las metricas de prueba de los cinco modelos desde los .joblib.

No entrena ni modifica nada. Carga cada modelo, predice sobre los poligonos de prueba
de data/training/spatial_partitions.json y compara Accuracy, Kappa y Macro-F1 con
outputs/metrics/<modelo>.json.

Uso (desde la raiz del repositorio):
    python src/verify_metrics.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("logistic_regression", "decision_tree", "svm", "knn", "ann")
BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
TOLERANCE = 5e-5  # los JSON redondean a 4 decimales


def main() -> int:
    df = pd.read_csv(ROOT / "data/training/dataset.tsv", sep="\t", dtype={"Polygon_ID": str})
    parts = json.loads((ROOT / "data/training/spatial_partitions.json").read_text(encoding="utf-8"))
    dev, test = set(parts["dev_polygons"]), set(parts["test_polygons"])
    if dev & test:
        print("[ERROR] Hay poligonos en desarrollo y prueba a la vez")
        return 2
    test_df = df[df["Polygon_ID"].isin(test)]
    print(f"[OK] Prueba: {len(test)} poligonos, {len(test_df)} pixeles; sin cruce con desarrollo")

    failures = 0
    for name in MODELS:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = joblib.load(ROOT / f"models/{name}.joblib")
        version_warning = any("version" in str(w.message).lower() for w in caught)
        pred = model.predict(test_df[BANDS].to_numpy())
        y_true = test_df["Class_ID"] if np.issubdtype(np.asarray(pred).dtype, np.number) else test_df["Class"]
        got = {
            "test_overall_accuracy": accuracy_score(y_true, pred),
            "test_kappa": cohen_kappa_score(y_true, pred),
            "test_macro_f1": f1_score(y_true, pred, average="macro"),
        }
        ref = json.loads((ROOT / f"outputs/metrics/{name}.json").read_text(encoding="utf-8"))
        ok = all(abs(got[k] - float(ref[k])) <= TOLERANCE for k in got)
        failures += not ok
        note = "  (aviso: version de scikit-learn distinta a la de serializacion)" if version_warning else ""
        print(f"[{'OK' if ok else 'FALLA'}] {name:20s} acc={got['test_overall_accuracy']:.4f} "
              f"kappa={got['test_kappa']:.4f} macroF1={got['test_macro_f1']:.4f}{note}")
    print("[APROBADO] Las metricas de los cinco modelos se reproducen." if not failures
          else f"[ERROR] {failures} modelo(s) no reproducen sus metricas.")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
