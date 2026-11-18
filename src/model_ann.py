"""Entrena ANN/MLP con los folds espaciales oficiales del proyecto.

Desde la raiz del repositorio:
    python src/model_ann.py --check-only
    python src/model_ann.py --jobs 2
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, classification_report,
    cohen_kappa_score, confusion_matrix, f1_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from model_classical import load_inputs, build_official_splits, sha256

ROOT = Path(__file__).resolve().parents[1]


def make_ann_pipeline(seed: int) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(random_state=seed, early_stopping=False)),
    ])


def load_ann_grid(cfg: dict) -> dict:
    configured = cfg["models"]["ann"]["hpo_grid"]
    required = {"clf__hidden_layer_sizes", "clf__alpha", "clf__max_iter"}
    if not required.issubset(configured):
        raise ValueError("Faltan hiperparametros de ANN en config/project.yaml")
    # No alteramos el conjunto oficial de experimentos.
    return configured


def validate_metric_keys(metrics: dict, schema_file: Path) -> None:
    schema = json.loads(schema_file.read_text(encoding="utf-8"))
    missing = set(schema["required"]) - set(metrics)
    if missing:
        raise ValueError(f"Metricas no cumplen el esquema, faltan: {sorted(missing)}")
    if len(metrics["classes"]) != len(metrics["test_confusion_matrix"]):
        raise ValueError("Dimensiones invalidas de la matriz de confusion")


def train(root: Path = ROOT, jobs: int = 2, check_only: bool = False) -> dict | None:
    cfg, partitions, data, bands, dataset_path, partitions_path = load_inputs(root)
    seed = int(cfg["project"]["random_seed"])
    folds_expected = int(cfg["validation_strategy"]["n_cv_splits"])
    development, test, classes, folds = build_official_splits(data, partitions, folds_expected)
    if len(bands) != 10 or len(set(bands)) != 10:
        raise ValueError("Se esperan diez bandas espectrales distintas")
    grid_values = load_ann_grid(cfg)
    print(f"[ANN] Desarrollo: {len(development)} muestras, test: {len(test)}, folds: {len(folds)}")
    print(f"[ANN] Bandas: {', '.join(bands)}")
    print(f"[ANN] Grid: {grid_values}")
    if check_only:
        print("[OK] Datos, bandas, folds y grid comprobados; no se entreno ni escribio nada")
        return None

    x_dev = development[bands].to_numpy(dtype=float)
    y_dev = development["Class"].to_numpy()
    x_test = test[bands].to_numpy(dtype=float)
    y_test = test["Class"].to_numpy()
    grid = GridSearchCV(
        estimator=make_ann_pipeline(seed), param_grid=grid_values,
        cv=folds, scoring="f1_macro", n_jobs=jobs,
        refit=True, error_score="raise", return_train_score=False, verbose=1,
    )
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        grid.fit(x_dev, y_dev)
    duration = round(time.perf_counter() - t0, 2)
    convergence_messages = sorted({str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)})
    # La muestra de test participa solo aqui, despues de fijar la mejor configuracion.
    predictions = grid.best_estimator_.predict(x_test)
    cm = confusion_matrix(y_test, predictions, labels=classes)
    class_report = classification_report(y_test, predictions, labels=classes,
                                         output_dict=True, zero_division=0)
    metrics = {
        "model_name": "ANN / MLP",
        "best_params": grid.best_params_,
        "cv_macro_f1_mean": round(float(grid.best_score_), 4),
        "cv_macro_f1_std": round(float(grid.cv_results_["std_test_score"][grid.best_index_]), 4),
        "test_overall_accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "test_kappa": round(float(cohen_kappa_score(y_test, predictions)), 4),
        "test_macro_f1": round(float(f1_score(y_test, predictions, average="macro", labels=classes)), 4),
        "test_weighted_f1": round(float(f1_score(y_test, predictions, average="weighted", labels=classes)), 4),
        "test_confusion_matrix": cm.tolist(),
        "class_metrics": {
            c: {
                "precision": round(float(class_report[c]["precision"]), 4),
                "recall": round(float(class_report[c]["recall"]), 4),
                "f1_score": round(float(class_report[c]["f1-score"]), 4),
                "support": int(class_report[c]["support"]),
            } for c in classes
        },
        "training_time_seconds": duration,
        "classes": classes,
    }
    validate_metric_keys(metrics, root / "config/metrics_schema.json")
    model_path = root / "models/ann.joblib"
    metric_path = root / "outputs/metrics/ann.json"
    figure_path = root / "outputs/figures/ann_confusion_matrix.png"
    for path in (model_path, metric_path, figure_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(grid.best_estimator_, model_path)
    # Comprobacion de reproducibilidad del pipeline completo guardado.
    reloaded = joblib.load(model_path)
    if not np.array_equal(predictions, reloaded.predict(x_test)):
        raise RuntimeError("Predicciones distintas tras recargar ann.joblib")
    metric_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(
        ax=ax, xticks_rotation=30, cmap="Blues", colorbar=False,
    )
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    provenance = {
        "dataset_sha256": sha256(dataset_path),
        "partitions_sha256": sha256(partitions_path),
        "cv_source": "data/training/spatial_partitions.json:cv_folds",
        "cv_folds": len(folds),
        "dev_polygons": int(development["Polygon_ID"].nunique()),
        "test_polygons": int(test["Polygon_ID"].nunique()),
        "feature_order": bands,
        "seed": seed,
        "grid_search": grid_values,
        "best_params": grid.best_params_,
        "convergence_warnings_detected_on_main_process": convergence_messages,
        "notes": "Con n_jobs>1, los warnings de subprocesos pueden no quedar capturados aqui.",
    }
    (root / "models/ann_provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[OK] ANN CV Macro-F1: {metrics['cv_macro_f1_mean']:.4f}")
    print(f"[OK] ANN test Macro-F1: {metrics['test_macro_f1']:.4f}")
    print(f"[OK] ANN guardada: {model_path}")
    if convergence_messages:
        print("[AVISO] Advertencias de convergencia detectadas, revisar ann_provenance.json")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.jobs == 0 or args.jobs < -1:
        parser.error("--jobs debe ser -1 o entero positivo")
    train(jobs=args.jobs, check_only=args.check_only)


if __name__ == "__main__":
    main()
