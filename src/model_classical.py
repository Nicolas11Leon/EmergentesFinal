"""Entrena Decision Tree, SVM o KNN con las particiones oficiales del proyecto.

Ejemplo (desde la raiz del repositorio):
    .\\.venv\\Scripts\\python.exe src/model_classical.py --model decision_tree

La validacion cruzada utiliza literalmente train_polygons/val_polygons del JSON.
No recrea folds con GroupKFold y no utiliza el conjunto test para HPO.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # Entorno de terminal, sin ventana grafica.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "decision_tree": ("Decision Tree", "decision_tree"),
    "svm": ("SVM (RBF)", "svm"),
    "knn": ("K-Nearest Neighbors", "knn"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_inputs(root: Path):
    config_file = root / "config/project.yaml"
    data_file = root / "data/training/dataset.tsv"
    split_file = root / "data/training/spatial_partitions.json"
    for path in (config_file, data_file, split_file):
        if not path.is_file():
            raise FileNotFoundError(f"Archivo obligatorio ausente: {path}")

    cfg = yaml.safe_load(config_file.read_text(encoding="utf-8-sig"))
    part = json.loads(split_file.read_text(encoding="utf-8-sig"))
    df = pd.read_csv(data_file, sep="\t", dtype={"Polygon_ID": str})
    bands = cfg["spectral_bands"]["order"]
    required = set(bands) | {"Class", "Polygon_ID", "Class_ID"}
    if df.empty or not required.issubset(df.columns):
        raise ValueError(f"TSV vacio o faltan columnas: {sorted(required - set(df.columns))}")
    if df[list(required)].isna().any().any():
        raise ValueError("TSV contiene valores nulos en columnas necesarias")
    values = df[bands].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("TSV contiene bandas no finitas")
    if df.groupby("Polygon_ID")["Class"].nunique().max() != 1:
        raise ValueError("Polygon_ID con varias clases")
    if df.groupby("Polygon_ID")["Class_ID"].nunique().max() != 1:
        raise ValueError("Polygon_ID con varios Class_ID")
    return cfg, part, df, bands, data_file, split_file


def build_official_splits(df: pd.DataFrame, part: dict, expected_folds: int):
    """Devuelve indices de filas de desarrollo, test y splits relativos a dev."""
    all_polygons = set(df["Polygon_ID"].astype(str))
    dev_polygons = set(map(str, part["dev_polygons"]))
    test_polygons = set(map(str, part["test_polygons"]))

    if dev_polygons & test_polygons or dev_polygons | test_polygons != all_polygons:
        raise ValueError("Desarrollo/test tienen fugas o cobertura incompleta")
    if part.get("total_polygons") != len(all_polygons):
        raise ValueError("total_polygons no coincide con el TSV")

    groups = df["Polygon_ID"].astype(str)
    dev = df.loc[groups.isin(dev_polygons)].copy().reset_index(drop=True)
    test = df.loc[groups.isin(test_polygons)].copy().reset_index(drop=True)
    classes = sorted(df["Class"].unique().tolist())
    for label, subset in (("desarrollo", dev), ("test", test)):
        if set(subset["Class"]) != set(classes):
            raise ValueError(f"Faltan clases en {label}")

    folds = part["cv_folds"]
    if not isinstance(folds, dict) or len(folds) != expected_folds:
        raise ValueError("El numero de folds difiere de config/project.yaml")
    if part.get("n_cv_splits") != expected_folds:
        raise ValueError("n_cv_splits del JSON no coincide con la configuracion")

    dev_groups = dev["Polygon_ID"].astype(str)
    cv_indices = []
    all_val_groups = []
    for i in range(expected_folds):
        fold_name = f"fold_{i}"
        if fold_name not in folds:
            raise ValueError(f"Falta {fold_name}")
        fold = folds[fold_name]
        training_groups = set(map(str, fold["train_polygons"]))
        validation_groups = set(map(str, fold["val_polygons"]))
        if training_groups & validation_groups:
            raise ValueError(f"Fuga de poligonos en {fold_name}")
        if training_groups | validation_groups != dev_polygons:
            raise ValueError(f"Cobertura incorrecta de {fold_name}")
        if not validation_groups:
            raise ValueError(f"{fold_name} vacio")

        idx_train = np.flatnonzero(dev_groups.isin(training_groups).to_numpy())
        idx_val = np.flatnonzero(dev_groups.isin(validation_groups).to_numpy())
        if not len(idx_train) or not len(idx_val):
            raise ValueError(f"{fold_name} sin muestras")
        for label, idx in (("train", idx_train), ("validacion", idx_val)):
            if set(dev.iloc[idx]["Class"]) != set(classes):
                raise ValueError(f"Faltan clases en {fold_name}/{label}")
        cv_indices.append((idx_train, idx_val))
        all_val_groups.extend(validation_groups)

    if len(all_val_groups) != len(set(all_val_groups)) or set(all_val_groups) != dev_polygons:
        raise ValueError("Cada poligono de desarrollo debe validarse una sola vez")
    return dev, test, classes, cv_indices


def make_pipeline(name: str, seed: int) -> Pipeline:
    if name == "decision_tree":
        return Pipeline([("clf", DecisionTreeClassifier(random_state=seed))])
    if name == "svm":
        return Pipeline([("scaler", StandardScaler()), ("clf", SVC())])
    if name == "knn":
        return Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier())])
    raise ValueError(f"Modelo no reconocido: {name}")


def train_one(name: str, root: Path, jobs: int, dry_run: bool = False) -> dict | None:
    cfg, part, df, bands, dataset_path, partitions_path = load_inputs(root)
    cv_count = int(cfg["validation_strategy"]["n_cv_splits"])
    dev, test, classes, cv = build_official_splits(df, part, cv_count)
    params = cfg["models"][name]["hpo_grid"]
    seed = int(cfg["project"]["random_seed"])
    estimator = make_pipeline(name, seed)

    X_dev = dev[bands].to_numpy(dtype=float)
    y_dev = dev["Class"].to_numpy()
    X_test = test[bands].to_numpy(dtype=float)
    y_test = test["Class"].to_numpy()

    print(f"\n[MODELO] {MODELS[name][0]}")
    print(f"[DATOS] desarrollo={len(dev)}, prueba={len(test)}, folds oficiales={len(cv)}")
    print(f"[GRID] {params}")
    print(f"[HUELLAS] TSV={sha256(dataset_path)[:16]}... JSON={sha256(partitions_path)[:16]}...")
    if dry_run:
        print("[OK] Auditoria completa. Sin entrenamiento ni escritura.")
        return None

    grid = GridSearchCV(
        estimator=estimator,
        param_grid=params,
        cv=cv,  # Indices EXACTOS del archivo compartido.
        scoring="f1_macro",
        n_jobs=jobs,
        refit=True,
        error_score="raise",
        return_train_score=False,
        verbose=1,
    )
    t0 = time.perf_counter()
    grid.fit(X_dev, y_dev)
    elapsed = round(time.perf_counter() - t0, 2)

    # Test final: no participa en la busqueda ni seleccion de hiperparametros.
    prediction = grid.best_estimator_.predict(X_test)
    report = classification_report(
        y_test, prediction, labels=classes,
        output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_test, prediction, labels=classes)
    metrics = {
        "model_name": MODELS[name][0],
        "best_params": grid.best_params_,
        "cv_macro_f1_mean": round(float(grid.best_score_), 4),
        "cv_macro_f1_std": round(float(grid.cv_results_["std_test_score"][grid.best_index_]), 4),
        "test_overall_accuracy": round(float(accuracy_score(y_test, prediction)), 4),
        "test_kappa": round(float(cohen_kappa_score(y_test, prediction)), 4),
        "test_macro_f1": round(float(f1_score(y_test, prediction, average="macro", labels=classes)), 4),
        "test_weighted_f1": round(float(f1_score(y_test, prediction, average="weighted", labels=classes)), 4),
        "test_confusion_matrix": cm.tolist(),
        "class_metrics": {
            c: {
                "precision": round(float(report[c]["precision"]), 4),
                "recall": round(float(report[c]["recall"]), 4),
                "f1_score": round(float(report[c]["f1-score"]), 4),
                "support": int(report[c]["support"]),
            }
            for c in classes
        },
        "training_time_seconds": elapsed,
        "classes": classes,
    }

    models_dir = root / cfg["paths"]["models_dir"]
    metrics_dir = root / cfg["paths"]["outputs_metrics"]
    figures_dir = root / cfg["paths"]["outputs_figures"]
    for folder in (models_dir, metrics_dir, figures_dir):
        folder.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / f"{name}.joblib"
    metrics_path = metrics_dir / f"{name}.json"
    joblib.dump(grid.best_estimator_, model_path)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(9, 7))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(
        ax=ax, xticks_rotation=30, cmap="Blues", colorbar=False
    )
    fig.tight_layout()
    fig.savefig(figures_dir / f"{name}_confusion_matrix.png", dpi=180)
    plt.close(fig)

    if name == "decision_tree":
        importances = grid.best_estimator_.named_steps["clf"].feature_importances_
        fig, ax = plt.subplots(figsize=(8, 5))
        order = np.argsort(importances)
        ax.barh(np.asarray(bands)[order], importances[order])
        ax.set_xlabel("Importancia por reduccion de impureza")
        ax.set_title("Importancia de bandas - Arbol de Decision")
        fig.tight_layout()
        fig.savefig(figures_dir / "decision_tree_band_importance.png", dpi=180)
        plt.close(fig)

    # Trazabilidad por separado: evaluate_models.py solo debe leer JSON de metricas.
    provenance = {
        "dataset_sha256": sha256(dataset_path),
        "partitions_sha256": sha256(partitions_path),
        "cv_source": "data/training/spatial_partitions.json:cv_folds",
        "cv_folds": len(cv),
        "dev_polygons": len(set(dev["Polygon_ID"])),
        "test_polygons": len(set(test["Polygon_ID"])),
        "feature_order": bands,
        "seed": seed,
    }
    (models_dir / f"{name}_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[OK] CV Macro-F1={metrics['cv_macro_f1_mean']:.4f}")
    print(f"[OK] TEST Macro-F1={metrics['test_macro_f1']:.4f}; "
          f"Accuracy={metrics['test_overall_accuracy']:.4f}; "
          f"Kappa={metrics['test_kappa']:.4f}")
    print(f"[OK] Modelo: {model_path}")
    print(f"[OK] Metricas: {metrics_path}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=[*MODELS, "all"], default="decision_tree")
    parser.add_argument("--jobs", type=int, default=2,
                        help="Paralelismo de GridSearchCV; en Windows se recomienda 2")
    parser.add_argument("--check-only", action="store_true",
                        help="Auditar dataset y folds sin entrenar ni escribir archivos")
    args = parser.parse_args()
    if args.jobs == 0 or args.jobs < -1:
        parser.error("--jobs debe ser -1 o un entero positivo")
    targets = list(MODELS) if args.model == "all" else [args.model]
    for model_name in targets:
        train_one(model_name, ROOT, args.jobs, dry_run=args.check_only)


if __name__ == "__main__":
    main()
