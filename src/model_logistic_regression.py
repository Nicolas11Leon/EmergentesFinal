"""
Módulo: Modelo 1 - Regresión Logística (Baseline Obligatorio)
Implementa el entrenamiento, optimización de hiperparámetros (HPO con GridSearchCV),
evaluación rigurosa (Accuracy, Kappa, Macro-F1, Matriz de Confusión) y exportación
conforme a config/metrics_schema.json.
"""

import os
import sys
import time
import json
import yaml
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)
import joblib

def load_config(config_path: str = "config/project.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def train_logistic_regression(dataset_path: str = None, config_path: str = "config/project.yaml"):
    cfg = load_config(config_path)
    if dataset_path is None:
        # Usar dataset real si existe; si no, usar dataset sintético
        dataset_path = cfg["paths"]["dataset_tsv"]
        if not os.path.isfile(dataset_path):
            dataset_path = cfg["paths"]["synthetic_tsv"]
            print(f"[INFO] Usando dataset sintético para prueba técnica: {dataset_path}")
            
    print(f"[*] Cargando dataset desde: {dataset_path}")
    df = pd.read_csv(dataset_path, sep="\t")
    
    bands = cfg["spectral_bands"]["order"]
    X = df[bands].values
    y = df["Class"].values
    groups = df["Polygon_ID"].values
    unique_classes = sorted(list(df["Class"].unique()))
    
    # Separación espacial desarrollo / prueba independiente (80% dev, 20% test)
    partitions_file = "data/training/spatial_partitions.json"
    if os.path.isfile(partitions_file):
        with open(partitions_file, "r", encoding="utf-8") as pf:
            part_meta = json.load(pf)
        test_polys = set(str(p) for p in part_meta["test_polygons"])
        str_groups = np.array([str(g) for g in groups])
        test_mask = np.isin(str_groups, list(test_polys))
        dev_mask = ~test_mask
        dev_idx = np.where(dev_mask)[0]
        test_idx = np.where(test_mask)[0]
        print(f"[*] Utilizando particiones espaciales predefinidas desde: {partitions_file}")
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=cfg["validation_strategy"]["test_size"], random_state=cfg["project"]["random_seed"])
        dev_idx, test_idx = next(gss.split(X, y, groups=groups))
    
    X_dev, y_dev, groups_dev = X[dev_idx], y[dev_idx], groups[dev_idx]
    X_test, y_test, groups_test = X[test_idx], y[test_idx], groups[test_idx]
    
    print(f"    Muestras de desarrollo: {len(X_dev)} ({len(np.unique(groups_dev))} polígonos)")
    print(f"    Muestras de prueba reservada: {len(X_test)} ({len(np.unique(groups_test))} polígonos)")
    
    # Construcción del Pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(random_state=cfg["project"]["random_seed"], max_iter=1000))
    ])
    
    # HPO mediante GridSearchCV sobre datos de desarrollo utilizando los folds exactos del JSON
    param_grid = cfg["models"]["logistic_regression"]["hpo_grid"]
    
    if os.path.isfile(partitions_file):
        custom_cv = []
        for fold_name, fold_data in part_meta["cv_folds"].items():
            val_polys = set(str(p) for p in fold_data["val_polygons"])
            val_mask_f = np.isin(groups_dev, list(val_polys))
            train_idx_f = np.where(~val_mask_f)[0]
            val_idx_f = np.where(val_mask_f)[0]
            custom_cv.append((train_idx_f, val_idx_f))
        cv = custom_cv
        print(f"[*] Folds de validación cruzada configurados desde JSON: {len(custom_cv)} particiones exactas.")
    else:
        cv = GroupKFold(n_splits=cfg["validation_strategy"]["n_cv_splits"])
    
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        refit=True
    )
    
    print("[*] Ejecutando GridSearchCV sobre particiones espaciales...")
    start_time = time.time()
    grid_search.fit(X_dev, y_dev, groups=groups_dev)
    training_time = round(time.time() - start_time, 2)
    
    best_pipeline = grid_search.best_estimator_
    best_params = grid_search.best_params_
    cv_mean = float(grid_search.best_score_)
    cv_std = float(grid_search.cv_results_["std_test_score"][grid_search.best_index_])
    
    print(f"[✓] Entrenamiento completado en {training_time} s.")
    print(f"    Mejores parámetros: {best_params}")
    print(f"    Macro-F1 promedio en CV: {cv_mean:.4f} (± {cv_std:.4f})")
    
    # Evaluación sobre el conjunto de prueba independiente
    print("[*] Evaluando sobre conjunto de prueba independiente...")
    y_pred = best_pipeline.predict(X_test)
    
    acc = float(accuracy_score(y_test, y_pred))
    kappa = float(cohen_kappa_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted"))
    cm = confusion_matrix(y_test, y_pred, labels=unique_classes).tolist()
    
    report_dict = classification_report(y_test, y_pred, labels=unique_classes, output_dict=True, zero_division=0)
    class_metrics = {}
    for c in unique_classes:
        class_metrics[c] = {
            "precision": round(float(report_dict[c]["precision"]), 4),
            "recall": round(float(report_dict[c]["recall"]), 4),
            "f1_score": round(float(report_dict[c]["f1-score"]), 4),
            "support": int(report_dict[c]["support"])
        }
        
    print(f"    Overall Accuracy: {acc:.4f}")
    print(f"    Kappa: {kappa:.4f}")
    print(f"    Macro-F1: {macro_f1:.4f}")
    
    # Guardar modelo serializado
    os.makedirs(cfg["paths"]["models_dir"], exist_ok=True)
    model_save_path = os.path.join(cfg["paths"]["models_dir"], "logistic_regression.joblib")
    joblib.dump(best_pipeline, model_save_path)
    print(f"[✓] Modelo guardado en: {model_save_path}")
    
    # Exportar resultados en JSON conforme a metrics_schema.json
    os.makedirs(cfg["paths"]["outputs_metrics"], exist_ok=True)
    metrics_payload = {
        "model_name": "Logistic Regression (Baseline)",
        "best_params": best_params,
        "cv_macro_f1_mean": round(cv_mean, 4),
        "cv_macro_f1_std": round(cv_std, 4),
        "test_overall_accuracy": round(acc, 4),
        "test_kappa": round(kappa, 4),
        "test_macro_f1": round(macro_f1, 4),
        "test_weighted_f1": round(weighted_f1, 4),
        "test_confusion_matrix": cm,
        "class_metrics": class_metrics,
        "training_time_seconds": training_time,
        "classes": unique_classes
    }
    
    json_save_path = os.path.join(cfg["paths"]["outputs_metrics"], "logistic_regression.json")
    with open(json_save_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=False)
        
    print(f"[✓] Métricas exportadas a: {json_save_path}")
    return metrics_payload

if __name__ == "__main__":
    tsv_arg = sys.argv[1] if len(sys.argv) > 1 else None
    train_logistic_regression(tsv_arg)
