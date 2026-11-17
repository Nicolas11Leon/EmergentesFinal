"""
Módulo de Particiones Espaciales Compartidas (Zero Data Leakage).
Garantiza que todos los modelos (Rol 1: Logistic Regression, Rol 2: Decision Tree / SVM / KNN,
Rol 3: Red Neuronal MLP) utilicen exactamente las mismas particiones espaciales a nivel de polígono (Polygon_ID).
"""

import os
import sys
import json
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold

def load_config(config_path: str = "config/project.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def generate_spatial_splits(dataset_path: str = None, config_path: str = "config/project.yaml", output_json: str = "data/training/spatial_partitions.json"):
    cfg = load_config(config_path)
    if dataset_path is None:
        dataset_path = cfg["paths"]["dataset_tsv"]
        if not os.path.isfile(dataset_path):
            dataset_path = cfg["paths"]["synthetic_tsv"]
            print(f"[INFO] Usando dataset sintético para generar particiones de prueba: {dataset_path}")
            
    print(f"[*] Generando particiones espaciales desde: {dataset_path}")
    df = pd.read_csv(dataset_path, sep="\t")
    
    unique_polygons = df.groupby("Polygon_ID").agg(
        Class_ID=("Class_ID", "first"),
        Class=("Class", "first"),
        pixel_count=("Polygon_ID", "count")
    ).reset_index()
    
    print(f"    Total polígonos: {len(unique_polygons)}")
    for cls_name, grp in unique_polygons.groupby("Class"):
        print(f"      - {cls_name}: {len(grp)} polígonos, {grp['pixel_count'].sum()} píxeles")
        
    random_seed = cfg["project"]["random_seed"]
    test_size = cfg["validation_strategy"]["test_size"]
    n_splits = cfg["validation_strategy"]["n_cv_splits"]
    
    # 1. Separación Train-Dev / Test independiente a nivel de polígono estratificada por clase
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
    dev_poly_idx, test_poly_idx = next(sss.split(unique_polygons, unique_polygons["Class"]))
    
    dev_poly_ids = unique_polygons.iloc[dev_poly_idx]["Polygon_ID"].tolist()
    test_poly_ids = unique_polygons.iloc[test_poly_idx]["Polygon_ID"].tolist()
    
    print(f"[✓] Polígonos de prueba reservada (Test set): {len(test_poly_ids)} polígonos")
    print("    Composición del Test set por clase:")
    for cls_name, grp in unique_polygons.iloc[test_poly_idx].groupby("Class"):
        print(f"      - {cls_name:22s}: {len(grp)} polígonos ({grp['pixel_count'].sum()} píxeles)")
    print(f"[✓] Polígonos de desarrollo (Dev set): {len(dev_poly_ids)} polígonos")
    
    # 2. Generar folds de GroupKFold dentro del Dev set
    dev_polygons_df = unique_polygons[unique_polygons["Polygon_ID"].isin(dev_poly_ids)].reset_index(drop=True)
    gkf = GroupKFold(n_splits=n_splits)
    
    folds = {}
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(dev_polygons_df, groups=dev_polygons_df["Polygon_ID"])):
        train_p = dev_polygons_df.iloc[train_idx]["Polygon_ID"].tolist()
        val_p = dev_polygons_df.iloc[val_idx]["Polygon_ID"].tolist()
        folds[f"fold_{fold_idx}"] = {
            "train_polygons": [str(x) for x in train_p],
            "val_polygons": [str(x) for x in val_p]
        }
        
    partition_data = {
        "dataset_path": dataset_path,
        "random_seed": random_seed,
        "n_cv_splits": n_splits,
        "total_polygons": len(unique_polygons),
        "test_polygons": [str(x) for x in test_poly_ids],
        "dev_polygons": [str(x) for x in dev_poly_ids],
        "cv_folds": folds
    }
    
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(partition_data, f, indent=2)
        
    print(f"[✓] Particiones espaciales exportadas exitosamente a: {output_json}")
    return partition_data

if __name__ == "__main__":
    generate_spatial_splits()
