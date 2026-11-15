"""
Generador de Dataset Sintético Robusto para EmergentesFinal.
Implementado usando ÚNICAMENTE la biblioteca estándar de Python (sin dependencias externas).
Permite a los frentes de Machine Learning e Inferencia trabajar y validar
sus pipelines en paralelo sin esperar el dataset real.
"""

import os
import csv
import math
import random

def generate_synthetic_dataset(output_path: str = "data/training/synthetic_dataset.tsv", n_samples: int = 1500, random_seed: int = 42):
    random.seed(random_seed)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 5 Clases con firmas espectrales sintéticas realistas (escala de reflectancia 0.0 a 1.0)
    # Bandas: B02 (Blue), B03 (Green), B04 (Red), B05 (RE1), B06 (RE2), B07 (RE3), B08 (NIR), B8A (NNIR), B11 (SWIR1), B12 (SWIR2)
    profiles = {
        "Bosque": {
            "id": 1,
            "mean": [0.03, 0.05, 0.03, 0.09, 0.25, 0.35, 0.40, 0.42, 0.18, 0.08],
            "std":  [0.01, 0.01, 0.01, 0.02, 0.03, 0.04, 0.04, 0.04, 0.02, 0.02],
            "weight": 0.35,
            "n_polygons": 8
        },
        "Pastos y Cultivos": {
            "id": 2,
            "mean": [0.05, 0.08, 0.06, 0.12, 0.22, 0.28, 0.32, 0.33, 0.20, 0.12],
            "std":  [0.01, 0.02, 0.02, 0.02, 0.03, 0.03, 0.04, 0.04, 0.03, 0.02],
            "weight": 0.25,
            "n_polygons": 6
        },
        "Suelo Desnudo y Roca": {
            "id": 3,
            "mean": [0.10, 0.13, 0.16, 0.18, 0.20, 0.22, 0.23, 0.24, 0.30, 0.28],
            "std":  [0.02, 0.02, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.04, 0.04],
            "weight": 0.20,
            "n_polygons": 6
        },
        "Urbano y Vías": {
            "id": 4,
            "mean": [0.12, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.22, 0.21],
            "std":  [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03],
            "weight": 0.10,
            "n_polygons": 4
        },
        "Agua": {
            "id": 5,
            "mean": [0.08, 0.07, 0.04, 0.02, 0.01, 0.01, 0.01, 0.01, 0.005, 0.003],
            "std":  [0.02, 0.02, 0.01, 0.01, 0.005, 0.005, 0.005, 0.005, 0.002, 0.002],
            "weight": 0.10,
            "n_polygons": 4
        }
    }
    
    bands = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
    header = ["Latitude", "Longitude", "Polygon_ID", "Class_ID"] + bands + ["Class"]
    
    rows = []
    poly_counter = 1
    # Centroide aproximado en San José del Palmar: Lat ~ 4.90, Lon ~ -76.23
    base_lat, base_lon = 4.903, -76.233
    class_counts = {}
    
    for class_name, cfg in profiles.items():
        class_samples = int(n_samples * cfg["weight"])
        samples_per_poly = class_samples // cfg["n_polygons"]
        class_counts[class_name] = 0
        
        for p in range(cfg["n_polygons"]):
            poly_id = f"poly_{poly_counter:03d}_{class_name[:3].upper()}"
            poly_counter += 1
            
            c_lat = base_lat + random.uniform(-0.04, 0.04)
            c_lon = base_lon + random.uniform(-0.04, 0.04)
            
            for _ in range(samples_per_poly):
                lat = round(c_lat + random.gauss(0, 0.0005), 6)
                lon = round(c_lon + random.gauss(0, 0.0005), 6)
                
                band_vals = []
                for m, s in zip(cfg["mean"], cfg["std"]):
                    val = random.gauss(m, s)
                    clamped = max(0.0001, min(0.9999, val))
                    band_vals.append(round(clamped, 4))
                
                row = [lat, lon, poly_id, cfg["id"]] + band_vals + [class_name]
                rows.append(row)
                class_counts[class_name] += 1
                
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows)
        
    print(f"✓ Dataset sintético generado con éxito en: {output_path}")
    print(f"  Total de muestras: {len(rows)}")
    print(f"  Total de polígonos independientes: {poly_counter - 1}")
    print(f"  Distribución por clase:")
    for c, count in class_counts.items():
        print(f"    - {c}: {count} píxeles ({count / len(rows) * 100:.1f}%)")

if __name__ == "__main__":
    generate_synthetic_dataset()
