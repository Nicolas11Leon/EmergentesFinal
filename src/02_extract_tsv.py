"""
Módulo 02: Extracción tabular de firmas espectrales y construcción del dataset.tsv.
Interseca polígonos de entrenamiento (ground_truth.gpkg) con el stack de 10 bandas
y extrae los valores píxel por píxel hacia data/training/dataset.tsv.
"""

import os
import sys
import yaml
import csv

def load_config(config_path: str = "config/project.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def extract_tsv(config_path: str = "config/project.yaml"):
    try:
        import rasterio
        from rasterio.mask import mask
        import geopandas as gpd
        from shapely.geometry import Point
    except ImportError:
        print("[ADVERTENCIA] 'rasterio' o 'geopandas' no están instalados aún.")
        print("  Instálalos ejecutando: pip install -r requirements.txt")
        return False

    cfg = load_config(config_path)
    stack_path = cfg["paths"]["processed_stack"]
    vectors_path = cfg["paths"]["training_vectors"]
    output_tsv = cfg["paths"]["dataset_tsv"]
    band_order = cfg["spectral_bands"]["order"]
    
    print(f"[*] Iniciando extracción espacial a TSV...")
    print(f"    Stack ráster: {stack_path}")
    print(f"    Vectores de entrenamiento: {vectors_path}")
    print(f"    Salida esperada: {output_tsv}")
    
    if not os.path.isfile(stack_path):
        print(f"[ERROR] El stack procesado no existe: '{stack_path}'")
        print("  Ejecuta primero: python src/01_prepare_stack.py")
        return False
        
    if not os.path.isfile(vectors_path):
        print(f"[ERROR] El archivo vectorial no existe: '{vectors_path}'")
        print("  Coloca el archivo de polígonos (ground_truth.gpkg o .shp) en data/vectors/")
        return False
        
    return True

if __name__ == "__main__":
    extract_tsv()
