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

"""
Módulo 02: Extracción tabular de firmas espectrales y construcción de dataset.tsv.
Interseca polígonos de verdad de terreno (ground_truth.gpkg) con el stack de 10 bandas
y extrae los valores píxel por píxel hacia data/training/dataset.tsv, preservando la
identidad del polígono (Polygon_ID) para la validación espacial sin fugas.
"""

import os
import sys
import yaml
import csv
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import pyproj

def load_config(config_path: str = "config/project.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def extract_tsv(config_path: str = "config/project.yaml") -> bool:
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
        print("  Coloca el archivo de polígonos (ground_truth.gpkg) en data/vectors/")
        return False
        
    # 1. Cargar geometrías vectoriales
    gdf = gpd.read_file(vectors_path)
    print(f"[✓] Polígonos cargados: {len(gdf)} registros.")
    
    # Validar columnas requeridas
    req_cols = ["Polygon_ID", "Class_ID", "Class"]
    for c in req_cols:
        if c not in gdf.columns:
            print(f"[ERROR] Columna requerida '{c}' ausente en el archivo vectorial.")
            return False
            
    # 2. Abrir stack ráster y verificar CRS
    with rasterio.open(stack_path) as src:
        raster_crs = src.crs
        raster_transform = src.transform
        raster_nodata = src.nodata
        num_bands = src.count
        
        if num_bands != len(band_order):
            print(f"[ERROR] El stack tiene {num_bands} bandas, pero se esperaban {len(band_order)}.")
            return False
            
        # Reproyectar geometrías al CRS del ráster si difieren
        if gdf.crs != raster_crs:
            print(f"    Reproyectando geometrías desde {gdf.crs} a {raster_crs}...")
            gdf = gdf.to_crs(raster_crs)
            
        # Preparar transformador a WGS84 para Latitude/Longitude
        proj_to_wgs84 = pyproj.Transformer.from_crs(raster_crs, "EPSG:4326", always_xy=True)
        
        # 3. Extraer píxeles por polígono
        records = []
        os.makedirs(os.path.dirname(output_tsv), exist_ok=True)
        
        for idx, row in gdf.iterrows():
            poly_id = int(row["Polygon_ID"])
            class_id = int(row["Class_ID"])
            class_name = str(row["Class"])
            geom = [row.geometry]
            
            try:
                # all_touched=False extrae píxeles cuyo centro cae dentro del polígono (firmas puras)
                out_image, out_transform = mask(src, geom, crop=True, all_touched=False, nodata=raster_nodata or -9999)
            except ValueError:
                # Si el polígono no interseca el ráster
                continue
                
            # out_image tiene forma (bands, H_sub, W_sub)
            c, h_sub, w_sub = out_image.shape
            # Máscara de píxeles válidos dentro del polígono
            valid_mask = out_image[0] != (raster_nodata or -9999)
            
            # Si all_touched=False no capturó píxeles (polígono muy angosto), reintentar con all_touched=True
            if not np.any(valid_mask):
                out_image, out_transform = mask(src, geom, crop=True, all_touched=True, nodata=raster_nodata or -9999)
                valid_mask = out_image[0] != (raster_nodata or -9999)
                
            rows, cols = np.where(valid_mask)
            for r, col in zip(rows, cols):
                pixel_vals = out_image[:, r, col]
                # Coordenadas proyectadas del centro del píxel
                x_coord, y_coord = rasterio.transform.xy(out_transform, r, col, offset="center")
                # Coordenadas geográficas WGS84
                lon, lat = proj_to_wgs84.transform(x_coord, y_coord)
                
                # Descartar píxeles con valores anómalos o NaN
                if np.any(np.isnan(pixel_vals)) or np.any(np.isinf(pixel_vals)):
                    continue
                if np.any(pixel_vals < 0.0) or np.any(pixel_vals > 1.0):
                    continue
                    
                rec = {
                    "Latitude": round(lat, 6),
                    "Longitude": round(lon, 6),
                    "Polygon_ID": poly_id,
                    "Class_ID": class_id,
                    "Class": class_name
                }
                for b_idx, b_name in enumerate(band_order):
                    rec[b_name] = round(float(pixel_vals[b_idx]), 5)
                    
                records.append(rec)
                
    print(f"[✓] Extracción finalizada: {len(records)} píxeles extraídos de {len(gdf)} polígonos.")
    
    # 4. Escribir archivo TSV según contrato oficial de validación
    fieldnames = ["Latitude", "Longitude", "Polygon_ID", "Class_ID"] + band_order + ["Class"]
    with open(output_tsv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(records)
        
    print(f"[✓] Archivo exportado con éxito a: {output_tsv}")
    return True

if __name__ == "__main__":
    success = extract_tsv()
    sys.exit(0 if success else 1)
