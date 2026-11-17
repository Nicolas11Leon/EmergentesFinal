"""
Módulo de Inferencia Ráster Espacial Completa (Full-Scene Inference).
Despliega el modelo de Machine Learning entrenado sobre el stack de 10 bandas
para clasificar toda el área de estudio y exportar un GeoTIFF categórico con su
respectiva cuantificación de áreas (hectáreas y km²).
"""

import os
import sys
import json
import yaml
import numpy as np
import rasterio
import joblib

def load_config(config_path: str = "config/project.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def predict_full_scene(model_path: str = "models/logistic_regression.joblib",
                       stack_path: str = "data/processed/stack_10bands.tif",
                       output_raster: str = "outputs/rasters/landcover_logistic_regression.tif",
                       output_stats: str = "outputs/metrics/area_stats_logistic_regression.json",
                       config_path: str = "config/project.yaml"):
    print(f"[*] Iniciando inferencia ráster espacial completa...")
    print(f"    Modelo: {model_path}")
    print(f"    Stack: {stack_path}")
    print(f"    Salida: {output_raster}")
    
    if not os.path.isfile(model_path):
        print(f"[ERROR] No se encuentra el modelo: {model_path}")
        return False
    if not os.path.isfile(stack_path):
        print(f"[ERROR] No se encuentra el stack ráster: {stack_path}")
        return False
        
    cfg = load_config(config_path)
    model = joblib.load(model_path)
    
    # Cargar clases oficiales
    with open("config/classes.json", "r", encoding="utf-8") as f:
        classes_cfg = json.load(f)["classes"]
    class_name_to_id = {c["name"]: c["id"] for c in classes_cfg}
    class_id_to_name = {c["id"]: c["name"] for c in classes_cfg}
    
    os.makedirs(os.path.dirname(output_raster), exist_ok=True)
    os.makedirs(os.path.dirname(output_stats), exist_ok=True)
    
    with rasterio.open(stack_path) as src:
        profile = src.profile.copy()
        h, w = src.height, src.width
        crs = src.crs
        transform = src.transform
        
        # Leer todas las bandas (10, H, W)
        stack_data = src.read()
        
    # Máscara de píxeles válidos (evitar background fuera de ROI o nodata)
    valid_mask = stack_data[0] > 0
    # Reordenar a (H * W, 10)
    flat_features = stack_data.reshape(stack_data.shape[0], -1).T
    flat_valid = valid_mask.flatten()
    
    print(f"[*] Píxeles totales: {len(flat_features)} | Píxeles válidos a clasificar: {np.sum(flat_valid)}")
    
    # Predecir solo sobre píxeles válidos para máxima eficiencia
    valid_features = flat_features[flat_valid]
    print(f"[*] Ejecutando inferencia por lotes...")
    predictions_str = model.predict(valid_features)
    
    # Convertir nombres de clase a IDs numéricos (1 a 5)
    predictions_int = np.array([class_name_to_id.get(p, 0) for p in predictions_str], dtype=np.uint8)
    
    # Reconstruir ráster completo
    out_flat = np.zeros(h * w, dtype=np.uint8)
    out_flat[flat_valid] = predictions_int
    out_raster_data = out_flat.reshape(h, w)
    
    # Configurar perfil de salida para GeoTIFF categórico de 1 sola banda
    profile.update({
        "count": 1,
        "dtype": rasterio.uint8,
        "nodata": 0,
        "compress": "deflate"
    })
    
    with rasterio.open(output_raster, "w", **profile) as dst:
        dst.write(out_raster_data, 1)
        dst.set_band_description(1, "Clasificación de Cobertura de la Tierra")
        
    print(f"[✓] Ráster temático guardado exitosamente en: {output_raster}")
    
    # Cuantificación de Áreas (Hectáreas y km²)
    # Resolución de píxel en EPSG:9377 es 10m x 10m = 100 m² por píxel
    pixel_area_m2 = abs(transform[0] * transform[4])
    area_stats = {}
    total_valid_pixels = np.sum(flat_valid)
    
    print("\n--- CUANTIFICACIÓN ESPACIAL DE ÁREAS ---")
    for cls in classes_cfg:
        cid = cls["id"]
        cname = cls["name"]
        px_count = int(np.sum(out_raster_data == cid))
        area_ha = round((px_count * pixel_area_m2) / 10000.0, 2)
        area_km2 = round((px_count * pixel_area_m2) / 1000000.0, 4)
        pct = round((px_count / total_valid_pixels) * 100, 2) if total_valid_pixels > 0 else 0
        
        area_stats[cname] = {
            "class_id": cid,
            "pixel_count": px_count,
            "area_ha": area_ha,
            "area_km2": area_km2,
            "percentage": pct,
            "color": cls.get("color", "#cccccc")
        }
        print(f"  {cname:22s} (ID {cid}): {px_count:7d} px | {area_ha:8.2f} ha ({area_km2:6.3f} km²) | {pct:5.2f}%")
        
    with open(output_stats, "w", encoding="utf-8") as f:
        json.dump(area_stats, f, indent=2)
        
    print(f"[✓] Estadísticas de área exportadas a: {output_stats}")
    return True

if __name__ == "__main__":
    predict_full_scene()
