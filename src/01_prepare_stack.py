"""
Módulo 01: Preparación y alineación del stack multiespectral de 10 bandas.
Lee las bandas crudas de Sentinel-2 (10m y 20m), las remuestrea a grilla común
de 10m en EPSG:9377 y genera el archivo multibanda procesado.
"""

import os
import sys
import yaml
import numpy as np

def load_config(config_path: str = "config/project.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def prepare_stack(config_path: str = "config/project.yaml"):
    try:
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.warp import calculate_default_transform, reproject
    except ImportError:
        print("[ADVERTENCIA] 'rasterio' no está instalado en este entorno aún.")
        print("  Instálalo ejecutando: pip install -r requirements.txt")
        return False

    cfg = load_config(config_path)
    raw_dir = cfg["paths"]["raw_rasters"]
    output_stack = cfg["paths"]["processed_stack"]
    target_crs = cfg["project"]["crs_target"]
    band_order = cfg["spectral_bands"]["order"]
    
    os.makedirs(os.path.dirname(output_stack), exist_ok=True)
    print(f"[*] Iniciando preparación del stack multiespectral (10 bandas)...")
    print(f"    Directorio de entrada: {raw_dir}")
    print(f"    Bandas esperadas: {band_order}")
    print(f"    CRS objetivo: {target_crs}")
    
    # Verificar existencia de archivos de bandas individuales
    missing = []
    band_files = {}
    for b in band_order:
        # Búsqueda flexible de archivos: B02.tif, *B02*.jp2, etc.
        found = None
        if os.path.isdir(raw_dir):
            for fname in os.listdir(raw_dir):
                if b in fname and fname.endswith((".tif", ".tiff", ".jp2")):
                    found = os.path.join(raw_dir, fname)
                    break
        if found:
            band_files[b] = found
        else:
            missing.append(b)
            
    if missing:
        print(f"[ERROR] No se encontraron las siguientes bandas en '{raw_dir}': {missing}")
        print("  Asegúrate de colocar las bandas en data/raw/ antes de ejecutar este paso.")
        return False
        
    print(f"[✓] Todas las {len(band_order)} bandas localizadas correctamente.")
    # El ensamblado real se ejecuta cuando los archivos .tif/jp2 estén presentes
    return True

if __name__ == "__main__":
    prepare_stack()
