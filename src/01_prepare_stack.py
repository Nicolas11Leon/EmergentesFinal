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

"""
Módulo 01: Preparación y alineación del stack multiespectral de 10 bandas.
Lee las bandas crudas de Sentinel-2 (10m y 20m), las reproyecta y remuestrea a la
grilla común de referencia de 10m en EPSG:9377 (681 x 1054) y genera el archivo multibanda procesado.
"""

import os
import sys
import yaml
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, calculate_default_transform

def load_config(config_path: str = "config/project.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def find_band_file(raw_dir: str, band_name: str) -> str | None:
    """Busca de forma flexible el archivo correspondiente a una banda."""
    if not os.path.isdir(raw_dir):
        return None
    for fname in sorted(os.listdir(raw_dir)):
        lower = fname.lower()
        # Coincidencia exacta con sufijos comunes como B02, _b02, t2_b02, etc.
        if (f"_{band_name.lower()}." in lower or 
            f"_{band_name.lower()}_" in lower or
            lower.startswith(f"{band_name.lower()}.") or
            lower.startswith(f"t2_{band_name.lower()}.") or
            lower.startswith(f"t1_{band_name.lower()}.")):
            return os.path.join(raw_dir, fname)
        if band_name.lower() in lower and lower.endswith((".tif", ".tiff", ".jp2")):
            return os.path.join(raw_dir, fname)
    return None

def prepare_stack(config_path: str = "config/project.yaml") -> bool:
    cfg = load_config(config_path)
    raw_dir = cfg["paths"]["raw_rasters"]
    output_stack = cfg["paths"]["processed_stack"]
    target_crs = cfg["project"]["crs_target"]
    band_order = cfg["spectral_bands"]["order"]
    
    os.makedirs(os.path.dirname(output_stack), exist_ok=True)
    print(f"[*] Iniciando preparación del stack multiespectral ({len(band_order)} bandas)...")
    print(f"    Directorio de entrada: {raw_dir}")
    print(f"    Bandas esperadas: {band_order}")
    print(f"    CRS objetivo: {target_crs}")
    
    # 1. Localizar archivos de bandas
    band_files = {}
    missing = []
    for b in band_order:
        bf = find_band_file(raw_dir, b)
        if bf:
            band_files[b] = bf
        else:
            missing.append(b)
            
    if missing:
        print(f"[ERROR] No se encontraron las siguientes bandas en '{raw_dir}': {missing}")
        print("  Asegúrate de colocar las bandas en data/raw/ antes de ejecutar este paso.")
        return False
        
    print(f"[✓] Todas las {len(band_order)} bandas localizadas correctamente:")
    for b in band_order:
        print(f"    {b:4s} -> {os.path.basename(band_files[b])}")
        
    # 2. Determinar la grilla de referencia usando una banda de 10m (ej. B04)
    ref_band = band_files.get("B04") or band_files[band_order[0]]
    with rasterio.open(ref_band) as src_ref:
        ref_crs = src_ref.crs
        ref_transform = src_ref.transform
        ref_width = src_ref.width
        ref_height = src_ref.height
        
    print(f"[*] Grilla de referencia: {ref_width}x{ref_height}, CRS={ref_crs}, Res={ref_transform[0]}x{-ref_transform[4]}")
    
    # 3. Preparar array de salida (10, H, W)
    stacked_data = np.zeros((len(band_order), ref_height, ref_width), dtype=np.float32)
    
    # 4. Reproyectar/Remuestrear cada banda a la grilla de referencia si es necesario
    for idx, b in enumerate(band_order):
        bfile = band_files[b]
        with rasterio.open(bfile) as src:
            if src.crs == ref_crs and src.transform == ref_transform and src.shape == (ref_height, ref_width):
                # Grilla idéntica, lectura directa
                stacked_data[idx] = src.read(1).astype(np.float32)
            else:
                # Reproyección o remuestreo bilineal a la grilla de referencia
                print(f"    Alineando {b} desde {src.shape} (CRS: {src.crs}) a grilla de referencia...")
                reproject(
                    source=rasterio.band(src, 1),
                    destination=stacked_data[idx],
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    resampling=Resampling.bilinear
                )
                
    # 5. Escribir GeoTIFF multibanda
    profile = {
        "driver": "GTiff",
        "height": ref_height,
        "width": ref_width,
        "count": len(band_order),
        "dtype": rasterio.float32,
        "crs": ref_crs,
        "transform": ref_transform,
        "nodata": 0.0,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256
    }
    
    with rasterio.open(output_stack, "w", **profile) as dst:
        for idx, b in enumerate(band_order):
            dst.write(stacked_data[idx], idx + 1)
            dst.set_band_description(idx + 1, b)
            
    print(f"[✓] Stack multiespectral exportado con éxito a: {output_stack}")
    print(f"    Dimensiones: {len(band_order)} bandas x {ref_height} x {ref_width}")
    return True

if __name__ == "__main__":
    success = prepare_stack()
    sys.exit(0 if success else 1)
