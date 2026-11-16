"""
Módulo de Consolidación de Polígonos de Verdad de Terreno (ground_truth.gpkg).
Integra las 181 cicatrices reales de deslizamientos (Clase 3: Suelo Desnudo y Roca)
con polígonos representativos e independientes para Bosque (1), Pastos y Cultivos (2),
Urbano y Vías (4) y Agua (5).
"""

import os
import sys
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape, Polygon, MultiPolygon
from shapely.ops import unary_union
from scipy.ndimage import label, binary_opening

def create_ground_truth(output_path: str = "data/vectors/ground_truth.gpkg"):
    print("[*] Iniciando consolidación de verdad de terreno (ground_truth.gpkg)...")
    
    # 1. Cargar deslizamientos reales (Clase 3)
    landslides_path = "data/vectors/landslides.gpkg"
    ls_gdf = gpd.read_file(landslides_path)
    crs = ls_gdf.crs
    print(f"[✓] Deslizamientos cargados: {len(ls_gdf)} polígonos.")
    
    features = []
    
    # Agregar deslizamientos como Clase 3
    for idx, row in ls_gdf.iterrows():
        geom = row.geometry
        features.append({
            "Polygon_ID": f"poly_sue_{idx+1:03d}",
            "Class_ID": 3,
            "Class": "Suelo Desnudo y Roca",
            "area_ha": round(float(row.get("area_ha", geom.area / 10000.0)), 4),
            "geometry": geom
        })
        
    # Crear unión espacial de deslizamientos para evitar traslapes
    ls_union = unary_union(ls_gdf.geometry)
    
    # 2. Cargar bandas Sentinel-2 de referencia para muestreo de las otras 4 clases
    with rasterio.open("data/raw/t2_B02.tif") as f2, \
         rasterio.open("data/raw/t2_B03.tif") as f3, \
         rasterio.open("data/raw/t2_B04.tif") as f4, \
         rasterio.open("data/raw/t2_B08.tif") as f8, \
         rasterio.open("data/raw/t2_B11.tif") as f11:
        b2 = f2.read(1)
        b3 = f3.read(1)
        b4 = f4.read(1)
        b8 = f8.read(1)
        b11 = f11.read(1)
        transform = f2.transform
        
    valid = b8 > 0
    ndvi = np.where((b8 + b4) > 0, (b8 - b4) / (b8 + b4 + 1e-6), 0)
    ratio = np.where(valid, b8 / (b3 + 1e-4), 999)
    
    def mask_to_polygons(binary_mask, min_pixels=25, max_polygons=25):
        """Convierte una máscara binaria en polígonos vectoriales filtrados por tamaño."""
        clean_mask = binary_opening(binary_mask, structure=np.ones((2, 2)))
        lbl, n_feats = label(clean_mask)
        sizes = [np.sum(lbl == i) for i in range(1, n_feats + 1)]
        
        # Ordenar componentes por tamaño descendente
        sorted_indices = np.argsort(sizes)[::-1]
        
        extracted_polys = []
        for rank, s_idx in enumerate(sorted_indices):
            comp_idx = s_idx + 1
            pix_count = sizes[s_idx]
            if pix_count < min_pixels:
                continue
            comp_mask = (lbl == comp_idx).astype(np.int16)
            for geom_dict, val in shapes(comp_mask, mask=(comp_mask == 1), transform=transform):
                s = shape(geom_dict)
                if s.is_valid and not s.is_empty and s.area >= min_pixels * 100:
                    # Descartar traslapes con deslizamientos
                    if not s.intersects(ls_union):
                        extracted_polys.append(s)
            if len(extracted_polys) >= max_polygons:
                break
        return extracted_polys

    # --- CLASE 5: AGUA (Río Ingará y tributarios) ---
    water_mask = valid & (ratio < 2.0) & (b8 < 0.18) & (b11 < 0.18)
    water_polys = mask_to_polygons(water_mask, min_pixels=10, max_polygons=12)
    print(f"[✓] Polígonos de Agua generados: {len(water_polys)}")
    for idx, poly in enumerate(water_polys):
        features.append({
            "Polygon_ID": f"poly_agu_{idx+1:03d}",
            "Class_ID": 5,
            "Class": "Agua",
            "area_ha": round(poly.area / 10000.0, 4),
            "geometry": poly
        })

    # --- CLASE 4: URBANO Y VÍAS (Casco urbano San José del Palmar y corredores viales) ---
    urban_mask = valid & (b11 > 0.11) & (b2 > 0.035) & (ndvi < 0.48) & (b8 < 0.28)
    urban_polys = mask_to_polygons(urban_mask, min_pixels=15, max_polygons=12)
    print(f"[✓] Polígonos de Urbano generados: {len(urban_polys)}")
    for idx, poly in enumerate(urban_polys):
        features.append({
            "Polygon_ID": f"poly_urb_{idx+1:03d}",
            "Class_ID": 4,
            "Class": "Urbano y Vías",
            "area_ha": round(poly.area / 10000.0, 4),
            "geometry": poly
        })

    # --- CLASE 2: PASTOS Y CULTIVOS ---
    pasture_mask = valid & (ndvi >= 0.42) & (ndvi <= 0.68) & (b8 > 0.16) & (b8 < 0.32) & (b4 > 0.045)
    pasture_polys = mask_to_polygons(pasture_mask, min_pixels=20, max_polygons=15)
    print(f"[✓] Polígonos de Pastos generados: {len(pasture_polys)}")
    for idx, poly in enumerate(pasture_polys):
        features.append({
            "Polygon_ID": f"poly_pas_{idx+1:03d}",
            "Class_ID": 2,
            "Class": "Pastos y Cultivos",
            "area_ha": round(poly.area / 10000.0, 4),
            "geometry": poly
        })

    # --- CLASE 1: BOSQUE ---
    forest_mask = valid & (ndvi > 0.86) & (b8 > 0.38) & (b4 < 0.028)
    forest_polys = mask_to_polygons(forest_mask, min_pixels=40, max_polygons=20)
    print(f"[✓] Polígonos de Bosque generados: {len(forest_polys)}")
    for idx, poly in enumerate(forest_polys):
        features.append({
            "Polygon_ID": f"poly_bos_{idx+1:03d}",
            "Class_ID": 1,
            "Class": "Bosque",
            "area_ha": round(poly.area / 10000.0, 4),
            "geometry": poly
        })

    # 3. Consolidar GeoDataFrame y guardar en GPKG
    out_gdf = gpd.GeoDataFrame(features, crs=crs)
    out_gdf.to_file(output_path, layer="ground_truth", driver="GPKG")
    print(f"\n[✓] Capa consolidada exitosamente en: {output_path}")
    print(f"    Total polígonos: {len(out_gdf)}")
    print("    Distribución por clase:")
    for cls_name, grp in out_gdf.groupby("Class"):
        print(f"      - {cls_name:22s} (ID {grp['Class_ID'].iloc[0]}): {len(grp):3d} polígonos, {grp['area_ha'].sum():8.2f} ha")

if __name__ == "__main__":
    create_ground_truth()
