"""Inferencia geoespacial verificable con mascara oficial (sin cambiar el script de Nicolas).

Ejecutar desde la raiz del repo:
    python src/predict_raster_hania.py --check-inputs
    python src/predict_raster_hania.py --model ann
    python src/predict_raster_hania.py --model all
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import rasterio
import yaml

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("logistic_regression", "decision_tree", "svm", "knn", "ann")


def load_valid_pixels(stack_file: Path, mask_file: Path, cfg: dict):
    if not stack_file.is_file() or not mask_file.is_file():
        raise FileNotFoundError("Faltan stack_10bands.tif o mask_valid.tif. Descargar ZIP oficial del Drive")
    with rasterio.open(stack_file) as src, rasterio.open(mask_file) as mask:
        expected_bands = cfg["spectral_bands"]["order"]
        expected_crs = cfg["project"]["crs_target"]
        expected_resolution = float(cfg["project"]["resolution_m"])
        if src.count != len(expected_bands) or src.count != 10:
            raise ValueError(f"El stack debe contener 10 bandas; tiene {src.count}")
        if str(src.crs) != expected_crs:
            raise ValueError(f"CRS incorrecto: {src.crs}, esperado {expected_crs}")
        if not np.isclose(abs(src.res[0]), expected_resolution) or not np.isclose(abs(src.res[1]), expected_resolution):
            raise ValueError(f"Resolucion incorrecta: {src.res}")
        if src.width != 1054 or src.height != 681:
            raise ValueError(f"Dimension distinta a la escena entregada: {src.width} x {src.height}")
        if mask.count != 1 or mask.width != src.width or mask.height != src.height:
            raise ValueError("La mascara debe tener una sola banda y mismas dimensiones")
        if mask.crs != src.crs or not mask.transform.almost_equals(src.transform):
            raise ValueError("Mascara no comparte CRS o transformacion con el stack")
        descriptions = src.descriptions
        if any(descriptions):
            if list(descriptions) != expected_bands:
                raise ValueError(f"Orden de bandas informado por el TIFF no coincide: {descriptions}")
        else:
            print("[AVISO] El GeoTIFF no tiene descripciones de bandas; confirmar orden con la documentacion de Nicolas")
        cube = src.read()
        official_mask = mask.read(1) > 0
        valid = official_mask & np.all(np.isfinite(cube), axis=0)
        if src.nodata is not None and np.isfinite(src.nodata):
            valid &= np.all(cube != src.nodata, axis=0)
        profile = src.profile.copy()
        transform = src.transform
        if not np.any(valid):
            raise ValueError("No hay pixeles validos tras aplicar mascara y NoData")
        print(f"[OK] Raster: {src.width}x{src.height}, {src.count} bandas, {src.crs}, {src.res}")
        print(f"[OK] Pixeles validos oficiales: {int(official_mask.sum())}; finalmente clasificables: {int(valid.sum())}")
        return cube, valid, profile, transform


def infer_one(name: str, cube: np.ndarray, valid: np.ndarray, profile: dict,
              transform, classes_cfg: list, root: Path, batch_size: int = 25000):
    if name not in MODELS:
        raise ValueError(f"Modelo desconocido: {name}")
    model_file = root / f"models/{name}.joblib"
    if not model_file.is_file():
        raise FileNotFoundError(f"Falta modelo: {model_file}")
    model = joblib.load(model_file)  # Solamente archivos joblib generados por nuestro equipo
    class_to_id = {item["name"]: item["id"] for item in classes_cfg}
    valid_positions = np.flatnonzero(valid.ravel())
    features = cube.reshape(cube.shape[0], -1).T
    result = np.zeros(valid.size, dtype=np.uint8)
    for offset in range(0, len(valid_positions), batch_size):
        positions = valid_positions[offset:offset + batch_size]
        predicted_names = model.predict(features[positions])
        unseen = set(predicted_names) - set(class_to_id)
        if unseen:
            raise ValueError(f"El modelo {name} predijo clases desconocidas: {sorted(unseen)}")
        result[positions] = np.array([class_to_id[v] for v in predicted_names], dtype=np.uint8)
    result = result.reshape(valid.shape)
    if np.any(result[~valid] != 0) or np.any(result[valid] == 0):
        raise RuntimeError("Inconsistencia entre pixeles clasificados y mascara")
    raster_file = root / f"outputs/rasters/landcover_{name}.tif"
    stats_file = root / f"outputs/metrics/inferencia_10bandas/area_stats_{name}.json"
    raster_file.parent.mkdir(parents=True, exist_ok=True)
    stats_file.parent.mkdir(parents=True, exist_ok=True)
    out_profile = profile.copy()
    out_profile.update(count=1, dtype="uint8", nodata=0, compress="deflate")
    with rasterio.open(raster_file, "w", **out_profile) as dst:
        dst.write(result, 1)
        dst.set_band_description(1, "Clasificacion de cobertura terrestre")
    pixel_area = abs(transform.a * transform.e - transform.b * transform.d)
    total = int(valid.sum())
    areas = {}
    for item in classes_cfg:
        cid = item["id"]
        count = int((result == cid).sum())
        areas[item["name"]] = {
            "class_id": cid, "pixel_count": count,
            "area_ha": round(count * pixel_area / 10000, 2),
            "area_km2": round(count * pixel_area / 1_000_000, 4),
            "percentage": round(100 * count / total, 2),
            "color": item.get("color"),
        }
    if sum(row["pixel_count"] for row in areas.values()) != total:
        raise RuntimeError("Las areas no suman el total valido")
    stats_file.write_text(json.dumps(areas, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] {name}: {total} px clasificados | {raster_file} | {stats_file}")
    return raster_file, stats_file


def run(root: Path = ROOT, model: str = "all", check_inputs: bool = False):
    cfg = yaml.safe_load((root / "config/project.yaml").read_text(encoding="utf-8"))
    classes_cfg = json.loads((root / "config/classes.json").read_text(encoding="utf-8"))["classes"]
    expected_ids = set(range(1, 6))
    if {x["id"] for x in classes_cfg} != expected_ids:
        raise ValueError("Se esperan exactamente las clases 1..5 de classes.json")
    cube, valid, profile, transform = load_valid_pixels(
        root / cfg["paths"]["processed_stack"], root / cfg["paths"]["quality_mask"], cfg,
    )
    if check_inputs:
        return []
    names = MODELS if model == "all" else (model,)
    return [infer_one(n, cube, valid, profile, transform, classes_cfg, root) for n in names]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=[*MODELS, "all"], default="all")
    parser.add_argument("--check-inputs", action="store_true")
    args = parser.parse_args()
    run(model=args.model, check_inputs=args.check_inputs)


if __name__ == "__main__":
    main()
