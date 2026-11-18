#!/usr/bin/env python3
"""Auditoría estricta y reproducible de cinco GeoTIFF de cobertura.

No entrena modelos ni modifica las imágenes. Requiere únicamente numpy y rasterio.

Ejemplos:
  python audit_rasters_final.py --zip Emergentes_Rasters_Inferencia.zip --source-only
  python audit_rasters_final.py --zip Emergentes_Rasters_Inferencia.zip \
      --rasters-dir outputs/rasters --stats-dir outputs/metrics/inferencia_10bandas
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import rasterio

MODELS = (
    "logistic_regression", "decision_tree", "svm", "knn", "ann"
)
BANDS = ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12")
IDS = (1, 2, 3, 4, 5)
EXPECTED = 342_525


class AuditError(ValueError):
    """Un artefacto incumple el contrato de inferencia."""


def need(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_official_zip(source_zip: Path, dest: Path) -> tuple[Path, Path]:
    source_zip = source_zip.resolve()
    need(source_zip.is_file(), f"No existe el ZIP oficial: {source_zip}")
    paths = {}
    with zipfile.ZipFile(source_zip) as zf:
        for filename in ("stack_10bands.tif", "mask_valid.tif"):
            candidates = [x for x in zf.namelist() if Path(x).name == filename and not x.endswith("/")]
            need(len(candidates) == 1, f"El ZIP debe contener un único {filename} (se encontraron {len(candidates)})")
            target = dest / filename
            with zf.open(candidates[0]) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            paths[filename] = target
    return paths["stack_10bands.tif"], paths["mask_valid.tif"]


def inspect_source(stack_path: Path, mask_path: Path, expected: int = EXPECTED) -> dict:
    """Devuelve georreferenciación y matriz booleana de píxeles clasficables."""
    with rasterio.open(stack_path) as stack, rasterio.open(mask_path) as mask:
        need(stack.count == len(BANDS), f"El stack tiene {stack.count} bandas, no 10")
        need(tuple(stack.descriptions) == BANDS, f"Orden de bandas no oficial: {stack.descriptions!r}")
        need(str(stack.crs) == "EPSG:9377", f"CRS inesperado: {stack.crs}")
        need(tuple(round(v, 8) for v in stack.res) == (10.0, 10.0), f"Resolución inesperada: {stack.res}")
        need(mask.count == 1, "La máscara debe tener una sola banda")
        need((stack.width, stack.height) == (mask.width, mask.height), "Dimensiones stack/máscara diferentes")
        need(stack.crs == mask.crs and stack.transform == mask.transform, "Stack y máscara no están alineados")
        data = stack.read()
        quality = mask.read(1) > 0
        nodata = stack.nodata
        finite_positive = np.isfinite(data) & (data > 0)
        if nodata is not None and np.isfinite(nodata):
            finite_positive &= data != nodata
        valid = quality & np.all(finite_positive, axis=0)
        n_quality = int(quality.sum())
        n_valid = int(valid.sum())
        need(n_valid == expected, f"Hay {n_valid} píxeles válidos; se esperaban {expected}")
        pixel_m2 = abs(stack.transform.a * stack.transform.e - stack.transform.b * stack.transform.d)
        need(abs(pixel_m2 - 100.0) < 1e-8, f"Área/píxel inesperada: {pixel_m2}")
        return {
            "valid": valid,
            "width": stack.width,
            "height": stack.height,
            "crs": stack.crs,
            "transform": stack.transform,
            "quality_pixels": n_quality,
            "valid_pixels": n_valid,
            "area_ha": n_valid * pixel_m2 / 10000,
            "pixel_m2": pixel_m2,
            "nodata": nodata,
        }


def find_one(directory: Path, filename: str) -> Path:
    need(directory.is_dir(), f"No existe la carpeta: {directory}")
    matches = [p for p in directory.rglob(filename) if p.is_file()]
    need(len(matches) == 1,
         f"Se esperaba un único {filename} debajo de {directory}; encontrados: {len(matches)}")
    return matches[0]


def check_raster(raster_path: Path, source: dict) -> dict[int, int]:
    with rasterio.open(raster_path) as raster:
        need(raster.count == 1, f"{raster_path.name}: no es un GeoTIFF de una banda")
        need((raster.width, raster.height) == (source["width"], source["height"]),
             f"{raster_path.name}: dimensiones incorrectas")
        need(raster.crs == source["crs"] and raster.transform == source["transform"],
             f"{raster_path.name}: CRS o transformación diferente al stack")
        need(raster.nodata == 0, f"{raster_path.name}: NoData debe ser 0")
        need(np.issubdtype(np.dtype(raster.dtypes[0]), np.integer),
             f"{raster_path.name}: el ráster categórico debe contener enteros")
        classified = raster.read(1)
    valid = source["valid"]
    need(np.all(classified[~valid] == 0),
         f"{raster_path.name}: hay predicciones fuera de la máscara válida")
    need(np.all(np.isin(classified[valid], IDS)),
         f"{raster_path.name}: códigos no válidos o NoData dentro de zona clasificable")
    return {i: int((classified == i).sum()) for i in IDS}


def check_stats(stats_path: Path, counts: dict[int, int], source: dict) -> dict:
    data = json.loads(stats_path.read_text(encoding="utf-8-sig"))
    need(isinstance(data, dict), f"{stats_path.name}: formato JSON incorrecto")
    by_id = {}
    for class_name, record in data.items():
        need(isinstance(record, dict) and "class_id" in record,
             f"{stats_path.name}: falta class_id en {class_name}")
        cid = int(record["class_id"])
        need(cid in IDS and cid not in by_id, f"{stats_path.name}: clase duplicada o desconocida {cid}")
        by_id[cid] = record
    need(set(by_id) == set(IDS), f"{stats_path.name}: no contiene las cinco clases oficiales")
    pixel_m2 = source["pixel_m2"]
    total = source["valid_pixels"]
    for cid in IDS:
        record = by_id[cid]
        observed = counts[cid]
        need(int(record["pixel_count"]) == observed,
             f"{stats_path.name}: clase {cid}, JSON={record['pixel_count']} vs GeoTIFF={observed}")
        area = observed * pixel_m2 / 10000
        need(abs(float(record["area_ha"]) - area) <= 0.00501,
             f"{stats_path.name}: hectáreas incorrectas en clase {cid}")
        need(abs(float(record["area_km2"]) - area / 100) <= 0.000051,
             f"{stats_path.name}: km² incorrectos en clase {cid}")
        need(abs(float(record["percentage"]) - 100 * observed / total) <= 0.00501,
             f"{stats_path.name}: porcentaje incorrecto en clase {cid}")
    need(sum(counts.values()) == total,
         f"{stats_path.name}: la suma por clase no coincide con la superficie válida")
    return {"pixels": total, "ha": source["area_ha"], "per_class_pixels": counts}


def audit(source_zip: Path, raster_dir: Path | None, stats_dir: Path | None,
          source_only: bool = False, expected: int = EXPECTED) -> dict:
    with tempfile.TemporaryDirectory(prefix="auditoria_geo_") as folder:
        stack, mask = copy_official_zip(source_zip, Path(folder))
        info = inspect_source(stack, mask, expected)
        print("[OK] ZIP oficial: bandas, CRS, resolución y alineación")
        print(f"[OK] Máscara original: {info['quality_pixels']:,} píxeles")
        print(f"[OK] Máscara final (diez bandas): {info['valid_pixels']:,} píxeles = {info['area_ha']:.2f} ha")
        report = {
            "source_zip": str(source_zip.resolve()),
            "zip_sha256": sha256(source_zip),
            "quality_pixels": info["quality_pixels"],
            "valid_pixels": info["valid_pixels"],
            "valid_area_ha": info["area_ha"],
            "method": "mask>0 AND finite(all 10 bands) AND all bands>0 AND not NoData",
            "models": {},
        }
        if source_only:
            print("[OK] Auditoría de fuentes terminada. NO se verificaron GeoTIFF finales.")
            return report
        need(raster_dir is not None and stats_dir is not None,
             "Debe indicar ambas carpetas de GeoTIFF y de JSON")
        for model in MODELS:
            tif = find_one(raster_dir, f"landcover_{model}.tif")
            json_path = find_one(stats_dir, f"area_stats_{model}.json")
            counts = check_raster(tif, info)
            row = check_stats(json_path, counts, info)
            report["models"][model] = {
                "raster": str(tif.resolve()), "stats": str(json_path.resolve()),
                "raster_sha256": sha256(tif), "stats_sha256": sha256(json_path),
                **row,
            }
            print(f"[OK] {model}: GeoTIFF, máscara y JSON coinciden ({row['pixels']:,} px)")
        print("[APROBADO] Los CINCO GeoTIFF reales y sus JSON corresponden al mismo dominio válido.")
        return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True, help="ZIP original de Nicolás")
    parser.add_argument("--rasters-dir", type=Path, default=Path("outputs/rasters"))
    parser.add_argument("--stats-dir", type=Path, default=Path("outputs/metrics/inferencia_10bandas"))
    parser.add_argument("--source-only", action="store_true", help="Verificar solo stack y máscara")
    parser.add_argument("--expected-pixels", type=int, default=EXPECTED)
    parser.add_argument("--report", type=Path, help="Opcional: exportar evidencia de auditoría a JSON")
    args = parser.parse_args()
    try:
        result = audit(args.zip, args.rasters_dir, args.stats_dir,
                       source_only=args.source_only, expected=args.expected_pixels)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"[OK] Reporte: {args.report}")
        return 0
    except (AuditError, OSError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
