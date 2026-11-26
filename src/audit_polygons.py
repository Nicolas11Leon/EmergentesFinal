"""Auditoria por Polygon_ID: por que algunos poligonos de ground_truth.gpkg no aportan pixeles.

Replica la regla de src/02_extract_tsv.py (centro del pixel dentro del poligono, reintento
con all_touched=True, B02 distinto de NoData, bandas finitas y en [0, 1]) y, para cada
poligono, cuenta los pixeles en cada etapa. Ademas cruza con mask_valid.tif para informar
cuantos pixeles de entrenamiento caen fuera de la mascara de calidad.

No modifica datos. Requiere el stack y la mascara del ZIP oficial en data/processed/.

Uso (desde la raiz del repositorio):
    python src/audit_polygons.py
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import raster_geometry_mask
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]


def window_values(src, geom, all_touched: bool, fill=None):
    """Valores crudos de la ventana del poligono y mascara booleana de la geometria."""
    outside, _, window = raster_geometry_mask(src, [geom], crop=True, all_touched=all_touched)
    return src.read(window=window), ~outside


def audit(stack_path: Path, mask_path: Path, gpkg: Path, tsv: Path, out_csv: Path) -> int:
    gdf = gpd.read_file(gpkg)
    tsv_counts = pd.read_csv(tsv, sep="\t", dtype={"Polygon_ID": str})["Polygon_ID"].value_counts()
    rows = []
    with rasterio.open(stack_path) as src, rasterio.open(mask_path) as msk:
        if msk.transform != src.transform or msk.shape != src.shape or msk.crs != src.crs:
            print("[ERROR] Stack y mascara no estan alineados")
            return 2
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)
        nodata = src.nodata
        fill = nodata if nodata is not None else -9999
        scene = box(*src.bounds)
        for _, r in gdf.iterrows():
            pid = str(r["Polygon_ID"])
            rec = {"Polygon_ID": pid, "Class": r["Class"], "area_ha": round(r.geometry.area / 1e4, 4),
                   "en_tsv": int(tsv_counts.get(pid, 0))}
            if not r.geometry.intersects(scene):
                rec.update(causa="fuera_de_escena", px_geometria=0, px_b02_valido=0, px_extraibles=0,
                           px_en_mascara=0, px_10_bandas_sin_mascara=0, px_validos_10_bandas=0)
                rows.append(rec)
                continue
            all_touched = False
            data, inside = window_values(src, r.geometry, False, fill)
            b02_ok = inside & (data[0] != fill)
            if not b02_ok.any():
                all_touched = True
                data, inside = window_values(src, r.geometry, True, fill)
                b02_ok = inside & (data[0] != fill)
            mdata, _ = window_values(msk, r.geometry, all_touched)
            finite = np.all(np.isfinite(data), axis=0)
            in_range = finite & np.all((data >= 0) & (data <= 1), axis=0)
            extract = b02_ok & in_range
            positive10 = b02_ok & finite & np.all(data > 0, axis=0)
            if nodata is not None and np.isfinite(nodata):
                positive10 &= np.all(data != nodata, axis=0)
            in_mask = mdata[0] > 0
            rec.update(px_geometria=int(inside.sum()), px_b02_valido=int(b02_ok.sum()),
                       px_extraibles=int(extract.sum()), px_en_mascara=int((extract & in_mask).sum()),
                       px_10_bandas_sin_mascara=int(positive10.sum()),
                       px_validos_10_bandas=int((positive10 & in_mask).sum()), all_touched=all_touched)
            if rec["px_geometria"] == 0:
                causa = "sin_pixeles_en_geometria"
            elif rec["px_b02_valido"] == 0:
                causa = "nodata_en_B02"
            elif rec["px_extraibles"] == 0:
                causa = "bandas_fuera_de_rango_o_no_finitas"
            elif rec["px_validos_10_bandas"] > 0:
                causa = "con_pixeles"
            elif rec["px_en_mascara"] == 0:
                causa = "todo_fuera_de_mascara_calidad"
            else:
                causa = "nodata_en_alguna_banda_dentro_de_mascara"
            rec["causa"] = causa
            rows.append(rec)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["Polygon_ID", "Class", "area_ha", "causa", "all_touched", "px_geometria", "px_b02_valido",
              "px_extraibles", "en_tsv", "px_en_mascara", "px_10_bandas_sin_mascara", "px_validos_10_bandas"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    df = pd.DataFrame(rows)
    print(f"[OK] Poligonos auditados: {len(df)}; en el TSV: {int((df['en_tsv'] > 0).sum())}")
    print("[OK] Causa (criterio de inferencia: 10 bandas > 0 y mascara):", df["causa"].value_counts().to_dict())
    fuera = df[df["en_tsv"] == 0]
    print(f"[OK] Poligonos fuera del TSV: {len(fuera)} -> causas {fuera['causa'].value_counts().to_dict()}")
    print("     por clase:", fuera.groupby("Class").size().to_dict())
    print("[OK] Coincidencia por poligono con el TSV segun regla:")
    best = None
    for col in ("px_extraibles", "px_en_mascara", "px_10_bandas_sin_mascara", "px_validos_10_bandas"):
        n = int((df[col] == df["en_tsv"]).sum())
        print(f"     {col:26s} {n}/{len(df)} poligonos | total {int(df[col].sum())} vs TSV {int(df['en_tsv'].sum())}")
        best = max(best or (0, col), (n, col))
    replica = best[0] == len(df)
    col = best[1]
    diff = df[df[col] != df["en_tsv"]]
    if len(diff):
        print(f"     Diferencias con la mejor regla ({col}): {len(diff)} poligonos")
        print(diff[["Polygon_ID", "Class", "en_tsv", "px_validos_10_bandas", "px_10_bandas_sin_mascara",
                    "px_extraibles"]].to_string(index=False))
    print(f"[OK] Detalle: {out_csv}")
    return 0 if replica else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stack", type=Path, default=ROOT / "data/processed/stack_10bands.tif")
    p.add_argument("--mask", type=Path, default=ROOT / "data/processed/mask_valid.tif")
    p.add_argument("--gpkg", type=Path, default=ROOT / "data/vectors/ground_truth.gpkg")
    p.add_argument("--tsv", type=Path, default=ROOT / "data/training/dataset.tsv")
    p.add_argument("--out", type=Path, default=ROOT / "outputs/auditorias/auditoria_poligonos.csv")
    a = p.parse_args()
    for path in (a.stack, a.mask):
        if not path.is_file():
            print(f"[ERROR] Falta {path}. Extraer el ZIP oficial en data/processed/")
            return 2
    return audit(a.stack, a.mask, a.gpkg, a.tsv, a.out)


if __name__ == "__main__":
    sys.exit(main())
