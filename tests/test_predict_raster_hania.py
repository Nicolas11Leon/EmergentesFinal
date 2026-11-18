"""Comprueba mascara, georreferenciacion, NoData y estadisticas de area."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import rasterio
import yaml
from rasterio.transform import from_origin
from sklearn.dummy import DummyClassifier

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
from predict_raster_hania import run


class RasterInferenceTests(unittest.TestCase):
    def test_mask_excludes_invalid_pixels_and_preserves_georef(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for folder in ("config", "models", "data/processed"):
                (root / folder).mkdir(parents=True, exist_ok=True)
            cfg = {
                "project": {"crs_target": "EPSG:9377", "resolution_m": 10},
                "spectral_bands": {"order": ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]},
                "paths": {"processed_stack": "data/processed/stack_10bands.tif", "quality_mask": "data/processed/mask_valid.tif"},
            }
            (root / "config/project.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
            (root / "config/classes.json").write_text(json.dumps({"classes": [
                {"id": 1, "name": "Bosque"}, {"id": 2, "name": "Pastos y Cultivos"},
                {"id": 3, "name": "Suelo Desnudo y Roca"}, {"id": 4, "name": "Urbano y Vías"},
                {"id": 5, "name": "Agua"},
            ]}), encoding="utf-8")
            shape = (681, 1054)
            cube = np.full((10, *shape), 10, dtype="float32")
            mask = np.zeros(shape, dtype="uint8")
            mask[0, 0] = 1
            mask[0, 1] = 1
            cube[3, 0, 1] = -9999  # Mask oficial sola no es suficiente: NoData.
            transform = from_origin(1000, 2000, 10, 10)
            with rasterio.open(root / "data/processed/stack_10bands.tif", "w", driver="GTiff",
                               height=shape[0], width=shape[1], count=10,
                               dtype="float32", crs="EPSG:9377", transform=transform,
                               nodata=-9999) as dst:
                dst.write(cube)
                for band, name in enumerate(cfg["spectral_bands"]["order"], 1):
                    dst.set_band_description(band, name)
            with rasterio.open(root / "data/processed/mask_valid.tif", "w", driver="GTiff",
                               height=shape[0], width=shape[1], count=1,
                               dtype="uint8", crs="EPSG:9377", transform=transform) as dst:
                dst.write(mask, 1)
            model = DummyClassifier(strategy="constant", constant="Bosque")
            model.fit(np.ones((2, 10)), ["Bosque", "Bosque"])
            joblib.dump(model, root / "models/ann.joblib")
            results = run(root, model="ann")
            self.assertEqual(len(results), 1)
            raster, stats = results[0]
            with rasterio.open(raster) as src:
                self.assertEqual(src.count, 1)
                self.assertEqual(src.nodata, 0)
                self.assertEqual(str(src.crs), "EPSG:9377")
                self.assertTrue(src.transform.almost_equals(transform))
                data = src.read(1)
                self.assertEqual(np.count_nonzero(data), 1)
                self.assertEqual(int(data[0, 0]), 1)
                self.assertEqual(int(data[0, 1]), 0)
            areas = json.loads(stats.read_text(encoding="utf-8"))
            self.assertEqual(areas["Bosque"]["pixel_count"], 1)
            self.assertEqual(areas["Bosque"]["area_ha"], 0.01)
            self.assertAlmostEqual(sum(v["percentage"] for v in areas.values()), 100)


if __name__ == "__main__":
    unittest.main()
