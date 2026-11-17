"""Pruebas unitarias del consumo de folds espaciales compartidos.

Ejecutar desde la raiz:
    .\\.venv\\Scripts\\python.exe -m unittest discover -s tests -p test_model_classical.py -v
"""

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from model_classical import build_official_splits  # noqa: E402


class OfficialPartitionsTests(unittest.TestCase):
    def setUp(self):
        self.classes = ["Agua", "Bosque", "Pastos y Cultivos", "Suelo Desnudo y Roca", "Urbano y Vías"]
        self.development = []
        self.test = []
        rows = []
        for cls in self.classes:
            for group_idx in range(6):
                polygon = f"{cls}-{group_idx}"
                (self.development if group_idx < 4 else self.test).append(polygon)
                for _ in range(3):
                    rows.append({"Polygon_ID": polygon, "Class": cls, "Class_ID": self.classes.index(cls) + 1})
        self.df = pd.DataFrame(rows)
        self.part = {
            "dev_polygons": self.development,
            "test_polygons": self.test,
            "total_polygons": len(self.development) + len(self.test),
            "n_cv_splits": 4,
            "cv_folds": {
                f"fold_{i}": {
                    "train_polygons": [p for p in self.development if not p.endswith(f"-{i}")],
                    "val_polygons": [p for p in self.development if p.endswith(f"-{i}")],
                }
                for i in range(4)
            },
        }

    def test_valid_splits_cover_each_group_once(self):
        dev, test, classes, splits = build_official_splits(self.df, self.part, 4)
        self.assertEqual(classes, sorted(self.classes))
        self.assertEqual(len(splits), 4)
        self.assertEqual(len(dev), 60)
        self.assertEqual(len(test), 30)
        validation_groups = []
        for train_idx, val_idx in splits:
            a = set(dev.iloc[train_idx]["Polygon_ID"])
            b = set(dev.iloc[val_idx]["Polygon_ID"])
            self.assertFalse(a & b)
            self.assertEqual(a | b, set(self.development))
            validation_groups.extend(b)
        self.assertCountEqual(validation_groups, self.development)

    def test_overlap_between_dev_and_test_rejected(self):
        self.part["test_polygons"].append(self.development[0])
        with self.assertRaisesRegex(ValueError, "fugas"):
            build_official_splits(self.df, self.part, 4)

    def test_missing_polygon_rejected(self):
        self.part["dev_polygons"].pop()
        with self.assertRaisesRegex(ValueError, "cobertura"):
            build_official_splits(self.df, self.part, 4)

    def test_duplicate_validation_group_rejected(self):
        # Mover un poligono de validacion a otra validacion sin alterar cobertura del fold.
        left = self.part["cv_folds"]["fold_0"]
        right = self.part["cv_folds"]["fold_1"]
        polygon = left["val_polygons"][0]
        right["val_polygons"].append(polygon)
        right["train_polygons"].remove(polygon)
        with self.assertRaisesRegex(ValueError, "una sola vez"):
            build_official_splits(self.df, self.part, 4)

    def test_missing_class_in_validation_rejected(self):
        first = self.part["cv_folds"]["fold_0"]
        first["train_polygons"].extend(first["val_polygons"][:1])
        first["val_polygons"] = first["val_polygons"][1:]
        with self.assertRaisesRegex(ValueError, "Faltan clases"):
            build_official_splits(self.df, self.part, 4)


if __name__ == "__main__":
    unittest.main()
