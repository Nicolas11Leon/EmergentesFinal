"""Verificaciones basicas de la implementacion ANN (no entrenan los 24 fits)."""
import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
from model_ann import load_ann_grid, make_ann_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


class AnnContractTests(unittest.TestCase):
    def test_pipeline_contains_scaler_and_mlp(self):
        pipe = make_ann_pipeline(42)
        self.assertIsInstance(pipe.named_steps["scaler"], StandardScaler)
        self.assertIsInstance(pipe.named_steps["clf"], MLPClassifier)
        self.assertEqual(pipe.named_steps["clf"].random_state, 42)

    def test_grid_keeps_six_configured_combinations(self):
        cfg = {"models": {"ann": {"hpo_grid": {
            "clf__hidden_layer_sizes": [[64], [128, 64], [64, 32]],
            "clf__alpha": [0.0001, 0.001],
            "clf__max_iter": [500],
        }}}}
        grid = load_ann_grid(cfg)
        self.assertEqual(len(grid["clf__hidden_layer_sizes"]) *
                         len(grid["clf__alpha"]) *
                         len(grid["clf__max_iter"]), 6)


if __name__ == "__main__":
    unittest.main()
