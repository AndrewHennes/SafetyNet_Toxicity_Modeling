"""Regression tests for table workflows and model-interface correctness."""

# Standalone discovery adds the checkout root before analysis imports.
# ruff: noqa: E402

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from other.evaluate_model_performance import evaluate_predictions
from other.plot_tox_vs_activity import quadrant_counts
from other.tsne_from_representation import load_stack, parse_rep_column
from safetynet.cli._common import predict_probabilities, read_model_table
from safetynet.cli.safetynet_score import aggregate_predictions
from safetynet.featurization.compute_morgan_fingerprints import compute_fingerprints
from safetynet.featurization.minimol_featurization import (
    MinimolFeaturizer,
    featurize_file,
)
from safetynet.featurization.table_utils import (
    representation_matrix,
    resolve_path,
    separator,
)
from safetynet.toxicity.model import MultiTaskFeedForward


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="safetynet_test_")
        self.work = Path(self.directory.name)
        self.addCleanup(self.directory.cleanup)

    def run_script(self, relative, *arguments):
        result = subprocess.run(
            [sys.executable, str(REPOSITORY_ROOT / relative), *map(str, arguments)],
            cwd=self.work,
            capture_output=True,
            text=True,
            env={**os.environ, "MPLBACKEND": "Agg"},
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        return result

    def test_relative_paths_do_not_depend_on_working_directory(self):
        expected = resolve_path("datasets/example.csv")
        previous = Path.cwd()
        try:
            os.chdir(self.work)
            self.assertEqual(resolve_path("datasets/example.csv"), expected)
        finally:
            os.chdir(previous)

    def test_separator_normalizes_escaped_tab(self):
        self.assertEqual(separator("file.tsv"), "\t")
        self.assertEqual(separator("file.csv", r"\t"), "\t")
        with self.assertRaises(ValueError):
            separator("file.csv", "two")

    def test_representations_reject_invalid_rows_and_widths(self):
        for values in [
            ["[1,2]", "[3]"],
            ["[1,2]", "[]"],
            ["bad"],
            ["[[1,2]]"],
            [[1, np.inf]],
        ]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                representation_matrix(pd.Series(values))
        np.testing.assert_array_equal(
            representation_matrix(pd.Series(["[1,2]", [3, 4]])), [[1, 2], [3, 4]]
        )

    def test_input_identifiers_preserve_leading_zeroes(self):
        source = self.work / "features.csv"
        source.write_text('ID,Minimol_Fingerprint\n001,"[1,2]"\n002,"[3,4]"\n')
        frame, matrix = read_model_table(source, "Minimol_Fingerprint")
        self.assertEqual(frame.ID.tolist(), ["001", "002"])
        self.assertEqual(matrix.shape, (2, 2))

    def test_minimol_input_validation_and_batch_order(self):
        featurizer = MinimolFeaturizer()
        featurizer.model = lambda smiles: [
            torch.tensor([len(value), 1.0]) for value in smiles
        ]
        actual = featurizer.embed(["C", "CC", "CCC"], batch_size=2, show_progress=False)
        np.testing.assert_array_equal(actual[:, 0], [1, 2, 3])
        for smiles in [[""], [None], ["*"], ["not_smiles"]]:
            with self.assertRaises(ValueError):
                featurizer.embed(smiles, show_progress=False)
        with self.assertRaises(ValueError):
            featurizer.embed(["C"], batch_size=0)

    def test_minimol_rejects_row_count_mismatch(self):
        featurizer = MinimolFeaturizer()
        featurizer.model = lambda smiles: []
        with self.assertRaisesRegex(ValueError, "number of rows"):
            featurizer.embed(["C"], show_progress=False)

    def test_featurization_writes_tsv_from_csv_and_safe_arrays(self):
        source, table, archive = (
            self.work / name for name in ["input.csv", "output.tsv", "output.npz"]
        )
        source.write_text("SMILES,ID\nC,001\nCC,002\n")
        with patch(
            "safetynet.featurization.minimol_featurization.featurize_smiles",
            return_value=np.ones((2, 3)),
        ):
            featurize_file(source, output_table=table, output_npz=archive)
        self.assertIn("\t", table.read_text().splitlines()[0])
        frame = pd.read_csv(table, sep="\t", dtype=str)
        self.assertEqual(frame.ID.tolist(), ["001", "002"])
        with np.load(archive, allow_pickle=False) as arrays:
            self.assertEqual(arrays["features"].shape, (2, 3))
            self.assertEqual(arrays["smiles"].tolist(), ["C", "CC"])
        with self.assertRaises(ValueError):
            featurize_file(source, output_table=source)

    def test_morgan_cache_and_source_row_mapping(self):
        source, output = self.work / "molecules.csv", self.work / "fingerprints.npz"
        source.write_text("SMILES,ID\nC,001\n*,002\nCC,003\n,004\n")
        compute_fingerprints(source, output, fp_bits=64)
        with np.load(output, allow_pickle=False) as archive:
            self.assertEqual(archive["X"].shape, (2, 64))
            self.assertEqual(archive["row_indices"].tolist(), [0, 2])
            self.assertEqual(archive["names"].tolist(), ["001", "003"])
        compute_fingerprints(source, output, fp_bits=64)
        with self.assertRaises(ValueError):
            compute_fingerprints(source, output, fp_bits=64, use_chirality=False)
        source.write_text(source.read_text() + "CCC,005\n")
        with self.assertRaises(ValueError):
            compute_fingerprints(source, output, fp_bits=64)

    def test_metrics_known_values_and_invalid_labels(self):
        frame = pd.DataFrame({"label": [0, 0, 1, 1], "score": [0.1, 0.4, 0.35, 0.8]})
        metrics, _, _ = evaluate_predictions(frame, "label", "score")
        self.assertAlmostEqual(metrics["roc_auc"], 0.75)
        self.assertAlmostEqual(metrics["average_precision"], 5 / 6)
        for labels in [[1, 1, 1, 1], [0, 1, 2, 0]]:
            with self.assertRaises(ValueError):
                evaluate_predictions(frame.assign(label=labels), "label", "score")
        with self.assertRaises(ValueError):
            evaluate_predictions(frame, "label", "score", pos_label=5)

    def test_metrics_report_nonfinite_rows(self):
        frame = pd.DataFrame(
            {"label": [0, 1, 0, 1], "score": [0.1, 0.9, np.inf, np.nan]}
        )
        metrics, _, _ = evaluate_predictions(frame, "label", "score")
        self.assertEqual(metrics["n_dropped"], 2)
        self.assertEqual(metrics["n_samples"], 2)

    def test_metric_files_with_dotted_prefix(self):
        source, prefix = self.work / "scores.tsv", self.work / "model.v1"
        pd.DataFrame({"label": [0, 0, 1, 1], "score": [0.1, 0.4, 0.35, 0.8]}).to_csv(
            source, sep="\t", index=False
        )
        self.run_script(
            "other/evaluate_model_performance.py",
            "--input",
            source,
            "--sep",
            r"\t",
            "--label-col",
            "label",
            "--score-col",
            "score",
            "--out-prefix",
            prefix,
        )
        self.assertTrue((self.work / "model.v1_roc_prc.png").is_file())
        self.assertAlmostEqual(
            json.loads((self.work / "model.v1_metrics.json").read_text())["roc_auc"],
            0.75,
        )

    def test_plot_threshold_equality_and_outputs(self):
        counts = quadrant_counts(
            np.array([0.5, 0.5, 0.4, 0.4]), np.array([0.2, 0.1, 0.2, 0.1])
        )
        self.assertEqual(set(counts.values()), {1})
        source = self.work / "plot.csv"
        pd.DataFrame({"activity": [0.5, 0.4], "toxicity": [0.2, 0.1]}).to_csv(
            source, index=False
        )
        self.run_script(
            "other/plot_tox_vs_activity.py",
            "--input",
            source,
            "--activity-col",
            "activity",
            "--tox-col",
            "toxicity",
            "--out-prefix",
            self.work / "plot.v1",
        )
        self.assertTrue((self.work / "plot.v1.svg").is_file())

    def test_tsne_uses_minimol_arrays_and_retains_rows(self):
        first, second = self.work / "first.npz", self.work / "second.csv"
        np.savez(first, features=np.array([[1, 2], [3, 4]]))
        matrix, _, _, _, rows = load_stack(self.work, [first.name], "npz")
        self.assertEqual(matrix.shape, (2, 2))
        self.assertEqual(rows.tolist(), [0, 1])
        pd.DataFrame({"rep": ["[1,2]", "bad", "[3,4]"]}).to_csv(second, index=False)
        with self.assertWarns(UserWarning):
            matrix, _, _, _, rows = load_stack(self.work, [second.name], "table", "rep")
        self.assertEqual(rows.tolist(), [0, 2])
        with self.assertRaises(ValueError):
            parse_rep_column(pd.Series(["[1,2]", "[3]"]))

    def test_tsne_modern_iteration_argument_runs(self):
        source = self.work / "features.npz"
        np.savez(source, X=np.random.default_rng(42).normal(size=(12, 4)))
        output = self.work / "nested" / "embedding.npz"
        self.run_script(
            "other/tsne_from_representation.py",
            "--mode",
            "npz",
            "--data-dir",
            self.work,
            "--npz-files",
            source,
            "--perplexity",
            "3",
            "--n-iter",
            "250",
            "--output",
            output,
        )
        with np.load(output, allow_pickle=False) as archive:
            self.assertEqual(archive["Y"].shape, (12, 2))
            self.assertTrue(np.isfinite(archive["Y"]).all())

    def test_scoring_averages_all_ten_toxicity_members(self):
        frame = pd.DataFrame({"ID": ["001"]})
        antibiotics = [
            (Path(f"ab_am_mtl_rep_1_col_{col}_is_pred.pt"), np.full((1, 4), 0.2))
            for col in range(1, 5)
        ]
        toxicities = [np.zeros((1, 3)) for _ in range(6)] + [
            np.ones((1, 3)) for _ in range(4)
        ]
        result = aggregate_predictions(frame, antibiotics, toxicities, True)
        self.assertAlmostEqual(result.loc[0, "Mean_HepG2_Toxicity"], 0.4)
        self.assertAlmostEqual(result.loc[0, "Non_Toxic_Score"], 0.6)
        self.assertEqual(result.ID.tolist(), ["001"])
        self.assertIn("Ensemble_Member_9_prediction_task_2", result)
        with self.assertRaises(ValueError):
            aggregate_predictions(result, antibiotics, toxicities, False)

    def test_model_honors_explicit_training_settings(self):
        model = MultiTaskFeedForward(
            input_dim=4,
            output_dim=2,
            hidden_dims=(7, 5, 3),
            learning_rate=0.003,
            L2_weight_norm=0.02,
            dropout_rate=0.1,
        )
        self.assertEqual(
            [layer.linear.out_features for layer in model.shared_layers], [7, 5, 3]
        )
        optimizers, _ = model.configure_optimizers()
        self.assertEqual(optimizers[0].defaults["lr"], 0.003)
        self.assertEqual(optimizers[0].defaults["weight_decay"], 0.02)
        actual = predict_probabilities(
            model, np.ones((3, 4), dtype=np.float32), 2, "cpu"
        )
        self.assertEqual(actual.shape, (3, 2))
        with self.assertRaises(ValueError):
            predict_probabilities(model, np.ones((3, 5), dtype=np.float32), 2, "cpu")
        loss = model._compute_loss(
            torch.zeros((2, 2)), torch.tensor([[1.0, -1.0], [0.0, 1.0]])
        )
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
