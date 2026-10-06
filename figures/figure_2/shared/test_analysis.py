"""Small scientific checks for metric conventions, missingness and cutoff ties."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('metric_analysis_under_test', HERE / 'analysis.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class ScientificMetricChecks(unittest.TestCase):
    def test_rank_example_and_confusion_table(self):
        result = analysis.evaluated_metrics([0, 0, 1, 1], [.1, .4, .35, .8])
        self.assertAlmostEqual(result['auroc'], .75)
        self.assertAlmostEqual(result['average_precision'], 5 / 6)
        self.assertAlmostEqual(result['pr_trapezoid_auc'], 19 / 24)
        self.assertEqual([result[name] for name in ['tn', 'fp', 'fn', 'tp']], [1, 1, 0, 2])
        self.assertAlmostEqual(result['balanced_accuracy'], .75)
        self.assertAlmostEqual(result['precision'], 2 / 3)
        self.assertAlmostEqual(result['f1'], .8)
        self.assertAlmostEqual(result['mcc'], 1 / np.sqrt(3))

    def test_tied_scores_have_chance_roc_and_prevalence_ap(self):
        result = analysis.evaluated_metrics([0, 1, 0, 1], [.5, .5, .5, .5])
        self.assertAlmostEqual(result['auroc'], .5)
        self.assertAlmostEqual(result['average_precision'], .5)

    def test_exact_cutoff_is_toxic(self):
        result = analysis.evaluated_metrics([0, 1], [.199, .2])
        self.assertEqual(result['exact_score_cutoff_count'], 1)
        self.assertEqual((result['tn'], result['tp']), (1, 1))

    def test_invalid_labels_or_scores_raise(self):
        for labels, scores in [([0, 2], [.1, .8]), ([0, 1], [.1, np.nan]),
                               ([0, 1], [.1, 1.2]), ([0], [.1, .2])]:
            with self.assertRaises(ValueError):
                analysis.evaluated_metrics(labels, scores)

    def test_missing_labels_remain_missing(self):
        frame = pd.DataFrame({'label': [0, np.nan, 1, -1], 'score': [.1, .2, .9, .7]})
        self.assertEqual(analysis.eligibility(frame, 'label', 'score').tolist(), [True, False, True, False])

    def test_single_class_is_excluded_from_discrimination(self):
        result = analysis.evaluated_metrics([1, 1], [.2, .8])
        self.assertEqual(result['status'], 'single_class')
        self.assertTrue(np.isnan(result['auroc']))
        self.assertTrue(np.isnan(result['average_precision']))

    def test_similarity_boundary_and_missingness(self):
        values = pd.Series([.899, .9, .901, np.nan, 1.0, -.1, -np.inf])
        self.assertEqual(analysis.similarity_mask(values, 'similarity_le_0_9').tolist(), [True, True, False, False, False, False, False])
        self.assertEqual(analysis.similarity_mask(values, 'similarity_lt_0_9').tolist(), [True, False, False, False, False, False, False])
        self.assertEqual(analysis.similarity_mask(values, 'similarity_lt_1').tolist(), [True, True, True, False, False, False, False])

    def test_max_mean_and_mean_max_differ(self):
        # Two real model-head arrangements need not share their highest task.
        member_scores = np.array([[[.9, .1, .2], [.1, .9, .2]]])
        self.assertAlmostEqual(member_scores.mean(axis=1).max(axis=1)[0], .5)
        self.assertAlmostEqual(member_scores.max(axis=2).mean(axis=1)[0], .9)


if __name__ == '__main__':
    unittest.main()
