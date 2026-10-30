"""Checks for experiment metadata and the validation-only selection boundary."""

import unittest

import pandas as pd

from src.experiments import _record_params, select_final_model
from src.model import DEFAULT_PARAMS, create_model
from src.tuning import tune_catboost


class ExperimentChecks(unittest.TestCase):
    def test_default_metadata_matches_configured_catboost_options(self):
        for transform in ("none", "log1p"):
            with self.subTest(transform=transform):
                recorded = _record_params({}, transform)
                configured = create_model().get_params()
                for key in DEFAULT_PARAMS:
                    self.assertEqual(recorded[key], configured[key])
                self.assertEqual(recorded["target_transform"], transform)

    def test_tuned_metadata_matches_overrides_without_mutating_defaults(self):
        original = DEFAULT_PARAMS.copy()
        overrides = {
            "iterations": 800, "depth": 8,
            "learning_rate": 0.03, "l2_leaf_reg": 5,
        }
        recorded = _record_params(overrides, "log1p")
        configured = create_model(overrides).get_params()
        for key in original.keys() | overrides.keys():
            self.assertEqual(recorded[key], configured[key])
        self.assertEqual(DEFAULT_PARAMS, original)

    def test_final_selection_is_unchanged_by_reversed_test_ranking(self):
        metrics = pd.DataFrame([
            {"model": "catboost_log", "validation_scheme": "expanding_window", "mape": 8.0},
            {"model": "catboost_raw", "validation_scheme": "expanding_window", "mape": 15.0},
            {"model": "catboost_log", "validation_scheme": "final_test", "mape": 1000.0},
            {"model": "catboost_raw", "validation_scheme": "final_test", "mape": 0.0},
        ])
        self.assertEqual(select_final_model(metrics), "catboost_log")
        metrics.loc[metrics["validation_scheme"].eq("final_test"), "mape"] = [0.0, 1000.0]
        self.assertEqual(select_final_model(metrics), "catboost_log")

    def test_tuning_rejects_test_rows_before_training(self):
        test_rows = pd.DataFrame({"date": pd.to_datetime(["2016-01-01"])})
        for transform in ("none", "log1p"):
            with self.subTest(transform=transform):
                with self.assertRaisesRegex(ValueError, "independent test"):
                    tune_catboost(test_rows, transform, grid=[], verbose=False)


if __name__ == "__main__":
    unittest.main()
