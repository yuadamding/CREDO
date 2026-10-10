"""Synthetic saved-risk rows only; no neural models or biological inputs."""

import copy
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import selection as module


def fixture():
    contract = {
        "schema": module.SCHEMA,
        "selection_donor": "D2",
        "seeds": [0, 1, 2],
        "state_estimator": "weighted_biased_v_energy",
        "count_score": "full_catalog_categorical_ce_nats",
        "catalog_size": 26504,
        "prefix": 4096,
        "support_id": "support-v1",
        "catalog_id": "catalog-v1",
        "state_axis_id": "axis-v1",
        "truth_id": "truth-D2-v1",
        "prefix_qualification_id": "prefix4096-qualified",
        "nonnegative_atol": 1e-10,
        "tie_atol": 1e-12,
        "state_support": [
            {"record_id": "a", "target_id": "A"},
            {"record_id": "b", "target_id": "A"},
            {"record_id": "c", "target_id": "B"},
        ],
        "candidates": [
            {
                "id": name,
                "complexity_rank": i,
                "parameter_count": (i + 1) * 100,
                "qualified": True,
                "qualification_id": name + "-qualified",
            }
            for i, name in enumerate(module.EXPECTED_CANDIDATES)
        ],
        "state_reference": {
            "qualified": True,
            "qualification_id": "state-ref-qualified",
        },
        "count_reference": {
            "qualified": True,
            "qualification_id": "count-ref-qualified",
        },
    }
    qualified = {
        item["id"]: item["qualification_id"] for item in contract["candidates"]
    }
    qualified[module.STATE_REFERENCE] = "state-ref-qualified"
    qualified[module.COUNT_REFERENCE] = "count-ref-qualified"

    def common(name, seed):
        return {
            "donor": "D2",
            "procedure_id": name,
            "seed": seed,
            "qualified": True,
            "qualification_id": qualified[name],
            **{key: contract[key] for key in module.IDENTITIES},
        }

    state, counts = [], []
    for name in (*module.EXPECTED_CANDIDATES, module.STATE_REFERENCE):
        value = 2 if name == module.STATE_REFERENCE else 1
        for seed in range(3):
            for record in contract["state_support"]:
                state.append({**common(name, seed), **record, "energy": value})
    for name in (*module.EXPECTED_CANDIDATES, module.COUNT_REFERENCE):
        value = 8 if name == module.COUNT_REFERENCE else 7
        for seed in range(3):
            counts.append(
                {
                    **common(name, seed),
                    "cross_entropy_nats": value,
                    "endpoint_entropy_nats": 6,
                }
            )
    return contract, state, counts


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.contract, self.state, self.counts = fixture()

    def result(self):
        return module.select_architecture(self.contract, self.state, self.counts)

    def test_balanced_equal_channel_entropy_subtracted_selection(self):
        result = self.result()
        self.assertEqual(result["selected_architecture"], module.EXPECTED_CANDIDATES[0])
        self.assertEqual(result["reference_count_ce_minus_entropy"], 2)
        self.assertEqual(result["candidates"][0]["balanced_risk"], 0.5)
        self.assertTrue(result["selected_balanced_developmental_advancement"])
        self.assertIsNone(result["selected_seed"])
        self.assertFalse(result["prediction_ensembling"])
        self.assertFalse(result["independent_validation"])

    def test_equal_targets_not_equal_all_guides(self):
        for row in self.state:
            if row["procedure_id"] == module.EXPECTED_CANDIDATES[0]:
                row["energy"] = 1 if row["target_id"] == "A" else 3
        result = self.result()["candidates"][0]
        self.assertEqual(result["state_energy"], 2)
        self.assertEqual(result["state_ratio_to_persistence"], 1)

    def test_mean_seed_scores_not_best_seed(self):
        for row in self.state:
            if row["procedure_id"] == module.EXPECTED_CANDIDATES[0]:
                row["energy"] = (0, 3, 6)[row["seed"]]
        result = self.result()
        self.assertEqual(result["candidates"][0]["state_energy"], 3)
        self.assertEqual(result["selected_architecture"], module.EXPECTED_CANDIDATES[1])

    def test_tradeoff_selected_but_not_advancement(self):
        for row in self.state:
            if row["procedure_id"] in module.EXPECTED_CANDIDATES:
                row["energy"] = 2.2
        for row in self.counts:
            if row["procedure_id"] in module.EXPECTED_CANDIDATES:
                row["cross_entropy_nats"] = 6.01
        result = self.result()
        self.assertLess(result["candidates"][0]["balanced_risk"], 1)
        self.assertFalse(result["selected_balanced_developmental_advancement"])
        self.assertFalse(result["any_candidate_balanced_developmental_advancement"])
        self.assertFalse(result["candidates"][0]["state_preserved"])
        self.assertTrue(result["candidates"][0]["count_preserved"])

    def test_exact_persistence_tie_is_not_advancement(self):
        for row in self.state:
            row["energy"] = 2
        for row in self.counts:
            row["cross_entropy_nats"] = 8
        self.assertFalse(self.result()["selected_balanced_developmental_advancement"])

    def test_deterministic_complexity_tie_and_input_order_invariance(self):
        original = self.result()
        self.state.reverse()
        self.counts.reverse()
        self.assertEqual(original, self.result())
        self.assertEqual(original["tie_candidates"], list(module.EXPECTED_CANDIDATES))

    def test_missing_seed_or_guide_rejected(self):
        self.state.pop()
        with self.assertRaisesRegex(module.SelectionError, "incomplete"):
            self.result()

    def test_missing_count_rejected(self):
        self.counts.pop()
        with self.assertRaisesRegex(module.SelectionError, "incomplete"):
            self.result()

    def test_duplicate_state_or_count_rejected(self):
        self.state.append(copy.deepcopy(self.state[0]))
        with self.assertRaisesRegex(module.SelectionError, "Duplicate state"):
            self.result()
        self.state.pop()
        self.counts.append(copy.deepcopy(self.counts[0]))
        with self.assertRaisesRegex(module.SelectionError, "Duplicate count"):
            self.result()

    def test_reserved_d1_outcome_rejected_in_both_channels(self):
        for rows in (self.state, self.counts):
            rows[0]["donor"] = "D1"
            with self.assertRaisesRegex(module.SelectionError, "D1"):
                self.result()
            rows[0]["donor"] = "D2"

    def test_candidate_specific_support_or_axis_rejected(self):
        for key in module.IDENTITIES:
            previous = self.state[0][key]
            self.state[0][key] = "different"
            with self.assertRaises(module.SelectionError):
                self.result()
            self.state[0][key] = previous

    def test_support_target_mapping_rejected(self):
        self.state[0]["target_id"] = "better-subset"
        with self.assertRaisesRegex(module.SelectionError, "common-support"):
            self.result()

    def test_unqualified_candidate_cannot_be_dropped(self):
        self.contract["candidates"][2]["qualified"] = False
        with self.assertRaisesRegex(module.SelectionError, "incomplete"):
            self.result()

    def test_score_qualification_must_match(self):
        self.state[0]["qualification_id"] = "another-procedure"
        with self.assertRaisesRegex(module.SelectionError, "identities"):
            self.result()

    def test_negative_energy_and_kl_rejected_beyond_tolerance(self):
        self.state[0]["energy"] = -1e-8
        with self.assertRaisesRegex(module.SelectionError, "negative beyond"):
            self.result()
        self.state[0]["energy"] = 1
        self.counts[0]["cross_entropy_nats"] = 6 - 1e-8
        with self.assertRaisesRegex(module.SelectionError, "negative beyond"):
            self.result()

    def test_tiny_negative_roundoff_is_explicit(self):
        self.state[0]["energy"] = -1e-12
        self.counts[0]["cross_entropy_nats"] = 6 - 1e-12
        result = self.result()
        self.assertEqual(len(result["numerical_roundoff_corrections"]), 2)
        self.assertTrue(
            all(
                item["canonical"] == 0
                for item in result["numerical_roundoff_corrections"]
            )
        )

    def test_zero_reference_normalizer_unresolved_no_epsilon(self):
        for row in self.state:
            if row["procedure_id"] == module.STATE_REFERENCE:
                row["energy"] = 0
        with self.assertRaisesRegex(module.SelectionError, "unresolved"):
            self.result()

    def test_zero_count_reference_normalizer_unresolved(self):
        for row in self.counts:
            if row["procedure_id"] == module.COUNT_REFERENCE:
                row["cross_entropy_nats"] = 6
        with self.assertRaisesRegex(module.SelectionError, "unresolved"):
            self.result()

    def test_entropy_identical_and_catalog_bounded(self):
        self.counts[0]["endpoint_entropy_nats"] = 6.1
        with self.assertRaisesRegex(module.SelectionError, "identical"):
            self.result()
        self.counts[0]["endpoint_entropy_nats"] = math.log(26504) + 0.1
        with self.assertRaisesRegex(module.SelectionError, "probability bounds"):
            self.result()

    def test_nan_infinity_and_booleans_are_not_scores(self):
        for value in (float("nan"), float("inf"), True, None):
            self.state[0]["energy"] = value
            with self.assertRaises(module.SelectionError):
                self.result()

    def test_persistence_not_seed_selected(self):
        self.counts[-1]["cross_entropy_nats"] = 8.01
        with self.assertRaisesRegex(module.SelectionError, "cannot vary"):
            self.result()

    def test_primary_prefix_and_fixed_numeric_policy(self):
        self.contract["prefix"] = 8192
        with self.assertRaisesRegex(module.SelectionError, "primary"):
            self.result()
        self.contract["prefix"] = 4096
        self.contract["nonnegative_atol"] = 0.1
        with self.assertRaisesRegex(module.SelectionError, "tolerances"):
            self.result()

    def test_bound_parameter_counts_match_prospective_complexity(self):
        self.contract["candidates"][0]["parameter_count"] = 400
        with self.assertRaisesRegex(module.SelectionError, "complexity"):
            self.result()

    def test_unbound_contract_cannot_select(self):
        self.contract["support_id"] = "UNBOUND_pending"
        with self.assertRaisesRegex(module.SelectionError, "bound"):
            self.result()


if __name__ == "__main__":
    unittest.main()
