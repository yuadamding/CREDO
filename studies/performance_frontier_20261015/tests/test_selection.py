import unittest

from frontier.selection import (CHANNELS, CandidateQuality, UnavailableSelection,
    normalized_risks, normalization_metadata, quality, choose, nondominated, developmental_margins)


def vector(value):
    return dict.fromkeys(CHANNELS, value)


class SelectionTests(unittest.TestCase):
    def test_incumbent_is_one_and_scale_invariance(self):
        self.assertEqual(normalized_risks(vector(2), vector(2), vector(3)), vector(1))
        self.assertEqual(normalized_risks(vector(1), vector(2), vector(3)), vector(.5))
        self.assertEqual(normalized_risks(vector(10), vector(20), vector(30)), vector(.5))

    def test_near_zero_fallback(self):
        result = normalized_risks(vector(1e-7), vector(0), vector(1))
        self.assertEqual(result, vector(1.1))
        with self.assertRaises(UnavailableSelection):
            normalized_risks(vector(1), vector(1), vector(0))

    def test_missing_required_channel_never_zero_filled(self):
        values = vector(1)
        del values["pooled_effect_mse"]
        with self.assertRaises(UnavailableSelection):
            normalized_risks(values, vector(1), vector(1))
        values["pooled_effect_mse"] = None
        with self.assertRaises(UnavailableSelection):
            normalized_risks(values, vector(1), vector(1))

    def test_worst_channel_matters(self):
        first = vector(.5)
        first["count_ce_minus_entropy"] = 4
        self.assertGreater(quality([first, vector(.5), vector(.5)]), quality([vector(1)] * 3))
        with self.assertRaises(UnavailableSelection):
            quality([vector(1)] * 2)

    def test_cost_then_complexity_ties(self):
        rows = [CandidateQuality("F0-V0", 1, 2, 10),
                CandidateQuality("F1-V0", 1 + 1e-7, 1, 20),
                CandidateQuality("F2-V0", 1 + 2e-7, 1, 15)]
        self.assertEqual(choose(rows).configuration_id, "F2-V0")

    def test_unregistered_recipe_is_not_a_candidate(self):
        for row in (CandidateQuality("F0-V4", 1, 1, 10),
                    CandidateQuality("F0-V0", 1, 1, 10, alpha=.2),
                    CandidateQuality("F0-V0", 1, 1, 10, spread_scale=2),
                    CandidateQuality("F0-V0", 1, 1, 10.5),
                    CandidateQuality("F0-V0", 1, 1, 10, alpha=True)):
            with self.assertRaises(ValueError):
                choose([row])

    def test_frontier_preserves_tradeoffs(self):
        a, b = vector(1), vector(1)
        a["state_energy"] = .5
        b["count_ce_minus_entropy"] = .5
        self.assertEqual(nondominated({"A": a, "B": b, "C": vector(1)}), ["A", "B"])

    def test_advancement_requires_all_marginals(self):
        risks = {donor: vector(1) for donor in ("D1", "D2", "D3", "D4")}
        flags = {donor: dict.fromkeys(CHANNELS, False) for donor in risks}
        for row in risks.values():
            row["state_energy"] = .94
        self.assertTrue(developmental_margins(risks, fallback_flags=flags)["developmental_rule_met"])
        risks["D1"]["spread_absolute_error"] = 1.1
        self.assertFalse(developmental_margins(risks, fallback_flags=flags)["developmental_rule_met"])

    def test_absolute_fallback_is_not_percentage_evidence(self):
        metadata = normalization_metadata(vector(0), vector(1))
        self.assertTrue(all(row["absolute_fallback"] for row in metadata.values()))
        flags = {donor: {k: row["absolute_fallback"] for k, row in metadata.items()}
                 for donor in ("D1", "D2", "D3", "D4")}
        result = developmental_margins({donor: vector(.9) for donor in flags}, fallback_flags=flags)
        self.assertIsNone(result["developmental_rule_met"])
        self.assertEqual(result["improved_5percent_channels"], [])


if __name__ == "__main__":
    unittest.main()
