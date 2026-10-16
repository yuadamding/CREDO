"""Metadata and supplied-value tests only: no models or optimizer calls."""

from dataclasses import replace
from hashlib import sha256
import json
import math
from pathlib import Path
import unittest

from frontier.configuration import configurations
from frontier.objective_registry import (FittingScaleBinding, ObjectiveRules,
    RULES_SHA256, SCALE_NAMES, SourceSupportRecord, build_objective_protocol,
    source_support_factors)


class ObjectiveRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = (Path(__file__).resolve().parents[1] / "MAIN_OBJECTIVE_RULES.json").read_bytes()
        cls.rules = ObjectiveRules(cls.payload, RULES_SHA256)

    def scales(self, **changes):
        values = dict(fitting_donors=("D3", "D4"), rules_sha256=RULES_SHA256,
                      observation_artifact_sha256="1" * 64, fitting_input_sha256="2" * 64,
                      support_inventory_sha256="3" * 64, scale_artifact_sha256="4" * 64,
                      producer_sha256="5" * 64, sampler_sha256="6" * 64,
                      values=dict(zip(SCALE_NAMES, (1., .2, .3, .4, .5))))
        values.update(changes)
        return FittingScaleBinding(**values)

    def protocol(self, identifier="F0-V0", scales=None, **changes):
        values = dict(rules=self.rules, scales=scales or self.scales(), fitting_donors=("D3", "D4"),
                      observation_artifact_sha256="1" * 64, fitting_input_sha256="2" * 64)
        values.update(changes)
        return build_objective_protocol(identifier, **values)

    def rows(self):
        return (
            SourceSupportRecord("D3", "Stim8hr", "gA", "A", 500, False),
            SourceSupportRecord("D3", "Stim8hr", "gB", "B", 64, False),
            SourceSupportRecord("D3", "Stim8hr", "gZero", "ZERO", 0, False),
            SourceSupportRecord("D3", "Stim8hr", "gN", "NTC", 1000, True),
            SourceSupportRecord("D4", "Stim8hr", "gA", "A", 64, False),
            SourceSupportRecord("D4", "Stim8hr", "gA2", "A", 16, False),
            SourceSupportRecord("D4", "Stim8hr", "gB", "B", 64, False),
            SourceSupportRecord("D4", "Stim8hr", "gN", "NTC", 100, True),
        )

    def support(self, records=None, **changes):
        values = dict(fitting_donors=("D3", "D4"), control_target_ids=("NTC", "__NTC__"),
                      source_inventory_sha256="7" * 64, rules=self.rules)
        values.update(changes)
        return source_support_factors(self.rows() if records is None else records, **values)

    def test_all_24_configuration_protocols_follow_exact_family_matrix(self):
        identifiers = set()
        for configuration in configurations():
            with self.subTest(configuration=configuration.configuration_id):
                protocol = self.protocol(configuration.configuration_id)
                identifiers.add(protocol.protocol_id)
                self.assertEqual((protocol.state_coefficient, protocol.count_coefficient), (.5, .5))
                self.assertEqual(protocol.shape_coefficient, 0. if configuration.family == "F0" else .1)
                molecular = .1 if configuration.family in ("F3", "F4", "F5") else 0.
                self.assertEqual((protocol.endpoint_coefficient, protocol.effect_coefficient), (molecular, molecular))
                target = .01 if configuration.family in ("F3", "F5") else 0.
                self.assertEqual((protocol.target_embedding_penalty, protocol.target_growth_penalty), (target, target))
                self.assertEqual((protocol.deterministic_action, protocol.stochastic_action,
                                  protocol.reaction_action, protocol.reference_embedding_penalty),
                                 (1e-4, 1e-5, 1e-4, 1e-4))
                self.assertEqual(protocol.shape_bandwidths, (.25, .5, 1., 2.))
        self.assertEqual(len(identifiers), 24)

    def test_rule_bytes_cannot_be_changed_even_with_a_matching_new_hash(self):
        for payload in (self.payload + b"\n", self.payload.replace(b'"state_coefficient": 0.5', b'"state_coefficient": 1.0')):
            with self.assertRaises(ValueError):
                ObjectiveRules(payload, sha256(payload).hexdigest())
        with self.assertRaises(ValueError):
            ObjectiveRules(self.payload, "0" * 64)
        with self.assertRaises(ValueError):
            ObjectiveRules(self.payload.decode(), RULES_SHA256)
        document = self.rules.document
        document["core"]["state_coefficient"] = 99
        self.assertEqual(self.rules.document["core"]["state_coefficient"], .5)

    def test_positive_complete_scales_required_even_for_F0(self):
        for bad in (0., -1., math.nan, math.inf, True, "0.1"):
            for field in SCALE_NAMES:
                with self.subTest(field=field, bad=bad), self.assertRaises(ValueError):
                    values = dict(self.scales().values)
                    values[field] = bad
                    self.scales(values=values)
        for values in ({"state": 1.}, {**dict(self.scales().values), "extra": 1.}):
            with self.assertRaises(ValueError):
                self.scales(values=values)

    def test_matching_fold_observation_and_input_context_required(self):
        for changes in (dict(fitting_donors=("D2", "D3")),
                        dict(observation_artifact_sha256="8" * 64),
                        dict(fitting_input_sha256="9" * 64)):
            with self.assertRaises(ValueError):
                self.protocol(**changes)
        for donors in (("D4", "D3"), ("D3", "D3"), ("D1",), ("D1", "D2", "D3", "D4")):
            with self.assertRaises(ValueError):
                self.scales(fitting_donors=donors)
        with self.assertRaises(ValueError):
            self.scales(rules_sha256="f" * 64)
        triple = ("D1", "D3", "D4")
        self.assertEqual(self.protocol(scales=self.scales(fitting_donors=triple), fitting_donors=triple).training_donors, triple)
        with self.assertRaises(ValueError):
            self.protocol("F6-V0")

    def test_scales_immutable_identity_bound_but_not_claimed_verified(self):
        supplied = dict(self.scales().values)
        scales = self.scales(values=supplied)
        supplied["state"] = 10.
        self.assertEqual(scales.values["state"], 1.)
        with self.assertRaises(TypeError):
            scales.values["state"] = 10.
        changed = self.scales(values={**dict(scales.values), "state": 10.})
        self.assertNotEqual(changed.identity, scales.identity)
        self.assertNotEqual(self.protocol(scales=changed).protocol_id, self.protocol(scales=scales).protocol_id)
        self.assertFalse(scales.as_binding()["numerical_values_verified_by_this_helper"])
        self.assertFalse(scales.as_binding()["scientific_launch_authority"])
        self.assertEqual(json.loads(json.dumps(scales.as_binding()))["fitting_donors"], ["D3", "D4"])

    def test_source_support_caps_each_law_and_has_exact_control_unseen_zero(self):
        result = self.support()
        self.assertEqual(result.target_ids, ("A", "B"))
        self.assertEqual(result.support_scores, (208, 128))
        self.assertEqual(result.factors, (math.sqrt(208 / 464), math.sqrt(128 / 384)))
        for target in ("NTC", "__NTC__", "ZERO", "UNSEEN"):
            self.assertEqual(result.factor_for(target), 0.)
        self.assertEqual(result.factor_for("A"), result.factors[0])

    def test_source_order_does_not_change_factors_or_binding(self):
        first = self.support()
        reverse = self.support(tuple(reversed(self.rows())), control_target_ids=("__NTC__", "NTC"))
        self.assertEqual(first, reverse)
        self.assertEqual(first.identity, reverse.identity)
        changed = self.support(source_inventory_sha256="8" * 64)
        self.assertNotEqual(first.identity, changed.identity)

    def test_endpoint_or_query_support_and_duplicate_rows_rejected(self):
        with self.assertRaises(ValueError):
            replace(self.rows()[0], condition="Stim48hr")
        with self.assertRaises(ValueError):
            self.support((replace(self.rows()[0], donor_id="D1"), *self.rows()[1:]))
        with self.assertRaises(ValueError):
            self.support((*self.rows(), self.rows()[0]))
        with self.assertRaises(ValueError):
            self.support(tuple(row for row in self.rows() if row.donor_id == "D3"))
        for count in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                replace(self.rows()[0], positive_rna_cells=count)

    def test_inconsistent_guide_identity_or_control_metadata_rejected(self):
        rows = list(self.rows())
        rows[4] = replace(rows[4], target_id="CHANGED")
        with self.assertRaises(ValueError):
            self.support(rows)
        rows = list(self.rows())
        rows[3] = replace(rows[3], is_control=False)
        with self.assertRaises(ValueError):
            self.support(rows)
        with self.assertRaises(ValueError):
            self.support(control_target_ids=("NTC", "NTC"))

    def test_only_control_or_zero_source_support_creates_no_target_parameters(self):
        rows = tuple(row for row in self.rows() if row.is_control or row.positive_rna_cells == 0)
        result = self.support(rows)
        self.assertEqual(result.target_ids, ())
        self.assertEqual(result.factors, ())
        self.assertEqual(result.factor_for("A"), 0.)


if __name__ == "__main__":
    unittest.main()
