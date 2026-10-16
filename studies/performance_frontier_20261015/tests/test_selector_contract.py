"""Fabricated scalar/identity evidence only, with no data access or fitting."""

from dataclasses import replace
from hashlib import sha256
import unittest

from frontier.configuration import ALPHA_GRID, DONORS, FAMILIES, SPREAD_GRID
from frontier.planner import family_plan, finalist_plan, screen_plan
from frontier.selection import CHANNELS
from frontier.selector_contract import (ChannelCoverage, CostEvidence, EvidenceContext,
    Recipe, ScoreEvidence, SelectorContract, TrainingScales)


def identity(value):
    return sha256(str(value).encode()).hexdigest()


def family_choices():
    return {d: tuple(f"{family}-V1" for family in FAMILIES) for d in DONORS}


def make_stage(name):
    if name == "screen":
        return screen_plan()
    families = family_plan(family_choices())
    if name == "family":
        return families
    return finalist_plan(families, {d: ("F0-V1", "F2-V1", "F5-V1") for d in DONORS})


def evidence_for(plan, selector, recipe, risk=1.0):
    rows = []
    for task in plan.scores_for(selector):
        if task.key.configuration_id != recipe.configuration_id:
            continue
        pair = task.key.training_donors
        context = EvidenceContext(
            plan.name, identity("runtime"), identity(("representation", pair)),
            identity(("fitting", pair)), identity(("source", pair, task.validation_donor)),
            identity(("truth", task.validation_donor)), identity("scorecode"),
            4096 if plan.name == "finalist" else 512, pair)
        coverage = {c: ChannelCoverage(identity(("support", c, task.validation_donor)),
                                     1 if c == "pooled_control_endpoint_mse" else 100)
                    for c in CHANNELS}
        rows.append(ScoreEvidence(task, recipe, context, coverage, {c: risk for c in CHANNELS}))
    return tuple(rows)


def scales_for(baseline):
    return tuple(TrainingScales(identity(("scales", row.task.key.training_donors)),
                row.task.key.training_donors, row.context.fitting_input_identity,
                row.context.representation_identity, {c: 1.0 for c in CHANNELS}) for row in baseline)


def contract_for(name="screen", selector="D1"):
    plan = make_stage(name)
    baseline = evidence_for(plan, selector, Recipe("F0-V0" if name == "screen" else "F0-V1"))
    return SelectorContract(plan, selector, baseline, scales_for(baseline))


def cost_for(recipe):
    return CostEvidence(recipe.configuration_id, identity("runtime"), identity("profile"), 1.0, 100)


def all_rows(contract, preferred=None):
    result = []
    for configuration in contract.plan.eligible_for(contract.selector):
        for alpha in ALPHA_GRID:
            for spread in SPREAD_GRID:
                recipe = Recipe(configuration, alpha, spread)
                risk = .5 if (alpha, spread) == preferred else 1.0
                evidence = evidence_for(contract.plan, contract.selector, recipe, risk)
                result.append(contract.evaluate(recipe, evidence, cost_for(recipe)))
    return tuple(result)


class SelectorContractTests(unittest.TestCase):
    def test_distinct_exact_edges_and_initial_baseline(self):
        contract = contract_for()
        self.assertEqual(contract.baseline_configuration, "F0-V0")
        recipe = Recipe("F1-V0", .25, 1.3)
        evidence = evidence_for(contract.plan, "D1", recipe, .8)
        row = contract.evaluate(recipe, evidence, cost_for(recipe))
        self.assertAlmostEqual(row.candidate_quality.quality, .8)
        with self.assertRaisesRegex(ValueError, "three distinct"):
            contract.evaluate(recipe, (evidence[0],) * 3, cost_for(recipe))
        with self.assertRaisesRegex(ValueError, "allowed edges"):
            contract.evaluate(recipe, evidence_for(contract.plan, "D2", recipe), cost_for(recipe))

    def test_same_prefix_winner_baseline_is_uncalibrated(self):
        for name in ("family", "finalist"):
            with self.subTest(name=name):
                contract = contract_for(name)
                self.assertEqual(contract.baseline_configuration, "F0-V1")
                calibrated = evidence_for(contract.plan, "D1", Recipe("F0-V1", .25, 1.3))
                with self.assertRaisesRegex(ValueError, "exact recipe"):
                    SelectorContract(contract.plan, "D1", calibrated, scales_for(calibrated))
        plan = make_stage("screen")
        wrong = evidence_for(plan, "D1", Recipe("F0-V1"))
        with self.assertRaisesRegex(ValueError, "allowed edges"):
            SelectorContract(plan, "D1", wrong, scales_for(wrong))

    def test_registered_recipe_grids(self):
        for arguments in (("UNKNOWN", 1., 1.), ("F0-V0", 2., 1.),
                          ("F0-V0", .25, .5), ("F0-V0", True, 1.)):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                Recipe(*arguments)

    def test_missing_or_changed_channel_support_rejected(self):
        contract = contract_for()
        recipe = Recipe("F1-V0")
        rows = list(evidence_for(contract.plan, "D1", recipe))
        coverage = dict(rows[0].coverage)
        coverage.pop(CHANNELS[0])
        with self.assertRaisesRegex(ValueError, "nine channel"):
            replace(rows[0], coverage=coverage)
        coverage = dict(rows[0].coverage)
        coverage[CHANNELS[0]] = ChannelCoverage(identity("changed"), 99)
        rows[0] = replace(rows[0], coverage=coverage)
        with self.assertRaisesRegex(ValueError, "identical support"):
            contract.evaluate(recipe, rows, cost_for(recipe))

    def test_missing_nonfinite_negative_channels_fail(self):
        contract = contract_for()
        row = evidence_for(contract.plan, "D1", Recipe("F1-V0"))[0]
        for replacement in (None, float("nan"), float("inf"), -1., True):
            risks = dict(row.risks)
            risks[CHANNELS[-1]] = replacement
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                replace(row, risks=risks)
        risks = dict(row.risks)
        risks.pop(CHANNELS[0])
        with self.assertRaises(ValueError):
            replace(row, risks=risks)

    def test_context_identity_mismatch(self):
        contract = contract_for()
        recipe = Recipe("F1-V0")
        original = evidence_for(contract.plan, "D1", recipe)
        for field in ("runtime_identity", "representation_identity", "fitting_input_identity",
                      "prediction_source_identity", "truth_identity", "scoring_identity"):
            rows = list(original)
            rows[0] = replace(rows[0], context=replace(rows[0].context, **{field: identity("different")}))
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "context differs"):
                contract.evaluate(recipe, rows, cost_for(recipe))

    def test_representation_and_scale_donors_cannot_include_outer(self):
        plan = make_stage("screen")
        baseline = evidence_for(plan, "D1", Recipe("F0-V0"))
        row = baseline[0]
        leaked_pair = tuple(sorted(("D1", row.task.key.training_donors[0])))
        with self.assertRaisesRegex(ValueError, "Representation fitting"):
            replace(row, context=replace(row.context, representation_fitting_donors=leaked_pair))
        scales = list(scales_for(baseline))
        scales[0] = replace(scales[0], fitting_donors=leaked_pair)
        with self.assertRaisesRegex(ValueError, "Missing fitting-only"):
            SelectorContract(plan, "D1", baseline, scales)

    def test_scale_identity_and_positivity(self):
        plan = make_stage("screen")
        baseline = evidence_for(plan, "D1", Recipe("F0-V0"))
        scales = list(scales_for(baseline))
        for field in ("fitting_input_identity", "representation_identity"):
            changed = list(scales)
            changed[0] = replace(changed[0], **{field: identity("different")})
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "same fitting-only"):
                SelectorContract(plan, "D1", baseline, changed)
        with self.assertRaises(ValueError):
            replace(scales[0], values={c: 0. for c in CHANNELS})
        with self.assertRaisesRegex(ValueError, "three distinct"):
            SelectorContract(plan, "D1", baseline, (scales[0],) * 3)

    def test_stage_and_resolution_rejected(self):
        contract = contract_for()
        recipe = Recipe("F1-V0")
        rows = list(evidence_for(contract.plan, "D1", recipe))
        with self.assertRaisesRegex(ValueError, "scoring resolution"):
            replace(rows[0].context, scoring_particles=4096)
        rows[0] = replace(rows[0], context=replace(rows[0].context, stage="family"))
        with self.assertRaisesRegex(ValueError, "different allocation stage"):
            contract.evaluate(recipe, rows, cost_for(recipe))

    def test_all_calibrations_required_and_family_roles(self):
        contract = contract_for()
        rows = all_rows(contract)
        self.assertEqual(len(rows), 24 * 24)
        winners = contract.family_winners(rows)
        self.assertEqual([row.recipe.configuration_id.split("-")[0] for row in winners], list(FAMILIES))
        with self.assertRaisesRegex(ValueError, "all 24 calibrations"):
            contract.family_winners(rows[:-1])
        with self.assertRaisesRegex(ValueError, "all 24 calibrations"):
            contract.family_winners(rows[:-1] + rows[:1])
        with self.assertRaises(ValueError):
            contract.freeze_outer(rows)

    def test_family_selection_and_outer_calibration_retention(self):
        family = contract_for("family")
        retained = family.finalists(all_rows(family, (.25, 1.3)))
        self.assertEqual(len(retained), 3)
        self.assertEqual(retained[0].recipe.configuration_id, "F0-V1")
        final = contract_for("finalist")
        outer = final.freeze_outer(all_rows(final, (.25, 1.3)))
        self.assertEqual(len(outer), 3)
        for recipe in outer:
            self.assertEqual(recipe.alpha, .25)
            self.assertEqual(recipe.spread_scale, 1.3)
            self.assertEqual(recipe.assessment_donor, "D1")
            self.assertEqual(recipe.fitting_donors, ("D2", "D3", "D4"))
            self.assertEqual(recipe.seeds, (0, 1, 2))
            self.assertEqual(recipe.updates, 9216)
            self.assertEqual(recipe.selection_contract_identity, final.identity)
            self.assertEqual(len(recipe.selection_evidence_identity), 64)
        self.assertEqual(len({row.selection_evidence_identity for row in outer}), 1)

    def test_cross_selector_or_altered_quality_rejected(self):
        first = contract_for("family", "D1")
        rows = list(all_rows(first))
        altered = replace(rows[0].candidate_quality, quality=-10.)
        rows[0] = replace(rows[0], candidate_quality=altered)
        with self.assertRaisesRegex(ValueError, "was altered"):
            first.finalists(rows)
        second = contract_for("family", "D2")
        with self.assertRaisesRegex(ValueError, "another selector"):
            second.finalists(all_rows(first))

    def test_cost_and_content_identity_guards(self):
        recipe = Recipe("F1-V0")
        contract = contract_for()
        evidence = evidence_for(contract.plan, "D1", recipe)
        with self.assertRaisesRegex(ValueError, "wrong model/runtime"):
            contract.evaluate(recipe, evidence, replace(cost_for(recipe), runtime_identity=identity("different")))
        for count in (0, 1.5, True):
            with self.subTest(count=count), self.assertRaises(ValueError):
                replace(cost_for(recipe), active_parameters=count)
        with self.assertRaises(ValueError):
            replace(cost_for(recipe), profile_identity="not-a-hash")


if __name__ == "__main__":
    unittest.main()
