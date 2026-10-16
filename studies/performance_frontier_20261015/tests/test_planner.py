"""Fabricated metadata tests only: no observations, optimizer or scheduler."""

import unittest
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

from frontier.configuration import DONORS, FAMILIES
from frontier.planner import (
    ACTIVE_STATUSES, H100_CAP, INNER_PAIRS, MAIN_UPDATE_CAP, OUTER_TRIPLES,
    AdmissionLedger, ScoreTask, SelectorBarrier, StagePlan, TrainingTask,
    TrajectoryKey, family_plan, finalist_plan, inner_edges, main_budget,
    outer_plan, screen_plan, training_pairs,
)


def choices(unique=False):
    return {d: tuple(f"{f}-V{i if unique else 0}" for f in FAMILIES)
            for i, d in enumerate(DONORS)}


def finalists(unique=False):
    return {d: tuple(f"{f}-V{i if unique else 0}" for f in ("F0", "F1", "F4"))
            for i, d in enumerate(DONORS)}


class PlannerTests(unittest.TestCase):
    def test_exact_subset_inventory(self):
        self.assertEqual(len(INNER_PAIRS), 6)
        self.assertEqual(len(OUTER_TRIPLES), 4)
        self.assertEqual(len(inner_edges()), 12)
        for selector, pair, validation in inner_edges():
            self.assertNotIn(selector, pair)
            self.assertNotIn(validation, pair)
            self.assertNotEqual(selector, validation)
            self.assertEqual(set(pair) | {selector, validation}, set(DONORS))
        for selector in DONORS:
            self.assertEqual(len(training_pairs(selector)), 3)

    def test_initial_screen_exact_counts(self):
        plan = screen_plan()
        self.assertEqual(len(plan.training), 144)
        self.assertEqual(len(plan.scores), 288)
        self.assertEqual(plan.additional_updates, 147456)
        for task in plan.training:
            self.assertEqual(len(task.requesting_selectors), 2)
            self.assertEqual(task.target_updates, 1024)
        for donor in DONORS:
            self.assertEqual(len(plan.scores_for(donor)), 72)

    def test_storage_identity_is_safe_distinct_and_preserves_logical_id(self):
        plans = [screen_plan(), family_plan(choices())]
        plans.append(finalist_plan(plans[1], finalists()))
        plans.append(outer_plan(plans[2]))
        keys = {task.key for plan in plans for task in plan.training}
        self.assertEqual(len({key.storage_identifier for key in keys}), len(keys))
        for key in keys:
            self.assertIsNotNone(re.fullmatch(r"[A-Za-z0-9_.-]+", key.storage_identifier))
            self.assertIn("/", key.identifier)
            self.assertNotIn("/", key.storage_identifier)
        self.assertEqual(TrajectoryKey(("D3", "D4"), "F0-V0", 0).storage_identifier,
                         "D3-D4--F0-V0--seed0")

    def test_maximum_updates_and_directed_scoring(self):
        first = screen_plan()
        second = family_plan(choices(unique=True))
        third = finalist_plan(second, finalists(unique=True))
        fourth = outer_plan(third)
        report = main_budget((first, second, third, fourth))
        self.assertEqual([len(p.training) for p in (first, second, third, fourth)], [144, 72, 36, 36])
        self.assertEqual([len(p.scores) for p in (first, second, third, fourth)], [288, 72, 36, 36])
        self.assertEqual([p.additional_updates for p in (first, second, third, fourth)],
                         [147456, 147456, 221184, 331776])
        self.assertEqual(report["planned_main_updates"], MAIN_UPDATE_CAP)
        self.assertEqual(MAIN_UPDATE_CAP, 92 * 9216)
        self.assertEqual(report["unused_updates_not_reallocatable"], 0)

    def test_identical_promotions_share_training_not_selectors(self):
        second = family_plan(choices())
        third = finalist_plan(second, finalists())
        fourth = outer_plan(third)
        report = main_budget((screen_plan(), second, third, fourth))
        self.assertEqual(len(second.training), 36)
        self.assertEqual(len(second.scores), 72)
        self.assertEqual(len(third.training), 18)
        self.assertEqual(len(third.scores), 36)
        self.assertEqual(len(fourth.training), 36)
        self.assertEqual(report["planned_main_updates"], 663552)
        self.assertEqual(report["unused_updates_not_reallocatable"], 184320)

    def test_union_promotions_cannot_leak_into_other_selector(self):
        second = family_plan(choices(unique=True))
        selected = finalists(unique=True)
        selected["D1"] = ("F0-V1", "F1-V0", "F4-V0")
        self.assertTrue(any(t.key.configuration_id == "F0-V1" for t in second.training))
        with self.assertRaisesRegex(ValueError, "another selector"):
            finalist_plan(second, selected)

    def test_family_winner_rules(self):
        for bad in (("F0-V0", "F0-V1", "F2-V0", "F3-V0", "F4-V0", "F5-V0"),
                    ("F0-V0",) * 6, ("F0-V0",)):
            with self.subTest(bad=bad):
                selected = choices()
                selected["D1"] = bad
                with self.assertRaises(ValueError):
                    family_plan(selected)
        with self.assertRaises(ValueError):
            family_plan({"D1": choices()["D1"]})

    def test_finalist_class_rules(self):
        second = family_plan(choices())
        for bad in (("F1-V0", "F2-V0", "F4-V0"),
                    ("F0-V0", "F4-V0", "F5-V0"),
                    ("F0-V0", "F1-V0", "F2-V0")):
            selected = finalists()
            selected["D1"] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                finalist_plan(second, selected)
        with self.assertRaises(ValueError):
            finalist_plan(screen_plan(), finalists())
        with self.assertRaises(ValueError):
            outer_plan(second)

    def test_score_role_validation(self):
        key = TrajectoryKey(("D3", "D4"), "F0-V0", 0)
        for selector, validation in (("D1", "D1"), ("D1", "D3"), ("D3", "D1")):
            with self.subTest(selector=selector, validation=validation), self.assertRaises(ValueError):
                ScoreTask(key, selector, validation, 1024)
        self.assertEqual(ScoreTask(key, "D1", "D2", 1024).phase, "inner")
        with self.assertRaises(ValueError):
            ScoreTask(key, "D1", "D2", 100)
        with self.assertRaises(ValueError):
            ScoreTask(key, "D1", "D2", 1024.0)
        with self.assertRaises(ValueError):
            ScoreTask(key, "D1", "D2", 9216, "outer")

    def test_key_and_transition_validation(self):
        self.assertEqual(TrajectoryKey(("D4", "D3"), "F0-V0", 0).training_donors, ("D3", "D4"))
        for args in ((("D3", "D3"), "F0-V0", 0),
                     (("D3", "D4"), "F6-V0", 0),
                     (("D3", "D4"), "F0-V0", 1),
                     (("D3", "D4"), "F0-V0", False)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                TrajectoryKey(*args)
        task = screen_plan().training[0]
        with self.assertRaises(ValueError):
            replace(task, start_updates=0, target_updates=9216)
        with self.assertRaises(ValueError):
            replace(task, start_updates=False)
        with self.assertRaises(ValueError):
            replace(task, requesting_selectors=(task.key.training_donors[0],))

    def test_plan_rejects_duplicate_or_unauthorized_tasks(self):
        plan = screen_plan()
        with self.assertRaises(ValueError):
            replace(plan, training=plan.training + plan.training[:1])
        with self.assertRaises(ValueError):
            replace(plan, scores=plan.scores + plan.scores[:1])
        with self.assertRaises(ValueError):
            replace(plan, training=())

    def test_budget_revalidates_whole_graph(self):
        plans = [screen_plan(), family_plan(choices())]
        plans.append(finalist_plan(plans[1], finalists()))
        plans.append(outer_plan(plans[2]))
        with self.assertRaises(ValueError):
            main_budget(plans + plans[-1:])
        with self.assertRaisesRegex(ValueError, "every exact"):
            replace(plans[1], scores=plans[1].scores[:-1])

    def test_empty_stage_cannot_create_ready_barrier(self):
        with self.assertRaisesRegex(ValueError, "every exact"):
            SelectorBarrier(replace(screen_plan(), scores=()))

    def test_selector_barriers_do_not_share_outcomes(self):
        plan = screen_plan()
        barrier = SelectorBarrier(plan)
        d1 = plan.scores_for("D1")
        with self.assertRaises(ValueError):
            barrier.require_ready("D1")
        with self.assertRaisesRegex(ValueError, "permitted view"):
            barrier.record("D2", d1[0].identifier)
        for task in d1:
            barrier.record("D1", task.identifier)
        barrier.require_ready("D1")
        self.assertFalse(barrier.ready("D2"))
        self.assertEqual(len(barrier.missing("D2")), 72)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            barrier.record("D1", d1[0].identifier)
        with self.assertRaises(ValueError):
            SelectorBarrier(outer_plan(finalist_plan(family_plan(choices()), finalists())))


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.tasks = screen_plan().training

    def test_capacity_includes_pending_running_uncertain(self):
        ledger = AdmissionLedger()
        for i in range(H100_CAP):
            ledger.reserve(f"owner{i}", self.tasks[i])
        ledger.transition("owner1", "running")
        ledger.transition("owner2", "uncertain")
        self.assertEqual(ledger.active_count, 4)
        self.assertEqual({item["status"] for item in ledger.snapshot()["owners"]}, ACTIVE_STATUSES)
        with self.assertRaisesRegex(ValueError, "slot"):
            ledger.reserve("owner4", self.tasks[4])

    def test_exclusive_trajectory_ownership_and_identity(self):
        ledger = AdmissionLedger()
        ledger.reserve("owner", self.tasks[0])
        with self.assertRaisesRegex(ValueError, "owner identity"):
            ledger.reserve("owner", self.tasks[1])
        with self.assertRaisesRegex(ValueError, "already has"):
            ledger.reserve("duplicate", self.tasks[0])
        ledger.transition("owner", "uncertain")
        with self.assertRaisesRegex(ValueError, "already has"):
            ledger.reserve("duplicate2", self.tasks[0])

    def test_completed_prefix_continuation(self):
        ledger = AdmissionLedger()
        first = self.tasks[0]
        second = replace(first, start_updates=1024, target_updates=3072)
        third = replace(first, start_updates=3072, target_updates=9216)
        with self.assertRaisesRegex(ValueError, "prefix"):
            ledger.reserve("premature", second)
        ledger.reserve("first", first)
        ledger.transition("first", "running")
        with self.assertRaisesRegex(ValueError, "exact registered"):
            ledger.complete("first", 1023)
        ledger.complete("first", 1024)
        self.assertEqual(ledger.active_count, 0)
        with self.assertRaisesRegex(ValueError, "prefix"):
            ledger.reserve("restart", first)
        ledger.reserve("second", second)
        ledger.transition("second", "running")
        ledger.transition("second", "uncertain")
        ledger.complete("second", 3072)
        ledger.reserve("third", third)
        ledger.transition("third", "running")
        ledger.complete("third", 9216)
        self.assertEqual(ledger.committed_updates, 9216)

    def test_failed_and_cancelled_do_not_retry_or_recycle_budget(self):
        for status in ("failed", "cancelled"):
            with self.subTest(status=status):
                ledger = AdmissionLedger()
                ledger.reserve("owner", self.tasks[0])
                ledger.transition("owner", status)
                self.assertEqual(ledger.active_count, 0)
                self.assertEqual(ledger.committed_updates, 1024)
                with self.assertRaisesRegex(ValueError, "separately reviewed"):
                    ledger.reserve("retry", self.tasks[0])
                with self.assertRaisesRegex(ValueError, "Terminal"):
                    ledger.transition("owner", "running")

    def test_update_budget_admission_and_resource_caps(self):
        ledger = AdmissionLedger(maximum_updates=1024)
        ledger.reserve("owner", self.tasks[0])
        with self.assertRaisesRegex(ValueError, "allowance"):
            ledger.reserve("over", self.tasks[1])
        self.assertEqual(ledger.committed_updates, 1024)
        for cap in (0, 5, True):
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                AdmissionLedger(maximum_active=cap)
        with self.assertRaises(ValueError):
            AdmissionLedger(maximum_updates=MAIN_UPDATE_CAP + 1)

    def test_atomic_threaded_reservation(self):
        ledger = AdmissionLedger()

        def reserve(i):
            try:
                ledger.reserve(f"owner{i}", self.tasks[i])
                return True
            except ValueError:
                return False

        # Local scope permits at most two CPU threads; this tests metadata only.
        with ThreadPoolExecutor(max_workers=2) as pool:
            admitted = list(pool.map(reserve, range(8)))
        self.assertEqual(sum(admitted), 4)
        self.assertEqual(ledger.active_count, 4)

    def test_completion_and_status_guardrails(self):
        ledger = AdmissionLedger()
        with self.assertRaises(ValueError):
            ledger.transition("unknown", "running")
        with self.assertRaises(ValueError):
            ledger.reserve("", self.tasks[0])
        ledger.reserve("owner", self.tasks[0])
        with self.assertRaises(ValueError):
            ledger.complete("owner", 1024)
        with self.assertRaises(ValueError):
            ledger.transition("owner", "complete")
        ledger.transition("owner", "running")
        with self.assertRaises(ValueError):
            ledger.transition("owner", "pending")
        ledger.complete("owner", 1024)
        with self.assertRaises(ValueError):
            ledger.complete("owner", 1024)


if __name__ == "__main__":
    unittest.main()
