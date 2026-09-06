"""Tests for unit-test sandbox, families, grader, and load_environment."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import unit_fix as ut  # noqa: E402


class EdgeCaseTests(unittest.TestCase):
    def test_fifteen_edge_cases(self) -> None:
        self.assertEqual(len(ut.EDGE_CASES), 15)
        families = [spec["family"] for spec in ut.EDGE_CASES]
        self.assertEqual(sorted(families), sorted(ut.FAMILIES))

    def test_each_edge_materializes(self) -> None:
        for spec in ut.EDGE_CASES:
            with self.subTest(family=spec["family"]):
                row = ut.example_from_spec(spec)
                self.assertEqual(row["info"]["family"], spec["family"])
                self.assertIn("question", row)
                self.assertTrue(
                    ut.run_tests(
                        row["info"]["gold_code"],
                        row["info"]["func_name"],
                        row["info"]["tests"],
                    ).all_passed
                )
                self.assertFalse(
                    ut.run_tests(
                        row["info"]["buggy_code"],
                        row["info"]["func_name"],
                        row["info"]["tests"],
                    ).visible_all_passed
                )


class SandboxTests(unittest.TestCase):
    def test_rejects_import_os(self) -> None:
        code = "import os\ndef f():\n    return 1\n"
        with self.assertRaises(ut.SandboxError):
            ut.load_function(code, "f")

    def test_rejects_from_import(self) -> None:
        code = "from os import path\ndef f():\n    return 1\n"
        with self.assertRaises(ut.SandboxError):
            ut.load_function(code, "f")

    def test_rejects_dunder_escape(self) -> None:
        code = "def f():\n    return ().__class__\n"
        with self.assertRaises(ut.SandboxError):
            ut.load_function(code, "f")

    def test_rejects_dunder_name(self) -> None:
        code = "def f(__x):\n    return __x\n"
        with self.assertRaises(ut.SandboxError):
            ut.load_function(code, "f")

    def test_rejects_while_true(self) -> None:
        code = "def f():\n    while True:\n        pass\n"
        with self.assertRaises(ut.SandboxError):
            ut.load_function(code, "f")

    def test_rejects_open(self) -> None:
        code = "def f():\n    open('/etc/passwd')\n"
        with self.assertRaises(ut.SandboxError):
            ut.load_function(code, "f")

    def test_run_tests_reports_load_error(self) -> None:
        result = ut.run_tests("import os\ndef f():\n    return 1\n", "f", {"visible": [], "hidden": []})
        self.assertTrue(any("load:" in e for e in result.errors))


class FamilyImplTests(unittest.TestCase):
    def test_all_families_registered(self) -> None:
        self.assertEqual(len(ut.FAMILY_IMPL), 15)
        for name in ut.FAMILIES:
            self.assertIn(name, ut.FAMILY_IMPL)

    def test_gold_passes_buggy_fails_visible_on_random(self) -> None:
        import random

        rng = random.Random(0)
        for family in ut.FAMILIES:
            with self.subTest(family=family):
                row = ut.generate_example(rng, family=family, difficulty="medium")
                info = row["info"]
                self.assertTrue(
                    ut.run_tests(info["gold_code"], info["func_name"], info["tests"]).all_passed
                )
                self.assertFalse(
                    ut.run_tests(
                        info["buggy_code"], info["func_name"], info["tests"]
                    ).visible_all_passed
                )


class GraderTests(unittest.TestCase):
    def _edge(self, family: str) -> dict:
        for spec in ut.EDGE_CASES:
            if spec["family"] == family:
                return ut.example_from_spec(spec)
        raise AssertionError(family)

    def test_gold_reward_is_1_2(self) -> None:
        row = self._edge("clamp")
        info = row["info"]
        scores = ut.grade(ut.gold_completion(info["gold_code"]), info["gold_code"], info)
        self.assertEqual(scores["exact_match"], 1.0)
        self.assertEqual(scores["format"], 1.0)
        self.assertEqual(scores["partial_credit"], 0.0)
        self.assertAlmostEqual(scores["reward"], 1.2)

    def test_naive_reward_is_0_2(self) -> None:
        row = self._edge("clamp")
        info = row["info"]
        scores = ut.grade(ut.naive_completion(info), info["gold_code"], info)
        self.assertEqual(scores["exact_match"], 0.0)
        self.assertEqual(scores["format"], 1.0)
        self.assertEqual(scores["partial_credit"], 0.0)
        self.assertAlmostEqual(scores["reward"], 0.2)

    def test_gold_naive_all_edges(self) -> None:
        for spec in ut.EDGE_CASES:
            with self.subTest(family=spec["family"]):
                row = ut.example_from_spec(spec)
                info = row["info"]
                g = ut.grade(ut.gold_completion(info["gold_code"]), info["gold_code"], info)
                n = ut.grade(ut.naive_completion(info), info["gold_code"], info)
                self.assertAlmostEqual(g["reward"], 1.2)
                self.assertAlmostEqual(n["reward"], 0.2)

    def test_hardcoding_visible_scores_0_3(self) -> None:
        """Hardcoding the visible expected value gets format+partial, not exact."""
        row = self._edge("nth_item")
        info = row["info"]
        # Visible: nth_item([10,20,30], 1) -> 10. Hidden asks for other indices.
        hardcoded = (
            "def nth_item(items, n):\n"
            "    return 10\n"
        )
        completion = f"<answer>\n{hardcoded}</answer>"
        scores = ut.grade(completion, info["gold_code"], info)
        self.assertEqual(scores["exact_match"], 0.0)
        self.assertEqual(scores["format"], 1.0)
        self.assertEqual(scores["partial_credit"], 0.5)
        self.assertAlmostEqual(scores["reward"], 0.3)

    def test_hardcoding_not_1_2(self) -> None:
        row = self._edge("sum_through")
        info = row["info"]
        hardcoded = "def sum_through(lo, hi):\n    return 6\n"
        scores = ut.grade(f"<answer>\n{hardcoded}</answer>", info["gold_code"], info)
        self.assertNotAlmostEqual(scores["reward"], 1.2)
        self.assertAlmostEqual(scores["reward"], 0.3)

    def test_no_tags_scores_zero(self) -> None:
        row = self._edge("safe_ratio")
        info = row["info"]
        scores = ut.grade(info["gold_code"], info["gold_code"], info)
        self.assertEqual(scores["format"], 0.0)
        self.assertEqual(scores["reward"], 0.0)

    def test_markdown_fence_inside_answer_stripped(self) -> None:
        row = self._edge("window_count")
        info = row["info"]
        body = info["gold_code"]
        completion = f"<answer>\n```python\n{body}```\n</answer>"
        scores = ut.grade(completion, info["gold_code"], info)
        self.assertAlmostEqual(scores["reward"], 1.2)

    def test_message_list_completion(self) -> None:
        row = self._edge("chunks")
        info = row["info"]
        completion = [{"role": "assistant", "content": ut.gold_completion(info["gold_code"])}]
        scores = ut.grade(completion, info["gold_code"], info)
        self.assertAlmostEqual(scores["reward"], 1.2)


class DatasetTests(unittest.TestCase):
    def test_build_rows_eval_prepends_edges(self) -> None:
        rows = ut.build_rows(n=15, seed=1, include_edge_cases=True)
        self.assertEqual(len(rows), 15)
        edge_families = [r["info"]["family"] for r in rows]
        self.assertEqual(edge_families, [s["family"] for s in ut.EDGE_CASES])

    def test_rows_for_dataset_stringifies_tests(self) -> None:
        rows = ut.build_rows(n=3, seed=7)
        ds_rows = ut.rows_for_dataset(rows)
        for row in ds_rows:
            self.assertIsInstance(row["info"]["tests"], str)
            parsed = json.loads(row["info"]["tests"])
            self.assertIn("visible", parsed)
            self.assertIn("hidden", parsed)
            self.assertNotIn("task", row)
            self.assertNotIn("task", row["info"])

    def test_family_pin(self) -> None:
        rows = ut.build_rows(n=5, seed=3, family="clamp")
        self.assertTrue(all(r["info"]["family"] == "clamp" for r in rows))

    def test_difficulty_pin(self) -> None:
        rows = ut.build_rows(n=5, seed=4, difficulty="easy")
        self.assertTrue(all(r["info"]["difficulty"] == "easy" for r in rows))


class LoadEnvironmentTests(unittest.TestCase):
    def test_load_environment_small(self) -> None:
        pytest.importorskip("verifiers")
        pytest.importorskip("datasets")
        env = ut.load_environment(
            num_train_examples=8,
            num_eval_examples=15,
            seed=42,
        )
        self.assertIsNotNone(env)
        # eval should start with the 15 curated edges
        eval_ds = env.eval_dataset
        self.assertEqual(len(eval_ds), 15)
        families = [row["info"]["family"] if isinstance(row["info"], dict) else json.loads(row["info"])["family"] for row in eval_ds]
        # HuggingFace datasets may nest info differently
        infos = []
        for i in range(len(eval_ds)):
            info = eval_ds[i]["info"]
            if isinstance(info, str):
                info = json.loads(info)
            infos.append(info["family"])
        self.assertEqual(infos, [s["family"] for s in ut.EDGE_CASES])
        for i in range(len(eval_ds)):
            info = eval_ds[i]["info"]
            if isinstance(info, str):
                info = json.loads(info)
            self.assertIsInstance(info["tests"], str)
            self.assertNotIn("task", info)

    def test_public_api_exports(self) -> None:
        for name in ("run_tests", "gold_completion", "naive_completion", "grade", "load_environment"):
            self.assertTrue(callable(getattr(ut, name)))


class BehaviorSpotChecks(unittest.TestCase):
    def test_same_letters_casefold(self) -> None:
        gold = ut.FAMILY_IMPL["same_letters"]["gold"]
        buggy = ut.FAMILY_IMPL["same_letters"]["buggy"]
        tests = {
            "visible": [{"args": ["Hello", "hello"], "expected": True}],
            "hidden": [{"args": ["straße", "STRASSE"], "expected": True}],
        }
        self.assertTrue(ut.run_tests(gold, "same_letters", tests).all_passed)
        self.assertFalse(ut.run_tests(buggy, "same_letters", tests).all_passed)

    def test_field_count_nbsp(self) -> None:
        gold = ut.FAMILY_IMPL["field_count"]["gold"]
        buggy = ut.FAMILY_IMPL["field_count"]["buggy"]
        tests = {
            "visible": [{"args": ["one two"], "expected": 2}],
            "hidden": [{"args": ["one\u00a0two"], "expected": 2}],
        }
        self.assertTrue(ut.run_tests(gold, "field_count", tests).all_passed)
        buggy_res = ut.run_tests(buggy, "field_count", tests)
        self.assertFalse(buggy_res.all_passed)

    def test_with_item_preserves_args(self) -> None:
        buggy = ut.FAMILY_IMPL["with_item"]["buggy"]
        tests = {
            "visible": [{"args": [[1, 2], 3], "expected": [1, 2, 3], "preserve_args": True}],
            "hidden": [],
        }
        self.assertFalse(ut.run_tests(buggy, "with_item", tests).visible_all_passed)

    def test_rotate_left_empty(self) -> None:
        gold = ut.FAMILY_IMPL["rotate_left"]["gold"]
        buggy = ut.FAMILY_IMPL["rotate_left"]["buggy"]
        tests = {
            "visible": [{"args": [[1, 2], 1], "expected": [2, 1]}],
            "hidden": [{"args": [[], 3], "expected": []}],
        }
        self.assertTrue(ut.run_tests(gold, "rotate_left", tests).all_passed)
        self.assertFalse(ut.run_tests(buggy, "rotate_left", tests).all_passed)

    def test_first_index_not_last(self) -> None:
        gold = ut.FAMILY_IMPL["first_index"]["gold"]
        buggy = ut.FAMILY_IMPL["first_index"]["buggy"]
        tests = {
            "visible": [{"args": [[1, 2, 1], 1], "expected": 0}],
            "hidden": [{"args": [[1, 2, 1, 2, 1], 2], "expected": 1}],
        }
        self.assertTrue(ut.run_tests(gold, "first_index", tests).all_passed)
        # buggy returns last match -> fails visible (last 1 is index 2)
        self.assertFalse(ut.run_tests(buggy, "first_index", tests).visible_all_passed)

    def test_json_string_tests_accepted(self) -> None:
        gold = ut.FAMILY_IMPL["clamp"]["gold"]
        tests = json.dumps(
            {
                "visible": [{"args": [5, 0, 10], "expected": 5}],
                "hidden": [{"args": [11, 0, 10], "expected": 10}],
            }
        )
        self.assertTrue(ut.run_tests(gold, "clamp", tests).all_passed)


if __name__ == "__main__":
    unittest.main()


class ExtraSandboxAndRewardTests(unittest.TestCase):
    def test_range_cap(self) -> None:
        code = "def f():\n    return list(range(10001))\n"
        with self.assertRaises(ut.SandboxError):
            ut.load_function(code, "f")()

    def test_partial_zero_when_exact(self) -> None:
        row = ut.example_from_spec(ut.EDGE_CASES[0])
        info = row["info"]
        partial = ut.partial_credit_score(
            ut.gold_completion(info["gold_code"]), info["gold_code"], info
        )
        self.assertEqual(partial, 0.0)

    def test_build_rows_train_size(self) -> None:
        rows = ut.build_rows(n=20, seed=99)
        self.assertEqual(len(rows), 20)

    def test_eval_edges_then_fill(self) -> None:
        rows = ut.build_rows(n=20, seed=5, include_edge_cases=True)
        self.assertEqual(len(rows), 20)
        self.assertEqual(
            [r["info"]["family"] for r in rows[:15]],
            [s["family"] for s in ut.EDGE_CASES],
        )

    def test_chunks_remainder_discriminator(self) -> None:
        gold = ut.FAMILY_IMPL["chunks"]["gold"]
        buggy = ut.FAMILY_IMPL["chunks"]["buggy"]
        tests = {
            "visible": [{"args": [[1, 2, 3, 4, 5], 2], "expected": [[1, 2], [3, 4], [5]]}],
            "hidden": [{"args": [[1, 2, 3], 2], "expected": [[1, 2], [3]]}],
        }
        self.assertTrue(ut.run_tests(gold, "chunks", tests).all_passed)
        self.assertFalse(ut.run_tests(buggy, "chunks", tests).visible_all_passed)
