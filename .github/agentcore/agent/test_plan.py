"""Unit tests for the deterministic plan validator (``plan.py``).

    python -m unittest test_plan
"""

from __future__ import annotations

import unittest

from plan import Command, Requirements, command_violation, parse_plan, validate_plan

TREE = {"mitra/src/lexicon/vocabulary.py", "mitra/tests/conftest.py", "mitra/tests/test_sanskrit.py"}
REQS = ["R1", "R2"]
ACCEPT = {"cwd": "mitra", "command": "python -m pytest tests/test_vocabulary.py -q"}


def task(id, files=(), new_files=(), depends_on=(), covers=("R1",), acceptance=ACCEPT, title=None):
    return {"id": id, "title": title or f"task {id}", "goal": "g", "files": list(files),
            "new_files": list(new_files), "depends_on": list(depends_on), "covers": list(covers),
            "acceptance": acceptance}


def check(*tasks, reqs=REQS):
    return validate_plan(parse_plan({"subtasks": list(tasks)}), reqs, TREE)


class PlanValidatorTests(unittest.TestCase):
    def test_good_plan_passes(self):
        self.assertEqual(check(
            task(1, new_files=["mitra/tests/test_vocabulary.py"], covers=["R1"]),
            task(2, files=[], new_files=[], covers=["R2"]) | {"files": ["mitra/tests/conftest.py"]},
        ), [])

    def test_single_subtask_is_fine_for_a_small_ask(self):
        self.assertEqual(check(task(1, new_files=["mitra/tests/test_vocabulary.py"], covers=["R1", "R2"])), [])

    def test_pr25_style_inspect_and_run_steps_are_rejected(self):
        v = check(
            task(1, title="Inspect vocabulary.py", covers=["R1"]),
            task(2, new_files=["mitra/tests/test_vocabulary.py"], covers=["R1", "R2"], depends_on=[1]),
            task(3, title="Run targeted and full test suites", covers=["R2"], depends_on=[2]),
        )
        self.assertEqual(sum("changes no files" in x for x in v), 2)

    def test_every_requirement_must_be_covered(self):
        v = check(task(1, new_files=["mitra/tests/test_vocabulary.py"], covers=["R1"]))
        self.assertTrue(any("['R2'] are not covered" in x for x in v))

    def test_unknown_requirement_ids(self):
        v = check(task(1, new_files=["mitra/tests/test_vocabulary.py"], covers=["R1", "R2", "R9"]))
        self.assertTrue(any("unknown requirement" in x for x in v))

    def test_dependencies_must_point_backwards(self):
        v = check(task(1, new_files=["a/x.py"], depends_on=[2], covers=["R1"]),
                  task(2, new_files=["a/y.py"], covers=["R2"]))
        self.assertTrue(any("not an EARLIER" in x for x in v))

    def test_shared_file_needs_ordering(self):
        f = {"files": ["mitra/tests/conftest.py"]}
        self.assertTrue(any("both change" in x for x in check(task(1, covers=["R1"]) | f,
                                                               task(2, covers=["R2"]) | f)))
        self.assertEqual(check(task(1, covers=["R1"]) | f, task(2, covers=["R2"], depends_on=[1]) | f), [])

    def test_file_lists_must_match_the_repo(self):
        v = check(task(1, files=["mitra/tests/test_vocabulary.py"],
                       new_files=["mitra/tests/conftest.py"], covers=REQS))
        self.assertTrue(any("not in the repo" in x for x in v))
        self.assertTrue(any("already exists" in x for x in v))

    def test_size_and_forbidden_paths(self):
        many = [f"mitra/tests/t{i}.py" for i in range(5)]
        v = check(task(1, new_files=many + [".github/workflows/x.yml"], covers=REQS))
        self.assertTrue(any("split it" in x for x in v))
        self.assertTrue(any("may not change .github/workflows/x.yml" in x for x in v))

    def test_acceptance_command_must_be_runnable(self):
        for bad in ({"cwd": "mitra", "command": "cd mitra && pytest"},
                    {"cwd": "mitra", "command": "bash run_tests.sh"},
                    {"cwd": "nope", "command": "pytest -q"}):
            with self.subTest(bad=bad):
                v = check(task(1, new_files=["mitra/tests/test_vocabulary.py"], covers=REQS, acceptance=bad))
                self.assertTrue(any("acceptance" in x for x in v))

    def test_py_compile_is_rejected_as_subtask_acceptance(self):
        for interp in ("python", "python3"):
            with self.subTest(interp=interp):
                v = check(task(1, new_files=["mitra/x.py"], covers=REQS,
                               acceptance={"cwd": "mitra", "command": f"{interp} -m py_compile x.py"}))
                self.assertTrue(any("acceptance" in x and "py_compile" in x and "verify" in x for x in v))

    def test_py_compile_allowed_for_verify_only(self):
        tree = TREE | {"mitra/x.py"}
        for interp in ("python", "python3"):
            cmd = Command("mitra", f"{interp} -m py_compile x.py")
            self.assertNotEqual(command_violation(cmd, tree), "")
            self.assertEqual(command_violation(cmd, tree, allow_compile=True), "")

    def test_pytest_and_unittest_still_pass(self):
        for c in ("pytest -q", "python -m pytest -q", "python3 -m unittest test_sanskrit"):
            with self.subTest(c=c):
                self.assertEqual(command_violation(Command("mitra", c), TREE), "")
                self.assertEqual(command_violation(Command("mitra", c), TREE, allow_compile=True), "")

    def test_requirements_parse(self):
        r = Requirements.parse({"requirements": [{"id": "R1", "text": "t"}],
                                "verify": [{"cwd": "mitra", "command": "pytest -q"}], "constraints": ["tests only"]})
        self.assertEqual(r.verify, [Command("mitra", "pytest -q")])


if __name__ == "__main__":
    unittest.main()
