"""Tests for the orchestrator and jobs against a real git repo and real commands.

Claude is replaced by :class:`ScriptedRunner`, which returns scripted
structured output for read-only (requirements/plan) sessions and performs
scripted file edits for implement sessions — so everything the orchestrator
itself does (validating plans, reverting out-of-scope edits, running
acceptance commands, committing, re-splitting, pushing) runs for real.

    python -m unittest test_orchestrator
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

try:
    import requests  # noqa: F401
except ImportError:  # git_pr only needs it for real HTTP calls, which these tests patch out
    sys.modules["requests"] = types.ModuleType("requests")

import git_pr
import jobs
from claude_runner import RunResult
from orchestrator import DONE, FAILED, SKIPPED, Ask, OrchestrationError, Orchestrator

PASSING_TEST = "import unittest\nfrom calc import add\n\nclass T(unittest.TestCase):\n    def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n"
CALC_OK = "def add(a, b):\n    return a + b\n"


def git(*args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


def make_repo(root: str, branch: str = "main") -> str:
    repo = os.path.join(root, "work")
    os.makedirs(os.path.join(repo, "pkg"))
    git("init", "-q", "-b", branch, cwd=repo)
    git("config", "user.name", "t", cwd=repo)
    git("config", "user.email", "t@t", cwd=repo)
    with open(os.path.join(repo, "pkg", "calc.py"), "w") as fh:
        fh.write("def add(a, b):\n    return a - b  # bug\n")
    with open(os.path.join(repo, "pkg", "README"), "w") as fh:
        fh.write("calc\n")
    git("add", "-A", cwd=repo)
    git("commit", "-qm", "init", cwd=repo)
    return repo


def accept(test_module="test_calc"):
    return {"cwd": "pkg", "command": f"python3 -m unittest {test_module}"}


def subtask(id, title, *, files=(), new_files=(), depends_on=(), covers=("R1",), acceptance=None):
    return {"id": id, "title": title, "goal": f"do {title}", "files": list(files),
            "new_files": list(new_files), "depends_on": list(depends_on), "covers": list(covers),
            "acceptance": acceptance or accept()}


REQS = {"requirements": [{"id": "R1", "text": "add() adds"}, {"id": "R2", "text": "tested"}],
        "verify": [{"cwd": "pkg", "command": "python3 -m unittest test_calc"}],
        "constraints": []}


class ScriptedRunner:
    """Pops one scripted step per session. A step is a RunResult (read-only
    sessions) or a callable(cwd, write_paths) that edits files (implement)."""

    def __init__(self, cwd, steps):
        self.cwd, self.steps, self.prompts, self.write_paths = cwd, list(steps), [], []

    def run(self, prompt, *, system_append, write_paths=None, allow_tests=True, schema=None,
            max_turns=40, max_budget_usd=2.0):
        self.prompts.append(prompt)
        self.write_paths.append(write_paths)
        step = self.steps.pop(0)
        if callable(step):
            step(self.cwd)
            return RunResult(ok=True, subtype="success", text="edited", cost_usd=0.1)
        return step


def structured(obj):
    return RunResult(ok=True, subtype="success", structured=obj, cost_usd=0.1)


def write(path, content):
    def step(cwd):
        abs_path = os.path.join(cwd, path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w") as fh:
            fh.write(content)
    return step


def both(*steps):
    return lambda cwd: [s(cwd) for s in steps]


class OrchestratorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = make_repo(self.tmp.name)
        self.ask = Ask(kind="issue", repo="org/repo", title="#1 agent-fix-add", body="fix add and test it")

    def tearDown(self):
        self.tmp.cleanup()

    def orch(self, steps, **kw):
        runner = ScriptedRunner(self.repo, steps)
        return Orchestrator(self.repo, runner, **kw), runner

    def test_happy_path_commits_each_subtask_and_verifies(self):
        plan = {"subtasks": [
            subtask(1, "fix add and test", files=["pkg/calc.py"], new_files=["pkg/test_calc.py"],
                    covers=["R1", "R2"]),
            subtask(2, "more tests", new_files=["pkg/test_more.py"], depends_on=[1], covers=["R2"],
                    acceptance=accept("test_more")),
        ]}
        orch, runner = self.orch([structured(REQS), structured(plan),
                                  both(write("pkg/calc.py", CALC_OK), write("pkg/test_calc.py", PASSING_TEST)),
                                  write("pkg/test_more.py", PASSING_TEST)],
                                 commit_trailers=["Agent-Fix-Comment: 7"])
        out = orch.run(self.ask)

        self.assertTrue(out.complete)
        self.assertEqual([r.status for r in out.records], [DONE, DONE])
        self.assertEqual(len(out.commits), 2)
        self.assertFalse(out.baseline[0].ok)  # test_calc did not exist before
        self.assertTrue(out.verify[0].ok)
        self.assertIn("Agent-Fix-Comment: 7", git("log", "-1", "--format=%B", cwd=self.repo))
        self.assertEqual(runner.write_paths[2], ["pkg/calc.py", "pkg/test_calc.py"])  # scope handed to the session
        self.assertIsNone(runner.write_paths[0])  # requirements session is read-only

    def test_rejected_plan_is_replanned_with_violations(self):
        bad = {"subtasks": [subtask(1, "Inspect calc.py", covers=["R1"]),
                            subtask(2, "fix + test", files=["pkg/calc.py"], new_files=["pkg/test_calc.py"],
                                    covers=["R1"])]}
        good = {"subtasks": [subtask(1, "fix + test", files=["pkg/calc.py"], new_files=["pkg/test_calc.py"],
                                     covers=["R1", "R2"])]}
        orch, runner = self.orch([structured(REQS), structured(bad), structured(good),
                                  both(write("pkg/calc.py", CALC_OK), write("pkg/test_calc.py", PASSING_TEST))])
        out = orch.run(self.ask)
        self.assertTrue(out.complete)
        self.assertIn("REJECTED by the validator", runner.prompts[2])
        self.assertIn("changes no files", runner.prompts[2])
        self.assertIn("['R2'] are not covered", runner.prompts[2])

    def test_no_acceptable_plan_raises(self):
        bad = structured({"subtasks": [subtask(1, "look around", covers=["R1", "R2"])]})
        orch, _ = self.orch([structured(REQS), bad, bad, bad])
        with self.assertRaises(OrchestrationError):
            orch.run(self.ask)

    def test_out_of_scope_edits_are_reverted(self):
        plan = {"subtasks": [subtask(1, "fix + test", files=["pkg/calc.py"], new_files=["pkg/test_calc.py"],
                                     covers=["R1", "R2"])]}
        sneaky = both(write("pkg/calc.py", CALC_OK), write("pkg/test_calc.py", PASSING_TEST),
                      write("pkg/README", "rewritten\n"), write("pkg/extra.py", "x = 1\n"))
        orch, _ = self.orch([structured(REQS), structured(plan), sneaky])
        out = orch.run(self.ask)
        self.assertTrue(out.complete)
        changed = git("show", "--name-only", "--format=", "HEAD", cwd=self.repo).split()
        self.assertEqual(sorted(changed), ["pkg/calc.py", "pkg/test_calc.py"])
        self.assertTrue(any("reverted out-of-scope" in n for n in out.notes))

    def test_failing_subtask_is_resplit(self):
        big = {"subtasks": [subtask(1, "do everything", files=["pkg/calc.py"], new_files=["pkg/test_calc.py"],
                                    covers=["R1", "R2"])]}
        children = {"subtasks": [
            subtask(1, "fix add and test", files=["pkg/calc.py"], new_files=["pkg/test_calc.py"],
                    covers=["R1", "R2"]),
        ]}
        broken_test = write("pkg/test_calc.py", PASSING_TEST)  # calc still buggy -> fails
        orch, runner = self.orch([structured(REQS), structured(big), broken_test, broken_test,
                                  structured(children),
                                  both(write("pkg/calc.py", CALC_OK), write("pkg/test_calc.py", PASSING_TEST))])
        out = orch.run(self.ask)
        self.assertTrue(out.complete)
        self.assertEqual([(r.task.title, r.status, r.depth) for r in out.records],
                         [("do everything", DONE, 0), ("fix add and test", DONE, 1)])
        self.assertIn("too big to get passing", runner.prompts[4])
        self.assertIn("AssertionError", runner.prompts[3])  # 2nd attempt saw the failure

    def test_failure_skips_dependents_and_is_incomplete(self):
        plan = {"subtasks": [
            subtask(1, "add test", new_files=["pkg/test_calc.py"], covers=["R2"]),
            subtask(2, "docs", files=["pkg/README"], depends_on=[1], covers=["R1"]),
        ]}
        fail = write("pkg/test_calc.py", PASSING_TEST)
        no_split = structured({"subtasks": []})
        orch, _ = self.orch([structured(REQS), structured(plan), fail, fail, no_split, no_split, no_split])
        out = orch.run(self.ask)
        self.assertFalse(out.complete)
        self.assertEqual([r.status for r in out.records], [FAILED, SKIPPED])
        self.assertEqual(out.commits, [])
        self.assertEqual(git("status", "--porcelain", cwd=self.repo), "")  # partial work discarded

    def test_acceptance_passing_without_the_change_is_rejected(self):
        plan = {"subtasks": [subtask(1, "fix + test", files=["pkg/calc.py"], new_files=["pkg/test_calc.py"],
                                     covers=["R1", "R2"])]}
        vacuous = both(write("pkg/calc.py", "def add(a, b):\n    return a - b  # still a bug\n"),
                       write("pkg/test_calc.py", "import unittest\n\nclass T(unittest.TestCase):\n"
                                                 "    def test_nothing(self):\n        self.assertTrue(True)\n"))
        no_split = structured({"subtasks": []})
        orch, runner = self.orch([structured(REQS), structured(plan), vacuous, vacuous,
                                  no_split, no_split, no_split])
        out = orch.run(self.ask)
        self.assertFalse(out.complete)
        self.assertEqual([r.status for r in out.records], [FAILED])
        self.assertEqual(out.commits, [])
        self.assertIn("does not exercise the change", runner.prompts[3])  # retry prompt
        self.assertIn("too big to get passing", runner.prompts[4])  # re-planned
        self.assertEqual(git("status", "--porcelain", cwd=self.repo), "")
        self.assertEqual(git("stash", "list", cwd=self.repo), "")

    def test_real_fix_and_test_is_committed_and_stash_is_clean(self):
        plan = {"subtasks": [subtask(1, "fix + test", files=["pkg/calc.py"], new_files=["pkg/test_calc.py"],
                                     covers=["R1", "R2"])]}
        orch, _ = self.orch([structured(REQS), structured(plan),
                             both(write("pkg/calc.py", CALC_OK), write("pkg/test_calc.py", PASSING_TEST))])
        out = orch.run(self.ask)
        self.assertEqual([r.status for r in out.records], [DONE])
        self.assertEqual(len(out.commits), 1)
        self.assertEqual(git("stash", "list", cwd=self.repo), "")

    def test_budget_is_enforced(self):
        orch, _ = self.orch([], budget_usd=0.0)
        with self.assertRaises(OrchestrationError):
            orch.run(self.ask)


class JobTests(unittest.TestCase):
    """run_issue_job / run_fix_job against a local bare repo standing in for GitHub."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        seed = make_repo(self.tmp.name)
        self.remote = os.path.join(self.tmp.name, "remote.git")
        git("init", "-q", "--bare", self.remote, cwd=self.tmp.name)
        git("push", "-q", self.remote, "main", cwd=seed)
        self.api = {}
        self.comments, self.prs = [], []
        patches = [
            mock.patch.object(git_pr, "_authed_remote", lambda repo, token: self.remote),
            mock.patch.object(git_pr, "_public_remote", lambda repo: self.remote),
            mock.patch.object(git_pr, "github_api", lambda m, path, token, payload=None: self.api[path]),
            mock.patch.object(git_pr, "post_issue_comment",
                              lambda repo, n, body, token: self.comments.append((n, body))),
            mock.patch.object(git_pr, "create_pr",
                              lambda **kw: (self.prs.append(kw) or ("https://pr/9", 9))),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(self.tmp.cleanup)

    def good_steps(self):
        plan = {"subtasks": [subtask(1, "fix + test", files=["pkg/calc.py"], new_files=["pkg/test_calc.py"],
                                     covers=["R1", "R2"])]}
        return [structured(REQS), structured(plan),
                both(write("pkg/calc.py", CALC_OK), write("pkg/test_calc.py", PASSING_TEST))]

    def run_issue(self, issue, steps):
        self.api["/repos/org/repo/issues/1"] = issue
        factory = lambda clone: ScriptedRunner(clone, steps)  # noqa: E731
        return jobs.run_issue_job({"repo": "org/repo", "issue_number": 1, "title": issue["title"],
                                   "base_branch": "main"}, get_token=lambda: "tok",
                                  runner_factory=factory, workdir=tempfile.mkdtemp(dir=self.tmp.name))

    def test_issue_job_pushes_branch_and_opens_pr(self):
        result = self.run_issue({"title": "agent-fix-add", "state": "open", "body": "fix add"}, self.good_steps())
        self.assertEqual(result["status"], "ok")
        branch = "agentcore/agent-fix-add-1"
        self.assertIn(branch, git("branch", "--list", cwd=self.remote))
        self.assertFalse(self.prs[0]["draft"])
        self.assertIn("Closes #1", self.prs[0]["body"])
        self.assertIn("✅ **R2**", self.prs[0]["body"])

    def test_closed_or_non_agent_issue_exits_without_work(self):
        for issue in ({"title": "agent-x", "state": "closed"}, {"title": "fix x", "state": "open"}):
            with self.subTest(issue=issue):
                self.assertEqual(self.run_issue(issue, [])["status"], "skipped")
        self.assertEqual(self.prs, [])

    def test_fix_job_pushes_to_pr_branch_once(self):
        branch = "agentcore/agent-fix-add-1"
        seed = os.path.join(self.tmp.name, "work")
        git("push", "-q", self.remote, f"main:{branch}", cwd=seed)
        self.api.update({
            "/repos/org/repo/pulls/9": {"state": "open", "title": "[agentcore] agent-fix-add", "body": "Closes #1",
                                        "head": {"ref": branch, "repo": {"full_name": "org/repo"}}},
            "/repos/org/repo/issues/comments/55": {"user": {"login": "taufiq"},
                                                   "body": "/agent fix\r\nadd() subtracts"},
            "/repos/org/repo/pulls/9/comments?per_page=100": [],
            "/repos/org/repo/pulls/9/files?per_page=100": [{"filename": "pkg/calc.py", "status": "modified"}],
            "/repos/org/repo/issues/1": {"title": "agent-fix-add", "body": "fix add"},
        })
        payload = {"repo": "org/repo", "pr_number": 9, "comment_id": 55}
        steps = self.good_steps()
        runner = {}

        def factory(clone):
            runner["r"] = ScriptedRunner(clone, steps)
            return runner["r"]

        result = jobs.run_fix_job(payload, get_token=lambda: "tok", runner_factory=factory,
                                  workdir=tempfile.mkdtemp(dir=self.tmp.name))
        self.assertEqual(result["status"], "ok")
        self.assertIn("add() subtracts", runner["r"].prompts[0])
        self.assertIn("Agent-Fix-Comment: 55", git("log", "-1", "--format=%B", branch, cwd=self.remote))
        self.assertIn("🔧 Pushed 1 commit", self.comments[-1][1])

        again = jobs.run_fix_job(payload, get_token=lambda: "tok", runner_factory=factory,
                                 workdir=tempfile.mkdtemp(dir=self.tmp.name))
        self.assertEqual(again["status"], "already_applied")


if __name__ == "__main__":
    unittest.main()
