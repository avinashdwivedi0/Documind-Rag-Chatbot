import json
import os
import tempfile
import time
import unittest

import backend.evaluation_service as es


def _write_runs(dirpath, workspace_id, runs):
    os.makedirs(dirpath, exist_ok=True)
    path = os.path.join(dirpath, f"{workspace_id}.runs.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        for r in runs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


class CompareRunsTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        # monkeypatch evaluation_service EVAL_DIR
        self.orig_eval_dir = es.EVAL_DIR
        es.EVAL_DIR = self.tmpdir.name

    def tearDown(self):
        es.EVAL_DIR = self.orig_eval_dir
        self.tmpdir.cleanup()

    def test_detects_hit_rate_regression(self):
        workspace_id = "ws-test"
        # run A: 2 cases, both hits
        run_a = {
            "id": "run-a",
            "workspace_id": workspace_id,
            "created_at": int(time.time()),
            "summary": {"total": 2, "passed": 2},
            "results": [
                {"case_id": "c1", "metrics": {"hit": True, "mrr": 1.0}, "trace": {"total_ms": 100}},
                {"case_id": "c2", "metrics": {"hit": True, "mrr": 1.0}, "trace": {"total_ms": 120}},
            ],
        }
        # run B: 2 cases, one hit only -> hit_rate drops from 1.0 to 0.5
        run_b = {
            "id": "run-b",
            "workspace_id": workspace_id,
            "created_at": int(time.time()),
            "summary": {"total": 2, "passed": 1},
            "results": [
                {"case_id": "c1", "metrics": {"hit": True, "mrr": 1.0}, "trace": {"total_ms": 110}},
                {
                    "case_id": "c2",
                    "metrics": {"hit": False, "mrr": 0.0},
                    "trace": {"total_ms": 130},
                },
            ],
        }
        _write_runs(self.tmpdir.name, workspace_id, [run_a, run_b])
        comp = es.compare_runs(workspace_id, "run-a", "run-b")
        self.assertIn("gate_pass", comp)
        self.assertFalse(comp["gate_pass"])
        self.assertIn("hit_rate", comp.get("alerts", []))

    def test_passes_with_higher_threshold(self):
        workspace_id = "ws-test-2"
        run_a = {
            "id": "run-a2",
            "workspace_id": workspace_id,
            "created_at": int(time.time()),
            "summary": {"total": 2, "passed": 2},
            "results": [
                {"case_id": "c1", "metrics": {"hit": True, "mrr": 1.0}, "trace": {"total_ms": 100}},
                {"case_id": "c2", "metrics": {"hit": True, "mrr": 1.0}, "trace": {"total_ms": 120}},
            ],
        }
        run_b = {
            "id": "run-b2",
            "workspace_id": workspace_id,
            "created_at": int(time.time()),
            "summary": {"total": 2, "passed": 1},
            "results": [
                {"case_id": "c1", "metrics": {"hit": True, "mrr": 1.0}, "trace": {"total_ms": 110}},
                {
                    "case_id": "c2",
                    "metrics": {"hit": False, "mrr": 0.0},
                    "trace": {"total_ms": 130},
                },
            ],
        }
        _write_runs(self.tmpdir.name, workspace_id, [run_a, run_b])
        # use a high hit_drop threshold so the small drop does not fail
        comp = es.compare_runs(
            workspace_id, "run-a2", "run-b2", thresholds={"hit_drop": 0.6, "mrr_drop": 0.6}
        )
        self.assertTrue(comp["gate_pass"])


if __name__ == "__main__":
    unittest.main()
