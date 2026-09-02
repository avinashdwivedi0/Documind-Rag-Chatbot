import json
import os
import pathlib
import sys
import tempfile

# Ensure project root is on sys.path
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from backend import evaluation_service as es

# Use a temporary evaluation dir
tmp = tempfile.TemporaryDirectory()
es.EVAL_DIR = tmp.name

workspace_id = "smoke-ws"
case = {"question": "What is revenue?", "expected_sources": ["Annual.pdf"]}
added = es.add_case(workspace_id, case)
dataset = es.load_dataset(workspace_id)
if len(dataset) != 1:
    print("FAILED: dataset length != 1")
    raise SystemExit(2)


class DummyDoc:
    def __init__(self, source):
        self.metadata = {"source": source}


retrieved = [
    (DummyDoc("Other.pdf"), 0.9),
    (DummyDoc("Annual.pdf"), 0.85),
    (DummyDoc("Note.pdf"), 0.5),
]
metrics = es.evaluate_retrieval_case(retrieved, dataset[0], k=3)
if not metrics.get("hit"):
    print("FAILED: hit expected True")
    raise SystemExit(3)
if metrics.get("rank") != 2:
    print("FAILED: rank != 2", metrics)
    raise SystemExit(4)
print("SMOKE TEST OK")
