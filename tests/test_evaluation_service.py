import os
import shutil
import json

from backend import evaluation_service as es


class DummyDoc:
    def __init__(self, source):
        self.metadata = {"source": source}


def test_metrics_and_dataset(tmp_path):
    workspace_id = "test-ws"
    # point EVAL_DIR to tmp path for test isolation
    es.EVAL_DIR = str(tmp_path)
    # ensure empty dataset
    path = os.path.join(es.EVAL_DIR, f"{workspace_id}.eval.json")
    if os.path.exists(path):
        os.remove(path)

    case = {"question": "What is revenue?", "expected_sources": ["Annual.pdf"]}
    added = es.add_case(workspace_id, case)
    dataset = es.load_dataset(workspace_id)
    assert len(dataset) == 1

    # create retrieved list where Annual.pdf is rank 2
    retrieved = [(DummyDoc("Other.pdf"), 0.9), (DummyDoc("Annual.pdf"), 0.85), (DummyDoc("Note.pdf"), 0.5)]
    metrics = es.evaluate_retrieval_case(retrieved, dataset[0], k=3)
    assert metrics["hit"] is True
    assert metrics["rank"] == 2
    assert metrics["recall@k"] == 1
    assert isinstance(metrics["precision@k"], float)

    # cleanup
    if os.path.exists(path):
        os.remove(path)
