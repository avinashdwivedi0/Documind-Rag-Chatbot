import os
import pathlib
import sys
import tempfile

# Ensure project root is on sys.path
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from backend import evaluation_service as es


class DummyDoc:
    def __init__(self, source, page=1, content="Dummy content for testing."):
        self.metadata = {"source": source, "page": page}
        self.page_content = content


class DummyVectorStore:
    def similarity_search_with_relevance_scores(self, query, k=5):
        query = str(query).lower()
        if "revenue" in query:
            docs = [DummyDoc("Annual.pdf", page=3), DummyDoc("Other.pdf", page=1)]
        elif "summary" in query:
            docs = [DummyDoc("Summary.pdf", page=1), DummyDoc("Other.pdf", page=2)]
        else:
            docs = [DummyDoc("Other.pdf", page=1), DummyDoc("Summary.pdf", page=2)]
        return [(doc, float(1.0 / (i + 1))) for i, doc in enumerate(docs[:k])]


class DummyChain:
    def invoke(self, payload):
        question = payload.get("question", "")
        answer = f"Response for: {question}"
        return {
            "answer": answer,
            "usage": {"prompt_tokens": 8, "completion_tokens": 16, "total_tokens": 24},
            "source_documents": [DummyDoc("Annual.pdf", page=3, content="Answer evidence.")],
        }


def run():
    tmp = tempfile.TemporaryDirectory()
    es.EVAL_DIR = tmp.name
    workspace_id = "ci-regression-ws"
    vectorstore = DummyVectorStore()
    chain = DummyChain()
    memory = None

    cases = [
        {
            "id": "case-1",
            "question": "What is the revenue?",
            "expected_sources": ["Annual.pdf"],
            "expected_answer": "Response for: What is the revenue?",
        },
        {
            "id": "case-2",
            "question": "Where is the summary?",
            "expected_sources": ["Summary.pdf"],
        },
    ]

    print("Running baseline evaluation...")
    baseline_run = es.run_evaluation_dataset(
        workspace_id, vectorstore, chain, memory, cases=cases, top_k=3
    )
    print(f"Baseline run created: {baseline_run['id']}")

    print("Running comparison evaluation...")
    comparison_run = es.run_evaluation_dataset(
        workspace_id, vectorstore, chain, memory, cases=cases, top_k=3
    )
    print(f"Comparison run created: {comparison_run['id']}")

    thresholds = {"hit_drop": 0.2, "mrr_drop": 0.2, "latency_increase": 1.0}
    result = es.compare_runs(
        workspace_id, baseline_run["id"], comparison_run["id"], thresholds=thresholds
    )

    print("Comparison summary:")
    print(f"  gate_pass: {result['gate_pass']}")
    print(f"  alerts: {result['alerts']}")
    print(f"  warnings: {result['warnings']}")
    print("  summary_a:", result["summary_a"])
    print("  summary_b:", result["summary_b"])

    if not result["gate_pass"]:
        print("Regression gate failed. See results above.")
        sys.exit(1)

    print("CI evaluation regression check passed.")


if __name__ == "__main__":
    run()
