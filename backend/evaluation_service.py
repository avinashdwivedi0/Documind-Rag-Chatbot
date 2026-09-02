"""Evaluation service for RAG pipelines: dataset management, metrics, and traces.

This module provides:
- dataset persistence per workspace
- retrieval metrics (hit@K, recall@K, precision@K, MRR)
- simple instrumentation/tracing around retrieval + LLM invocation
- run orchestration for evaluation datasets

All persisted evaluation data is stored under the `EVAL_DIR` configured in `backend.config`.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from backend.config import EVAL_DIR
from backend.evaluator import LLMEvaluator, RuleBasedEvaluator

logger = logging.getLogger(__name__)


def _dataset_path(workspace_id: str) -> str:
    """Get the dataset file path for a workspace."""
    return os.path.join(EVAL_DIR, f"{workspace_id}.eval.json")


def load_dataset(workspace_id: str) -> List[Dict[str, Any]]:
    """Load evaluation dataset for a workspace.

    Args:
        workspace_id: Workspace identifier

    Returns:
        List of evaluation cases, empty list if file not found

    Example:
        >>> cases = load_dataset("workspace-123")
        >>> len(cases)
        5
    """
    try:
        with open(_dataset_path(workspace_id), "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_dataset(workspace_id: str, dataset: List[Dict[str, Any]]) -> None:
    """Save evaluation dataset for a workspace.

    Uses atomic write (temp file + replace) to prevent corruption.

    Args:
        workspace_id: Workspace identifier
        dataset: List of evaluation cases

    Example:
        >>> cases = [{"id": "1", "question": "What is AI?"}]
        >>> save_dataset("workspace-123", cases)
    """
    path: str = _dataset_path(workspace_id)
    tmp: str = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(dataset, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def add_case(workspace_id: str, case: Dict[str, Any]) -> Dict[str, Any]:
    """Add an evaluation case to dataset.

    Args:
        workspace_id: Workspace identifier
        case: Evaluation case dictionary (auto-generates id if missing)

    Returns:
        The added case with id field populated

    Example:
        >>> case = {"question": "What is AI?", "expected_source": "ai.pdf"}
        >>> added = add_case("workspace-123", case)
        >>> "id" in added
        True
    """
    dataset: List[Dict[str, Any]] = load_dataset(workspace_id)
    case = dict(case)
    case.setdefault("id", str(uuid.uuid4()))
    dataset.append(case)
    save_dataset(workspace_id, dataset)
    return case


def delete_case(workspace_id: str, case_id: str) -> None:
    """Delete an evaluation case from dataset.

    Args:
        workspace_id: Workspace identifier
        case_id: Case ID to delete
    """
    dataset: List[Dict[str, Any]] = load_dataset(workspace_id)
    dataset = [c for c in dataset if c.get("id") != case_id]
    save_dataset(workspace_id, dataset)


def _rank_list_index(
    retrieved: List[Tuple[Any, float]], expected_source: Optional[str]
) -> Optional[int]:
    """Find rank position of expected source in retrieved results.

    Args:
        retrieved: List of (Document, score) tuples
        expected_source: Expected document source name

    Returns:
        1-based rank position or None if not found
    """
    if not expected_source:
        return None
    for idx, (doc, _) in enumerate(retrieved, start=1):
        meta = getattr(doc, "metadata", {})
        source: str = os.path.basename(meta.get("source", ""))
        if source == expected_source:
            return idx
    return None


def recall_at_k(
    retrieved: List[Tuple[Any, float]], expected_source: Optional[str], k: int
) -> Optional[int]:
    """Calculate recall@k metric (0 or 1).

    Args:
        retrieved: List of (Document, score) tuples
        expected_source: Expected document source
        k: Cutoff position

    Returns:
        1 if expected source in top-k, 0 otherwise, None if not applicable
    """
    if expected_source is None:
        return None
    topk = retrieved[:k]
    return 1 if _rank_list_index(topk, expected_source) is not None else 0


def precision_at_k(
    retrieved: List[Tuple[Any, float]], relevant_sources: List[str], k: int
) -> Optional[float]:
    """Calculate precision@k metric (0.0 to 1.0).

    Args:
        retrieved: List of (Document, score) tuples
        relevant_sources: List of relevant document sources
        k: Cutoff position

    Returns:
        Fraction of relevant documents in top-k, None if not applicable
    """
    if not relevant_sources:
        return None
    topk = retrieved[:k]
    hits: int = 0
    for doc, _ in topk:
        source: str = os.path.basename(getattr(doc, "metadata", {}).get("source", ""))
        if source in relevant_sources:
            hits += 1
    return hits / k


def mrr(retrieved: List[Tuple[Any, float]], expected_sources: List[str]) -> Optional[float]:
    """Calculate Mean Reciprocal Rank (0.0 to 1.0).

    Args:
        retrieved: List of (Document, score) tuples
        expected_sources: List of relevant document sources

    Returns:
        1/rank of first relevant document, 0 if none found, None if not applicable
    """
    if not expected_sources:
        return None
    for idx, (doc, _) in enumerate(retrieved, start=1):
        source: str = os.path.basename(getattr(doc, "metadata", {}).get("source", ""))
        if source in expected_sources:
            return 1.0 / idx
    return 0.0


def evaluate_retrieval_case(
    retrieved: List[Tuple[Any, float]], case: Dict[str, Any], k: int = 5
) -> Dict[str, Any]:
    """Compute retrieval-focused metrics for a single case.

    Args:
        retrieved: List of (Document, score) tuples from vectorstore
        case: Evaluation case with expected_sources or expected_source
        k: Cutoff position for metrics (default: 5)

    Returns:
        Dictionary with hit, rank, recall@k, precision@k, MRR, avg_retrieval_score

    Example:
        >>> case = {"expected_source": "document.pdf"}
        >>> metrics = evaluate_retrieval_case(retrieved, case, k=5)
        >>> metrics["hit"]
        True
    """
    # Normalize expected sources to list
    expected_sources: List[str] = case.get("expected_sources") or (
        [case.get("expected_source")] if case.get("expected_source") else []
    )
    expected_sources = [s for s in expected_sources if s]

    # Calculate metrics
    rank: Optional[int] = (
        _rank_list_index(retrieved, expected_sources[0]) if expected_sources else None
    )
    hit: bool = bool(rank)
    rec_at_k: Optional[int] = recall_at_k(
        retrieved, expected_sources[0] if expected_sources else None, k
    )
    prec_at_k: Optional[float] = precision_at_k(retrieved, expected_sources, k)
    mrr_score: Optional[float] = mrr(retrieved, expected_sources)

    # Average relevance score
    avg_score: Optional[float] = None
    if retrieved:
        scores: List[float] = [float(score) for (_, score) in retrieved[:k]]
        if scores:
            avg_score = sum(scores) / len(scores)

    return {
        "hit": hit,
        "rank": rank,
        "recall@k": rec_at_k if rec_at_k is not None else "Not available",
        "precision@k": round(prec_at_k, 3) if prec_at_k is not None else "Not available",
        "mrr": round(mrr_score, 3) if mrr_score is not None else "Not available",
        "avg_retrieval_score": round(avg_score, 4) if avg_score is not None else "Not available",
    }


def instrument_query(
    vectorstore: Any, chain: Any, memory: Any, workspace_id: str, question: str, top_k: int = 5
) -> Dict[str, Any]:
    """Run retrieval and chain invocation while collecting trace metadata and timings.

    Args:
        vectorstore: FAISS vectorstore instance
        chain: LangChain conversation chain
        memory: Conversation memory buffer
        workspace_id: Workspace identifier
        question: User query
        top_k: Number of documents to retrieve (default: 5)

    Returns:
        Dictionary with 'trace' (metadata) and 'result' (chain output)

    Example:
        >>> output = instrument_query(vs, chain, mem, "ws-123", "What is AI?")
        >>> output["trace"]["retrieval_ms"]
        145
    """
    trace: Dict[str, Any] = {
        "trace_id": str(uuid.uuid4()),
        "workspace_id": workspace_id,
        "question": question if len(question) < 256 else question[:256] + "...",
    }

    # Retrieval with timing
    t0: float = time.perf_counter()
    try:
        retrieved: List[Tuple[Any, float]] = vectorstore.similarity_search_with_relevance_scores(
            question, k=top_k
        )
    except Exception:
        retrieved = []
    t1: float = time.perf_counter()
    trace["retrieval_ms"] = int((t1 - t0) * 1000)
    trace["retrieved_chunks"] = len(retrieved)

    # LLM / chain invoke with timing
    t2: float = time.perf_counter()
    try:
        result: Dict[str, Any] = chain.invoke({"question": question})
    except Exception as e:
        result = {"error": str(e)}
    t3 = time.perf_counter()
    trace["llm_ms"] = int((t3 - t2) * 1000)
    trace["total_ms"] = int((t3 - t0) * 1000)

    # Safe summary of context
    trace["final_context_chunks"] = min(len(retrieved), top_k)
    trace["retrieval_top_k"] = top_k

    # Token & cost tracking (provider-agnostic). Prefer precise usage if returned by the chain.
    tokens_info = None
    cost_info = None
    try:
        # Common locations for usage info
        usage = None
        if isinstance(result, dict):
            usage = result.get("usage") or result.get("token_usage") or result.get("tokens")
        if usage and isinstance(usage, dict):
            prompt_t = (
                usage.get("prompt_tokens") or usage.get("prompt") or usage.get("input_tokens")
            )
            comp_t = (
                usage.get("completion_tokens")
                or usage.get("completion")
                or usage.get("output_tokens")
            )
            total_t = (
                usage.get("total_tokens") or usage.get("total") or (prompt_t or 0) + (comp_t or 0)
            )
            tokens_info = {
                "prompt_tokens": int(prompt_t) if prompt_t is not None else None,
                "completion_tokens": int(comp_t) if comp_t is not None else None,
                "total": int(total_t) if total_t is not None else None,
                "estimated": False,
            }
        else:
            # Estimate tokens conservatively from word counts if no usage provided
            answer = None
            try:
                answer = result.get("answer") or result.get("output")
            except Exception:
                answer = None
            words_q = len(str(question).split()) if question else 0
            words_a = len(str(answer).split()) if answer else 0
            # approx tokens ≈ words * 1.33
            prompt_est = int(words_q * 1.33)
            comp_est = int(words_a * 1.33)
            tokens_info = {
                "prompt_tokens": prompt_est,
                "completion_tokens": comp_est,
                "total": prompt_est + comp_est,
                "estimated": True,
            }

        # Cost calculation using optional env var TOKEN_COST_PER_1K (USD per 1000 tokens)
        cost_per_1k = os.getenv("TOKEN_COST_PER_1K")
        if cost_per_1k:
            try:
                cost_per_1k = float(cost_per_1k)
                amount = None
                if tokens_info and tokens_info.get("total") is not None:
                    amount = round((tokens_info["total"] / 1000.0) * cost_per_1k, 6)
                cost_info = {
                    "provider": os.getenv("LLM_PROVIDER") or "unknown",
                    "amount_usd": amount,
                    "per_1k_usd": cost_per_1k,
                    "estimated": tokens_info.get("estimated", True),
                }
            except Exception:
                cost_info = {
                    "provider": os.getenv("LLM_PROVIDER") or "unknown",
                    "amount_usd": None,
                    "per_1k_usd": None,
                    "estimated": True,
                }
        else:
            cost_info = {
                "provider": os.getenv("LLM_PROVIDER") or "unknown",
                "amount_usd": None,
                "per_1k_usd": None,
                "estimated": tokens_info.get("estimated", True),
            }
    except Exception:
        tokens_info = {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total": None,
            "estimated": True,
        }
        cost_info = {
            "provider": os.getenv("LLM_PROVIDER") or "unknown",
            "amount_usd": None,
            "per_1k_usd": None,
            "estimated": True,
        }

    trace["tokens"] = tokens_info
    trace["cost"] = cost_info

    return {"trace": trace, "retrieved": retrieved, "result": result}


def run_evaluation_dataset(
    workspace_id: str,
    vectorstore,
    chain,
    memory,
    cases: Optional[List[Dict]] = None,
    top_k: int = 5,
) -> Dict:
    """Run a workspace evaluation dataset and persist a structured run record."""
    dataset = cases if cases is not None else load_dataset(workspace_id)
    results = []
    summary = {"total": 0, "passed": 0}
    # Choose evaluator based on environment: 'llm' to enable LLMEvaluator when GROQ_API_KEY set, otherwise rule-based
    mode = os.getenv("EVALUATOR_MODE", "rule").lower()
    if mode == "llm":
        evaluator = LLMEvaluator()
    else:
        evaluator = RuleBasedEvaluator()
    for case in dataset:
        evaluation = {"case_id": case.get("id"), "question": case.get("question")}
        run = instrument_query(
            vectorstore, chain, memory, workspace_id, case.get("question", ""), top_k=top_k
        )
        metrics = evaluate_retrieval_case(run["retrieved"], case, k=top_k)
        evaluation.update({"metrics": metrics, "trace": run["trace"]})
        # Answer-level evaluation
        expected_answer = case.get("expected_answer")
        generated = None
        try:
            generated = run.get("result", {}).get("answer") or run.get("result", {}).get("output")
        except Exception:
            generated = None
        if expected_answer:
            answer_eval = evaluator.evaluate_answer(expected_answer, generated)
            evaluation["answer_evaluation"] = answer_eval

        # Citation evaluation: check expected source presence in retrieved documents or returned source_documents
        expected_sources = case.get("expected_sources") or (
            [case.get("expected_source")] if case.get("expected_source") else []
        )
        # Build structured retrieved source info: source, page, excerpt
        retrieved_sources_struct = []
        try:
            for doc, _ in run.get("retrieved", []):
                meta = getattr(doc, "metadata", {})
                src = meta.get("source") or meta.get("document_name")
                page = meta.get("page") if meta.get("page") is not None else None
                excerpt = doc.page_content[:360] if getattr(doc, "page_content", None) else ""
                if src:
                    retrieved_sources_struct.append(
                        {"source": src, "page": page, "excerpt": excerpt}
                    )
            # Also check chain-returned source_documents if present
            for sd in run.get("result", {}).get("source_documents") or []:
                meta = (
                    getattr(sd, "metadata", {})
                    if hasattr(sd, "metadata")
                    else sd.get("metadata", {})
                )
                src = meta.get("source") or meta.get("document_name")
                page = meta.get("page") if meta.get("page") is not None else None
                excerpt = (
                    sd.page_content[:360]
                    if getattr(sd, "page_content", None)
                    else sd.get("page_content", "") or ""
                )
                if src:
                    retrieved_sources_struct.append(
                        {"source": src, "page": page, "excerpt": excerpt}
                    )
        except Exception:
            retrieved_sources_struct = retrieved_sources_struct
        if expected_sources:
            # normalize expected into dicts with pages and optional excerpt
            expected_struct = []
            for es in expected_sources:
                if isinstance(es, dict):
                    expected_struct.append(es)
                else:
                    expected_struct.append(
                        {
                            "source": es,
                            "pages": case.get("expected_pages") or [],
                            "excerpt": case.get("expected_excerpt") or "",
                        }
                    )
            citation_eval = evaluator.evaluate_citations(expected_struct, retrieved_sources_struct)
            evaluation["citation_evaluation"] = citation_eval
        results.append(evaluation)
        summary["total"] += 1
        if metrics.get("hit"):
            summary["passed"] += 1

    run_record = {
        "id": str(uuid.uuid4()),
        "workspace_id": workspace_id,
        "created_at": int(time.time()),
        "summary": summary,
        "results": results,
    }
    # persist run
    runs_path = os.path.join(EVAL_DIR, f"{workspace_id}.runs.jsonl")
    with open(runs_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(run_record, ensure_ascii=False) + "\n")
    return run_record


def load_runs(workspace_id: str) -> List[Dict]:
    """Load past evaluation run records for a workspace (most recent last)."""
    runs_path = os.path.join(EVAL_DIR, f"{workspace_id}.runs.jsonl")
    runs = []
    try:
        with open(runs_path, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    runs.append(json.loads(line))
                except Exception:
                    continue
    except FileNotFoundError:
        pass
    return runs


def compare_runs(
    workspace_id: str, run_id_a: str, run_id_b: str, thresholds: Optional[Dict] = None
) -> Dict:
    """Compare two persisted runs and return summary deltas and per-case diffs.

    Returns a dict with: run_a, run_b, summary_a, summary_b, deltas, warnings, per_case list
    """
    runs = load_runs(workspace_id)
    run_map = {r.get("id"): r for r in runs}
    a = run_map.get(run_id_a)
    b = run_map.get(run_id_b)
    if not a or not b:
        raise ValueError("One or both run ids not found")

    def agg(run):
        results = run.get("results", [])
        total = len(results)
        hits = sum(1 for r in results if r.get("metrics", {}).get("hit"))
        mrrs = [
            r.get("metrics", {}).get("mrr")
            for r in results
            if isinstance(r.get("metrics", {}).get("mrr"), (int, float))
        ]
        traces = [r.get("trace", {}) for r in results]
        avg_total = int(sum(int(t.get("total_ms", 0)) for t in traces) / total) if total else 0
        return {
            "total_cases": total,
            "passed": hits,
            "hit_rate": round(hits / total, 3) if total else None,
            "mrr": round(sum(mrrs) / len(mrrs), 3) if mrrs else None,
            "avg_total_ms": avg_total,
        }

    summary_a = agg(a)
    summary_b = agg(b)

    # deltas: b - a
    deltas = {}
    for key in ("hit_rate", "mrr", "avg_total_ms"):
        va = summary_a.get(key) or 0
        vb = summary_b.get(key) or 0
        # for rates, show absolute delta; for ms show percent change
        if key.endswith("_ms"):
            delta = None
            try:
                delta = round(((vb - va) / va) if va else float("inf"), 3)
            except Exception:
                delta = None
        else:
            delta = round(vb - va, 3) if va is not None and vb is not None else None
        deltas[key] = delta

    # warnings heuristics with configurable thresholds
    thr = thresholds or {}
    hit_drop_thr = float(thr.get("hit_drop", 0.05))
    mrr_drop_thr = float(thr.get("mrr_drop", 0.05))
    latency_increase_thr = float(thr.get("latency_increase", 0.2))
    warnings = []
    alerts = []
    gate_pass = True
    if summary_a.get("hit_rate") is not None and summary_b.get("hit_rate") is not None:
        hit_delta = summary_b["hit_rate"] - summary_a["hit_rate"]
        if hit_delta <= -hit_drop_thr:
            warnings.append(f"Hit rate dropped by {round(hit_delta, 3)} (threshold {hit_drop_thr})")
            alerts.append("hit_rate")
            gate_pass = False
    if summary_a.get("mrr") is not None and summary_b.get("mrr") is not None:
        mrr_delta = summary_b["mrr"] - summary_a["mrr"]
        if mrr_delta <= -mrr_drop_thr:
            warnings.append(f"MRR dropped by {round(mrr_delta, 3)} (threshold {mrr_drop_thr})")
            alerts.append("mrr")
            gate_pass = False
    # latency increase
    try:
        va = summary_a.get("avg_total_ms") or 0
        vb = summary_b.get("avg_total_ms") or 0
        if va and ((vb - va) / va) >= latency_increase_thr:
            pct = round(((vb - va) / va), 3)
            warnings.append(
                f"Average total latency increased by {pct} (threshold {latency_increase_thr})"
            )
            alerts.append("latency")
            gate_pass = False
    except Exception:
        pass

    # per-case diffs
    per_case = []
    cases_a = {r.get("case_id"): r for r in a.get("results", [])}
    cases_b = {r.get("case_id"): r for r in b.get("results", [])}
    all_case_ids = set(cases_a) | set(cases_b)
    for cid in sorted(all_case_ids):
        ra = cases_a.get(cid)
        rb = cases_b.get(cid)
        entry = {"case_id": cid, "question": (rb or ra or {}).get("question")}
        ma = ra.get("metrics") if ra else {}
        mb = rb.get("metrics") if rb else {}
        entry.update(
            {
                "a": {
                    "hit": ma.get("hit"),
                    "mrr": ma.get("mrr"),
                    "avg_retrieval_score": ma.get("avg_retrieval_score"),
                },
                "b": {
                    "hit": mb.get("hit"),
                    "mrr": mb.get("mrr"),
                    "avg_retrieval_score": mb.get("avg_retrieval_score"),
                },
            }
        )
        # compute simple deltas for numeric fields
        try:
            entry["delta_hit"] = (1 if mb.get("hit") else 0) - (1 if ma.get("hit") else 0)
        except Exception:
            entry["delta_hit"] = None
        try:
            entry["delta_mrr"] = (
                None
                if ma.get("mrr") is None or mb.get("mrr") is None
                else round(float(mb.get("mrr")) - float(ma.get("mrr")), 3)
            )
        except Exception:
            entry["delta_mrr"] = None
        per_case.append(entry)

    return {
        "run_a": {"id": a.get("id"), "created_at": a.get("created_at")},
        "run_b": {"id": b.get("id"), "created_at": b.get("created_at")},
        "summary_a": summary_a,
        "summary_b": summary_b,
        "deltas": deltas,
        "warnings": warnings,
        "alerts": alerts,
        "gate_pass": gate_pass,
        "per_case": per_case,
    }
