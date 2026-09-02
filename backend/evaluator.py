"""Evaluator abstraction for answer and citation evaluation.

Provides a simple rule-based evaluator and a placeholder LLM-backed evaluator interface.
Implementations must return a dict with at least: { 'score': float(0-1), 'label': str, 'reason': str }

Do NOT log sensitive content; evaluators return short reasons only.
"""
import json
import os
import re
from typing import Dict, Optional

import requests


class BaseEvaluator:
    def evaluate_answer(self, expected: str, generated: str) -> Dict:
        raise NotImplementedError()

    def evaluate_citations(
        self, expected_sources: Optional[list], retrieved_sources: Optional[list]
    ) -> Dict:
        raise NotImplementedError()


class RuleBasedEvaluator(BaseEvaluator):
    """Lightweight evaluator based on token overlap and simple heuristics."""

    def _tokens(self, text: str):
        return re.findall(r"\w+", (text or "").lower())

    def evaluate_answer(self, expected: str, generated: str) -> Dict:
        if not expected:
            return {"score": 0.0, "label": "Not available", "reason": "No expected answer provided"}
        if not generated:
            return {"score": 0.0, "label": "Unsupported", "reason": "No generated answer"}
        exp_tokens = set(self._tokens(expected))
        gen_tokens = set(self._tokens(generated))
        if not exp_tokens:
            return {
                "score": 0.0,
                "label": "Not available",
                "reason": "Expected answer had no tokens",
            }
        overlap = len(exp_tokens & gen_tokens)
        score = overlap / len(exp_tokens)
        if score >= 0.6:
            label = "Supported"
        elif score >= 0.3:
            label = "Partially Supported"
        else:
            label = "Unsupported"
        reason = f"Token overlap {overlap}/{len(exp_tokens)}"
        return {"score": round(float(score), 3), "label": label, "reason": reason}

    def evaluate_citations(
        self, expected_sources: Optional[list], retrieved_sources: Optional[list]
    ) -> Dict:
        # Support structured expected/retrieved sources
        # expected_sources: list[str] or list[dict]{'source':str,'pages':[int],'excerpt':str}
        # retrieved_sources: list[str] or list[dict]{'source':str,'page':int,'excerpt':str}
        if not expected_sources:
            return {
                "filename_matches": 0,
                "filename_total": 0,
                "page_matches": 0,
                "page_total": 0,
                "excerpt_match_rate": "Not available",
                "overall_match_rate": "Not available",
            }

        def _norm_expected(es):
            out = []
            for e in es:
                if isinstance(e, dict):
                    out.append(
                        {
                            "source": os.path.basename(e.get("source", ""))
                            if e.get("source")
                            else "",
                            "pages": e.get("pages") or [],
                            "excerpt": e.get("excerpt") or "",
                        }
                    )
                else:
                    out.append({"source": os.path.basename(str(e)), "pages": [], "excerpt": ""})
            return out

        def _norm_retrieved(rs):
            out = []
            for r in rs or []:
                if isinstance(r, dict):
                    out.append(
                        {
                            "source": os.path.basename(r.get("source", ""))
                            if r.get("source")
                            else "",
                            "page": r.get("page"),
                            "excerpt": r.get("excerpt", ""),
                        }
                    )
                else:
                    out.append({"source": os.path.basename(str(r)), "page": None, "excerpt": ""})
            return out

        expected = _norm_expected(expected_sources)
        retrieved = _norm_retrieved(retrieved_sources)

        filename_total = len(expected)
        filename_matches = 0
        page_total = 0
        page_matches = 0
        excerpt_checks = 0
        excerpt_matches = 0

        def toks(text: str):
            return set(re.findall(r"\w+", (text or "").lower()))

        for e in expected:
            esrc = e.get("source")
            matched = [r for r in retrieved if r.get("source") == esrc]
            if matched:
                filename_matches += 1
            pages = e.get("pages") or []
            if pages:
                page_total += 1
                found = False
                for r in matched:
                    try:
                        if r.get("page") is not None and (int(r.get("page")) + 1) in [
                            int(p) for p in pages
                        ]:
                            found = True
                            break
                    except Exception:
                        continue
                if found:
                    page_matches += 1
            if e.get("excerpt"):
                excerpt_checks += 1
                et = toks(e.get("excerpt"))
                best = 0
                for r in matched:
                    rt = toks(r.get("excerpt", ""))
                    if not et:
                        continue
                    overlap = len(et & rt) / len(et)
                    if overlap > best:
                        best = overlap
                if best >= 0.25:
                    excerpt_matches += 1

        filename_rate = (
            round(filename_matches / filename_total, 3) if filename_total else "Not available"
        )
        page_rate = round(page_matches / page_total, 3) if page_total else "Not available"
        excerpt_rate = (
            round(excerpt_matches / excerpt_checks, 3) if excerpt_checks else "Not available"
        )

        rates = [r for r in (filename_rate, page_rate, excerpt_rate) if isinstance(r, (int, float))]
        overall = round(sum(rates) / len(rates), 3) if rates else "Not available"

        return {
            "filename_matches": filename_matches,
            "filename_total": filename_total,
            "filename_rate": filename_rate,
            "page_matches": page_matches,
            "page_total": page_total,
            "page_rate": page_rate,
            "excerpt_matches": excerpt_matches,
            "excerpt_checks": excerpt_checks,
            "excerpt_match_rate": excerpt_rate,
            "overall_match_rate": overall,
        }


class LLMEvaluator(BaseEvaluator):
    """Placeholder for an LLM-backed evaluator. Implementations should call an LLM and return a compact score and reason.

    For safety, the evaluator must NOT return chain-of-thought or full comparison text—only a short reason.
    """

    def __init__(self, llm=None):
        # llm argument kept for compatibility; prefer GROQ_API_KEY env for simple HTTP call
        self.llm = llm
        self.api_key = os.getenv("GROQ_API_KEY")
        self.enabled = bool(self.api_key)

    def evaluate_answer(self, expected: str, generated: str) -> Dict:
        if not self.enabled:
            return {
                "score": 0.0,
                "label": "Unavailable",
                "reason": "LLM evaluator not configured (missing GROQ_API_KEY)",
            }
        # Build a compact prompt asking for JSON-only output.
        prompt = (
            "You are an evaluator. Compare the EXPECTED answer and the GENERATED answer. "
            "Respond ONLY with a compact JSON object with keys: score (0-1), label (Supported|Partially Supported|Unsupported), reason (one short sentence). "
            "Do NOT include chain-of-thought or any extra commentary.\n\n"
            f"EXPECTED: {expected}\n\nGENERATED: {generated}\n"
        )
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": "llama-3.1-4b-instant",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 200,
                "temperature": 0,
            }
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            # Try to extract JSON from response
            text = text.strip()
            # If the model returned surrounding markdown or text, attempt to find first JSON block
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                text = text[start : end + 1]
            parsed = json.loads(text)
            # Normalize keys
            score = float(parsed.get("score", 0.0))
            label = parsed.get("label", "")
            reason = parsed.get("reason", "")
            return {"score": round(score, 3), "label": label, "reason": reason}
        except Exception:
            return {
                "score": 0.0,
                "label": "Unavailable",
                "reason": "LLM evaluation failed or returned unparsable output",
            }

    def evaluate_citations(
        self, expected_sources: Optional[list], retrieved_sources: Optional[list]
    ) -> Dict:
        return {"valid": 0, "total": 0, "match_rate": "Not available"}
