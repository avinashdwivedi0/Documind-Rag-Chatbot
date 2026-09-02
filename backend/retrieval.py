"""Hybrid retrieval: FAISS semantic search plus dependency-free BM25-style ranking."""

import re
from collections import Counter
from typing import Any, List

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]{2,}", text.lower())


class HybridRetriever(BaseRetriever):
    """Fuses semantic and keyword candidates with weighted reciprocal rank fusion."""

    vectorstore: Any
    top_k: int = 2
    candidate_k: int = 6
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3
    max_context_chars: int = 640

    def _truncate_content(self, content: str) -> str:
        if content is None:
            return ""
        return (
            content
            if len(content) <= self.max_context_chars
            else content[: self.max_context_chars].rsplit(" ", 1)[0] + "..."
        )

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        semantic = self.vectorstore.similarity_search_with_relevance_scores(
            query, k=self.candidate_k
        )
        documents = list(getattr(self.vectorstore.docstore, "_dict", {}).values())
        terms = tokenize(query)
        keyword_scores = []
        for document in documents:
            counts = Counter(tokenize(document.page_content))
            score = sum(counts[term] / (1 + counts[term]) for term in terms)
            if score:
                keyword_scores.append((document, score))
        keyword_scores.sort(key=lambda item: item[1], reverse=True)

        fused = {}
        for rank, (document, score) in enumerate(semantic, 1):
            fused[id(document)] = [document, self.semantic_weight / (60 + rank), score]
        for rank, (document, _) in enumerate(keyword_scores[: self.candidate_k], 1):
            entry = fused.setdefault(id(document), [document, 0.0, 0.0])
            entry[1] += self.keyword_weight / (60 + rank)
        ranked = sorted(fused.values(), key=lambda item: item[1], reverse=True)[: self.top_k]
        results = []
        for document, _, semantic_score in ranked:
            metadata = dict(document.metadata)
            metadata["relevance_score"] = round(float(semantic_score), 3)
            truncated = self._truncate_content(document.page_content)
            results.append(Document(page_content=truncated, metadata=metadata))
        return results
