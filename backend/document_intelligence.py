"""Local, explainable document metadata extraction without external services."""

import re
from collections import Counter
from pathlib import Path
from typing import Dict, List


STOP_WORDS = {"this", "that", "with", "from", "have", "will", "into", "their", "about", "which", "were", "been", "than", "also", "document", "page", "your", "they", "when", "where"}


def extract_intelligence(text: str, filename: str) -> dict:
    words = re.findall(r"[A-Za-z]{4,}", text.lower())
    topics = [word for word, _ in Counter(word for word in words if word not in STOP_WORDS).most_common(8)]
    entities = []
    for entity in re.findall(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b", text):
        if entity not in entities:
            entities.append(entity)
    return {
        "title": Path(filename).stem.replace("_", " "),
        "topics": topics,
        "entities": entities[:12],
        "estimated_words": len(words),
    }


def build_knowledge_graph(documents: List[Dict]) -> str:
    """Build a compact Graphviz graph from persisted document intelligence."""
    lines = ["digraph G {", "rankdir=LR;", "bgcolor=transparent;", "node [fontname=Arial];"]
    for document in documents:
        document_name = document.get("name", "Document")
        document_id = "doc_" + str(abs(hash(document_name)))
        safe_name = document_name.replace('"', "'")[:34]
        lines.append(f'{document_id} [label="{safe_name}", shape=box, style="rounded,filled", fillcolor="#263451", fontcolor="#f7f9ff"];')
        for entity in document.get("intelligence", {}).get("entities", [])[:6]:
            entity_id = "entity_" + str(abs(hash(entity)))
            safe_entity = entity.replace('"', "'")[:30]
            lines.append(f'{entity_id} [label="{safe_entity}", shape=ellipse, style=filled, fillcolor="#7258ff", fontcolor="#ffffff"];')
            lines.append(f"{document_id} -> {entity_id} [color=\"#8da0c7\", label=\"mentions\", fontcolor=\"#8da0c7\"];")
    lines.append("}")
    return "\n".join(lines)
