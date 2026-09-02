"""Safe, transparent query interpretation and rewriting module.

This module provides functions for rewriting user queries to handle follow-up
questions more effectively by expanding pronouns with context from conversation history.
"""

from __future__ import annotations

import re
from typing import Any, List

FOLLOW_UP: re.Pattern[str] = re.compile(
    r"\b(it|its|they|them|that|this|those|these|former|latter)\b", re.IGNORECASE
)


def rewrite_query(question: str, history: List[dict[str, Any]]) -> str:
    """Rewrite query to expand follow-up questions with context.

    Detects when a user's question contains follow-up pronouns (it, they, this, etc.)
    and expands the query by including the previous user message for better retrieval.

    Args:
        question: The current user question/query
        history: List of chat history items with 'role' and 'content' keys

    Returns:
        Original or expanded query string (stripped of extra whitespace)

    Example:
        >>> history = [
        ...     {"role": "user", "content": "What is climate change?"},
        ...     {"role": "assistant", "content": "Climate change is..."},
        ... ]
        >>> question = "What are its causes?"
        >>> rewrite_query(question, history)
        'Context: What is climate change?\\nFollow-up: What are its causes?'

        >>> question = "Tell me more"
        >>> rewrite_query(question, history)
        'Context: What is climate change?\\nFollow-up: Tell me more'
    """
    # Extract all user messages from history
    earlier: List[str] = [item.get("content", "") for item in history if item.get("role") == "user"]

    # If no history or no follow-up pronouns, return original question
    if not earlier or not FOLLOW_UP.search(question):
        return question.strip()

    # Expand follow-up with previous context (limit to first 280 chars for brevity)
    return f"Context: {earlier[-1][:280]}\nFollow-up: {question.strip()}"
