"""Safe, transparent query interpretation without revealing private reasoning."""
import re

FOLLOW_UP = re.compile(r"\b(it|its|they|them|that|this|those|these|former|latter)\b", re.I)


def rewrite_query(question, history):
    """Expand an obvious follow-up with the last user question for retrieval."""
    earlier = [item.get("content", "") for item in history if item.get("role") == "user"]
    if not earlier or not FOLLOW_UP.search(question):
        return question.strip()
    return f"Context: {earlier[-1][:280]}\nFollow-up: {question.strip()}"
