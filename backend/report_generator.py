"""Create portable, readable research reports from a workspace conversation."""

from datetime import datetime
from typing import Dict, Iterable, List


def build_report(workspace_name: str, messages: Iterable[Dict[str, str]], evidence: List[Dict[str, str]]) -> str:
    lines = [
        f"# {workspace_name}",
        "", 
        "## Research conversation report",
        f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}",
        "",
        "This report contains assistant analysis, document-grounded discussion, and clearly labeled source evidence."
        " Validate important conclusions against the original documents.",
        "",
        "## Conversation",
        "",
    ]
    for message in messages:
        role = "Question" if message.get("role") == "user" else "Analysis"
        lines.extend([f"### {role}", message.get("content", ""), ""])
    if evidence:
        lines.extend(["## Evidence consulted", ""])
        for index, item in enumerate(evidence, 1):
            location = f" — page {item['page']}" if item.get("page") else ""
            lines.extend([f"### {index}. {item['source']}{location}", item["excerpt"], ""])
    return "\n".join(lines)
