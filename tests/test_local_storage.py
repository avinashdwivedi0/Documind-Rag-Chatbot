import unittest
from types import SimpleNamespace
from unittest.mock import patch

from backend import chat_history, workspace_store
from backend.report_generator import build_report
from backend.document_intelligence import build_knowledge_graph, extract_intelligence


class LocalStorageTests(unittest.TestCase):
    def test_turn_is_scoped_to_its_conversation(self):
        with patch.object(chat_history, "load_history", return_value=[]) as load, patch.object(chat_history, "save_history") as save:
            message = chat_history.append_turn("workspace", "chat-1", "user", "Hello")
        load.assert_called_once_with("workspace", "chat-1")
        save.assert_called_once_with("workspace", [message], "chat-1")
        self.assertEqual(message["content"], "Hello")

    def test_workspace_metadata_and_conversations(self):
        files = [SimpleNamespace(name="notes.pdf", size=123)]
        with patch.object(workspace_store, "get_workspace", return_value=None), patch.object(workspace_store, "_write"):
            workspace = workspace_store.upsert_workspace("abc123", files, "Notes")
        self.assertEqual(workspace["name"], "Notes")
        self.assertEqual(workspace["documents"][0]["name"], "notes.pdf")

        with patch.object(workspace_store, "get_workspace", return_value=workspace), patch.object(workspace_store, "_write"):
            chat = workspace_store.create_conversation("abc123", "Follow up")
            workspace_store.rename_conversation("abc123", chat["id"], "Review")
        self.assertEqual(workspace["conversations"][1]["title"], "Review")

    def test_report_contains_conversation_and_evidence(self):
        report = build_report(
            "Research", [{"role": "user", "content": "What changed?"}],
            [{"source": "notes.pdf", "page": "2", "excerpt": "A relevant finding."}],
        )
        self.assertIn("# Research", report)
        self.assertIn("What changed?", report)
        self.assertIn("notes.pdf — page 2", report)

    def test_document_intelligence_and_graph(self):
        intelligence = extract_intelligence("Acme Corporation acquired River Labs in London.", "deal.pdf")
        self.assertIn("Acme Corporation", intelligence["entities"])
        graph = build_knowledge_graph([{"name": "deal.pdf", "intelligence": intelligence}])
        self.assertIn("mentions", graph)


if __name__ == "__main__":
    unittest.main()
