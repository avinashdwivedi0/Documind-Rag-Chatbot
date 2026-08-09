# Document Copilot

A local-first Streamlit RAG assistant for asking grounded questions about PDF, DOCX, TXT, and CSV files. It uses FAISS for retrieval and Groq for responses.

## Highlights

- Persistent document workspaces: upload once, then reopen the indexed workspace later.
- Named conversations with locally cached history, export, clear, and deletion controls.
- Document-grounded response prompt that says when information is not in the uploaded files.
- Source filename, PDF page number, and retrieval relevance shown with each live answer.
- Cached embedding model, file-size checks, feedback storage, comfort-mode styling, and local analytics.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Set your Groq key in `.env`:

```env
GROQ_API_KEY=your_key_here
```

Run the app:

```powershell
streamlit run main.py
```

## How persistence works

All generated data stays on the machine running the app:

| Data | Location |
| --- | --- |
| Uploaded files | `data/uploads/` |
| FAISS indexes | `data/vectors/` |
| Workspaces and chats | `data/workspaces/`, `data/chat_history/` |
| Answer feedback | `data/feedback.jsonl` |

These paths are ignored by Git. Use **Delete workspace data** in the sidebar to remove a workspace, its files, index, and conversations.

## Testing

```powershell
python -m unittest discover -s tests
```

## Evaluation regression CI

A dedicated CI regression step validates the evaluation engine and compares two synthetic runs using the same evaluation logic.

```powershell
python scripts/ci_eval_regression.py
```

If the comparison gate fails, the script exits non-zero and the pipeline fails.

## Project structure

```text
backend/
  chat_history.py       # Persistent conversation cache
  feedback_store.py     # Feedback and basic analytics
  file_handler.py       # File loading, embeddings, FAISS index
  rag_chain.py          # Groq conversational retrieval chain
  workspace_store.py    # Workspace and conversation metadata
frontend/ui.py          # Streamlit interface
```
