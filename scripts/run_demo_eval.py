import json
import pathlib
import sys
import tempfile

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import os

from backend import evaluation_service as es
from backend.config import EVAL_DIR
from backend.file_handler import create_memory, load_vectorstore
from backend.rag_chain import create_chat_chain

workspace_id = "d6d15051ecc3e87b784fe197b6ec8168"
# Add a demo case
case = {
    "question": "What is the topic of the document?",
    "expected_sources": ["MLT_Unit-1_Introduction[1].pdf"],
}
print("Adding case...")
os.makedirs(EVAL_DIR, exist_ok=True)
es.add_case(workspace_id, case)
print("Loading vectorstore...")
vectorstore = load_vectorstore(workspace_id)
memory = create_memory()
print("Creating chain...")
chain = create_chat_chain(vectorstore, memory)
print("Running evaluation dataset...")
run = es.run_evaluation_dataset(workspace_id, vectorstore, chain, memory, top_k=4)
print("Run summary:")
print(json.dumps(run["summary"], indent=2))
print("Saved run id:", run["id"])
