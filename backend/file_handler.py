import os
import hashlib 
import logging
from typing import List, Tuple

import streamlit as st

from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory

from backend.config import UPLOAD_DIR, VECTOR_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_file_hash(file_list: List) -> str:
    hash_md5 = hashlib.md5()
    try:
        for file in file_list:
            content = file.read()
            file.seek(0)
            hash_md5.update(content)
    except Exception as e:
        logger.error(f"Error hashing files: {e}")
        raise
    return hash_md5.hexdigest()


def save_uploaded_files(file_list: List, file_hash: str) -> List[str]:
    save_dir = os.path.join(UPLOAD_DIR, file_hash)
    os.makedirs(save_dir, exist_ok=True)
    saved_paths = []
    try:
        for file in file_list:
            if not file.name or file.size == 0:
                raise ValueError(f"{file.name or 'A file'} is empty.")
            file_path = os.path.join(save_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.read())
            file.seek(0)
            saved_paths.append(file_path)
        logger.info(f"Saved {len(saved_paths)} files to {save_dir}")
    except Exception as e:
        logger.error(f"Error saving files: {e}")
        raise
    return saved_paths


def load_documents_from_paths(paths: List[str]):
    documents = []
    for path in paths:
        try:
            if path.endswith(".pdf"):
                loader = PyPDFLoader(path)
            elif path.endswith(".docx"):
                loader = UnstructuredWordDocumentLoader(path)
            elif path.endswith(".csv"):
                loader = CSVLoader(path)
            else:
                loader = TextLoader(path)
            loaded = loader.load()
            for document in loaded:
                document.metadata.update({
                    "document_name": os.path.basename(path),
                    "document_type": os.path.splitext(path)[1].lstrip(".").upper(),
                })
            documents.extend(loaded)
            logger.info(f"Loaded documents from {path}")
        except Exception as e:
            logger.error(f"Error loading document {path}: {e}")
    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    try:
        splits = splitter.split_documents(documents)
        for index, split in enumerate(splits):
            split.metadata["chunk_id"] = f"chunk-{index + 1}"
        return splits
    except Exception as e:
        logger.error(f"Error splitting documents: {e}")
        raise


def embed_documents(splits, save_path: str):
    try:
        embeddings = get_embeddings()
        vectorstore = FAISS.from_documents(splits, embeddings)
        vectorstore.save_local(save_path)
        logger.info(f"Saved FAISS index at {save_path}")
        return vectorstore
    except Exception as e:
        logger.error(f"Error embedding documents: {e}")
        raise


def load_or_create_vectorstore(file_list: List) -> Tuple[FAISS, ConversationBufferMemory, str]:
    try:
        file_hash = get_file_hash(file_list)
        index_path = os.path.join(VECTOR_DIR, file_hash)

        if os.path.exists(index_path):
            embeddings = get_embeddings()
            vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
            logger.info(f"Loaded existing FAISS index for hash {file_hash}")
        else:
            saved_paths = save_uploaded_files(file_list, file_hash)
            raw_docs = load_documents_from_paths(saved_paths)
            splits = split_documents(raw_docs)
            vectorstore = embed_documents(splits, index_path)

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        return vectorstore, memory, file_hash
    except Exception as e:
        logger.error(f"Failed to load or create vectorstore: {e}")
        raise


def load_vectorstore(workspace_id: str) -> FAISS:
    """Open an existing local index without needing a new file upload."""
    if not workspace_id or any(character not in "0123456789abcdef" for character in workspace_id):
        raise ValueError("Invalid workspace identifier.")
    index_path = os.path.join(VECTOR_DIR, workspace_id)
    if not os.path.isdir(index_path):
        raise FileNotFoundError("The vector index for this workspace is unavailable.")
    return FAISS.load_local(index_path, get_embeddings(), allow_dangerous_deserialization=True)


def create_memory() -> ConversationBufferMemory:
    return ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
@st.cache_resource(show_spinner="Loading the embedding model…")
def get_embeddings():
    """Load the model once per Streamlit server instead of once per upload."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
