"""File handling and document processing module.

This module provides functions for:
- Uploading and hashing files
- Loading documents from various formats (PDF, DOCX, CSV, TXT)
- Splitting documents into chunks
- Creating and managing FAISS vector stores
- Embedding documents using HuggingFace models
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any, List, Tuple

import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    CSVLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from backend.config import UPLOAD_DIR, VECTOR_DIR
from backend.core.exceptions import (
    FileProcessingException,
    FileTooLargeException,
    UnsupportedFileFormatException,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants
MAX_FILE_SIZE_MB = 50
SUPPORTED_FORMATS = {".pdf", ".docx", ".csv", ".txt"}
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def get_file_hash(file_list: List[Any]) -> str:
    """Compute MD5 hash of file contents.

    Args:
        file_list: List of uploaded file objects

    Returns:
        MD5 hash of combined file contents

    Raises:
        FileProcessingException: If hashing fails
    """
    hash_md5 = hashlib.md5()
    try:
        for file in file_list:
            content = file.read()
            file.seek(0)
            hash_md5.update(content)
    except Exception as e:
        logger.error(f"Error hashing files: {e}")
        raise FileProcessingException("Unable to hash files", details={"error": str(e)})
    return hash_md5.hexdigest()


def save_uploaded_files(file_list: List[Any], file_hash: str) -> List[str]:
    """Save uploaded files to disk.

    Args:
        file_list: List of uploaded file objects from Streamlit
        file_hash: Hash identifier for the file set

    Returns:
        List of saved file paths

    Raises:
        FileTooLargeException: If any file exceeds size limit
        FileProcessingException: If saving fails
    """
    save_dir = os.path.join(UPLOAD_DIR, file_hash)
    os.makedirs(save_dir, exist_ok=True)
    saved_paths: List[str] = []
    try:
        for file in file_list:
            if not file.name or file.size == 0:
                raise FileProcessingException(
                    f"File '{file.name or 'unknown'}' is empty",
                    details={"file_name": file.name, "size": file.size},
                )

            # Check file size (in bytes, convert MAX_FILE_SIZE_MB to bytes)
            if file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
                raise FileTooLargeException(file.name, file.size / (1024 * 1024), MAX_FILE_SIZE_MB)

            file_path = os.path.join(save_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.read())
            file.seek(0)
            saved_paths.append(file_path)
        logger.info(f"Saved {len(saved_paths)} files to {save_dir}")
    except (FileTooLargeException, FileProcessingException):
        raise
    except Exception as e:
        logger.error(f"Error saving files: {e}")
        raise FileProcessingException("Failed to save files", details={"error": str(e)})
    return saved_paths


def load_documents_from_paths(paths: List[str]) -> List[Document]:
    """Load documents from file paths.

    Args:
        paths: List of file paths to load

    Returns:
        List of Document objects with metadata

    Raises:
        UnsupportedFileFormatException: If file format not supported
        FileProcessingException: If loading fails
    """
    documents: List[Document] = []
    for path in paths:
        try:
            # Check file format
            file_ext = os.path.splitext(path)[1].lower()
            if file_ext not in SUPPORTED_FORMATS:
                raise UnsupportedFileFormatException(file_ext, list(SUPPORTED_FORMATS), path)

            if path.endswith(".pdf"):
                loader = PyPDFLoader(path)
            elif path.endswith(".docx"):
                loader = UnstructuredWordDocumentLoader(path)
            elif path.endswith(".csv"):
                loader = CSVLoader(path)
            else:
                loader = TextLoader(path)

            loaded: List[Document] = loader.load()
            for document in loaded:
                document.metadata.update(
                    {
                        "document_name": os.path.basename(path),
                        "document_type": file_ext.lstrip(".").upper(),
                    }
                )
            documents.extend(loaded)
            logger.info(f"Loaded documents from {path}")
        except UnsupportedFileFormatException:
            raise
        except Exception as e:
            logger.error(f"Error loading document {path}: {e}")
            raise FileProcessingException(
                f"Failed to load document: {path}", details={"path": path, "error": str(e)}
            )
    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    """Split documents into chunks.

    Args:
        documents: List of documents to split

    Returns:
        List of split document chunks with chunk_id metadata

    Raises:
        FileProcessingException: If splitting fails
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    try:
        splits: List[Document] = splitter.split_documents(documents)
        for index, split in enumerate(splits):
            split.metadata["chunk_id"] = f"chunk-{index + 1}"
        logger.info(f"Split {len(documents)} documents into {len(splits)} chunks")
        return splits
    except Exception as e:
        logger.error(f"Error splitting documents: {e}")
        raise FileProcessingException("Failed to split documents", details={"error": str(e)})


def embed_documents(splits: List[Document], save_path: str) -> FAISS:
    """Embed documents and create FAISS index.

    Args:
        splits: List of document chunks to embed
        save_path: Path to save FAISS index

    Returns:
        FAISS vectorstore instance

    Raises:
        FileProcessingException: If embedding fails
    """
    try:
        embeddings = get_embeddings()
        vectorstore: FAISS = FAISS.from_documents(splits, embeddings)
        vectorstore.save_local(save_path)
        logger.info(f"Saved FAISS index at {save_path}")
        return vectorstore
    except Exception as e:
        logger.error(f"Error embedding documents: {e}")
        raise FileProcessingException(
            "Failed to embed documents", details={"save_path": save_path, "error": str(e)}
        )


def load_or_create_vectorstore(file_list: List[Any]) -> Tuple[FAISS, ConversationBufferMemory, str]:
    """Load or create a FAISS vectorstore from uploaded files.

    Args:
        file_list: List of uploaded files

    Returns:
        Tuple of (vectorstore, memory, file_hash)

    Raises:
        FileTooLargeException: If any file too large
        FileProcessingException: If processing fails
    """
    try:
        file_hash: str = get_file_hash(file_list)
        index_path: str = os.path.join(VECTOR_DIR, file_hash)

        if os.path.exists(index_path):
            embeddings: HuggingFaceEmbeddings = get_embeddings()
            vectorstore: FAISS = FAISS.load_local(
                index_path, embeddings, allow_dangerous_deserialization=True
            )
            logger.info(f"Loaded existing FAISS index for hash {file_hash}")
        else:
            saved_paths: List[str] = save_uploaded_files(file_list, file_hash)
            raw_docs: List[Document] = load_documents_from_paths(saved_paths)
            splits: List[Document] = split_documents(raw_docs)
            vectorstore = embed_documents(splits, index_path)

        memory: ConversationBufferMemory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True, output_key="answer"
        )
        return vectorstore, memory, file_hash
    except (FileTooLargeException, FileProcessingException):
        raise
    except Exception as e:
        logger.error(f"Failed to load or create vectorstore: {e}")
        raise FileProcessingException(
            "Failed to process files and create vectorstore", details={"error": str(e)}
        )


def load_vectorstore(workspace_id: str) -> FAISS:
    """Load an existing vectorstore by workspace ID.

    Args:
        workspace_id: Hexadecimal workspace identifier

    Returns:
        FAISS vectorstore instance

    Raises:
        FileProcessingException: If workspace ID invalid
        FileNotFoundError: If index not found
    """
    if not workspace_id or any(character not in "0123456789abcdef" for character in workspace_id):
        raise FileProcessingException(
            "Invalid workspace identifier", details={"workspace_id": workspace_id}
        )
    index_path: str = os.path.join(VECTOR_DIR, workspace_id)
    if not os.path.isdir(index_path):
        raise FileNotFoundError(f"Vector index not found for workspace: {workspace_id}")
    return FAISS.load_local(index_path, get_embeddings(), allow_dangerous_deserialization=True)


def create_memory() -> ConversationBufferMemory:
    """Create a fresh conversation memory buffer.

    Returns:
        ConversationBufferMemory instance
    """
    return ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key="answer"
    )


@st.cache_resource(show_spinner="Loading the embedding model…")
def get_embeddings() -> HuggingFaceEmbeddings:
    """Load HuggingFace embedding model.

    Uses Streamlit caching to load the model once per server session.
    This significantly improves performance for repeated calls.

    Returns:
        HuggingFaceEmbeddings instance with all-MiniLM-L6-v2 model

    Example:
        >>> embeddings = get_embeddings()
        >>> vector = embeddings.embed_query("Hello world")
    """
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
