# backend/rag_chain.py
"""RAG chain creation and LLM configuration module.

This module provides functions to create and configure conversational retrieval chains
with Groq LLM backend. Handles API key validation, model fallback, and response mode
configuration.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Literal, Optional

import streamlit as st
from dotenv import load_dotenv

try:
    from langchain.chains import ConversationalRetrievalChain
except ImportError:  # pragma: no cover - compatibility for newer LangChain versions
    from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from backend.core.exceptions import APIKeyMissingException, ModelUnavailableException
from backend.retrieval import HybridRetriever

# Load environment variables from .env
load_dotenv()

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load the API key from environment variables first, then fallback to Streamlit secrets if available.
GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
try:
    if not GROQ_API_KEY:
        GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
except Exception:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Tunable defaults for Groq requests.
# This API key exposes a different model catalog than the stale llama IDs we previously used,
# so we must align to the currently available Groq model names for this account.
DEFAULT_GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
DEFAULT_LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))
DEFAULT_RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "3"))
DEFAULT_RETRIEVAL_CANDIDATE_K: int = int(os.getenv("RETRIEVAL_CANDIDATE_K", "8"))
DEFAULT_CONTEXT_TRUNCATION_CHARS: int = int(os.getenv("CONTEXT_TRUNCATION_CHARS", "2048"))

FALLBACK_GROQ_MODEL: str = "openai/gpt-oss-20b"

# Response mode type hint
ResponseMode = Literal[
    "Deep research", "Executive brief", "Compare viewpoints", "Study guide", "Action plan"
]


def create_chat_chain(
    vectorstore: Any,
    memory: Any,
    response_mode: ResponseMode = "Deep research",  # type: ignore
) -> ConversationalRetrievalChain:
    """Create a conversational retrieval chain with Groq LLM.

    Args:
        vectorstore: FAISS vectorstore instance with embedded documents
        memory: Conversation memory buffer for maintaining chat history
        response_mode: Response generation mode (default: "Deep research")

    Returns:
        ConversationalRetrievalChain configured with LLM, retriever, and memory

    Raises:
        APIKeyMissingException: If GROQ_API_KEY is not configured
        ModelUnavailableException: If neither default nor fallback model is available

    Example:
        >>> chain = create_chat_chain(vectorstore, memory, response_mode="Executive brief")
        >>> response = chain({"question": "What is AI?"})
    """
    try:
        if not GROQ_API_KEY:
            raise APIKeyMissingException("GROQ_API_KEY")

        model_name: str = DEFAULT_GROQ_MODEL
        logger.info(f"Using Groq model: {model_name}")
        try:
            llm = ChatGroq(
                model_name=model_name,
                temperature=0,
                max_tokens=DEFAULT_LLM_MAX_TOKENS,
                api_key=GROQ_API_KEY,  # ✅ Correct argument name for ChatGroq
            )
        except Exception as e:
            message: str = str(e).lower()
            if "does not exist" in message or "model not found" in message:
                logger.warning(
                    f"Groq model {model_name} unavailable; falling back to {FALLBACK_GROQ_MODEL}"
                )
                model_name = FALLBACK_GROQ_MODEL
                try:
                    llm = ChatGroq(
                        model_name=model_name,
                        temperature=0,
                        max_tokens=DEFAULT_LLM_MAX_TOKENS,
                        api_key=GROQ_API_KEY,
                    )
                except Exception as fallback_error:
                    raise ModelUnavailableException(
                        DEFAULT_GROQ_MODEL, [FALLBACK_GROQ_MODEL], str(fallback_error)
                    )
            else:
                raise

        mode_instructions: dict[str, str] = {
            "Deep research": "Explain the topic in depth, including mechanisms, nuance, implications, and trade-offs.",
            "Executive brief": "Lead with the decision-relevant conclusion, then give concise findings, risks, and next actions.",
            "Compare viewpoints": "Identify perspectives, agreements, differences, evidence, and unresolved tensions.",
            "Study guide": "Teach step by step with definitions, examples, key takeaways, and a short self-check.",
            "Action plan": "Turn findings into prioritised actions, owners or stakeholders where known, risks, and next steps.",
        }
        answer_prompt = PromptTemplate.from_template(
            """You are Document Copilot, a friendly and helpful research assistant. Answer naturally like a chatbot: start with a direct reply, keep the tone clear and conversational, and use short paragraphs.
Act as an expert teacher and research assistant. Use plain language, helpful examples, and concise summaries. Avoid overly formal or technical prose unless the user specifically asks for it.

Response mode: {mode_instruction}

Use only the text provided in the Context. Do not use any outside knowledge, web search results, or general domain expertise. Every claim in your answer must be supported by the context. If you cannot answer the question from the context alone, answer exactly: "I couldn't find relevant information in the uploaded documents."

Answer every question only from the documents. After the answer, include a section titled **What your documents say** that cites only the context text or document source metadata. If the context does not contain evidence, do not invent an answer.

First answer the user's question directly in one or two sentences, and complete every sentence fully. If the answer needs more detail, continue with a second paragraph.

Never present general knowledge as though it came from the uploaded documents. Do not invent document facts, quotes, or sources. Clearly identify uncertainty, assumptions, disagreements, or missing information.

Context:
{context}

Question: {question}
Helpful answer:"""
        )
        answer_prompt = answer_prompt.partial(
            mode_instruction=mode_instructions.get(
                response_mode, mode_instructions["Deep research"]
            )
        )
        qa_chain: ConversationalRetrievalChain = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=HybridRetriever(
                vectorstore=vectorstore,
                top_k=DEFAULT_RETRIEVAL_TOP_K,
                candidate_k=DEFAULT_RETRIEVAL_CANDIDATE_K,
                max_context_chars=DEFAULT_CONTEXT_TRUNCATION_CHARS,
            ),
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": answer_prompt},
        )

        logger.info("✅ Chat chain successfully created.")
        return qa_chain

    except APIKeyMissingException:
        logger.error("❌ GROQ_API_KEY not configured")
        raise
    except ModelUnavailableException:
        logger.error("❌ No available Groq models")
        raise
    except Exception as e:
        logger.error(f"❌ Failed to create chat chain: {e}")
        raise
