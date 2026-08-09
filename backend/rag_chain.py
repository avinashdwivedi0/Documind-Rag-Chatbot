# backend/rag_chain.py

import os
import logging
import streamlit as st   # ✅ Added this line
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from backend.retrieval import HybridRetriever

# Load environment variables from .env
load_dotenv()

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the API key from environment variables first, then fallback to Streamlit secrets if available.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
try:
    if not GROQ_API_KEY:
        GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
except Exception:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Tunable defaults for Groq requests.
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
DEFAULT_LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))
DEFAULT_RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "3"))
DEFAULT_RETRIEVAL_CANDIDATE_K = int(os.getenv("RETRIEVAL_CANDIDATE_K", "8"))
DEFAULT_CONTEXT_TRUNCATION_CHARS = int(os.getenv("CONTEXT_TRUNCATION_CHARS", "2048"))

FALLBACK_GROQ_MODEL = "llama-3.1-8b-instant"

def create_chat_chain(vectorstore, memory, response_mode: str = "Deep research"):
    try:
        if not GROQ_API_KEY:
            raise ValueError("❌ GROQ_API_KEY not found in environment variables or Streamlit secrets.")

        model_name = DEFAULT_GROQ_MODEL
        logger.info(f"Using Groq model: {model_name}")
        try:
            llm = ChatGroq(
                model_name=model_name,
                temperature=0,
                max_tokens=DEFAULT_LLM_MAX_TOKENS,
                api_key=GROQ_API_KEY  # ✅ Correct argument name for ChatGroq
            )
        except Exception as e:
            message = str(e).lower()
            if "does not exist" in message or "model not found" in message:
                logger.warning(f"Groq model {model_name} unavailable; falling back to {FALLBACK_GROQ_MODEL}")
                model_name = FALLBACK_GROQ_MODEL
                llm = ChatGroq(
                    model_name=model_name,
                    temperature=0,
                    max_tokens=DEFAULT_LLM_MAX_TOKENS,
                    api_key=GROQ_API_KEY
                )
            else:
                raise

        mode_instructions = {
            "Deep research": "Explain the topic in depth, including mechanisms, nuance, implications, and trade-offs.",
            "Executive brief": "Lead with the decision-relevant conclusion, then give concise findings, risks, and next actions.",
            "Compare viewpoints": "Identify perspectives, agreements, differences, evidence, and unresolved tensions.",
            "Study guide": "Teach step by step with definitions, examples, key takeaways, and a short self-check.",
            "Action plan": "Turn findings into prioritised actions, owners or stakeholders where known, risks, and next steps.",
        }
        answer_prompt = PromptTemplate.from_template("""You are Document Copilot, a friendly and helpful research assistant. Answer naturally like a chatbot: start with a direct reply, keep the tone clear and conversational, and use short paragraphs.
Act as an expert teacher and research assistant. Use plain language, helpful examples, and concise summaries. Avoid overly formal or technical prose unless the user specifically asks for it.

Response mode: {mode_instruction}

Use only the text provided in the Context. Do not use any outside knowledge, web search results, or general domain expertise. Every claim in your answer must be supported by the context. If you cannot answer the question from the context alone, answer exactly: "I couldn't find relevant information in the uploaded documents."

Answer every question only from the documents. After the answer, include a section titled **What your documents say** that cites only the context text or document source metadata. If the context does not contain evidence, do not invent an answer.

First answer the user's question directly in one or two sentences, and complete every sentence fully. If the answer needs more detail, continue with a second paragraph.

Never present general knowledge as though it came from the uploaded documents. Do not invent document facts, quotes, or sources. Clearly identify uncertainty, assumptions, disagreements, or missing information.

Context:
{context}

Question: {question}
Helpful answer:""")
        answer_prompt = answer_prompt.partial(mode_instruction=mode_instructions.get(response_mode, mode_instructions["Deep research"]))
        qa_chain = ConversationalRetrievalChain.from_llm(
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

    except Exception as e:
        logger.error(f"❌ Failed to create chat chain: {e}")
        raise
