"""Main Streamlit application entry point for the document intelligence workspace."""

from __future__ import annotations

import logging
import os
import shutil

import pandas as pd
import streamlit as st

from backend.chat_history import (
    append_turn,
    clear_history,
    load_history,
    restore_memory,
)
from backend.config import HISTORY_DIR, UPLOAD_DIR, VECTOR_DIR, ensure_dirs
from backend.document_intelligence import build_knowledge_graph, extract_intelligence
from backend.evaluation_service import (
    add_case,
    delete_case,
    load_dataset,
    load_runs,
    run_evaluation_dataset,
)
from backend.feedback_store import feedback_summary, record_feedback
from backend.file_handler import (
    create_memory,
    embed_documents,
    load_documents_from_paths,
    load_or_create_vectorstore,
    load_vectorstore,
    save_uploaded_files,
    split_documents,
)
from backend.query_service import rewrite_query
from backend.rag_chain import create_chat_chain
from backend.report_generator import build_report
from backend.workspace_store import (
    create_conversation,
    delete_workspace_metadata,
    get_workspace,
    list_workspaces,
    rename_conversation,
    touch_conversation,
    upsert_workspace,
)
from frontend.auth import _logout, render_auth_screen, require_auth
from frontend.theme import apply_theme

logger = logging.getLogger(__name__)
MAX_FILE_SIZE_MB = 25


def _reset_runtime() -> None:
    """Clear transient runtime state while preserving saved workspace data."""
    for key in (
        "qa_chain",
        "memory",
        "vectorstore",
        "workspace_id",
        "conversation_id",
        "workspace_files",
        "latest_answer",
        "chain_mode",
        "insights",
    ):
        st.session_state.pop(key, None)


def _load_conversation(workspace_id: str, conversation_id: str) -> None:
    """Load an existing workspace and reconstruct the chat runtime state."""
    vectorstore = st.session_state.get("vectorstore")
    if st.session_state.get("workspace_id") != workspace_id or vectorstore is None:
        vectorstore = load_vectorstore(workspace_id)
    memory = create_memory()
    restore_memory(memory, load_history(workspace_id, conversation_id))
    mode = st.session_state.get("response_mode", "Deep research")
    st.session_state.update(
        vectorstore=vectorstore,
        qa_chain=create_chat_chain(vectorstore, memory, mode),
        memory=memory,
        workspace_id=workspace_id,
        conversation_id=conversation_id,
    )
    st.session_state.chain_mode = mode
    workspace = get_workspace(workspace_id) or {}
    st.session_state.workspace_files = [item["name"] for item in workspace.get("documents", [])]


def _open_upload_workspace(files, name: str) -> None:
    """Create a brand-new workspace from uploaded files and initialize the runtime."""
    vectorstore, memory, workspace_id = load_or_create_vectorstore(files)
    intelligence = {}
    for file in files:
        try:
            loaded = load_documents_from_paths([os.path.join(UPLOAD_DIR, workspace_id, file.name)])
            raw = "\n".join(document.page_content for document in loaded)
            intelligence[file.name] = extract_intelligence(raw, file.name)
        except Exception:
            intelligence[file.name] = {
                "title": os.path.splitext(file.name)[0],
                "topics": [],
                "entities": [],
            }
    workspace = upsert_workspace(workspace_id, files, name, intelligence)
    restore_memory(memory, load_history(workspace_id, "default"))
    mode = st.session_state.get("response_mode", "Deep research")
    st.session_state.update(
        vectorstore=vectorstore,
        qa_chain=create_chat_chain(vectorstore, memory, mode),
        memory=memory,
        workspace_id=workspace_id,
        conversation_id="default",
        workspace_files=[item["name"] for item in workspace.get("documents", [])],
    )
    st.session_state.chain_mode = mode


def _remove_workspace(workspace_id: str) -> None:
    """Delete all local storage for a workspace including uploaded data and indexes."""
    for path in (os.path.join(UPLOAD_DIR, workspace_id), os.path.join(VECTOR_DIR, workspace_id)):
        if os.path.isdir(path):
            shutil.rmtree(path)
    for filename in os.listdir(HISTORY_DIR):
        if filename.startswith(f"{workspace_id}__") or filename == f"{workspace_id}.json":
            os.remove(os.path.join(HISTORY_DIR, filename))
    delete_workspace_metadata(workspace_id)


def _is_greeting(question: str) -> bool:
    """Detect very short greetings as a friendly, non-document-based interaction."""
    normalized = str(question or "").strip().lower()
    if not normalized:
        return False
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", "yo"]
    if normalized in greetings:
        return True
    return any(normalized.startswith(g + " ") or normalized == g for g in greetings)


def _render_sources(source_documents, question=None) -> None:
    """Render document evidence and relevance metadata in a compact panel."""
    sources = []
    if source_documents:
        sources = [
            (document, float(document.metadata.get("relevance_score", 0.0)))
            for document in source_documents
        ]
    else:
        try:
            sources = st.session_state.vectorstore.similarity_search_with_relevance_scores(
                question or "", k=4
            )
        except Exception:
            sources = []
    if sources:
        with st.expander("Sources and relevance"):
            for index, (document, score) in enumerate(sources, 1):
                source = os.path.basename(document.metadata.get("source", "Document"))
                page = document.metadata.get("page")
                page_text = f" · page {int(page) + 1}" if page is not None else ""
                st.markdown(f"**{index}. {source}{page_text}** — {score:.0%} relevance")
                st.caption(
                    document.page_content[:450] + ("…" if len(document.page_content) > 450 else "")
                )


def _ask(question: str, regenerate: bool = False) -> None:
    """Run a question against the active workspace and persist the chat turn."""
    workspace_id, conversation_id = st.session_state.workspace_id, st.session_state.conversation_id
    if not regenerate:
        append_turn(workspace_id, conversation_id, "user", question)
        with st.chat_message("user"):
            st.markdown(question)
    with st.chat_message("assistant"):
        if _is_greeting(question):
            greeting_answer = (
                "Hi there! I can help answer questions about the uploaded documents. "
                "Please ask a specific document-related question."
            )
            st.markdown(greeting_answer)
            append_turn(workspace_id, conversation_id, "assistant", greeting_answer)
            touch_conversation(workspace_id, conversation_id)
            st.session_state.latest_answer = {
                "question": question,
                "answer": greeting_answer,
                "evidence": [],
                "interpreted_query": question,
            }
            return
        with st.spinner("Searching your documents…"):
            try:
                mode = st.session_state.get("response_mode", "Deep research")
                if st.session_state.get("chain_mode") != mode:
                    st.session_state.qa_chain = create_chat_chain(
                        st.session_state.vectorstore, st.session_state.memory, mode
                    )
                    st.session_state.chain_mode = mode
                interpreted = rewrite_query(question, load_history(workspace_id, conversation_id))
                result = st.session_state.qa_chain.invoke({"question": interpreted})
                answer = (
                    result.get("answer")
                    or result.get("output")
                    or result.get("text")
                    or result.get("response")
                    or result.get("content")
                    or "No answer was generated."
                )
                if isinstance(answer, dict):
                    answer = answer.get("text") or answer.get("output") or str(answer)
                answer = str(answer).strip()
                source_documents = result.get("source_documents") or []
                if not source_documents:
                    answer = "I couldn't find relevant information in the uploaded documents."
                elif (
                    "What your documents say" not in answer
                    and "I couldn't find relevant information in the uploaded documents."
                    not in answer
                ):
                    answer = (
                        "I found documents that may be relevant, but I cannot safely answer "
                        "your question without a more specific document-based prompt. "
                        "Please ask a more precise question about the uploaded documents."
                    )
                st.markdown(answer)
                _render_sources(source_documents, question)
                append_turn(workspace_id, conversation_id, "assistant", answer)
                touch_conversation(workspace_id, conversation_id)
                evidence = []
                for document in source_documents:
                    evidence.append(
                        {
                            "source": os.path.basename(document.metadata.get("source", "Document")),
                            "page": str(int(document.metadata["page"]) + 1)
                            if document.metadata.get("page") is not None
                            else "",
                            "excerpt": document.page_content[:360],
                        }
                    )
                st.session_state.latest_answer = {
                    "question": question,
                    "answer": answer,
                    "evidence": evidence,
                    "interpreted_query": interpreted,
                }
            except Exception as error:
                logger.exception("Chat query failed")
                message = str(error)
                if (
                    "Request too large" in message
                    or "tokens per minute" in message
                    or "rate_limit_exceeded" in message
                ):
                    st.error(
                        "The request was too large for the Groq model or rate limit. "
                        "Try a shorter question, fewer documents, or reduce the retrieval window."
                    )
                else:
                    st.error(
                        f"I couldn't complete that request. Check your API key and try again. {message}"
                    )


def _generate_insight(kind: str) -> None:
    """Generate a reusable research artifact from the active workspace."""
    prompts = {
        "Research brief": "Create a document-wide research brief. Include: an executive summary, the most important findings, key evidence, implications, risks, and the most useful next actions. Use only the uploaded documents for factual claims.",
        "Evidence challenge": "Act as a rigorous reviewer of these documents. Identify the strongest claims, supporting evidence, counterevidence or tensions, assumptions, and what evidence would change each conclusion. Be explicit when the documents do not provide enough evidence.",
        "Knowledge gaps": "Build a research agenda from these documents. Identify unanswered questions, missing data, ambiguous terms, possible biases, stakeholders who need more information, and the five highest-value follow-up questions.",
    }
    with st.spinner(f"Creating {kind.lower()}…"):
        try:
            isolated_chain = create_chat_chain(
                st.session_state.vectorstore, create_memory(), "Deep research"
            )
            result = isolated_chain.invoke({"question": prompts[kind]})
            sources = []
            for document in result.get("source_documents", []):
                sources.append(
                    {
                        "source": os.path.basename(document.metadata.get("source", "Document")),
                        "page": str(int(document.metadata["page"]) + 1)
                        if document.metadata.get("page") is not None
                        else "",
                        "excerpt": document.page_content[:280],
                    }
                )
            insight_answer = (
                result.get("answer")
                or result.get("output")
                or result.get("text")
                or result.get("response")
                or result.get("content")
                or "No insight generated."
            )
            if isinstance(insight_answer, dict):
                insight_answer = (
                    insight_answer.get("text")
                    or insight_answer.get("output")
                    or str(insight_answer)
                )
            st.session_state.setdefault("insights", {})[kind] = {
                "content": insight_answer,
                "sources": sources,
            }
        except Exception:
            logger.exception("Insight generation failed")
            st.error("The insight could not be created. Check your API key and try again.")


def _generate_comparison(document_a: str, document_b: str) -> None:
    """Compare two documents with a structured research-style summary."""
    question = (
        f"Compare the uploaded documents '{document_a}' and '{document_b}'. Provide an "
        "executive comparison, similarities, differences, changed facts or numbers, "
        "contradictions, risks, and a compact Markdown comparison table. State clearly "
        "when the available evidence is insufficient."
    )
    with st.spinner("Comparing selected documents…"):
        try:
            result = create_chat_chain(
                st.session_state.vectorstore, create_memory(), "Compare viewpoints"
            ).invoke({"question": question})
            comparison_answer = (
                result.get("answer")
                or result.get("output")
                or result.get("text")
                or result.get("response")
                or result.get("content")
                or "No comparison generated."
            )
            if isinstance(comparison_answer, dict):
                comparison_answer = (
                    comparison_answer.get("text")
                    or comparison_answer.get("output")
                    or str(comparison_answer)
                )
            st.session_state.comparison = {
                "title": f"{document_a} vs {document_b}",
                "content": comparison_answer,
            }
        except Exception:
            logger.exception("Comparison failed")
            st.error("The comparison could not be generated. Check your API key and try again.")


def _generate_document_summary(document_name: str) -> None:
    """Create a summarization artifact for one document in the workspace."""
    prompt = (
        f"Create a detailed structured summary of the uploaded document named '{document_name}'. "
        "Include executive summary, key findings, important numbers if present, risks, "
        "recommendations, entities, and questions worth asking. Only attribute facts to that "
        "document when its evidence is retrieved."
    )
    with st.spinner("Creating document summary…"):
        try:
            result = create_chat_chain(
                st.session_state.vectorstore, create_memory(), "Deep research"
            ).invoke({"question": prompt})
            summary_answer = (
                result.get("answer")
                or result.get("output")
                or result.get("text")
                or result.get("response")
                or result.get("content")
                or "No summary generated."
            )
            if isinstance(summary_answer, dict):
                summary_answer = (
                    summary_answer.get("text")
                    or summary_answer.get("output")
                    or str(summary_answer)
                )
            st.session_state.document_summary = {"name": document_name, "content": summary_answer}
        except Exception:
            logger.exception("Summary failed")
            st.error("The summary could not be generated. Check your API key and try again.")


def _label(workspace: dict) -> str:
    """Create a short, human-friendly label for a saved workspace."""
    return f"{workspace.get('name', 'Untitled workspace')} · {len(workspace.get('documents', []))} files"


def _add_files_to_workspace(files, workspace_id: str, workspace_name: str = "") -> bool:
    """Save uploaded files into an existing workspace and rebuild its vector store."""
    try:
        save_uploaded_files(files, workspace_id)
    except Exception:
        raise
    upload_dir = os.path.join(UPLOAD_DIR, workspace_id)
    all_paths = [os.path.join(upload_dir, fname) for fname in os.listdir(upload_dir)]
    raw_docs = load_documents_from_paths(all_paths)
    splits = split_documents(raw_docs)
    index_path = os.path.join(VECTOR_DIR, workspace_id)
    embed_documents(splits, index_path)

    intelligence = {}
    for file in files:
        try:
            loaded = load_documents_from_paths([os.path.join(upload_dir, file.name)])
            raw = "\n".join(document.page_content for document in loaded)
            intelligence[file.name] = extract_intelligence(raw, file.name)
        except Exception:
            intelligence[file.name] = {
                "title": os.path.splitext(file.name)[0],
                "topics": [],
                "entities": [],
            }

    workspace = get_workspace(workspace_id) or {}
    existing = [
        type("F", (), {"name": d["name"], "size": d.get("size", 0)})()
        for d in workspace.get("documents", [])
    ]
    combined = existing + list(files)
    upsert_workspace(
        workspace_id, combined, workspace_name or workspace.get("name", ""), intelligence
    )
    return True


def _render_main_screen() -> None:
    """Render the main workspace UI for active documents and chat."""
    workspace_id = st.session_state.get("workspace_id")
    if not workspace_id:
        st.markdown(
            "<div class='brand-row'><span class='brand-mark'>D</span><span class='eyebrow'>Document intelligence, without clutter</span></div>",
            unsafe_allow_html=True,
        )
        st.title("Turn complex documents into clear answers.")
        st.markdown(
            "<p class='hero-copy'>A focused research workspace for searching reports, contracts, notes, and datasets. Every answer is grounded in your files and accompanied by source context.</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<span class='status-badge'>Local-first AI workspace</span>", unsafe_allow_html=True
        )
        quick = st.columns(3)
        for column, label, value in zip(
            quick,
            ["Grounded answers", "Evidence tracing", "Persistent workspaces"],
            [
                "Answers are tied back to the files you upload.",
                "Each response includes source snippets and relevance context.",
                "Keep your documents, chats, and conversations organized locally.",
            ],
        ):
            with column, st.container(border=True):
                st.markdown(f"<p class='eyebrow'>{label}</p>", unsafe_allow_html=True)
                st.caption(value)

        st.markdown("<p class='eyebrow'>How it flows</p>", unsafe_allow_html=True)
        workflow_cols = st.columns(3)
        workflow_steps = [
            ("1. Upload", "Drop in PDFs, tables, notes, and research files."),
            ("2. Ask", "Search across the corpus with source-aware follow-ups."),
            ("3. Export", "Turn the findings into a usable research brief or report."),
        ]
        for column, (step, text) in zip(workflow_cols, workflow_steps):
            with column, st.container(border=True):
                st.markdown(f"<p class='eyebrow'>{step}</p>", unsafe_allow_html=True)
                st.caption(text)

        st.markdown("<p class='eyebrow'>Popular workflows</p>", unsafe_allow_html=True)
        prompt_cols = st.columns(3)
        sample_prompts = [
            "Summarize the key findings from these documents.",
            "Compare the main arguments and contradictions across the files.",
            "List the most important decisions or open questions in this set.",
        ]
        for column, prompt in zip(prompt_cols, sample_prompts):
            with column, st.container(border=True):
                st.caption(prompt)
                if st.button(
                    "Try prompt", key=f"quick-prompt-{prompt[:12]}", use_container_width=True
                ):
                    st.session_state["demo_prompt"] = prompt
        if st.session_state.get("demo_prompt"):
            st.info(f"Example prompt ready: {st.session_state['demo_prompt']}")
        create, details = st.columns([1.35, 0.85], gap="large")
        with create, st.container(border=True):
            st.subheader("Start a new workspace")
            st.caption(
                "Choose a clear project name, then add the source material you want to explore."
            )
            name = st.text_input("Workspace name", placeholder="For example: Q3 market research")
            files = st.file_uploader(
                "Add your documents",
                type=["txt", "pdf", "docx", "csv"],
                accept_multiple_files=True,
                help=f"PDF, DOCX, TXT, or CSV · maximum {MAX_FILE_SIZE_MB} MB per file",
            )
            oversize = [
                file.name for file in files or [] if file.size > MAX_FILE_SIZE_MB * 1024 * 1024
            ]
            if oversize:
                st.error("File size limit exceeded: " + ", ".join(oversize))
            if st.button(
                "Build my workspace",
                type="primary",
                use_container_width=True,
                disabled=not files or bool(oversize),
            ):
                try:
                    with st.spinner("Building your searchable workspace…"):
                        _open_upload_workspace(files, name)
                    st.rerun()
                except Exception as error:
                    logger.exception("Workspace setup failed")
                    st.error(f"Could not prepare documents: {error}")
        with details, st.container(border=True):
            st.subheader("What you get")
            st.markdown(
                "<p class='feature'><b>Document-grounded answers</b><br>Responses use your files, not generic web knowledge.</p>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<p class='feature'><b>Transparent sources</b><br>Review the relevant file, page, excerpt, and relevance score.</p>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<p class='feature'><b>Continuity that lasts</b><br>Return to saved workspaces and keep the conversation moving.</p>",
                unsafe_allow_html=True,
            )
            st.divider()
            st.caption("Supported: PDF · DOCX · TXT · CSV")
        one, two, three = st.columns(3)
        one.metric("Storage", "Local-only")
        two.metric("Retrieval", "Top 4 sources")
        three.metric("Privacy", "Private by design")
        return

    workspace = get_workspace(workspace_id) or {}
    conversations = workspace.get("conversations", [])
    conversation_id = st.session_state.get("conversation_id", "default")
    history = load_history(workspace_id, conversation_id)
    if st.session_state.get("evaluation_mode"):
        _render_evaluation_ui()
        return
    with st.sidebar:
        st.divider()
        st.markdown("**CONVERSATIONS**")
        st.selectbox(
            "Research mode",
            [
                "Deep research",
                "Executive brief",
                "Compare viewpoints",
                "Study guide",
                "Action plan",
            ],
            key="response_mode",
            help="Controls the structure and emphasis of each new answer.",
        )
        st.caption("The mode changes how the answer is written; it does not change your documents.")
        labels = {item["title"]: item["id"] for item in conversations}
        current = next(
            (title for title, value in labels.items() if value == conversation_id),
            "New conversation",
        )
        selected = st.selectbox(
            "Saved conversations",
            list(labels),
            index=list(labels).index(current) if current in labels else 0,
        )
        if labels[selected] != conversation_id:
            _load_conversation(workspace_id, labels[selected])
            st.rerun()
        title = st.text_input("Conversation title", value=current)
        new, rename = st.columns(2)
        if new.button("New", use_container_width=True):
            conversation = create_conversation(workspace_id)
            _load_conversation(workspace_id, conversation["id"])
            st.rerun()
        if rename.button("Rename", use_container_width=True):
            rename_conversation(workspace_id, conversation_id, title)
            st.rerun()
        if st.button("Create new workspace", use_container_width=True):
            _reset_runtime()
            st.session_state.pop("workspace_id", None)
            st.rerun()
        export = "\n\n".join(f"{item['role'].title()}: {item['content']}" for item in history)
        st.download_button(
            "Download chat",
            export,
            "document-copilot-chat.txt",
            "text/plain",
            use_container_width=True,
        )
        if st.button("Clear this conversation", use_container_width=True):
            clear_history(workspace_id, conversation_id)
            _load_conversation(workspace_id, conversation_id)
            st.rerun()
        if st.button("Delete this workspace", use_container_width=True):
            st.session_state["confirm_delete_workspace"] = workspace_id
        if st.session_state.get("confirm_delete_workspace") == workspace_id:
            st.warning(
                "This will permanently remove the current workspace and all its local files. "
                "This action cannot be undone."
            )
            col_confirm, col_cancel = st.columns([1, 1])
            if col_confirm.button(
                "Confirm delete this workspace",
                use_container_width=True,
                key="confirm_delete_current",
            ):
                try:
                    _remove_workspace(workspace_id)
                    st.session_state.pop("confirm_delete_workspace", None)
                    _reset_runtime()
                    st.rerun()
                except Exception as error:
                    logger.exception("Workspace deletion failed")
                    st.error(f"Could not delete workspace: {error}")
            if col_cancel.button("Cancel", use_container_width=True, key="cancel_delete_current"):
                st.session_state.pop("confirm_delete_workspace", None)
        summary = feedback_summary()
        st.caption(f"Feedback: {summary['helpful']}/{summary['total']} helpful")

    st.markdown("<p class='eyebrow'>Active workspace</p>", unsafe_allow_html=True)
    with st.container(border=True):
        st.title(workspace.get("name", "Document workspace"))
        st.caption(
            f"{len(st.session_state.get('workspace_files', []))} indexed document(s) · {len(history)} saved messages · local-only storage"
        )
        pills = ["Local-first", "Evidence-backed", "Private-by-default"]
        st.markdown(
            " ".join(f"<span class='resource-pill'>{pill}</span>" for pill in pills),
            unsafe_allow_html=True,
        )
        action_left, action_mid, action_right = st.columns(3)
        if action_left.button("Ask a question", use_container_width=True):
            st.chat_input("Ask a question about your documents")
        if action_mid.button("Create brief", use_container_width=True):
            _generate_insight("Research brief")
        if action_right.button("Compare docs", use_container_width=True):
            st.session_state["comparison_tab_open"] = True
    workspace_documents = [item["name"] for item in workspace.get("documents", [])]
    metrics_left, metrics_middle, metrics_right = st.columns(3)
    metrics_left.metric("Documents", len(workspace_documents))
    metrics_middle.metric(
        "Research questions", len([item for item in history if item.get("role") == "user"])
    )
    metrics_right.metric("Conversations", len(conversations))
    with st.expander("Knowledge graph", expanded=False):
        st.caption(
            "A local entity-to-document map built from text extracted during ingestion. It shows mentions, not verified real-world relationships."
        )
        graph_documents = [
            item
            for item in workspace.get("documents", [])
            if item.get("intelligence", {}).get("entities")
        ]
        if graph_documents:
            st.graphviz_chart(build_knowledge_graph(graph_documents), use_container_width=True)
        else:
            st.info(
                "No named entities were extracted yet. Rebuild the workspace after uploading text-rich documents."
            )
    with st.expander("Workspace analytics", expanded=False):
        document_types = (
            pd.DataFrame(workspace.get("documents", []))
            .get("type", pd.Series(dtype=str))
            .value_counts()
        )
        if not document_types.empty:
            st.caption("Indexed documents by type")
            st.bar_chart(document_types)
        topics = []
        for document in workspace.get("documents", []):
            topics.extend(document.get("intelligence", {}).get("topics", []))
        if topics:
            topic_counts = pd.Series(topics).value_counts().head(10)
            st.caption("Most frequent extracted topics")
            st.bar_chart(topic_counts)
        else:
            st.info("Analytics will appear after document text is processed.")
    with st.expander("Document library", expanded=False):
        st.caption("Browse indexed documents and inspect locally extracted metadata.")
        st.markdown("---")
        add_files = st.file_uploader(
            "Upload new documents to this workspace",
            type=["txt", "pdf", "docx", "csv"],
            accept_multiple_files=True,
            key="add_files",
        )
        if add_files:
            oversize = [
                file.name for file in add_files or [] if file.size > MAX_FILE_SIZE_MB * 1024 * 1024
            ]
            if oversize:
                st.error("File size limit exceeded: " + ", ".join(oversize))
        if st.button(
            "Add to workspace",
            type="primary",
            use_container_width=True,
            disabled=not add_files
            or (add_files and any(f.size > MAX_FILE_SIZE_MB * 1024 * 1024 for f in add_files)),
        ):
            try:
                with st.spinner("Adding files and updating workspace…"):
                    _add_files_to_workspace(add_files, workspace_id, workspace.get("name", ""))
                st.success("Files added and workspace updated.")
                _load_conversation(workspace_id, conversation_id)
                st.rerun()
            except Exception as error:
                logger.exception("Failed to add files to workspace")
                st.error(f"Could not add files: {error}")
        search = st.text_input("Search documents", placeholder="Filter by name, type, or topic")
        for index, document in enumerate(workspace.get("documents", [])):
            details = document.get("intelligence", {})
            searchable = " ".join(
                [document["name"], document.get("type", ""), " ".join(details.get("topics", []))]
            ).lower()
            if search and search.lower() not in searchable:
                continue
            with st.container(border=True):
                info, action = st.columns([4, 1])
                info.markdown(f"**{document['name']}**  ")
                info.caption(
                    f"{document.get('type', 'FILE')} · {document.get('size', 0):,} bytes · {document.get('status', 'processed').title()}"
                )
                if details.get("topics"):
                    info.caption("Topics: " + " · ".join(details["topics"][:6]))
                if details.get("entities"):
                    info.caption("Entities: " + " · ".join(details["entities"][:5]))
                doc_key = document.get("id") or document.get("_id") or f"{index}-{document['name']}"
                summary_key = f"summary-{workspace_id}-{index}-{doc_key}".replace(" ", "_")
                if action.button("Summarize", key=summary_key):
                    _generate_document_summary(document["name"])
        if summary_result := st.session_state.get("document_summary"):
            st.divider()
            st.markdown(f"#### Summary: {summary_result['name']}")
            st.markdown(summary_result["content"])
            st.download_button(
                "Download summary (.md)",
                f"# {summary_result['name']}\n\n{summary_result['content']}",
                "document-summary.md",
                "text/markdown",
            )
    with st.expander("Compare documents", expanded=False):
        if len(workspace_documents) < 2:
            st.info("Upload at least two documents to compare them.")
        else:
            left_doc, right_doc = st.columns(2)
            document_a = left_doc.selectbox("Document A", workspace_documents, key="compare-a")
            document_b = right_doc.selectbox(
                "Document B",
                [name for name in workspace_documents if name != document_a],
                key="compare-b",
            )
            if st.button("Run comparison", type="primary"):
                _generate_comparison(document_a, document_b)
            if comparison := st.session_state.get("comparison"):
                st.markdown(f"#### {comparison['title']}")
                st.markdown(comparison["content"])
    csv_files = [name for name in workspace_documents if name.lower().endswith(".csv")]
    if csv_files:
        with st.expander("CSV data explorer", expanded=False):
            selected_csv = st.selectbox("Dataset", csv_files, key="csv-explorer")
            try:
                dataframe = pd.read_csv(os.path.join(UPLOAD_DIR, workspace_id, selected_csv))
                st.caption(f"{len(dataframe):,} rows · {len(dataframe.columns)} columns")
                st.dataframe(dataframe.head(100), use_container_width=True)
                numeric_columns = dataframe.select_dtypes(include="number").columns.tolist()
                if numeric_columns:
                    chart_column = st.selectbox("Visualize numeric column", numeric_columns)
                    st.bar_chart(dataframe[chart_column])
                    st.dataframe(dataframe[numeric_columns].describe().T, use_container_width=True)
                else:
                    st.info("No numeric columns were detected for charting.")
            except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
                st.error(f"Could not read this CSV safely: {error}")
    st.divider()
    st.markdown("### Insight Studio")
    st.caption(
        "Create structured research artifacts from the entire workspace. These are separate from your chat history."
    )
    brief_col, challenge_col, gaps_col = st.columns(3)
    if brief_col.button("Create research brief", use_container_width=True):
        _generate_insight("Research brief")
    if challenge_col.button("Challenge the evidence", use_container_width=True):
        _generate_insight("Evidence challenge")
    if gaps_col.button("Find knowledge gaps", use_container_width=True):
        _generate_insight("Knowledge gaps")
    insights = st.session_state.get("insights", {})
    if insights:
        tabs = st.tabs(list(insights))
        for tab, (kind, insight) in zip(tabs, insights.items()):
            with tab:
                st.markdown(insight["content"])
                if insight["sources"]:
                    with st.expander("Evidence used"):
                        for source in insight["sources"]:
                            page = f" · page {source['page']}" if source["page"] else ""
                            st.markdown(f"**{source['source']}{page}**")
                            st.caption(
                                source["excerpt"] + ("…" if len(source["excerpt"]) == 280 else "")
                            )
                st.download_button(
                    f"Download {kind.lower()} (.md)",
                    f"# {kind}\n\n{insight['content']}",
                    f"{kind.lower().replace(' ', '-')}.md",
                    "text/markdown",
                    key=f"download-{kind}",
                )
    st.divider()
    for message in history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if not history:
        with st.container(border=True):
            st.subheader("Where would you like to start?")
            st.caption("Try one of these prompts, or write your own question below.")
            prompt_one, prompt_two, prompt_three = st.columns(3)
            if prompt_one.button("Summarize the key points", use_container_width=True):
                _ask("Summarize the key points from these documents.")
            if prompt_two.button("Find important decisions", use_container_width=True):
                _ask(
                    "What are the most important decisions, findings, or conclusions in these documents?"
                )
            if prompt_three.button("List open questions", use_container_width=True):
                _ask(
                    "What important open questions or missing information do these documents reveal?"
                )
    latest = st.session_state.get("latest_answer")
    if latest:
        left, right = st.columns([1, 4])
        if left.button("Regenerate"):
            _ask(latest["question"], regenerate=True)
            st.rerun()
        with st.expander("Copy last answer"):
            st.code(latest["answer"], language=None)
        with st.expander("Evidence and claim check", expanded=False):
            st.caption(
                "Use these document excerpts to verify the answer's document-grounded claims. Broader explanations are intentionally labeled in the response."
            )
            if latest.get("evidence"):
                for index, item in enumerate(latest["evidence"], 1):
                    page = f" · page {item['page']}" if item.get("page") else ""
                    st.markdown(f"**Evidence {index}: {item['source']}{page}**")
                    st.caption(item["excerpt"] + ("…" if len(item["excerpt"]) == 360 else ""))
            else:
                st.info("No source excerpts were returned for the latest answer.")
        if latest.get("interpreted_query") and latest["interpreted_query"] != latest["question"]:
            with st.expander("Query interpretation"):
                st.caption(
                    "Your follow-up was expanded with the previous question to improve retrieval."
                )
                st.code(latest["interpreted_query"], language=None)
        rating = st.radio(
            "Was the latest answer helpful?",
            ["Helpful", "Needs improvement"],
            horizontal=True,
            label_visibility="collapsed",
        )
        if right.button("Send feedback"):
            record_feedback(
                workspace_id,
                conversation_id,
                latest["question"],
                latest["answer"],
                "helpful" if rating == "Helpful" else "not_helpful",
            )
            st.toast("Feedback saved")
    report = build_report(
        workspace.get("name", "Document workspace"),
        history,
        latest.get("evidence", []) if latest else [],
    )
    with st.expander("Export research report", expanded=False):
        st.caption(
            "Download a clean Markdown report with the conversation and evidence used for the latest answer."
        )
        st.download_button(
            "Download report (.md)",
            report,
            "documind-research-report.md",
            "text/markdown",
            use_container_width=True,
        )
    if query := st.chat_input("Ask a question about your documents"):
        _ask(query)


def _render_evaluation_ui() -> None:
    """Render the evaluation management and comparison console for a workspace."""
    workspace_id = st.session_state.get("workspace_id")
    st.title("RAG Evaluation")
    st.caption("Create, run, and manage evaluation datasets for this workspace.")
    dataset = load_dataset(workspace_id)
    st.markdown(f"**Cases:** {len(dataset)}")
    cols = st.columns([3, 1, 1])
    with cols[0]:
        if st.button("+ New Test Case"):
            st.session_state["new_case"] = True
    with cols[1]:
        if st.button("Import JSON"):
            uploaded = st.file_uploader(
                "Upload JSON evaluation file", type=["json"], key="import_eval"
            )
            if uploaded:
                try:
                    data = __import__("json").load(uploaded)
                    for case in data:
                        add_case(workspace_id, case)
                    st.success("Imported dataset")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Import failed: {exc}")
    with cols[2]:
        if st.button("Export JSON"):
            st.download_button(
                "Download dataset",
                __import__("json").dumps(dataset, ensure_ascii=False, indent=2),
                file_name=f"{workspace_id}.eval.json",
                mime="application/json",
            )

    if st.session_state.get("new_case"):
        st.subheader("New evaluation case")
        q = st.text_input("Question")
        expected_answer = st.text_area("Expected answer (optional)")
        expected_source = st.text_input("Expected source filename (optional)")
        expected_pages = st.text_input("Expected pages (comma-separated, optional)")
        if st.button("Save case"):
            case = {
                "question": q,
                "expected_answer": expected_answer or None,
                "expected_sources": [s.strip() for s in expected_source.split(",") if s.strip()],
                "expected_pages": [
                    int(p.strip()) for p in expected_pages.split(",") if p.strip().isdigit()
                ],
            }
            add_case(workspace_id, case)
            st.session_state.pop("new_case", None)
            st.success("Saved")
            st.rerun()

    st.subheader("Evaluation cases")
    if not dataset:
        st.info("No evaluation data yet.")
        return
    for case in dataset:
        with st.container(border=True):
            cols = st.columns([6, 1, 1])
            cols[0].markdown(
                f"**{case.get('question')}**\n\n_Source:_ {', '.join(case.get('expected_sources') or [])}"
            )
            if cols[1].button("Run", key=f"run_{case.get('id')}"):
                st.info("Running test...")
                run = run_evaluation_dataset(
                    workspace_id,
                    st.session_state.get("vectorstore"),
                    st.session_state.get("qa_chain"),
                    st.session_state.get("memory"),
                    cases=[case],
                )
                st.json(run)
            if cols[2].button("Delete", key=f"del_{case.get('id')}"):
                delete_case(workspace_id, case.get("id"))
                st.rerun()
    st.subheader("Evaluation runs")

    def _aggregate_runs(runs):
        total_cases = 0
        total_hits = 0
        mrr_total = 0.0
        mrr_count = 0
        total_retrieval_ms = 0
        total_llm_ms = 0
        total_tokens = 0
        total_cost = 0.0
        cost_count = 0
        for run in runs:
            for res in run.get("results", []):
                total_cases += 1
                metrics = res.get("metrics", {})
                if metrics.get("hit"):
                    total_hits += 1
                if isinstance(metrics.get("mrr"), (int, float)):
                    mrr_total += float(metrics.get("mrr"))
                    mrr_count += 1
                trace = res.get("trace", {})
                total_retrieval_ms += int(trace.get("retrieval_ms", 0))
                total_llm_ms += int(trace.get("llm_ms", 0))
                tokens = trace.get("tokens", {}) or {}
                t_total = tokens.get("total")
                if isinstance(t_total, (int, float)):
                    total_tokens += int(t_total)
                cost = trace.get("cost", {}) or {}
                amount = cost.get("amount_usd")
                if isinstance(amount, (int, float)):
                    total_cost += float(amount)
                    cost_count += 1
        return {
            "total_runs": len(runs),
            "total_cases": total_cases,
            "hit_rate": round(total_hits / total_cases, 3) if total_cases else "Not available",
            "mrr": round(mrr_total / mrr_count, 3) if mrr_count else "Not available",
            "avg_retrieval_ms": int(total_retrieval_ms / total_cases)
            if total_cases
            else "Not available",
            "avg_llm_ms": int(total_llm_ms / total_cases) if total_cases else "Not available",
            "avg_tokens": int(total_tokens / total_cases)
            if total_cases and total_tokens
            else "Not available",
            "avg_cost_usd": round(total_cost / cost_count, 6) if cost_count else "Not available",
        }

    runs = load_runs(workspace_id)
    if not runs:
        st.info("No evaluation runs yet.")
    else:
        agg = _aggregate_runs(runs)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Runs", agg.get("total_runs"))
        c2.metric("Cases", agg.get("total_cases"))
        c3.metric("Hit Rate", agg.get("hit_rate"))
        c4.metric("MRR", agg.get("mrr"))
        if st.button("Export runs (JSON)"):
            st.download_button(
                "Download JSON",
                __import__("json").dumps(runs, ensure_ascii=False, indent=2),
                file_name=f"{workspace_id}_runs.json",
                mime="application/json",
            )
        if st.button("Export runs (CSV)"):
            import csv
            import io

            out = io.StringIO()
            writer = csv.writer(out)
            writer.writerow(
                [
                    "run_id",
                    "case_id",
                    "question",
                    "hit",
                    "rank",
                    "recall@k",
                    "precision@k",
                    "mrr",
                    "avg_retrieval_score",
                    "retrieval_ms",
                    "llm_ms",
                    "total_ms",
                    "tokens_total",
                    "cost_usd",
                    "answer_score",
                    "answer_label",
                    "citation_match_summary",
                ]
            )
            for run in runs:
                for res in run.get("results", []):
                    metrics = res.get("metrics", {})
                    trace = res.get("trace", {})
                    ans = res.get("answer_evaluation", {})
                    cit = res.get("citation_evaluation", {})
                    tokens = (trace.get("tokens") or {}).get("total")
                    cost = (trace.get("cost") or {}).get("amount_usd")
                    writer.writerow(
                        [
                            run.get("id"),
                            res.get("case_id"),
                            res.get("question"),
                            metrics.get("hit"),
                            metrics.get("rank"),
                            metrics.get("recall@k"),
                            metrics.get("precision@k"),
                            metrics.get("mrr"),
                            metrics.get("avg_retrieval_score"),
                            trace.get("retrieval_ms"),
                            trace.get("llm_ms"),
                            trace.get("total_ms"),
                            tokens if tokens is not None else "",
                            cost if cost is not None else "",
                            ans.get("score") if ans else "",
                            ans.get("label") if ans else "",
                            cit.get("overall_match_rate") if cit else "",
                        ]
                    )
            st.download_button(
                "Download CSV",
                out.getvalue(),
                file_name=f"{workspace_id}_runs.csv",
                mime="text/csv",
            )


def main() -> None:
    """Entry point for the Streamlit dashboard."""
    ensure_dirs()
    st.set_page_config(
        page_title="DocuMind",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": None,
            "Report a bug": None,
            "About": None,
        },
    )
    apply_theme()

    # Ensure auth state is checked first
    if not require_auth():
        render_auth_screen()
        st.stop()

    user = st.session_state.get("user") or {}
    saved = list_workspaces()
    with st.sidebar:
        st.markdown("## ✦ DocuMind")
        st.caption(f"Signed in as {user.get('name', user.get('email', 'User'))}")
        if st.button("Logout", use_container_width=True, type="secondary"):
            _logout()
        st.divider()
        st.markdown("**WORKSPACES**")
        if saved:
            labels = {_label(item): item["id"] for item in saved}
            choice = st.selectbox("Recent workspaces", list(labels), label_visibility="collapsed")
            if st.button("Open workspace", type="primary", use_container_width=True):
                try:
                    _load_conversation(labels[choice], "default")
                    st.rerun()
                except Exception as error:
                    st.error(f"Could not open workspace: {error}")
            if st.button("Delete selected workspace", use_container_width=True):
                st.session_state["confirm_delete_workspace"] = labels[choice]
            if st.session_state.get("confirm_delete_workspace") == labels[choice]:
                st.warning(
                    "This will permanently remove the selected workspace and all its local files. "
                    "This action cannot be undone."
                )
                col_confirm, col_cancel = st.columns([1, 1])
                if col_confirm.button(
                    "Confirm delete selected workspace",
                    use_container_width=True,
                    key="confirm_delete_selected",
                ):
                    try:
                        _remove_workspace(labels[choice])
                        st.session_state.pop("confirm_delete_workspace", None)
                        _reset_runtime()
                        st.rerun()
                    except Exception as error:
                        logger.exception("Workspace deletion failed")
                        st.error(f"Could not delete workspace: {error}")
                if col_cancel.button(
                    "Cancel", use_container_width=True, key="cancel_delete_selected"
                ):
                    st.session_state.pop("confirm_delete_workspace", None)
        else:
            st.caption("Saved workspaces will appear here.")
        st.divider()
        st.markdown(
            "<p class='mini-note'>Files, indexes, and chat history stay on this machine.</p>",
            unsafe_allow_html=True,
        )

    _render_main_screen()


if __name__ == "__main__":
    main()
