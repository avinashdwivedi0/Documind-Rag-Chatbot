"""Shared styling and app theming for the professional document workspace."""

from __future__ import annotations

import streamlit as st


def apply_theme() -> None:
    """Apply the polished product styling used across the app."""
    st.markdown(
        """
        <style>
        :root {
            --bg: #0a1020;
            --bg-2: #101a2d;
            --panel: rgba(17, 25, 39, 0.9);
            --panel-strong: rgba(22, 31, 49, 0.96);
            --line: rgba(148, 163, 184, 0.22);
            --text: #eef5ff;
            --muted: #a9bbd9;
            --primary: #7c5cff;
            --primary-2: #5eead4;
            --success: #34d399;
            --warning: #fbbf24;
            --danger: #f87171;
            --shadow: rgba(15, 23, 42, 0.75);
        }

        html, body, [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #07111d 0%, #101a2d 35%, #0d1326 100%);
            color: var(--text);
        }

        .block-container {
            max-width: 1280px;
            padding-top: 2.5rem;
            padding-bottom: 6rem;
        }

        [data-testid="stSidebar"] {
            background: rgba(9, 14, 23, 0.92);
            border-right: 1px solid var(--line);
            box-shadow: 12px 0 40px rgba(0, 0, 0, 0.18);
        }

        #MainMenu {
            visibility: hidden;
        }

        footer {
            visibility: hidden;
        }

        [data-testid="stHeader"] {
            background: rgba(10, 16, 32, 0.12);
            backdrop-filter: blur(10px);
        }

        .stApp {
            color: var(--text);
        }

        .stAppHeader {
            background: transparent;
        }

        .stTabs [role="tablist"] {
            gap: 0.5rem;
        }

        .stTabs [role="tab"] {
            border: 1px solid var(--line);
            border-radius: 10px 10px 0 0;
            background: rgba(15, 23, 42, 0.7);
            color: var(--muted);
        }

        .stTabs [role="tab"][aria-selected="true"] {
            background: linear-gradient(135deg, rgba(124, 92, 255, 0.18), rgba(94, 234, 212, 0.08));
            color: var(--text);
            border-color: rgba(124, 92, 255, 0.4);
        }

        div[data-testid="stVerticalBlockBorderWrapper"],
        div[data-testid="stChatMessage"],
        div[data-testid="stFileUploaderDropzone"],
        div[data-testid="stExpander"],
        [data-testid="stMetric"],
        [data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            background: var(--panel);
            border-radius: 18px;
            box-shadow: 0 10px 25px rgba(15, 23, 42, 0.25);
        }

        [data-testid="stMetric"] {
            background: linear-gradient(135deg, rgba(124, 92, 255, 0.12), rgba(15, 23, 42, 0.6));
            border: 1px solid rgba(124, 92, 255, 0.22);
        }

        [data-testid="stChatMessage"] {
            background: rgba(15, 23, 42, 0.58);
            border: 1px solid rgba(148, 163, 184, 0.12);
            border-radius: 16px;
        }

        h1, h2, h3, h4, p, label, [data-testid="stMarkdownContainer"] {
            color: var(--text);
        }

        code {
            background: rgba(15, 23, 42, 0.9);
            color: var(--primary-2);
            border-radius: 8px;
            padding: 0.15rem 0.45rem;
        }

        .eyebrow {
            color: var(--primary-2) !important;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            font-weight: 700;
            font-size: 0.73rem;
        }

        .hero-copy {
            color: var(--muted) !important;
            font-size: 1.08rem;
            line-height: 1.7;
            max-width: 720px;
        }

        .feature {
            color: var(--muted) !important;
            line-height: 1.6;
        }

        .feature b {
            color: var(--text) !important;
        }

        .mini-note {
            color: var(--muted) !important;
            font-size: 0.9rem;
        }

        .brand-row {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 0.75rem;
        }

        .brand-mark {
            width: 2.5rem;
            height: 2.5rem;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 0.85rem;
            background: linear-gradient(135deg, rgba(124, 92, 255, 0.9), rgba(94, 234, 212, 0.7));
            color: white;
            font-weight: 700;
            box-shadow: 0 10px 20px rgba(124, 92, 255, 0.35);
        }

        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            border: 1px solid rgba(94, 234, 212, 0.3);
            background: rgba(94, 234, 212, 0.08);
            color: var(--primary-2);
            font-size: 0.75rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            font-weight: 700;
        }

        .stButton > button {
            border-radius: 12px;
            border: 1px solid rgba(148, 163, 184, 0.32);
            background: rgba(15, 23, 42, 0.8);
            color: var(--text);
            font-weight: 600;
            min-height: 2.8rem;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }

        .stButton > button:hover {
            border-color: rgba(124, 92, 255, 0.5);
            box-shadow: 0 0 0 1px rgba(124, 92, 255, 0.2);
            transform: translateY(-1px);
        }

        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #7258ff, #8a70ff);
            color: white;
            border-color: rgba(124, 92, 255, 0.7);
        }

        .workspace-shell {
            border: 1px solid rgba(94, 234, 212, 0.28);
            background: linear-gradient(135deg, rgba(124, 92, 255, 0.14), rgba(15, 23, 42, 0.82));
            border-radius: 18px;
            padding: 1.25rem 1.25rem 1rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.22);
        }

        .resource-pill {
            display: inline-block;
            border: 1px solid rgba(148, 163, 184, 0.22);
            background: rgba(9, 14, 23, 0.55);
            color: var(--muted);
            border-radius: 999px;
            padding: 0.35rem 0.7rem;
            margin-right: 0.5rem;
            margin-bottom: 0.5rem;
            font-size: 0.78rem;
        }

        .glass-panel {
            background: rgba(11, 17, 28, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.14);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            box-shadow: 0 12px 24px rgba(15, 23, 42, 0.18);
        }

        .auth-shell {
            background: linear-gradient(135deg, rgba(124, 92, 255, 0.09), rgba(15, 23, 42, 0.8));
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 22px;
            padding: 1.25rem;
            box-shadow: 0 18px 40px rgba(15, 23, 42, 0.26);
        }

        .auth-brand {
            display: flex;
            align-items: center;
            gap: 0.85rem;
            margin-bottom: 1rem;
        }

        .auth-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 3rem;
            height: 3rem;
            border-radius: 1rem;
            background: linear-gradient(135deg, #7c5cff, #5eead4);
            color: white;
            font-weight: 800;
            box-shadow: 0 12px 28px rgba(124, 92, 255, 0.3);
        }

        .auth-copy {
            color: var(--muted) !important;
            line-height: 1.6;
            font-size: 1rem;
        }

        .cta-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.65rem;
            margin: 0.5rem 0 1rem;
        }

        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #7258ff, #8a70ff);
            color: white;
            border-color: rgba(124, 92, 255, 0.7);
        }

        [data-baseweb="input"] > div,
        [data-baseweb="select"] > div,
        [data-testid="stFileUploaderDropzone"] {
            background: rgba(15, 23, 42, 0.7) !important;
            border: 1px solid rgba(148, 163, 184, 0.2) !important;
            color: var(--text) !important;
            border-radius: 12px;
        }

        [data-testid="stMetricLabel"] *,
        [data-testid="stMetricValue"] * {
            color: var(--text) !important;
        }

        .stChatInput textarea {
            border-radius: 12px;
            background: rgba(15, 23, 42, 0.75);
            border: 1px solid rgba(148, 163, 184, 0.22);
            color: var(--text);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
