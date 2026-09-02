"""Authentication pages and session flow for DocuMind."""

from __future__ import annotations

import streamlit as st

from backend.auth_service import authenticate_user, register_user


def _set_session_user(user: dict | None) -> None:
    """Persist user identity in Streamlit session state."""
    st.session_state["user"] = user


def _logout() -> None:
    """Clear the active login session."""
    st.session_state.pop("user", None)
    st.session_state.pop("auth_view", None)
    st.rerun()


def render_auth_screen() -> None:
    """Render login and registration screens for unauthenticated users."""
    st.markdown(
        """
        <div class='auth-shell'>
            <div class='auth-brand'>
                <span class='auth-badge'>D</span>
                <div>
                    <div class='eyebrow' style='margin-bottom: 0.2rem;'>Private Research Workspace</div>
                    <h1 style='margin: 0;'>Welcome to DocuMind</h1>
                </div>
            </div>
            <p class='auth-copy'>Sign in to continue to your secure workspace. Your files, chats, and research context stay local while your identity is managed through MongoDB.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.2, 1.2], gap="large")
    with left:
        with st.container(border=True):
            st.subheader("What you can do")
            st.markdown(
                """
                - Upload PDFs, DOCX, TXT, and CSV files
                - Ask grounded questions with source-backed answers
                - Save workspaces and continue research later
                - Export reports and evidence summaries
                """
            )
            st.caption(
                "Built for private document intelligence and local-first research workflows."
            )

    with right:
        tab1, tab2 = st.tabs(["Login", "Register"])
        with tab1:
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            remember = st.checkbox("Keep me signed in on this browser", value=True)
            if st.button("Login", type="primary", use_container_width=True):
                if not email or not password:
                    st.error("Email and password are required.")
                else:
                    user = authenticate_user(email, password)
                    if user:
                        _set_session_user(user)
                        st.session_state["remember_me"] = remember
                        st.session_state.pop("login_email", None)
                        st.session_state.pop("login_password", None)
                        st.rerun()
                    else:
                        st.error("Invalid email or password.")

        with tab2:
            name = st.text_input("Full name", key="register_name")
            register_email = st.text_input("Email", key="register_email")
            new_password = st.text_input("Password", type="password", key="register_password")
            confirm_password = st.text_input(
                "Confirm password", type="password", key="register_confirm"
            )
            if st.button("Create account", type="primary", use_container_width=True):
                if not name or not register_email or not new_password:
                    st.error("Name, email, and password are required.")
                elif new_password != confirm_password:
                    st.error("Passwords do not match.")
                else:
                    try:
                        user = register_user(name, register_email, new_password)
                        _set_session_user(user)
                        st.session_state["remember_me"] = True
                        st.session_state.pop("register_name", None)
                        st.session_state.pop("register_email", None)
                        st.session_state.pop("register_password", None)
                        st.session_state.pop("register_confirm", None)
                        st.success("Account created successfully.")
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))
                    except Exception as exc:  # pragma: no cover - UI safety
                        st.error(f"Could not create account: {exc}")


def require_auth() -> bool:
    """Return true when a user is active in session state."""
    return bool(st.session_state.get("user"))
