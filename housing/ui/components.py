# housing/ui/components.py — reusable UI bits
import streamlit as st


def page_header(
    title: str, author_md: str, intro_md: str, notice_md: str | None = None
):
    st.title(title)
    st.markdown(author_md)

    # Intro info box with expander
    with st.expander("ℹ About this page", expanded=True):
        st.info(intro_md)

    # Optional notice with expander
    if notice_md:
        with st.expander("⚠ Important notice", expanded=True):
            st.warning(notice_md)


def kpi(label: str, value, delta: str | None = None, help: str | None = None):
    st.metric(label, value, delta=delta, help=help)


def section_help(text: str):
    st.caption(text)
