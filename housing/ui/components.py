# housing/ui/components.py — reusable UI bits
import streamlit as st


def page_header(
    title: str, author_md: str, intro_md: str, notice_md: str | None = None
):
    st.title(title)
    st.markdown(author_md)
    st.info(intro_md)
    if notice_md:
        st.warning(notice_md)


def kpi(label: str, value, delta: str | None = None, help: str | None = None):
    st.metric(label, value, delta=delta, help=help)


def section_help(text: str):
    st.caption(text)
