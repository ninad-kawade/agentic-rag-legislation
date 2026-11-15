"""Minimal Streamlit client for the Evidence RAG API."""

from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000/query")
QUESTION_LIMIT = max(1, int(os.getenv("RAG_SESSION_QUESTION_LIMIT", "10")))

st.set_page_config(page_title="Evidence RAG", page_icon="📚", layout="centered")
st.title("Evidence RAG")
st.caption("Ask a question about the legislation corpus. Answers include source citations.")

if "question_count" not in st.session_state:
    st.session_state.question_count = 0
st.caption(f"Questions this session: {st.session_state.question_count}/{QUESTION_LIMIT}")

with st.form("query-form"):
    question = st.text_area(
        "Your question",
        placeholder="What is the enactment date of Commission Implementing Regulation (EU) 2020/2041?",
        height=110,
        max_chars=2000,
    )
    submitted = st.form_submit_button(
        "Ask", type="primary", disabled=st.session_state.question_count >= QUESTION_LIMIT
    )

if submitted:
    if not question.strip():
        st.warning("Enter a question first.")
    elif st.session_state.question_count >= QUESTION_LIMIT:
        st.warning("This session has reached its question limit.")
    else:
        try:
            st.session_state.question_count += 1
            with st.spinner("Searching the legislation and checking evidence…"):
                response = requests.post(
                    API_URL,
                    json={"question": question.strip()},
                    timeout=300,
                )
                response.raise_for_status()
                result = response.json()

            answer_text = result["text"]
            is_abstention = answer_text.lower().startswith("insufficient evidence")
            if is_abstention:
                st.warning(answer_text)
            else:
                st.subheader("Answer")
                st.write(answer_text)

            st.write(f"**Confidence:** {float(result['confidence']):.2f}")
            citations = result.get("citations", [])
            if citations:
                st.subheader("Citations")
                for index, citation in enumerate(citations, start=1):
                    with st.container(border=True):
                        st.markdown(f"**Source {index}:** [{citation['source']}]({citation['source']})")
                        st.caption(citation["snippet"])
            elif not is_abstention:
                st.info("The answer did not include a citation.")
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 429:
                st.error("service is busy, please try again later")
            else:
                st.error(f"Could not reach the answer service at {API_URL}: {exc}")
        except requests.RequestException as exc:
            st.error(f"Could not reach the answer service at {API_URL}: {exc}")
        except (KeyError, TypeError, ValueError) as exc:
            st.error(f"The answer service returned an invalid response: {exc}")
