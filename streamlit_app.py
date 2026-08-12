"""
Streamlit UI for the Medical Evidence RAG Agent.

Run with:
    streamlit run streamlit_app.py

Design direction: clinical evidence / medical journal, not a generic
"healthcare app" gradient. Serif headlines echo journal mastheads,
monospace numerals echo lab-report data readouts, and the ECG trace is
the one signature graphic -- used once, deliberately, not as a repeated
decoration.
"""

from __future__ import annotations
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.rag_chain import MedicalRAGAgent  # noqa: E402
from src.confidence import RETRIEVAL_SCORE_THRESHOLD, LLM_CONFIDENCE_THRESHOLD  # noqa: E402


st.set_page_config(
    page_title="Medical Evidence RAG",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Design tokens + CSS
# ---------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --paper:      #FAFAF8;
    --paper-dim:  #F1F0EB;
    --ink:        #16232E;
    --ink-dim:    #55606B;
    --chart-blue: #2C6E8E;
    --chart-blue-dim: #E4EEF2;
    --flag-red:   #B23A3A;
    --flag-red-dim: #F7E9E9;
    --line:       #DAD8CE;
}

.stApp {
    background: var(--paper);
    color: var(--ink);
    font-family: 'Inter', sans-serif;
}

/* Kill Streamlit's default top padding so our masthead sits high */
.block-container { padding-top: 2rem; max-width: 760px; }

/* ---- Masthead ---- */
.masthead-title {
    font-family: 'Source Serif 4', serif;
    font-weight: 700;
    font-size: 2.1rem;
    letter-spacing: -0.01em;
    color: var(--ink);
    margin-bottom: 0.1rem;
}
.masthead-sub {
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
    color: var(--ink-dim);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.6rem;
}
.masthead-disclaimer {
    font-size: 0.78rem;
    color: var(--ink-dim);
    border-left: 2px solid var(--line);
    padding-left: 0.6rem;
    margin: 0.9rem 0 1.2rem 0;
    line-height: 1.5;
}

/* ---- ECG divider (signature element) ---- */
.ecg-divider { margin: 0.4rem 0 1.4rem 0; }
.ecg-divider svg { width: 100%; height: 28px; display: block; }
.ecg-divider path {
    fill: none;
    stroke: var(--chart-blue);
    stroke-width: 1.6;
    stroke-linecap: round;
    stroke-linejoin: round;
}

/* ---- Query input ---- */
div[data-testid="stTextInput"] input {
    font-family: 'Inter', sans-serif;
    font-size: 1rem;
    border: 1px solid var(--line);
    border-radius: 3px;
    padding: 0.65rem 0.8rem;
    background: #FFFFFF;
}
div[data-testid="stTextInput"] input:focus {
    border-color: var(--chart-blue);
    box-shadow: 0 0 0 1px var(--chart-blue);
}

.stButton button {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 0.9rem;
    background: var(--ink);
    color: var(--paper);
    border: none;
    border-radius: 3px;
    padding: 0.55rem 1.4rem;
}
.stButton button:hover { background: var(--chart-blue); color: white; }

/* ---- Answer card ---- */
.answer-card {
    background: #FFFFFF;
    border: 1px solid var(--line);
    border-left: 3px solid var(--chart-blue);
    border-radius: 2px;
    padding: 1.3rem 1.5rem;
    margin-top: 1.2rem;
}
.answer-card.refused {
    border-left: 3px solid var(--flag-red);
    background: var(--flag-red-dim);
}
.answer-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--chart-blue);
    margin-bottom: 0.5rem;
}
.answer-label.refused { color: var(--flag-red); }
.answer-text {
    font-family: 'Source Serif 4', serif;
    font-size: 1.08rem;
    line-height: 1.6;
    color: var(--ink);
}

/* ---- Citations ---- */
.citation-block {
    margin-top: 1.1rem;
    padding-top: 0.9rem;
    border-top: 1px dashed var(--line);
}
.citation-item {
    font-size: 0.85rem;
    color: var(--ink-dim);
    margin-bottom: 0.5rem;
    line-height: 1.5;
}
.citation-item .doc-id {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    background: var(--chart-blue-dim);
    color: var(--chart-blue);
    padding: 0.1rem 0.4rem;
    border-radius: 2px;
    margin-right: 0.4rem;
}

/* ---- Confidence readout (lab-data style) ---- */
.confidence-strip {
    display: flex;
    gap: 1.4rem;
    margin-top: 1.1rem;
    padding-top: 0.9rem;
    border-top: 1px solid var(--line);
    font-family: 'IBM Plex Mono', monospace;
}
.confidence-metric { flex: 1; }
.confidence-metric .value {
    font-size: 1.15rem;
    font-weight: 500;
}
.confidence-metric .value.pass { color: var(--chart-blue); }
.confidence-metric .value.fail { color: var(--flag-red); }
.confidence-metric .metric-label {
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--ink-dim);
    margin-top: 0.1rem;
}

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {
    background: var(--paper-dim);
    border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] h3 {
    font-family: 'Source Serif 4', serif;
    font-size: 1rem;
}
</style>
"""

ECG_SVG = """
<div class="ecg-divider">
<svg viewBox="0 0 600 28" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M0,14 L120,14 L134,14 L142,4 L150,24 L158,10 L166,14 L180,14
           L300,14 L314,14 L322,4 L330,24 L338,10 L346,14 L360,14
           L480,14 L494,14 L502,4 L510,24 L518,10 L526,14 L600,14" />
</svg>
</div>
"""

EXAMPLE_QUESTIONS = [
    "Why is metformin used as first-line therapy for type 2 diabetes?",
    "How do statins reduce cardiovascular risk?",
    "What is the recommended pediatric dosage of amoxicillin for otitis media?",
]


@st.cache_resource(show_spinner=False)
def load_agent() -> MedicalRAGAgent:
    return MedicalRAGAgent()


def render_response(response) -> None:
    card_class = "answer-card refused" if response.refused else "answer-card"
    label_class = "answer-label refused" if response.refused else "answer-label"
    label_text = "EVIDENCE INSUFFICIENT — REFUSED" if response.refused else "GROUNDED ANSWER"

    st.markdown(
        f"""
        <div class="{card_class}">
            <div class="{label_class}">{label_text}</div>
            <div class="answer-text">{response.answer}</div>
        """,
        unsafe_allow_html=True,
    )

    if not response.refused and response.citations:
        citations_html = '<div class="citation-block">'
        for c in response.citations:
            citations_html += (
                f'<div class="citation-item"><span class="doc-id">{c.doc_id}</span>'
                f'<strong>{c.title}</strong> (PMID: {c.pmid})<br>'
                f'&ldquo;{c.quoted_snippet}&rdquo;</div>'
            )
        citations_html += "</div>"
        st.markdown(citations_html, unsafe_allow_html=True)

    retrieval_pass = response.retrieval_score >= RETRIEVAL_SCORE_THRESHOLD
    llm_pass = response.llm_self_confidence >= LLM_CONFIDENCE_THRESHOLD
    st.markdown(
        f"""
        <div class="confidence-strip">
            <div class="confidence-metric">
                <div class="value {'pass' if retrieval_pass else 'fail'}">{response.retrieval_score:.2f}</div>
                <div class="metric-label">Retrieval match</div>
            </div>
            <div class="confidence-metric">
                <div class="value {'pass' if llm_pass else 'fail'}">{response.llm_self_confidence:.2f}</div>
                <div class="metric-label">Model self-confidence</div>
            </div>
            <div class="confidence-metric">
                <div class="value">{response.confidence_score:.2f}</div>
                <div class="metric-label">Combined score</div>
            </div>
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if response.refused and response.refusal_reason:
        st.caption(f"Refusal reason: {response.refusal_reason}")


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)

    st.markdown('<div class="masthead-title">Medical Evidence RAG</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="masthead-sub">Citation-Grounded &middot; Confidence-Gated &middot; PubMed Corpus</div>',
        unsafe_allow_html=True,
    )
    st.markdown(ECG_SVG, unsafe_allow_html=True)
    st.markdown(
        '<div class="masthead-disclaimer">This is a research/portfolio demo, not a clinical '
        'tool. Answers are generated only from the retrieved corpus below and are not '
        'validated for patient care.</div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### Corpus & thresholds")
        st.caption("Loaded index: `index/faiss_medical`")
        st.markdown(
            f"""
            <div style="font-family:'IBM Plex Mono',monospace; font-size:0.8rem; line-height:1.9;">
            retrieval threshold&nbsp;&nbsp;<b>{RETRIEVAL_SCORE_THRESHOLD:.2f}</b><br>
            confidence threshold&nbsp;<b>{LLM_CONFIDENCE_THRESHOLD:.2f}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()
        st.markdown("### Try an example")
        for q in EXAMPLE_QUESTIONS:
            if st.button(q, key=q, use_container_width=True):
                st.session_state["pending_question"] = q

    try:
        agent = load_agent()
    except (RuntimeError, FileNotFoundError) as e:
        st.error(f"Setup error: {e}")
        st.stop()

    default_q = st.session_state.pop("pending_question", "")
    question = st.text_input(
        "Ask a medical question",
        value=default_q,
        placeholder="e.g. How do statins reduce cardiovascular risk?",
        label_visibility="collapsed",
    )
    submitted = st.button("Get grounded answer")

    if submitted and question.strip():
        with st.spinner("Retrieving evidence and generating a grounded answer..."):
            response = agent.answer(question.strip())
        render_response(response)
    elif submitted:
        st.warning("Enter a question first.")


if __name__ == "__main__":
    main()
