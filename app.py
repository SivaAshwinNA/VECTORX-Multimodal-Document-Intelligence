import streamlit as st
import fitz
import os

from model.inference import EvidenceRelevanceModel


# ============================================================
# VECTORX — MULTIMODAL DOCUMENT INTELLIGENCE
# ============================================================

st.set_page_config(
    page_title="VECTORX | Multimodal Document Intelligence",
    page_icon="📄",
    layout="wide"
)


# ------------------------------------------------------------
# LOAD VECTORX MODEL
# ------------------------------------------------------------

@st.cache_resource
def load_model():
    return EvidenceRelevanceModel(
        "model/saved_model/vectorx_distilbert"
    )


model = load_model()


# ------------------------------------------------------------
# DOCUMENT EXTRACTION
# ------------------------------------------------------------

def extract_pdf(uploaded_file):

    pdf_bytes = uploaded_file.read()

    document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    evidence = []

    for page_number, page in enumerate(
        document,
        start=1
    ):

        text = page.get_text("text").strip()

        if text:

            evidence.append({
                "document": uploaded_file.name,
                "page": page_number,
                "text": text
            })

    document.close()

    return evidence


# ------------------------------------------------------------
# PAGE HEADER
# ------------------------------------------------------------

st.title("📄 Multimodal Document Intelligence")

st.markdown(
    """
    ### VECTORX
    **Evidence-first document intelligence powered by a fine-tuned Transformer**
    """
)

st.divider()


# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------

with st.sidebar:

    st.header("VECTORX Model")

    st.success("DistilBERT Loaded")

    st.caption(
        "Task: Evidence Relevance Classification"
    )

    st.caption(
        "Fine-tuned on the VECTORX dataset"
    )


# ------------------------------------------------------------
# PDF UPLOAD
# ------------------------------------------------------------

uploaded_files = st.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)


documents = []


if uploaded_files:

    for uploaded_file in uploaded_files:

        pages = extract_pdf(
            uploaded_file
        )

        documents.extend(pages)

    st.success(
        f"{len(uploaded_files)} document(s) loaded — "
        f"{len(documents)} page(s) extracted."
    )


# ------------------------------------------------------------
# QUESTION
# ------------------------------------------------------------

question = st.text_input(
    "Ask a question about your documents",
    placeholder="Example: What was the Q4 profit?"
)


# ------------------------------------------------------------
# ANALYZE
# ------------------------------------------------------------

if st.button(
    "🔍 Analyze with VECTORX",
    type="primary"
):

    if not uploaded_files:

        st.warning(
            "Please upload at least one PDF."
        )

    elif not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        st.subheader("VECTORX Analysis")

        results = []

        # -----------------------------------------------
        # SCORE EVERY PAGE
        # -----------------------------------------------

        for item in documents:

            prediction = model.predict(
                question,
                item["text"]
            )

            results.append({
                **item,
                **prediction
            })


        # -----------------------------------------------
        # SORT BY RELEVANCE
        # -----------------------------------------------

        results.sort(
            key=lambda x: x["relevance_score"],
            reverse=True
        )


        # -----------------------------------------------
        # BEST EVIDENCE
        # -----------------------------------------------

        best = results[0]

        score = best["relevance_score"]


        st.metric(
            "Evidence Relevance",
            f"{score * 100:.2f}%"
        )


        if best["relevant"]:

            st.success(
                "VECTORX found relevant evidence."
            )

        else:

            st.warning(
                "VECTORX could not find sufficiently "
                "relevant evidence."
            )


        # -----------------------------------------------
        # SOURCE
        # -----------------------------------------------

        st.subheader("📌 Source")

        st.write(
            f"**Document:** {best['document']}"
        )

        st.write(
            f"**Page:** {best['page']}"
        )


        # -----------------------------------------------
        # EVIDENCE
        # -----------------------------------------------

        st.subheader("📖 Evidence")

        evidence_text = best["text"]

        if len(evidence_text) > 2000:

            evidence_text = (
                evidence_text[:2000]
                + "..."
            )

        st.info(evidence_text)


        # -----------------------------------------------
        # TOP EVIDENCE PAGES
        # -----------------------------------------------

        st.subheader(
            "🔎 Evidence Ranking"
        )

        for index, result in enumerate(
            results[:5],
            start=1
        ):

            st.write(
                f"**{index}. "
                f"{result['document']} — "
                f"Page {result['page']}**"
            )

            st.progress(
                min(
                    max(
                        result["relevance_score"],
                        0.0
                    ),
                    1.0
                )
            )

            st.caption(
                f"Relevance: "
                f"{result['relevance_score']:.4f}"
            )


# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------

st.divider()

st.caption(
    "VECTORX — Multimodal Document Intelligence | "
    "Fine-tuned DistilBERT Evidence Relevance Model"
)