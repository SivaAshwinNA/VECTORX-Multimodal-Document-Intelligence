from model.inference import EvidenceRelevanceModel

# Load VECTORX fine-tuned DistilBERT
VECTORX_MODEL = EvidenceRelevanceModel(
    "model/saved_model/vectorx_distilbert"
)


def vectorx_rerank(question, evidence_units):
    """
    Rerank retrieved document evidence using
    the fine-tuned VECTORX DistilBERT model.
    """

    ranked = []

    for evidence in evidence_units:

        if isinstance(evidence, dict):
            text = (
                evidence.get("text")
                or evidence.get("content")
                or evidence.get("evidence")
                or ""
            )
        else:
            text = str(evidence)

        if not text.strip():
            continue

        prediction = VECTORX_MODEL.predict(
            question,
            text
        )

        item = (
            evidence.copy()
            if isinstance(evidence, dict)
            else {"text": text}
        )

        item["vectorx_relevance_score"] = prediction[
            "relevance_score"
        ]

        item["vectorx_relevant"] = prediction[
            "relevant"
        ]

        ranked.append(item)

    ranked.sort(
        key=lambda x: x["vectorx_relevance_score"],
        reverse=True
    )

    return ranked

def rank_evidence(question, evidence_items, model):
    ranked = []
    for item in evidence_items:
        result = model.predict(question, item["text"])
        ranked.append({**item, "relevance_score": result["relevance_score"]})
    return sorted(ranked, key=lambda x: x["relevance_score"], reverse=True)
