def build_evidence_trace(question, evidence_items):
    return {"question": question, "evidence": evidence_items}

def build_evidence_graph(question, evidence_items):
    return {
        "question": question,
        "claims": [{"claim": "Retrieved evidence", "sources": evidence_items}],
    }
