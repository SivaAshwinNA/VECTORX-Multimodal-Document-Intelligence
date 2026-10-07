from fastapi import FastAPI
from pydantic import BaseModel

try:
    from model.inference import EvidenceRelevanceModel
except Exception:
    EvidenceRelevanceModel = None

app = FastAPI(title="VECTORX API", version="0.1.0")
_model = None

class PredictionRequest(BaseModel):
    question: str
    evidence: str

@app.get("/health")
def health():
    return {"status": "ok", "service": "VECTORX", "model_loaded": _model is not None}

@app.post("/predict")
def predict(request: PredictionRequest):
    global _model
    if EvidenceRelevanceModel is None:
        return {"relevance_score": 0.0, "relevant": False, "status": "model_not_available"}
    if _model is None:
        try:
            _model = EvidenceRelevanceModel()
        except Exception as exc:
            return {
                "relevance_score": 0.0,
                "relevant": False,
                "status": "model_not_trained",
                "message": str(exc),
            }
    return _model.predict(request.question, request.evidence)
