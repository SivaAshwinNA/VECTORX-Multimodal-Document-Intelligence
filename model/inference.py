import os
import joblib


class EvidenceRelevanceModel:

    def __init__(self, model_dir="model/saved_model"):
        self.model_dir = model_dir

        model_path = os.path.join(
            model_dir,
            "evidence_relevance_model.joblib"
        )

        vectorizer_path = os.path.join(
            model_dir,
            "tfidf_vectorizer.joblib"
        )

        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

    def predict(self, question, evidence):

        text = f"question: {question} evidence: {evidence}"

        vector = self.vectorizer.transform([text])

        probabilities = self.model.predict_proba(vector)[0]

        relevance_score = float(probabilities[1])

        return {
            "relevance_score": round(relevance_score, 4),
            "relevant": relevance_score >= 0.5,
            "status": "ok"
        }