# VECTORX — Multimodal Document Intelligence

Evidence-first multimodal document intelligence project for HACKNEX PSI01.

## Architecture

UPLOAD → EXTRACT → RETRIEVE → LEARNED EVIDENCE RANKING → REASON → VERIFY → ANSWER

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train the first VECTORX model

```bash
python model/train_model.py
```

## Start API

```bash
uvicorn api:app --reload --port 8000
```

## Start Streamlit

Open another terminal:

```bash
streamlit run app.py
```

The first model is a fine-tuned evidence relevance classifier. It receives a question and evidence and predicts whether the evidence is relevant.

Do not commit `.env` or API keys.
