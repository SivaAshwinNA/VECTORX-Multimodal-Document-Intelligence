import os
import pandas as pd
import numpy as np

from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)


# ============================================================
# VECTORX DISTILBERT FINE-TUNING
# ============================================================

DATA_PATH = "data/training_data.csv"

MODEL_NAME = "distilbert-base-uncased"

OUTPUT_DIR = "model/saved_model/vectorx_distilbert"


print("=" * 60)
print("VECTORX DISTILBERT FINE-TUNING")
print("=" * 60)


# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

print("\n[1/6] Loading VECTORX training dataset...")

df = pd.read_csv(DATA_PATH)

required_columns = ["question", "evidence", "label"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(
            f"Missing required column: {column}"
        )

df["question"] = df["question"].fillna("")
df["evidence"] = df["evidence"].fillna("")

# Combine question + evidence
df["text"] = (
    "question: "
    + df["question"]
    + " evidence: "
    + df["evidence"]
)

df["label"] = df["label"].astype(int)

print(f"Total examples: {len(df)}")
print(f"Positive examples: {(df['label'] == 1).sum()}")
print(f"Negative examples: {(df['label'] == 0).sum()}")


# ------------------------------------------------------------
# 2. TRAIN / TEST SPLIT
# ------------------------------------------------------------

print("\n[2/6] Creating train/test split...")

train_df, test_df = train_test_split(
    df,
    test_size=0.25,
    random_state=42,
    stratify=df["label"]
)

train_dataset = Dataset.from_pandas(
    train_df[["text", "label"]],
    preserve_index=False
)

test_dataset = Dataset.from_pandas(
    test_df[["text", "label"]],
    preserve_index=False
)

print(f"Training examples: {len(train_dataset)}")
print(f"Testing examples: {len(test_dataset)}")


# ------------------------------------------------------------
# 3. LOAD TOKENIZER
# ------------------------------------------------------------

print("\n[3/6] Loading DistilBERT tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


def tokenize_function(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=256
    )


train_dataset = train_dataset.map(
    tokenize_function,
    batched=True
)

test_dataset = test_dataset.map(
    tokenize_function,
    batched=True
)


# ------------------------------------------------------------
# 4. LOAD PRETRAINED MODEL
# ------------------------------------------------------------

print("\n[4/6] Loading pretrained DistilBERT...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label={
        0: "NOT_RELEVANT",
        1: "RELEVANT"
    },
    label2id={
        "NOT_RELEVANT": 0,
        "RELEVANT": 1
    }
)


# ------------------------------------------------------------
# 5. FINE-TUNE
# ------------------------------------------------------------

print("\n[5/6] Fine-tuning VECTORX model...")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    num_train_epochs=3,

    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,

    learning_rate=2e-5,
    weight_decay=0.01,

    logging_steps=10,

    save_strategy="no",

    report_to="none",

    fp16=False
)


trainer = Trainer(
    model=model,
    args=training_args,

    train_dataset=train_dataset,
    eval_dataset=test_dataset,

    processing_class=tokenizer
)


trainer.train()


# ------------------------------------------------------------
# 6. EVALUATION
# ------------------------------------------------------------

print("\n[6/6] Evaluating VECTORX model...")

predictions = trainer.predict(test_dataset)

predicted_labels = np.argmax(
    predictions.predictions,
    axis=1
)

true_labels = np.array(
    test_df["label"].tolist()
)

accuracy = accuracy_score(
    true_labels,
    predicted_labels
)

precision, recall, f1, _ = precision_recall_fscore_support(
    true_labels,
    predicted_labels,
    average="binary",
    zero_division=0
)


print("\n" + "=" * 60)
print("VECTORX TRANSFORMER PERFORMANCE")
print("=" * 60)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")


# ------------------------------------------------------------
# SAVE MODEL
# ------------------------------------------------------------

print("\nSaving fine-tuned VECTORX model...")

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

trainer.save_model(
    OUTPUT_DIR
)

tokenizer.save_pretrained(
    OUTPUT_DIR
)


print("\n" + "=" * 60)
print("VECTORX FINE-TUNING COMPLETE")
print("=" * 60)

print(f"\nModel saved to:")
print(OUTPUT_DIR)

print("\nModel type:")
print("DistilBERT - Task-specific Evidence Relevance")

print("\nTraining complete.")