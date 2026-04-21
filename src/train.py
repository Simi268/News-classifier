import os
import pandas as pd
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)

from dataset import NewsDataset


# ==============================
# Configuration
# ==============================

MODEL_NAME = "bert-base-uncased"
MAX_LEN = 64
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
TEST_SIZE = 0.1
RANDOM_SEED = 42



BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "NewsCategorizer.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "saved_models", "bert-base-uncased")


# ==============================
# Metrics
# ==============================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro")
    }


# ==============================
# Main Training Function
# ==============================

def main():

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    # Combine text fields
    df["text"] = df["headline"] + " " + df["short_description"]

    # Encode labels properly
    df["category"] = df["category"].astype("category")
    df["label"] = df["category"].cat.codes

    label_list = list(df["category"].cat.categories)

    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}

    num_labels = len(label_list)

    print("Number of classes:", num_labels)

    # Train-validation split
    train_df, val_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=df["label"]
    )

    print("Train size:", len(train_df))
    print("Validation size:", len(val_df))

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    # Dataset objects
    train_dataset = NewsDataset(
        train_df["text"].values,
        train_df["label"].values,
        tokenizer,
        MAX_LEN
    )

    val_dataset = NewsDataset(
        val_df["text"].values,
        val_df["label"].values,
        tokenizer,
        MAX_LEN
    )

    # Model
    print("Loading model...")
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_dir="logs",
        logging_steps=50,
        logging_strategy="steps",
        report_to="tensorboard",
        fp16=torch.cuda.is_available(),
        save_total_limit=2
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    print("Starting training...")
    trainer.train(resume_from_checkpoint=True)

    print("Saving best model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Training complete!")
    print(f"Model saved at: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()