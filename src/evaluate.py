# src/evaluate.py

import torch
from sklearn.metrics import classification_report


def evaluate_model(trainer, test_dataset, label_names):
    predictions = trainer.predict(test_dataset)
    preds = predictions.predictions.argmax(-1)
    labels = predictions.label_ids

    print(classification_report(labels, preds, target_names=label_names))