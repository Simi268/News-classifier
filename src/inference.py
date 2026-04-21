# src/inference.py

import torch
from transformers import BertTokenizer
from model import get_model
from config import MODEL_NAME, MAX_LEN


def predict(text, model_path, label_encoder):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = get_model(MODEL_NAME, len(label_encoder.classes_))
    model.load_state_dict(torch.load(f"{model_path}/pytorch_model.bin"))
    model.eval()

    encoding = tokenizer(
        text,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**encoding)

    prediction = torch.argmax(outputs.logits, dim=1).item()
    return label_encoder.inverse_transform([prediction])[0]
if __name__ == "__main__":
    import sys
    import joblib  # or pickle depending on how you saved it

    if len(sys.argv) < 2:
        print("Usage: python src/inference.py \"The government announced new economic reforms today.\"")
        exit()

    text = sys.argv[1]

    model_path = "saved_models"  # adjust if needed
    label_encoder = joblib.load(f"{model_path}/label_encoder.pkl")

    prediction = predict(text, model_path, label_encoder)
    print(f"Prediction: {prediction}")