from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F

app = FastAPI()

MODEL_PATH = "Simik26/news-classifier"

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

class Request(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "News Classifier API"}

@app.post("/predict")
def predict(request: Request):
    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs).item()
    confidence = probs[0][pred].item()

    return {
    "label": model.config.id2label[pred],
    "confidence": round(confidence, 3),
    "probs": probs[0].tolist()
}