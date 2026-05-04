from pathlib import Path
import json

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_DIR = Path("classifier_model_1")
EVIDENCE_PATH = Path("data/evidence.json")
MAX_LENGTH = 512
EVID_SEP = "[EVID_SEP]"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


evidence_db = _load_json(EVIDENCE_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()


def _ids_to_evidence_texts(evidence: list[int]) -> list[str]:
    texts = []
    for ev_id in evidence:
        key = f"evidence-{ev_id}"
        text = evidence_db.get(key)
        if text:
            texts.append(text)
    return texts


def predict_label(claim: str, evidence: list[int]) -> str:
    evidence_texts = _ids_to_evidence_texts(evidence)
    text_b = f" {EVID_SEP} ".join(evidence_texts)

    enc = tokenizer(
        claim,
        text_b,
        truncation="only_second",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    with torch.no_grad():
        outputs = model(**enc)
        pred_id = int(torch.argmax(outputs.logits, dim=-1).item())

    label_map = model.config.id2label
    if isinstance(label_map, dict):
        return label_map.get(pred_id, label_map.get(str(pred_id), "NOT_ENOUGH_INFO"))
    return "NOT_ENOUGH_INFO"
