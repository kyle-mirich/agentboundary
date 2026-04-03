from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import settings
from .models import ClassificationResponse, ExampleRecord, Label


LABELS = [Label.IN_SCOPE.value, Label.OUT_OF_SCOPE.value, Label.AMBIGUOUS.value]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_classification_explanation(
    label: Label,
    confidence: float,
    probability_map: dict[str, float],
) -> str:
    ranked = sorted(probability_map.items(), key=lambda item: item[1], reverse=True)
    runner_up_label = Label(ranked[1][0]) if len(ranked) > 1 else label
    confidence_pct = round(confidence * 100, 1)

    if label == Label.IN_SCOPE:
        return (
            f"Predicted in scope with {confidence_pct}% confidence. "
            f"The wording is closer to the in-scope examples than to {runner_up_label.value.replace('_', ' ')}."
        )
    if label == Label.OUT_OF_SCOPE:
        return (
            f"Predicted out of scope with {confidence_pct}% confidence. "
            f"The text is closer to outside-topic examples than to the in-scope set."
        )
    return (
        f"Predicted ambiguous with {confidence_pct}% confidence. "
        f"The text appears mixed or too close to more than one class."
    )


class EncodedTextDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer) -> None:
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=settings.max_sequence_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {key: value[index] for key, value in self.encodings.items()}
        item["labels"] = self.labels[index]
        return item


@dataclass
class TrainingResult:
    checkpoint_path: Path
    training_loss: float
    train_count: int


def _load_model(model_path: str):
    resolved_path = str(Path(model_path).resolve())
    tokenizer = AutoTokenizer.from_pretrained(resolved_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(resolved_path, local_files_only=True)
    return tokenizer, model


def train_model(train_examples: Iterable[ExampleRecord], checkpoint_dir: Path) -> TrainingResult:
    examples = list(train_examples)
    texts = [item.text for item in examples]
    labels = [LABEL_TO_ID[item.label.value] for item in examples]
    tokenizer = AutoTokenizer.from_pretrained(settings.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        settings.model_name,
        num_labels=len(LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )
    device = get_device()
    model.to(device)

    dataset = EncodedTextDataset(texts, labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=settings.batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=settings.learning_rate)
    model.train()
    losses: list[float] = []

    for _ in range(settings.epochs):
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    metadata = {
        "model_name": settings.model_name,
        "labels": LABELS,
        "epochs": settings.epochs,
        "batch_size": settings.batch_size,
        "learning_rate": settings.learning_rate,
    }
    (checkpoint_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2))
    average_loss = sum(losses) / len(losses) if losses else 0.0
    return TrainingResult(checkpoint_path=checkpoint_dir, training_loss=average_loss, train_count=len(examples))


def evaluate_model(eval_examples: Iterable[ExampleRecord], checkpoint_dir: Path) -> dict:
    examples = list(eval_examples)
    tokenizer, model = _load_model(str(checkpoint_dir))
    device = get_device()
    model.to(device)
    model.eval()
    dataset = EncodedTextDataset(
        [item.text for item in examples],
        [LABEL_TO_ID[item.label.value] for item in examples],
        tokenizer,
    )
    dataloader = DataLoader(dataset, batch_size=settings.batch_size, shuffle=False)
    predictions: list[int] = []
    probabilities: list[float] = []
    targets: list[int] = []

    with torch.no_grad():
        for batch in dataloader:
            labels = batch.pop("labels")
            targets.extend(labels.tolist())
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            probs = torch.softmax(logits, dim=-1)
            predictions.extend(torch.argmax(probs, dim=-1).cpu().tolist())
            probabilities.extend(torch.max(probs, dim=-1).values.cpu().tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(
        targets,
        predictions,
        labels=list(range(len(LABELS))),
        zero_division=0,
    )
    matrix = confusion_matrix(targets, predictions, labels=list(range(len(LABELS)))).tolist()
    misclassified = []
    for example, target, prediction, confidence in zip(examples, targets, predictions, probabilities):
        if target != prediction:
            misclassified.append(
                {
                    "text": example.text,
                    "true_label": ID_TO_LABEL[target],
                    "predicted_label": ID_TO_LABEL[prediction],
                    "confidence": round(confidence, 4),
                }
            )
    macro_f1 = sum(f1) / len(f1) if len(f1) else 0.0
    out_of_scope_precision = precision[LABEL_TO_ID[Label.OUT_OF_SCOPE.value]]
    return {
        "macro_f1": round(float(macro_f1), 4),
        "out_of_scope_precision": round(float(out_of_scope_precision), 4),
        "per_class": {
            LABELS[index]: {
                "precision": round(float(precision[index]), 4),
                "recall": round(float(recall[index]), 4),
                "f1": round(float(f1[index]), 4),
            }
            for index in range(len(LABELS))
        },
        "confusion_matrix": matrix,
        "misclassified": misclassified[:10],
        "eval_count": len(examples),
    }


def classify_text(text: str, checkpoint_dir: Path) -> ClassificationResponse:
    tokenizer, model = _load_model(str(checkpoint_dir))
    device = get_device()
    model.to(device)
    model.eval()
    encoded = tokenizer(
        [text],
        truncation=True,
        padding=True,
        max_length=settings.max_sequence_length,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        logits = model(**encoded).logits
        probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()
    best_index = max(range(len(probs)), key=lambda index: probs[index])
    label = Label(ID_TO_LABEL[best_index])
    probability_map = {ID_TO_LABEL[index]: round(float(value), 4) for index, value in enumerate(probs)}
    explanation = _build_classification_explanation(
        label=label,
        confidence=round(float(probs[best_index]), 4),
        probability_map=probability_map,
    )
    return ClassificationResponse(
        label=label,
        confidence=round(float(probs[best_index]), 4),
        probabilities=probability_map,
        explanation=explanation,
    )
