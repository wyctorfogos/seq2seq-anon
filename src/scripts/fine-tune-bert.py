import os
import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer
)
import numpy as np
from seqeval.metrics import classification_report

# =========================
# Caminhos dos arquivos
# =========================
DATA_DIR = "data_ner"
TRAIN_PATH = os.path.join(DATA_DIR, "train.json")
VAL_PATH = os.path.join(DATA_DIR, "validation.json")
LABELS_PATH = os.path.join(DATA_DIR, "labels.json")
MODEL_NAME = "pierreguillou/bert-base-cased-pt-lenerbr"
OUTPUT_DIR = "./bert-ner-model"

# =========================
# Carregar labels
# =========================
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    label_list = json.load(f)

label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}

print("Labels carregadas:", label_list)

# =========================
# Tokenizer
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# =========================
# Carregar dados
# =========================
def load_json_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Converte ner_tags_str -> ids
    for ex in data:
        ex["labels"] = [label2id[tag] for tag in ex["ner_tags_str"]]
    return Dataset.from_list(data)

train_dataset = load_json_dataset(TRAIN_PATH)
val_dataset = load_json_dataset(VAL_PATH)

print("Exemplo de treino:", train_dataset[0])

# =========================
# Modelo
# =========================
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True  # importante para adaptar a nova head
)

# =========================
# Data collator
# =========================
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# =========================
# Métricas
# =========================
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [id2label[pred] for (pred, lab) in zip(prediction, label) if lab != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[lab] for (pred, lab) in zip(prediction, label) if lab != -100]
        for prediction, label in zip(predictions, labels)
    ]

    report = classification_report(true_labels, true_predictions, digits=4)
    print(report)
    return {"f1": float(report.split()[-2])}  # pega F1 final

# =========================
# Configuração de treino
# =========================
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    save_total_limit=1,
    push_to_hub=False
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# =========================
# Treinar
# =========================
trainer.train()

# =========================
# Salvar modelo final
# =========================
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Modelo salvo em {OUTPUT_DIR}")