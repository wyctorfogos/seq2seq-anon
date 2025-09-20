import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import numpy as np
from sklearn.model_selection import KFold
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
import evaluate
import pandas as pd

# -----------------------------
# Carregar dataset JSONL único
# -----------------------------
dataset = load_dataset("json", data_files={"full": "data_ner/train.json"})["full"]

# -----------------------------
# Definição de labels
# -----------------------------
label_list = [
    "O",
    "B-PER", "I-PER",
    "B-CPF", "I-CPF",
    "B-ADDR", "I-ADDR",
    "B-ORG", "I-ORG",
    "B-AGE", "I-AGE",
    "B-CEP", "I-CEP",
    "B-LAW", "I-LAW",
    "B-LICENSE_PLATE", "I-LICENSE_PLATE",
    "B-OAB", "I-OAB",
    "B-EMAIL", "I-EMAIL",
    "B-PHONE", "I-PHONE",
    "B-DATE", "I-DATE",
    "B-RG", "I-RG",
    "B-CNPJ", "I-CNPJ",
    "B-MONEY", "I-MONEY",
    "B-PROCESS_ID", "I-PROCESS_ID",
    "B-CREDIT_CARD", "I-CREDIT_CARD",
    "B-IP", "I-IP"
]
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for l, i in label2id.items()}

# -----------------------------
# Tokenizer
# -----------------------------
model_name = "pierreguillou/bert-base-cased-pt-lenerbr"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=512,
        padding="max_length"
    )
    labels = []
    for i, label in enumerate(examples["ner_tags_str"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_name = label[word_idx]
                label_ids.append(label2id.get(label_name, label2id["O"]))
            else:
                label_name = label[word_idx]
                if label_name.startswith("B-"):
                    label_name = "I-" + label_name[2:]
                label_ids.append(label2id.get(label_name, label2id["O"]))
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# -----------------------------
# Métricas
# -----------------------------
seqeval = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = [
        [id2label[l] for l in label if l != -100]
        for label in labels
    ]
    true_preds = [
        [id2label[p] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_preds, references=true_labels)

    # mean token accuracy
    correct, total = 0, 0
    for pred, lab in zip(predictions, labels):
        for p_i, l_i in zip(pred, lab):
            if l_i != -100:
                total += 1
                if p_i == l_i:
                    correct += 1
    mean_token_accuracy = correct / total if total > 0 else 0.0

    return {
        "accuracy": results["overall_accuracy"],
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "mean_token_accuracy": mean_token_accuracy,
    }

# -----------------------------
# K-FOLD
# -----------------------------
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)
all_metrics = []

for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    print(f"\n===== Fold {fold+1}/{k} =====")

    train_ds = dataset.select(train_idx).map(tokenize_and_align_labels, batched=True)
    val_ds = dataset.select(val_idx).map(tokenize_and_align_labels, batched=True)

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id
    )

    args = TrainingArguments(
        output_dir=f"./ner-anon-model/fold-{fold+1}",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_dir=f"./logs/fold-{fold+1}",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=True
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()
    metrics = trainer.evaluate()
    all_metrics.append(metrics)

# -----------------------------
# Estatísticas finais
# -----------------------------
df = pd.DataFrame(all_metrics)
print("\n===== Resultados médios =====")
print(df.mean())
print("\n===== Desvios padrão =====")
print(df.std())
