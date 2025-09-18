from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
import numpy as np
import evaluate
seqeval = evaluate.load("seqeval")

# -----------------------------
# Labels (BIO correto)
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
    "B-PROCESS_ID", "I-PROCESS_ID"
]
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for l, i in label2id.items()}

# -----------------------------
# Dataset
# -----------------------------
dataset = load_dataset(
    "json",
    data_files={
        "train": "data_ner/train_ner.jsonl",
        "validation": "data_ner/val_ner.jsonl"
    }
)

model_name = "pierreguillou/bert-base-cased-pt-lenerbr"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        padding="max_length",
        max_length=512,
        is_split_into_words=True
    )
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            else:
                label_name = label[word_idx]
                if label_name not in label2id:
                    label_name = "O"
                if word_idx != previous_word_idx:
                    label_ids.append(label2id[label_name])
                else:
                    label_ids.append(-100)  # subword
                previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

# -----------------------------
# Modelo
# -----------------------------
model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

# -----------------------------
# Métricas
# -----------------------------
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
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
        "mean_token_accuracy": mean_token_accuracy
    }

# -----------------------------
# Treinamento
# -----------------------------
args = TrainingArguments(
    output_dir="./ner-anon-model",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=1,
    num_train_epochs=10,
    weight_decay=1e-2,
    logging_dir="./logs",
    logging_steps=1,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
)

data_collator = DataCollatorForTokenClassification(tokenizer)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

trainer.train()
