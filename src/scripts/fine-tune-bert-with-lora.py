from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model
import numpy as np
import evaluate

# -----------------------------
# Labels (BIO)
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
# Modelo base + Tokenizer
# -----------------------------
model_name = "pierreguillou/bert-base-cased-pt-lenerbr"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Dataset
dataset = load_dataset(
    "json",
    data_files={
        "train": "./data/data_ner/train_ner.jsonl",
        "validation": "./data/data_ner/val_ner.jsonl"
    }
)

# -----------------------------
# Tokenização e alinhamento
# -----------------------------
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
                    label_ids.append(-100)
                previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

# -----------------------------
# Modelo com LoRA
# -----------------------------
base_model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

# Configuração LoRA
lora_config = LoraConfig(
    r=16,  # rank do LoRA (quanto maior, mais capacidade)
    lora_alpha=32,
    target_modules=["query", "value"],  # aplica nos blocos de atenção
    lora_dropout=0.05,
    bias="none",
    task_type="TOKEN_CLS"
)

model = get_peft_model(base_model, lora_config)

print("Camadas treináveis com LoRA:")
model.print_trainable_parameters()

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

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# -----------------------------
# Treinamento
# -----------------------------
args = TrainingArguments(
    output_dir="./ner-anon-model/lora/",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
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

# -----------------------------
# Salvando modelo com LoRA
# -----------------------------
trainer.save_model("./ner-anon-model/lora/best")
