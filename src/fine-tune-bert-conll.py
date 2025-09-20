import os
os.environ["CUDA_LAUNCH_BLOCKING"]="1"
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

# -----------------------------
# Métrica
# -----------------------------
seqeval = evaluate.load("seqeval")

# -----------------------------
# Labels (CORRIGIDO: Lista completa com todas as entidades)
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
    "B-CREDIT_CARD", "I-CREDIT_CARD", # <-- Adicionado
    "B-IP", "I-IP"                   # <-- Adicionado
]
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for l, i in label2id.items()}

# -----------------------------
# Dataset (CORRIGIDO: Carregando os arquivos .json corretos)
# -----------------------------
dataset = load_dataset(
    "json",
    data_files={
        "train": "data_ner/train.json",
        "validation": "data_ner/validation.json"
    }
)

tokenizer_name = "pierreguillou/bert-base-cased-pt-lenerbr"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# -----------------------------
# Função de Tokenização (CORRIGIDA e OTIMIZADA)
# -----------------------------
def tokenize_and_align_labels(examples):
    # O tokenizer é aplicado aos tokens pré-divididos
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=512,
        padding="max_length"
    )

    labels = []
    # CORRIGIDO: usa "ner_tags_str" que é a chave correta do nosso JSON
    for i, label in enumerate(examples["ner_tags_str"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Atribui -100 para tokens especiais como [CLS] e [SEP]
            if word_idx is None:
                label_ids.append(-100)
            # Se estamos no primeiro token de uma palavra, usamos seu label
            elif word_idx != previous_word_idx:
                label_name = label[word_idx]
                label_ids.append(label2id.get(label_name, label2id["O"]))
            # OTIMIZADO: Para sub-tokens da mesma palavra, propagamos o label "I-"
            else:
                label_name = label[word_idx]
                # Se o label original era B-TAG, o sub-token se torna I-TAG
                if label_name.startswith("B-"):
                    label_name = "I-" + label_name[2:]
                label_ids.append(label2id.get(label_name, label2id["O"]))
            
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

# -----------------------------
# Modelo
# -----------------------------
model = AutoModelForTokenClassification.from_pretrained(
    tokenizer_name,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

# -----------------------------
# Função de Métricas (sem alterações, já estava correta)
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
    output_dir="./ner-anon-model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,          # Lote pequeno para caber na memória
    per_device_eval_batch_size=4,           # Lote de avaliação também pode precisar ser reduzido
    gradient_accumulation_steps=2,          # Efetivo batch_size = 2 * 4 = 8
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
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
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

print("Iniciando o treinamento do modelo de NER...")
trainer.train()
print("Treinamento concluído!")

# Salva o melhor modelo no final
trainer.save_model("./ner-anon-model/best-model")