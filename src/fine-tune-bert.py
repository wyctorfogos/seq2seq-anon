from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, TrainingArguments, Trainer

model_name = "neuralmind/bert-base-portuguese-cased"

# Dataset no formato HuggingFace (tokens + labels)
dataset = load_dataset("json", data_files={"train": "data_ner/train_ner.jsonl", "validation": "data_ner/val_ner.jsonl"})

label_list = ["O", "B-PER", "I-PER", "B-CPF", "B-O", "I-CPF", "B-ADDR", "I-ADDR", "B-ORG", "B-AGE", "I-ORG", "I-O", "B-EMAIL", "I-EMAIL", "B-PHONE", "I-PHONE", "B-DATE", "I-DATE", "B-RG", "I-RG", "B-CNPJ", "I-CNPJ"]
label2id = {l:i for i,l in enumerate(label_list)}
id2label = {i:l for l,i in label2id.items()}

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
                    label_name = "O"  # fallback para valores fora do esquema
                if word_idx != previous_word_idx:
                    label_ids.append(label2id[label_name])
                else:
                    # subword → marca como -100 para não calcular loss
                    label_ids.append(-100)
                previous_word_idx = word_idx
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(label_list), id2label=id2label, label2id=label2id)

args = TrainingArguments(
    output_dir="./ner-anon-model",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=5,
    weight_decay=0.01,
)

data_collator = DataCollatorForTokenClassification(tokenizer)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
