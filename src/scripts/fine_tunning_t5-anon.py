from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
from peft import LoraConfig, get_peft_model

# Dataset
dataset = load_dataset("json", data_files={"train": "data_seq2seq/train.jsonl", "validation": "data_seq2seq/val.jsonl"})

model_id = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)

def preprocess(example):
    return tokenizer(example["input"], text_target=example["output"], truncation=True)
tokenized = dataset.map(preprocess, batched=True)

# Modelo com LoRA
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, load_in_8bit=True, device_map="auto")
lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["k", "q", "v"], lora_dropout=0.05)
model = get_peft_model(model, lora_config)

# Treinamento
args = Seq2SeqTrainingArguments(
    output_dir="finetuned-anon",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=2,
    learning_rate=2e-4,
    num_train_epochs=5,
    do_eval=True,
    fp16=False,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_total_limit=2,
    predict_with_generate=True
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()
