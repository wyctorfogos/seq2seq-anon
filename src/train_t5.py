# train_t5.py
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="t5-small")
parser.add_argument("--data_dir", default="data_seq2seq")
parser.add_argument("--output_dir", default="./t5-anon")
parser.add_argument("--num_train_epochs", type=int, default=3)
parser.add_argument("--per_device_train_batch_size", type=int, default=2)
parser.add_argument("--learning_rate", type=float, default=5e-5)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

ds_tokenized = load_from_disk(os.path.join(args.data_dir, f"tokenized_{args.model.replace('/','_')}"))
train_ds = ds_tokenized["train"]
eval_ds = ds_tokenized["validation"]

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    output_dir=args.output_dir,
    num_train_epochs=args.num_train_epochs,
    per_device_train_batch_size=args.per_device_train_batch_size,
    per_device_eval_batch_size=args.per_device_train_batch_size,
    predict_with_generate=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=args.learning_rate,
    logging_steps=200,
    fp16=False 
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()
trainer.save_model(args.output_dir)
print("Treino finalizado. Modelo salvo em", args.output_dir)
