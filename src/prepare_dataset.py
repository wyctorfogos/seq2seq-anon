# prepare_dataset.py
from datasets import load_dataset
from transformers import AutoTokenizer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="t5-small")
parser.add_argument("--data_dir", default="data_seq2seq")
parser.add_argument("--max_input_len", type=int, default=256)
parser.add_argument("--max_target_len", type=int, default=256)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model)

def preprocess(example):
    inp = example["input"]
    tgt = example["output"]
    enc = tokenizer(inp, truncation=True, padding="max_length", max_length=args.max_input_len)
    with tokenizer.as_target_tokenizer():
        lab = tokenizer(tgt, truncation=True, padding="max_length", max_length=args.max_target_len)
    enc["labels"] = lab["input_ids"]
    return enc

ds = load_dataset("json", data_files={
    "train": f"{args.data_dir}/train.jsonl",
    "validation": f"{args.data_dir}/val.jsonl"
})

tok = ds.map(preprocess, batched=False, remove_columns=["input","output"])
tok.save_to_disk(f"{args.data_dir}/tokenized_{args.model.replace('/','_')}")
print("Dataset tokenizado salvo em", f"{args.data_dir}/tokenized_{args.model.replace('/','_')}")
