# inference_masking.py
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
import argparse
import os
import math

# -------------------
# Função de chunking por tokens
# -------------------
def chunk_text(text, tokenizer, max_tokens=250):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    num_chunks = math.ceil(len(tokens) / max_tokens)
    chunks = []
    for i in range(num_chunks):
        start = i * max_tokens
        end = (i + 1) * max_tokens
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
    return chunks

# -------------------
# Parser de argumentos
# -------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", default="./finetuned-anon")  # diretório do modelo fine-tuned
parser.add_argument("--input_file", required=True)   # arquivo com textos brutos (1 por linha ou JSONL com field "text")
parser.add_argument("--out_file", default="masked_outputs.jsonl")
parser.add_argument("--chunk_size", type=int, default=250, help="Tamanho máximo de tokens por chunk")
args = parser.parse_args()

# -------------------
# Carrega modelo
# -------------------
tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)

# -------------------
# Leitura de inputs
# -------------------
def read_inputs(path):
    # aceita linhas simples ou JSONL com field "text"
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
                text = obj.get("text") or obj.get("input") or obj.get("original") or line
            except Exception:
                text = line
            yield text

# -------------------
# Inferência com chunking
# -------------------
def anonymize_long_text(text, model, tokenizer, max_tokens=100):
    chunks = chunk_text(text, tokenizer, max_tokens=max_tokens)
    masked_chunks = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=max_tokens)
        outputs = model.generate(**inputs, max_length=max_tokens, do_sample=False)
        masked_chunk = tokenizer.decode(outputs[0], skip_special_tokens=True)
        masked_chunks.append(masked_chunk)
    return " ".join(masked_chunks)

# -------------------
# Loop principal
# -------------------
with open(args.out_file, "w", encoding="utf8") as fout:
    for text in read_inputs(args.input_file):
        masked = anonymize_long_text(text, model, tokenizer, max_tokens=args.chunk_size)
        fout.write(json.dumps({"masked": masked}, ensure_ascii=False) + "\n")

print("✅ Inferência concluída. Resultados salvos em", args.out_file)
