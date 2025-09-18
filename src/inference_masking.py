import json
import math
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -------------------
# Função de chunking com overlap
# -------------------
def chunk_text(text, tokenizer, max_tokens=250, stride=200):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
        if end == len(tokens):
            break
        start += stride
    return chunks

# -------------------
# Parser de argumentos
# -------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", default="./finetuned-anon")  # diretório do modelo fine-tuned
parser.add_argument("--input_file", required=True)   # arquivo com textos brutos (1 por linha ou JSONL com field "text")
parser.add_argument("--out_file", default="masked_outputs.jsonl")
parser.add_argument("--chunk_size", type=int, default=250, help="Tamanho máximo de tokens por chunk")
parser.add_argument("--stride", type=int, default=200, help="Overlap entre chunks em tokens")
args = parser.parse_args()

# -------------------
# Carrega modelo
# -------------------
tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# -------------------
# Leitura de inputs
# -------------------
def read_inputs(path):
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = obj.get("text") or obj.get("input") or obj.get("original") or line
            except Exception:
                text = line
            yield text

# -------------------
# Inferência com chunking + batch
# -------------------
def anonymize_long_text(text, model, tokenizer, max_tokens=250, stride=200):
    chunks = chunk_text(text, tokenizer, max_tokens=max_tokens, stride=stride)
    
    # Tokeniza em batch
    inputs = tokenizer(chunks, return_tensors="pt", padding=True, truncation=True, max_length=max_tokens).to(device)
    
    # Geração com beam search
    outputs = model.generate(
        **inputs,
        max_length=max_tokens*2,
        min_length=max_tokens//2,
        num_beams=4,
        do_sample=False
    )
    masked_chunks = [tokenizer.decode(out, skip_special_tokens=True) for out in outputs]

    # Pós-processamento simples: remove duplicações entre chunks
    final_text = []
    for i, chunk in enumerate(masked_chunks):
        if i > 0 and chunk in masked_chunks[i-1]:
            continue  # evita duplicata
        final_text.append(chunk)

    return " ".join(final_text)

# -------------------
# Loop principal
# -------------------
with open(args.out_file, "w", encoding="utf8") as fout:
    for text in read_inputs(args.input_file):
        masked = anonymize_long_text(text, model, tokenizer, max_tokens=args.chunk_size, stride=args.stride)
        fout.write(json.dumps({"masked": masked}, ensure_ascii=False) + "\n")

print("✅ Inferência concluída. Resultados salvos em", args.out_file)
