# inference_causal_masking.py
import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# -------------------
# Função de leitura de inputs
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
# Função de anonimização
# -------------------
def anonymize_text(text, model, tokenizer, device, max_new_tokens=256):
    # prompt do treino
    prompt = (
        "Anonimize os dados pessoais (nome completo, CPF, endereço, e-mail, telefone, etc.) "
        "no texto a seguir substituindo-os por tags como [NOME], [CPF], [EMAIL].\n\n"
        f"Texto original: {text}\n"
        "Texto anonimizado:"
    )
    messages = [{"role": "user", "content": prompt}]

    # aplica template de chat
    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    # remove tokens do input
    generated_tokens = outputs[0][inputs.shape[1]:]
    masked_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return masked_text

# -------------------
# Main
# -------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="Diretório do modelo ou repo HF")
    parser.add_argument("--input_file", required=True, help="Arquivo com textos (1 por linha ou JSONL)")
    parser.add_argument("--out_file", default="masked_outputs.jsonl", help="Arquivo de saída JSONL")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir).to(device)
    model.eval()

    with open(args.out_file, "w", encoding="utf8") as fout:
        for text in tqdm(read_inputs(args.input_file), desc="Anonimizando"):
            masked = anonymize_text(text, model, tokenizer, device, max_new_tokens=512)
            fout.write(json.dumps({"original": text, "masked": masked}, ensure_ascii=False) + "\n")

    print(f"✅ Inferência concluída. Resultados salvos em {args.out_file}")

if __name__ == "__main__":
    main()
