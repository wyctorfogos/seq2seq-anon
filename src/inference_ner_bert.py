from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import os
import json

# Caminho do checkpoint treinado
# model_dir = "./ner-anon-model/checkpoint-800"  # ajuste para o melhor checkpoint
model_dir = "celiudos/legal-bert-lgpd" # "pierreguillou/ner-bert-large-cased-pt-lenerbr" # "celiudos/legal-bert-lgpd"
# Carregar tokenizer e modelo
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForTokenClassification.from_pretrained(model_dir)

# Criar pipeline de NER
nlp = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple"  # junta tokens de uma mesma entidade
)

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
# Função de anonimização de um texto (sem chunking)
# -------------------
def anonymize_text(text: str):
    entities = nlp(text)
    anonymized_text = text

    # Substituir de trás para frente para não bagunçar os índices
    for ent in sorted(entities, key=lambda x: x["start"], reverse=True):
        label = ent["entity_group"]
        start, end = ent["start"], ent["end"]

        # Marcador genérico para a entidade
        replacement = f"<{label}>"
        anonymized_text = anonymized_text[:start] + replacement + anonymized_text[end:]

    return anonymized_text, entities

# -------------------
# Função para textos longos com chunking
# -------------------
def anonymize_long_text(text, tokenizer, max_tokens=250, stride=200):
    chunks = chunk_text(text, tokenizer, max_tokens=max_tokens, stride=stride)

    final_text = []
    all_entities = []

    for chunk in chunks:
        anon_chunk, ents = anonymize_text(chunk)
        final_text.append(anon_chunk)
        all_entities.extend(ents)

    return " ".join(final_text), all_entities

def make_json_serializable(obj):
    """
    Converte tipos não-serializáveis (ex: np.float32) para tipos nativos do Python.
    """
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    else:
        try:
            return float(obj)
        except Exception:
            return str(obj)

# ----------------------------
# Exemplo de uso
# ----------------------------
if __name__ == "__main__":
    with open(f"./results/anonymized_{str(model_dir).replace("/","-")}.jsonl", "w", encoding="utf8") as fout:
        for text in read_inputs("./data_seq2seq/to_test/sentences2test.txt"):
            masked, ents = anonymize_long_text(
                text.replace("\n"," ").replace("'\'",""),
                tokenizer,
                max_tokens=256,
                stride=50
            )
            # Converte entities para JSON serializable
            ents_serializable = make_json_serializable(ents)

            fout.write(json.dumps({
                "original": text,
                "masked": masked,
                "entities": ents_serializable
            }, ensure_ascii=False) + "\n")
