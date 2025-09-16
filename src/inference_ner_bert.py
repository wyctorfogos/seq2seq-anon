from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Caminho do checkpoint treinado
model_dir = "./ner-anon-model/checkpoint-250"  # ajuste para o checkpoint que deu melhor resultado

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

def anonymize_text(text: str):
    """
    Executa anonimização no texto substituindo entidades por placeholders.
    """
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


# ----------------------------
# Exemplo de uso
# ----------------------------
if __name__ == "__main__":
    text = "A empresa ACME Ltda., CNPJ 12.345.678/0001-90, tem como contato o responsável Carlos Pereira (tel: +55 21 99888-7766)."

    anon_text, ents = anonymize_text(text)

    print("Original:", text)
    print("Anonimizado:", anon_text)
    print("Entidades detectadas:")
    for e in ents:
        print(e)
