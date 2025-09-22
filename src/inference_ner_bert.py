import os
import json
import numpy as np
import pandas as pd
import models
from models.anonymizer_class import SentenceAnonymyzer
from typing import List, Dict, Any, Tuple
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

if __name__ == "__main__":
    model_dir = "./ner-anon-model/special-tokens/checkpoint-71" #"./ner-anon-model/checkpoint-10000" # "celiudos/legal-bert-lgpd"
    anonymizer_model = SentenceAnonymyzer(model_dir=model_dir, threshold=0.75)

    inference_final_results = f"./results/anonymized_{str(model_dir).replace('/', '-')}_added-tokens.jsonl"
    with open(inference_final_results, "w", encoding="utf8") as fout:
        for text in read_inputs("./data/to_test/sentences2test.txt"):
            # Anonimização com Regex sobre o texto

            # Anonimização com o modelo de IA
            masked, ner_entities = anonymizer_model.anonymize_text(
                text=text.replace("\n", " ").replace("'\'", ""),
                width=100
            )
            
            # A função agora retorna o texto final e as entidades encontradas pelo regex
            masked, regex_entities = anonymizer_model.apply_regex_anonymization(text=masked)
            
            # Combina as listas de entidades (IA + Regex)
            all_entities = ner_entities + regex_entities
            
            # Converte a lista completa de entidades para ser serializável em JSON
            ents_serializable = anonymizer_model.make_json_serializable(all_entities)

            fout.write(json.dumps({
                "original": text,
                "masked": masked, # Usa o 'masked' final, após IA e Regex
                "entities": ents_serializable
            }, ensure_ascii=False) + "\n")

    print(f"Dados salvos em:{inference_final_results}\n")