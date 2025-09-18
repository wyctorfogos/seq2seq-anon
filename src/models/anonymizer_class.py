from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import os
import json
import numpy as np
import re
import torch
from typing import List, Dict, Any, Tuple

class SentenceAnonymyzer():
    def __init__(self, model_dir:str = "pierreguillou/bert-base-cased-pt-lenerbr", threshold:float=0.75):
        # Caminho do checkpoint treinado
        # model_dir = "./ner-anon-model/checkpoint-800"  # ajuste para o melhor checkpoint
        self.model_dir = model_dir # "celiudos/legal-bert-lgpd"
        self.threshold = threshold
        # Carregar tokenizer e modelo
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)

        self.model = AutoModelForTokenClassification.from_pretrained(self.model_dir)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Criar pipeline de NER
        self.nlp = pipeline(
            "ner",
            model = self.model,
            tokenizer = self.tokenizer,
            aggregation_strategy="simple",
            device = self.device
        )

    def chunk_text(self, text, tokenizer, max_tokens=250, stride=200):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            # Use convert_tokens_to_string to avoid adding extra spaces
            chunk = tokenizer.convert_tokens_to_string(
                tokenizer.convert_ids_to_tokens(chunk_tokens)
            )
            chunks.append(chunk)
            if end == len(tokens):
                break
            start += stride
        return chunks

    # -------------------
    # Função de anonimização de um texto (sem chunking)
    # -------------------
    def anonymize_text(self, text: str):
        entities = self.nlp(text)
        anonymized_text = text
        for ent in sorted(entities, key=lambda x: x["start"], reverse=True):
            score = float(ent["score"])
            if score >= self.threshold:
                start, end = ent["start"], ent["end"]
                anonymized_text = anonymized_text[:start] + f"<{ent['entity_group']}>" + anonymized_text[end:]
        return anonymized_text, entities

    # -------------------
    # Função para textos longos com chunking
    # -------------------
    def anonymize_long_text(self, text, tokenizer, max_tokens=250, stride=200):
        chunks = self.chunk_text(text, tokenizer, max_tokens=max_tokens, stride=stride)

        final_text = []
        all_entities = []

        for chunk in chunks:
            anon_chunk, ents = self.anonymize_text(text=chunk)
            final_text.append(anon_chunk)
            all_entities.extend(ents)

        final_text = " ".join(final_text)
        final_text=str(final_text).replace(". ", ".").replace(" / ", "/").replace(" - ","-")
        return final_text, all_entities

    def make_json_serializable(self, obj):
        """
            Converte tipos não-serializáveis (ex: np.float32) para tipos nativos do Python.
        """
        if isinstance(obj, dict):
            return {k: self.make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.make_json_serializable(v) for v in obj]
        elif isinstance(obj, (float, int, str, bool)) or obj is None:
            return obj
        else:
            try:
                return float(obj)
            except Exception:
                return str(obj)

        
    def apply_regex_anonymization(self, text: str):
        """
        Aplica regras de Regex para anonimizar o texto e retorna tanto o texto
        modificado quanto as entidades que foram encontradas e substituídas.
        """
        regex_entities = []
        
        # Função que será chamada para cada match encontrado pelo re.sub
        # Ela salva a entidade e retorna a tag para substituição.
        def find_and_log_entity(match, tag):
            # Extrai a informação do objeto 'match'
            word = match.group(0)
            start, end = match.span()
            
            # Cria o dicionário da entidade, no mesmo formato do pipeline da Hugging Face
            entity = {
                "entity_group": tag.strip("<>"), # Salva o nome da tag sem '<' e '>'
                "score": 1.0,                     # Score de confiança 1.0, pois Regex é determinístico
                "word": word,
                "start": start,
                "end": end
            }
            regex_entities.append(entity)
            
            # Retorna a tag que substituirá o texto encontrado
            return tag

        substitutions = [
            (re.compile(r'\b(?:\d{3}\.\d{3}\.\d{3}-\d{2}|\d{11})\b'), "<CPF>"),
            (re.compile(r'\b(?:\d{2}\.\d{3}\.\d{3}\/\d{4}-\d{2}|\d{14})\b'), "<CNPJ>"),
            (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'), "<EMAIL>"),
            (re.compile(r'\b(?:\+55\s?)?(?:\(?\d{2}\)?\s?)?(?:9?\d{4}[-\s]?\d{4})\b'), "<PHONE>"),
            (re.compile(r'\b\d{5}-?\d{3}\b'), "<CEP>"),
            (re.compile(r'\b(?:\d{7}-\d{2}\.\d{4}\.\d{1}\.\d{2}\.\d{4}|\d{16})\b'), "<PROCESS_ID>"),
            (re.compile(r'\b\d{2}[\/\-]\d{2}[\/\-]\d{2,4}\b'), "<DATE>"),
            (re.compile(r'\b[A-Za-z]\s?\d{6}\b'), "<OAB>")
        ]

        # Itera sobre cada padrão e aplica a substituição
        for pattern, tag in substitutions:
            # Usamos uma função lambda para passar a 'tag' para nosso logger
            text = pattern.sub(lambda m: find_and_log_entity(m, tag), text)
                
        return text, regex_entities

