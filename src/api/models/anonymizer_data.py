import os
import re
import textwrap
import torch
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


class SentenceAnonymyzer():
    def __init__(self, model_dir: str = "pierreguillou/bert-large-cased-pt-lenerbr", threshold: float = 0.75):
        self.model_dir = model_dir
        self.threshold = threshold

        # Carregar tokenizer e modelo
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_dir)

        self.device = 0 if torch.cuda.is_available() else -1
        self.nlp = pipeline(
            "ner",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple",
            device=self.device
        )

    # -------------------
    # Divisão em blocos
    # -------------------
    def split_text(self, text: str, width: int = 445) -> list:
        """
        Divide o texto em blocos menores de até 'width' caracteres.
        Útil para textos longos que não cabem no modelo de uma vez.
        """
        return textwrap.wrap(text, width=width, replace_whitespace=False)

    # -------------------
    # Anonimização bloco a bloco
    # -------------------
    def anonymize_text(self, text: str, width: int = 445):
        """
        Anonimiza um texto usando o pipeline de NER.
        Passa a versão minúscula para o modelo,
        mas preserva o texto original ao substituir.
        """
        blocks = self.split_text(text, width=width)

        anonymized_blocks = []
        all_entities = []

        for block in blocks:
            entities = self.nlp(str(block).lower()  )
            all_entities.extend(entities)

            anon_block = block
            offset = 0

            # Ordena entidades para evitar desalinhamento
            for ent in sorted(entities, key=lambda x: x["start"]):
                score = float(ent["score"])
                if score < self.threshold:
                    continue

                start = ent["start"] + offset
                end = ent["end"] + offset
                label = ent["entity_group"]
                placeholder = f"[{label}]"

                original = anon_block[start:end]
                anon_block = anon_block[:start] + placeholder + anon_block[end:]

                offset += len(placeholder) - len(original)

            anonymized_blocks.append(anon_block)

        return " ".join(anonymized_blocks), all_entities


    # -------------------
    # Regex complementar
    # -------------------
    def apply_regex_anonymization(self, text: str):
        regex_entities = []

        def find_and_log_entity(match, tag):
            word = match.group(0)
            start, end = match.span()
            entity = {
                "entity_group": tag.strip("<>"),
                "score": 1.0,
                "word": word,
                "start": start,
                "end": end
            }
            regex_entities.append(entity)
            return tag

        substitutions = [
            # 1. Padrões com formatos muito distintos e longos
            (re.compile(r'\b(?:\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}|\d{20})\b'), "<PROCESS_ID>"),
            (re.compile(r'\b(?:\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}|\d{14})\b'), "<CNPJ>"),

            # 2. Padrões com palavras-chave específicas + números
            (re.compile(r'\b(?:cnh|habilitação|carteira)[\s\S]*?(\d{11})\b', re.IGNORECASE), "<CNH>"),
            (re.compile(r'\bBU\s+\d{8}\b', re.IGNORECASE), "<BU_ID>"),

            # 3. Padrões numéricos comuns (CPF é mais específico que t  elefone de 11 dígitos)
            (re.compile(r'\b(?:\d{3}\.\d{3}\.\d{3}-\d{2}|\d{11})\b'), "<CPF>"),
            (re.compile(r'\b(?:\+?55\s*)?(?:(?:\(\d{2}\)|\d{2})\s*)?(?:9\d{4}|\d{4})[-\s]?\d{4}\b'), "<PHONE>"),
            (re.compile(r'\b\d{1,2}\.?\d{3}\.?\d{3}-?[\dXx]\b', re.IGNORECASE), "<RG>"),

            # 4. Outros formatos específicos
            (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'), "<EMAIL>"),
            (re.compile(r'\b(?:https?://|www\.)[^\s/$.?#].[^\s]*\b', re.IGNORECASE), "<URL>"),
            (re.compile(r'\b[A-Z]{1,2}\s?\d{2,6}\b', re.IGNORECASE), "<OAB>"),
            (re.compile(r'\b\d{5}-?\d{3}\b'), "<CEP>"),
            (re.compile(r'\b\d{2}[/-]\d{2}[/-]\d{2,4}\b'), "<DATE>"),

            # 5. Padrão genérico por último para capturar o que sobrou
            (re.compile(r"\b\d{6,}\b"), "<ID>")
        ]
        for pattern, tag in substitutions:
            text = pattern.sub(lambda m: find_and_log_entity(m, tag), text)

        return text, regex_entities

    # -------------------
    # Utilitário de serialização
    # -------------------
    def make_json_serializable(self, obj):
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
            
        
    def pipeline(self, input_text:str, width:int=100):
        try:
           
            # Anonimização com o modelo de IA
            masked, ner_entities = self.anonymize_text(
                text=input_text.replace("\n", " ").replace("'\'", ""),
                width=width
            )
            # A função agora retorna o texto final e as entidades encontradas pelo regex
            masked, regex_entities = self.apply_regex_anonymization(text=masked)            
            # Combina as listas de entidades (IA + Regex)
            all_entities = ner_entities + regex_entities
            
            # Converte a lista completa de entidades para ser serializável em JSON
            ents_serializable = self.make_json_serializable(all_entities)

            final_response = {
                "original": input_text,
                "masked": masked, # Usa o 'masked' final, após IA e Regex
                "entities": ents_serializable
            }

            return final_response
        except Exception as e:
            raise Exception(f"Erro ao processar os dados!")
