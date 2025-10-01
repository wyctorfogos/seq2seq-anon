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
            Anonimiza o texto bloco a bloco, reutilizando IDs por entidade (ex: <PER_1>).
            Processa em minúsculas para consistência, mas preserva a estrutura e capitalização original.
            Mantém offsets globais e retorna mapa de entidades.
        """
        original_blocks = self.split_text(text=text, width=width)
        lower_blocks = self.split_text(text.lower(), width=width)

        anonymized_blocks = []
        all_entities = []
        entity_map = {}      
        entity_counters = {}  
        global_offset = 0  
        # Percorre blocos paralelamente (lowercase e original)
        for i, block in enumerate(lower_blocks):
            entities = self.nlp(block)
            anon_block = original_blocks[i]
            offset = 0

            for ent in sorted(entities, key=lambda x: x["start"]):
                score = float(ent["score"])
                if score < self.threshold:
                    continue

                start = ent["start"] + offset
                end = ent["end"] + offset
                label = ent["entity_group"]

                # Palavra original (mantém capitalização)
                original_word = anon_block[start:end]
                # Palavra minúscula (para mapear de forma consistente)
                normalized_word = original_word.lower().strip()

                key = (label, normalized_word)

                # Reutiliza ou cria ID novo
                if key not in entity_map:
                    count = entity_counters.get(label, 0) + 1
                    entity_counters[label] = count
                    entity_map[key] = f"<{label}_{count}>"

                placeholder = entity_map[key]

                # Substitui no texto original (mantendo maiúsculas/minúsculas)
                anon_block = anon_block[:start] + placeholder + anon_block[end:]
                offset += len(placeholder) - len(original_word)

                # Adiciona metadados da entidade
                all_entities.append({
                    "id": placeholder,
                    "entity_group": label,
                    "word": original_word,      
                    "word_lower": normalized_word,
                    "score": score,
                    "start": global_offset + start,
                    "end": global_offset + start + len(placeholder)
                })

            anonymized_blocks.append(anon_block)
            global_offset += len(original_blocks[i]) + 1

        # Mapa reverso: <TAG> -> palavra original (na primeira ocorrência)
        reverse_map = {v: k[1] for k, v in entity_map.items()}

        return " ".join(anonymized_blocks), all_entities, reverse_map

    # -------------------
    # Regex complementar
    # -------------------
    def apply_regex_anonymization(self, text: str):
        """
        Aplica anonimização baseada em regex, preservando o texto original.
        Reutiliza IDs por tipo (ex: <CPF_1>), registra forma original e minúscula.
        """
        regex_entities = []
        entity_counters = {}
        entity_map = {}

        def find_and_log_entity(match, tag_label):
            original_word = match.group(0)
            normalized_word = original_word.lower().strip()

            key = (tag_label, normalized_word)
            start, end = match.span()

            # Reutiliza ID para a mesma entidade
            if key not in entity_map:
                count = entity_counters.get(tag_label, 0) + 1
                entity_counters[tag_label] = count
                entity_map[key] = f"<{tag_label}_{count}>"

            placeholder = entity_map[key]

            # Armazena metadados da entidade
            regex_entities.append({
                "id": placeholder,
                "entity_group": tag_label,
                "score": 1.0,
                "word": original_word,
                "word_lower": normalized_word,
                "start": start,
                "end": start + len(placeholder)
            })

            return placeholder

        # Padrões de substituição
        substitutions = [
            (re.compile(r'\b(?:\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}|\d{20})\b'), "<PROCESS_ID>"),
            (re.compile(r'\b(?:\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}|\d{14})\b'), "<CNPJ>"),
            (re.compile(r'\b(?:\d{3}\.\d{3}\.\d{3}-\d{2}|\d{11})\b'), "<CPF>"),
            (re.compile(r'\b(?:https?://|www\.)[^\s/$.?#]+[^\s]*\.(?:com|br)\b', re.IGNORECASE), "<URL>"),
            (re.compile(r'\b(?:cnh|habilitação|carteira)[\s\S]*?(\d{11})\b', re.IGNORECASE), "<CNH>"),
            (re.compile(r'\bBU\s+\d{8}\b', re.IGNORECASE), "<BU_ID>"),
            (re.compile(r'\b(?:\+?55\s*)?(?:(?:\(\d{2}\)|\d{2})\s*)?(?:9\d{4}|\d{4})[-\s]?\d{4}\b'), "<PHONE>"),
            (re.compile(r'\b\d{1,2}\.?\d{3}\.?\d{3}-?[\dXx]\b', re.IGNORECASE), "<RG>"),
            (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'), "<EMAIL>"),
            (re.compile(r'\b[A-Z]{1,2}\s?\d{2,6}\b', re.IGNORECASE), "<OAB>"),
            (re.compile(r'\b\d{5}-?\d{3}\b'), "<CEP>"),
            (re.compile(r"\b\d{6,}\b"), "<ID>")
        ]

        # Executa substituições com IDs únicos por tipo
        for pattern, tag_label in substitutions:
            text = pattern.sub(lambda m: find_and_log_entity(m, tag_label), text)

        # Cria mapa reverso <TAG> -> palavra original (primeira ocorrência)
        reverse_map = {v: k[1] for k, v in entity_map.items()}

        return text, regex_entities, reverse_map


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
            masked, ner_entities, _ = self.anonymize_text(
                text=input_text,
                width=width
            )
            # A função agora retorna o texto final e as entidades encontradas pelo regex
            masked, regex_entities, _ = self.apply_regex_anonymization(text=masked)            
            # Combina as listas de entidades (IA + Regex)
            all_entities = ner_entities + regex_entities
            
            # Converte a lista completa de entidades para ser serializável em JSON
            ents_serializable = self.make_json_serializable(all_entities)

            final_response = {
                "original": input_text,
                "masked": masked, 
                "entities": ents_serializable
            }

            return final_response
        except Exception as e:
            raise Exception(f"Erro ao processar os dados!")
