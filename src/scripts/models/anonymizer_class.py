import os
import re
import textwrap
import torch
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


class SentenceAnonymyzer():
    def __init__(self, model_dir: str = "pierreguillou/bert-large-cased-pt-lenerbr", threshold: float = 0.85):
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
        """Divide o texto em blocos menores de até 'width' caracteres."""
        return textwrap.wrap(text, width=width, replace_whitespace=False)

    # -------------------
    # Anonimização via modelo NER
    # -------------------
    def anonymize_text(self, text: str, width: int = 445):
        """
        Anonimiza texto usando modelo NER, reutilizando IDs por entidade.
        Ignora placeholders já anonimizados (<TAG_1> etc.).
        """
        # Ignorar blocos que contêm apenas placeholders
        def clean_for_ner(segment):
            # Substitui tags por espaços (para manter offsets)
            return re.sub(r"<[A-Z_0-9]+>", lambda m: " " * len(m.group(0)), segment)

        original_blocks = self.split_text(text=text, width=width)
        lower_blocks = [clean_for_ner(b.lower()) for b in original_blocks]

        anonymized_blocks = []
        all_entities = []
        entity_map = {}
        entity_counters = {}
        global_offset = 0

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

                # Ignora se sobrepõe a placeholder já existente
                if re.search(r"<[A-Z_0-9]+>", anon_block[start:end]):
                    continue

                original_word = anon_block[start:end]
                normalized_word = original_word.lower().strip()
                key = (label, normalized_word)

                if key not in entity_map:
                    count = entity_counters.get(label, 0) + 1
                    entity_counters[label] = count
                    entity_map[key] = f"<{label}_{count}>"

                placeholder = entity_map[key]
                anon_block = anon_block[:start] + placeholder + anon_block[end:]
                offset += len(placeholder) - len(original_word)

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

        reverse_map = {v: k[1] for k, v in entity_map.items()}
        return " ".join(anonymized_blocks), all_entities, reverse_map

    # -------------------
    # Regex complementar
    # -------------------
    def apply_regex_anonymization(self, text: str):
        """Anonimiza texto via regex, criando placeholders numerados."""
        regex_entities = []
        entity_counters = {}
        entity_map = {}

        def find_and_log_entity(match, tag_label):
            original_word = match.group(0)
            normalized_word = original_word.lower().strip()
            key = (tag_label, normalized_word)
            start, _ = match.span()

            if key not in entity_map:
                count = entity_counters.get(tag_label, 0) + 1
                entity_counters[tag_label] = count
                entity_map[key] = f"<{tag_label}_{count}>"

            placeholder = entity_map[key]
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

        substitutions = [
            # URLs primeiro (para proteger links com números)
            (re.compile(r'(?:https?://|www\.)[^\s]+', re.IGNORECASE), "<URL>"),
            # Documentos e IDs oficiais
            (re.compile(r'\b\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}\b|\b\d{20}\b'), "<PROCESS_ID>"),
            (re.compile(r'\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b|\b\d{14}\b'), "<CNPJ>"),
            (re.compile(r'\b\d{3}\.\d{3}\.\d{3}-\d{2}\b|\b\d{11}\b'), "<CPF>"),
            # Demais padrões
            (re.compile(r'\bBU\s+\d{8}\b', re.IGNORECASE), "<BU_ID>"),
            (re.compile(r'\b(?:OAB\s*/?\s*)?[A-Z]{2}\s*\d{4,6}\b', re.IGNORECASE), "<OAB>"),
            (re.compile(r'\b\d{5}-?\d{3}\b'), "<CEP>"),
        ]


        for pattern, tag_label in substitutions:
            text = pattern.sub(lambda m: find_and_log_entity(m, tag_label), text)

        reverse_map = {v: k[1] for k, v in entity_map.items()}
        return text, regex_entities, reverse_map

    # -------------------
    # Serialização
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

    # -------------------
    # Pipeline principal (Regex → NER)
    # -------------------
    def pipeline(self, input_text: str, width: int = 100):
        try:
            url_pattern = re.compile(r'(https?://[^\s,;<>")]+|www\.[^\s,;<>")]+)', re.IGNORECASE)
            url_entities = []
            entity_counters = {"URL": 0}

            def protect_url(match):
                entity_counters["URL"] += 1
                placeholder = f"<URL_{entity_counters['URL']}>"
                url_entities.append({
                    "id": placeholder,
                    "entity_group": "URL",
                    "word": match.group(0),
                    "word_lower": match.group(0).lower(),
                    "score": 1.0,
                    "start": match.start(),
                    "end": match.end()
                })
                return placeholder

            protected_text = url_pattern.sub(protect_url, input_text)

            masked, ner_entities, _ = self.anonymize_text(
                text=protected_text,
                width=width
            )

            masked, regex_entities, _ = self.apply_regex_anonymization(text=masked)

            all_entities = url_entities + ner_entities + regex_entities

            return {
                "original": input_text,
                "masked": masked,
                "entities": self.make_json_serializable(all_entities)
            }

        except Exception as e:
            raise Exception(f"Erro ao processar os dados: {e}")
