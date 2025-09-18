"""
generate_masked_dataset.py

Gera um dataset sintético (pt_BR) com pares (original, masked),
e mantém mapeamentos seguros (HMAC por padrão; AES-GCM opcional).

Dependências:
  pip install faker cryptography

Uso:
  # gerar dataset de exemplo
  python generate_masked_dataset.py

Configurações:
  - NUM_EXAMPLES: quantos pares gerar
  - SAVE_ORIGINALS: se False, NÃO grava arquivo com textos originais (preserva privacidade)
  - ENABLE_AES_STORE: se True, grava mapeamento reversível (id -> ciphertext), requer AES_KEY_BASE64 env var
"""

import os
import json
import random
import re
import uuid
import base64
import hmac
import hashlib
from dataclasses import dataclass
from typing import List, Tuple, Dict

from faker import Faker

# opcional para criptografia reversível
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
except Exception:
    AESGCM = None

fake = Faker("pt_BR")
Faker.seed(42)
random.seed(42)

# ===== CONFIGURAÇÕES =====
NUM_EXAMPLES = 5000
SAVE_ORIGINALS = True   # se False, não salva arquivo com textos originais
OUTPUT_DIR = "output_masking"
MASK_TOKEN = "[DADO_MASCARADO]"

# HMAC key (para pseudonímia determinística)
# Em produção, troque por chave segura, preferencialmente vinda de KMS / env var
HMAC_KEY = os.environ.get("HMAC_KEY") or "troca_por_chave_segura_de_32_bytes_ou_mais"

# Se quiser transparência reversível (inseguro se chave vazada): ativa AES store
ENABLE_AES_STORE = False  # por padrão False. Se True, configurar AES_KEY_BASE64 no env
AES_KEY_BASE64 = os.environ.get("AES_KEY_BASE64")  # base64-encoded 32-byte key (AES-256)
# ==========================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# regexes de "sweep" final para segurança (garantir que padrões óbvios foram mascarados)
REGEXES = [
    re.compile(r'\b(?:\d{3}\.\d{3}\.\d{3}-\d{2}|\d{11})\b'),  # CPF
    re.compile(r'\b(?:\d{2}\.\d{3}\.\d{3}\/\d{4}-\d{2}|\d{14})\b'),  # CNPJ
    re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'),  # email
    re.compile(r'\b(?:\+55\s?)?(?:\(?\d{2}\)?\s?)?(?:9?\d{4}[-\s]?\d{4})\b'),  # telefone
    re.compile(r'\b\d{5}-?\d{3}\b'),  # CEP
    re.compile(r'\b\d{2}[\/\-]\d{2}[\/\-]\d{2,4}\b'),  # datas
    re.compile(r'\b[A-Z]{3}-?\d{4}\b|\b[A-Z]{3}\d[A-Z]\d{2}\b')  # placas
]

# ==== Helpers de criptografia / pseudonímia ====

def hmac_pseudonym(value: str, key: bytes, prefix: str = "P", length: int = 12) -> str:
    mac = hmac.new(key, value.encode("utf-8"), hashlib.sha256).digest()
    token = base64.urlsafe_b64encode(mac).decode("utf-8").rstrip("=")
    return f"[{prefix}_{token[:length]}]"

class AESStore:
    """Gerencia cifragem AES-GCM para guardar ciphertexts (id -> ciphertext_base64)."""
    def __init__(self, key_bytes: bytes):
        if AESGCM is None:
            raise RuntimeError("cryptography não disponível (instale cryptography).")
        if len(key_bytes) not in (16, 24, 32):
            raise ValueError("Key length must be 16/24/32 bytes (AES-128/192/256).")
        self.aesgcm = AESGCM(key_bytes)

    def encrypt(self, plaintext: str) -> str:
        nonce = os.urandom(12)
        ct = self.aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)
        payload = nonce + ct
        return base64.b64encode(payload).decode("utf-8")

    def decrypt(self, payload_b64: str) -> str:
        data = base64.b64decode(payload_b64)
        nonce, ct = data[:12], data[12:]
        pt = self.aesgcm.decrypt(nonce, ct, None)
        return pt.decode("utf-8")

# ==== Gerador sintético conhecido (sabemos os valores gerados, para mascarar com precisão) ====

TEMPLATES = [
    "O cliente {name} CPF {cpf} mora em {street}, {number}, {city}/{state}. Telefone: {phone}.",
    "{name}, nascido em {birth}, portador do CPF {cpf}, trabalha em {company}.",
    "Enviar documento para {name2} no endereço {street2}, CEP {cep}.",
    "Contato: {name} - email: {email} - telefone: {phone2}.",
    "O responsável {name3} (matrícula {user_id}) informou que o endereço {street3}, {number3} é válida.",
    "Empresa: {company2}, CNPJ {cnpj}, responsável: {name4}, fone {phone3}."
]

def gen_example_data():
    data = {
        "name": fake.name(),
        "cpf": fake.cpf(),
        "street": fake.street_name(),
        "number": str(random.randint(1,3000)),
        "city": fake.city(),
        "state": fake.estado_sigla(),
        "phone": fake.phone_number(),
        "birth": fake.date_of_birth(minimum_age=18, maximum_age=90).strftime("%d/%m/%Y"),
        "company": fake.company(),
        "name2": fake.name(),
        "street2": fake.street_name(),
        "cep": fake.postcode(),
        "email": fake.email(),
        "phone2": fake.phone_number(),
        "name3": fake.name(),
        "user_id": f"USR{random.randint(1000,99999)}",
        "street3": fake.street_name(),
        "number3": str(random.randint(1,3000)),
        "company2": fake.company(),
        "cnpj": fake.cnpj(),
        "name4": fake.name(),
        "phone3": fake.phone_number()
    }
    return data

# função principal de mascaramento: substitui os valores detectados pelos pseudônimos (HMAC) ou token fixo
def mask_using_generated_values(text: str, generated_values: Dict[str,str], 
                                use_hmac: bool=True, key_bytes: bytes=None, hmac_prefix_map: Dict[str,str]=None
                                ) -> Tuple[str, List[Tuple[str,str,str]]]:
    """
    Substitui ocorrências dos valores em generated_values no texto por MASK_TOKEN ou pseudônimo HMAC.
    Retorna (masked_text, list_of_spans) onde spans = [(label, raw_value, masked_token), ...]
    """
    spans = []
    out = text
    # ordernar por comprimento decrescente para evitar substituição parcial (ex: nome dentro de outro)
    sorted_items = sorted(generated_values.items(), key=lambda kv: len(kv[1] or ""), reverse=True)
    for key, val in sorted_items:
        if not val:
            continue
        # escape para regex seguro
        esc = re.escape(val)
        # pesquisar ocorrências exatas (case-sensitive) e substituir
        # usar função para preservar múltiplas ocorrências
        def repl(match):
            original = match.group(0)
            if use_hmac:
                prefix = (hmac_prefix_map.get(key) if hmac_prefix_map else key.upper()[:4])
                pseudo = hmac_pseudonym(original, key_bytes, prefix=prefix, length=12)
                spans.append((key, original, pseudo))
                return pseudo
            else:
                spans.append((key, original, MASK_TOKEN))
                return MASK_TOKEN
        out_new, n = re.subn(esc, repl, out)
        out = out_new
    # pós-sweep regex para capturar padrões que não estavam nos generated_values
    for rx in REGEXES:
        out = rx.sub(MASK_TOKEN, out)
    return out, spans

# ==== Função principal que gera dataset e grava arquivos ====
def generate_dataset(num_examples=1000, save_originals=True, enable_aes=False):
    key_bytes = HMAC_KEY.encode("utf-8")
    aes_store = None
    if enable_aes:
        if AES_KEY_BASE64 is None:
            raise RuntimeError("ENABLE_AES_STORE=True, mas AES_KEY_BASE64 não está definido no ambiente.")
        aes_key_bytes = base64.b64decode(AES_KEY_BASE64)
        aes_store = AESStore(aes_key_bytes)

    masked_only_path = os.path.join(OUTPUT_DIR, "masked_outputs.jsonl")
    full_dataset_path = os.path.join(OUTPUT_DIR, "dataset_with_originals.jsonl")
    mapping_hmac_path = os.path.join(OUTPUT_DIR, "mapping_hmac.json")
    mapping_aes_path = os.path.join(OUTPUT_DIR, "mapping_aes.json")

    # maps
    mapping_hmac: Dict[str, Dict] = {}  # hmac -> {"masked": pseudonym, "label": label}
    mapping_aes: Dict[str, Dict] = {}   # id -> {"ciphertext": base64, "label": label}

    # abrindo arquivos
    f_masked = open(masked_only_path, "w", encoding="utf8")
    f_full = open(full_dataset_path, "w", encoding="utf8") if save_originals else None

    for i in range(num_examples):
        template = random.choice(TEMPLATES)
        gen = gen_example_data()
        original_text = template.format(**gen)
        # construir dict com valores gerados e labels curtos
        gen_map = {
            "name": gen["name"],
            "cpf": gen["cpf"],
            "street": gen["street"],
            "phone": gen["phone"],
            "email": gen["email"],
            "cep": gen["cep"],
            "birth": gen["birth"],
            "company": gen["company"],
            "user_id": gen["user_id"],
            "cnpj": gen["cnpj"]
            # adiciona outros campos se desejar
        }
        # opcional: mapear prefixos apropriados para HMAC (ex.: CPF -> CPF)
        prefix_map = {k: k.upper() for k in gen_map.keys()}

        # mascarar usando HMAC pseudonimico (determinístico)
        masked_text, spans = mask_using_generated_values(original_text, gen_map, use_hmac=True, key_bytes=key_bytes, hmac_prefix_map=prefix_map)

        # gravar no arquivo apenas mascarados
        f_masked.write(json.dumps({"masked": masked_text}, ensure_ascii=False) + "\n")

        # gravar também dataset completo (opcional)
        if save_originals:
            f_full.write(json.dumps({"original": original_text, "masked": masked_text}, ensure_ascii=False) + "\n")

        # preencher mappings HMAC e AES (somente metadados e ciphertext; nunca salvar original em claro)
        for label, raw, masked_token in spans:
            # HMAC key for lookup
            h = hmac_pseudonym(raw, key_bytes, prefix="H")  # hmac-based index
            # armazenar mapping HMAC -> masked_token e label
            mapping_hmac[h] = {"masked": masked_token, "label": label}
            # AES (opcional) - gerar id único e salvar ciphertext
            if enable_aes:
                id_ = str(uuid.uuid4())
                ciphertext = aes_store.encrypt(raw)
                mapping_aes[id_] = {"ciphertext": ciphertext, "label": label, "masked": masked_token}
                # Nota: não gravamos raw em nenhum lugar; somente ciphertext

    f_masked.close()
    if f_full:
        f_full.close()

    # salvar mappings (JSON)
    with open(mapping_hmac_path, "w", encoding="utf8") as mf:
        json.dump(mapping_hmac, mf, ensure_ascii=False, indent=2)

    if enable_aes:
        with open(mapping_aes_path, "w", encoding="utf8") as mf:
            json.dump(mapping_aes, mf, ensure_ascii=False, indent=2)

    print(f"Gerado {num_examples} exemplos.")
    print(f"Masked-only: {masked_only_path}")
    if save_originals:
        print(f"Dataset completo (original+masked): {full_dataset_path}")
    print(f"HMAC mapping: {mapping_hmac_path}")
    if enable_aes:
        print(f"AES mapping (ciphertexts): {mapping_aes_path}")

    return {
        "masked_file": masked_only_path,
        "full_dataset_file": full_dataset_path if save_originals else None,
        "mapping_hmac": mapping_hmac_path,
        "mapping_aes": mapping_aes_path if enable_aes else None
    }

# ==== Funções utilitárias para consulta ====

def lookup_masked_from_original(original_value: str, hmac_key: bytes = None, mapping_path: str = None):
    """Calcula HMAC do original e busca mapping_hmac.json (se existir)."""
    if hmac_key is None:
        hmac_key = HMAC_KEY.encode("utf-8")
    h = hmac_pseudonym(original_value, hmac_key, prefix="H", length=12)
    if mapping_path is None:
        mapping_path = os.path.join(OUTPUT_DIR, "mapping_hmac.json")
    if not os.path.exists(mapping_path):
        raise FileNotFoundError("mapping_hmac.json não encontrado. Gere dataset primeiro.")
    with open(mapping_path, "r", encoding="utf8") as f:
        mp = json.load(f)
    return mp.get(h)  # retorna dict {"masked": ..., "label": ...} ou None

def recover_original_from_id(id_: str, aes_key_b64: str = None, mapping_aes_path: str = None):
    """Descriptografa ciphertext salvo em mapping_aes.json para recuperar original (reversível).
       Requer AES key (base64) e que mapping_aes.json exista."""
    if aes_key_b64 is None:
        aes_key_b64 = AES_KEY_BASE64
    if aes_key_b64 is None:
        raise RuntimeError("AES key não fornecida.")
    if mapping_aes_path is None:
        mapping_aes_path = os.path.join(OUTPUT_DIR, "mapping_aes.json")
    if not os.path.exists(mapping_aes_path):
        raise FileNotFoundError("mapping_aes.json não encontrado.")
    with open(mapping_aes_path, "r", encoding="utf8") as f:
        mp = json.load(f)
    entry = mp.get(id_)
    if not entry:
        return None
    aes_store = AESStore(base64.b64decode(aes_key_b64))
    pt = aes_store.decrypt(entry["ciphertext"])
    return {"original": pt, "label": entry["label"], "masked": entry["masked"]}


# ==== Execução principal quando executado como script ====
if __name__ == "__main__":
    print("Gerando dataset sintético de exemplo...")
    # controlar flags aqui
    results = generate_dataset(num_examples=NUM_EXAMPLES, save_originals=SAVE_ORIGINALS, enable_aes=ENABLE_AES_STORE)

    # EXEMPLO de uso das funções de lookup (demo)
    print("\nExemplo de lookup (sem desanonimizar):")
    sample_value = "João"  # normalmente coloca aqui um valor que exista no dataset gerado
    r = lookup_masked_from_original(sample_value)
    print("lookup for", sample_value, "->", r)

    if ENABLE_AES_STORE:
        print("\nExemplo de recovery (necessita id válido do mapping_aes.json):")
        # pega um id aleatório do arquivo
        with open(results["mapping_aes"], "r", encoding="utf8") as ff:
            mm = json.load(ff)
            some_id = next(iter(mm.keys()))
            recovered = recover_original_from_id(some_id)
            print("recovered:", recovered)

    print("\nTerminado. Veja arquivos em:", OUTPUT_DIR)
