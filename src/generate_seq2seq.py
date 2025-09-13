import os, json, random, re
from faker import Faker

fake = Faker("pt_BR")
Faker.seed(42)
random.seed(42)

NUM_EXAMPLES = 20000 
VAL_FRACTION = 0.05           
OUT_DIR = "data_seq2seq"
os.makedirs(OUT_DIR, exist_ok=True)
MASK = "[DADO_MASCARADO]"

TEMPLATES = [
    "O cliente {name} CPF {cpf} mora em {street}, {number}, {city}/{state}. Telefone: {phone}.",
    "{name}, nascido em {birth}, portador do CPF {cpf}, trabalha em {company}.",
    "Enviar documento para {name2} no endereço {street2}, CEP {cep}.",
    "Contato: {name} - email: {email} - telefone: {phone2}.",
    "O responsável {name3} (matrícula {user_id}) informou que o endereço {street3}, {number3} é válido.",
    "Empresa: {company2}, CNPJ {cnpj}, responsável: {name4}, fone {phone3}.",
]

# regexes para pós-sweep (garantia)
REGEXES = [
    re.compile(r'\b(?:\d{3}\.\d{3}\.\d{3}-\d{2}|\d{11})\b'),  # CPF
    re.compile(r'\b(?:\d{2}\.\d{3}\.\d{3}\/\d{4}-\d{2}|\d{14})\b'),  # CNPJ
    re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'),  # email
    re.compile(r'\b(?:\+55\s?)?(?:\(?\d{2}\)?\s?)?(?:9?\d{4}[-\s]?\d{4})\b'),  # telefone
    re.compile(r'\b\d{5}-?\d{3}\b'),  # CEP
    re.compile(r'\b\d{2}[\/\-]\d{2}[\/\-]\d{2,4}\b'),  # datas
]

def gen_row():
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
    template = random.choice(TEMPLATES)
    inp = template.format(**data)
    # substitui cada valor gerado por MASK (garantido, porque sabemos os valores)
    out = inp
    # ordenar por comprimento para não cortar substrings
    for v in sorted(set(data.values()), key=lambda s: len(s), reverse=True):
        if not isinstance(v, str) or len(v.strip()) == 0: 
            continue
        out = out.replace(v, MASK)
    # pós-sweep regex para segurança (cobre formatos não substituídos)
    for rx in REGEXES:
        out = rx.sub(MASK, out)
    return {"input": inp, "output": out}

def main():
    rows = [gen_row() for _ in range(NUM_EXAMPLES)]
    random.shuffle(rows)
    n_val = int(len(rows) * VAL_FRACTION)
    val = rows[:n_val]
    train = rows[n_val:]
    with open(os.path.join(OUT_DIR, "train.jsonl"), "w", encoding="utf8") as f:
        for r in train: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT_DIR, "val.jsonl"), "w", encoding="utf8") as f:
        for r in val: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Gerados {len(train)} train + {len(val)} val em {OUT_DIR}")

if __name__ == "__main__":
    main()
