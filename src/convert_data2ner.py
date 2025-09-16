import os, json, random, re
from faker import Faker
from transformers import BertTokenizerFast

fake = Faker("pt_BR")
Faker.seed(42)
random.seed(42)

NUM_EXAMPLES = 1000  # Ajuste conforme necessário
VAL_FRACTION = 0.2
OUT_DIR = "data_ner"
os.makedirs(OUT_DIR, exist_ok=True)

tokenizer = BertTokenizerFast.from_pretrained("neuralmind/bert-base-portuguese-cased")

MASK = "[DADO_MASCARADO]"

# Tipos de entidades para NER
ENTITY_TYPES = {
    "name": "PER", "name1": "PER", "name2": "PER", "name3": "PER",
    "name4": "PER", "name5": "PER", "name6": "PER", "name7": "PER",
    "name8": "PER",
    "age": "AGE", "idade": "AGE",
    "cpf": "CPF", "cnpj": "CNPJ", "rg": "RG",
    "email": "EMAIL",
    "phone": "PHONE", "phone1": "PHONE", "phone2": "PHONE", "phone3": "PHONE",
    "street": "ADDR", "street1": "ADDR", "street2": "ADDR", "street3": "ADDR",
    "address": "ADDR", "address1": "ADDR", "address2": "ADDR", "address3": "ADDR",
    "company": "ORG", "company2": "ORG",
    "date": "DATE", "birth": "DATE",
    "number": "O", "number3": "O"
}

# Suas templates longas (pode incluir todas)
TEMPLATES = [
    "O cliente {name} CPF {cpf} mora em {street}, {number}, {city}/{state}. Telefone: {phone}.",
    "{name}, nascido em {birth}, portador do CPF {cpf}, trabalha em {company}.",
    "Enviar documento para {name2} no endereço {street2}, CEP {cep}.",
    "Contato: {name} - email: {email} - telefone: {phone2}.",
    "O responsável {name3} (matrícula {user_id}) informou que o endereço {street3}, {number3} é válido.",
    "Empresa: {company2}, CNPJ {cnpj}, responsável: {name4}, fone {phone3}.",
    "O juiz {name5} deferiu o pedido feito ao requente {name6} após a agressão cometida por parte do agressor (a) {name7} na cidade de {city}/{state} no dia {date}",
    "Nº BU {number} Data de emissão {date} 13:56 Página {number} Histórico do fato A VÍTIMA, RETORNA NESTA DELEGACIA A FIM DE REGISTRAR NOVAMENTE EM DESFAVOR DO AUTOR {name}, NO QUAL HAVIA REGISTRADO PELOS CRIMES DE AGRESSÃO FÍSICA E AMEAÇAS E A VITIMA SOLICITOU MEDIDAS PROTETIVAS;QUE, A VÍTIMA RELATA, NO DIA DOS FATOS ELA SE ENCONTRAVA NA CASA DE SUA MÃE COMO SEMPRE FAZ, POIS ELA CUIDA DOS SEUS PAIS, COMO LEVAR AOS LUGARES QUE SEUS ELES NECESSITAM E OUTROS, PRINCIPALMENTE SUA MÃE.DISSE A VÍTIMA QUE O AUTOR TEM POR HÁBITO FREQUENTAR A CASA DE SUA MÃE QUANDO ELA NÃO SE ENCONTRA, MESMO ELE SABENDO DAS MEDIDAS PROTETIVAS. NO DIA DOS FATOS, A VÍTIMA SE ENCONTRAVA NA CASA DE SEUS PAIS COMO DE COSTUME, QUANDO O AUTOR APARECEU NO PORTÃO GRITANDO PELA MÃE DA VÍTIMA PARA PEDIR COMIDA, DISSE A VÍTIMA O AUTOR TAMBÉM PEDE DINHEIRO, POIS ELE É USUÁRIO DE DROGAS. A VÍTIMA DISSE QUE FOI ATÉ O PORTÃO E PERGUNTOU PARA O AUTOR SE ELE SABE DAS MEDIDAS, E O POR QUE ELE SE ENCONTRAVA NO LOCAL, DISSE A VÍTIMA QUE ELE RESPONDEU XINGANDO A MESMA, MANDANDO ELA , \" TOMAR NO CU\", DEPOIS ELE SAIU. DIANTE DOS FATOS ACIMA, A VITIMA DESEJA REGISTRAR PELO DESCUMPRIMENTO DAS MEDIDAS PROTETIVAS E QUE TEME PELA SUA VIDA, DISSE TAMBÉM QUE SEU PAI NÃO PERMITE A ENTRADA DO AUTOR NA CASA; QUE FOI PERGUNTADO A VÍTIMA SE O AUTOR FAZ USO INDEVIDO DE BEBIDA ALCOOLICA E OU DROGAS, RESPONDEU QUE SIM PARA DROGAS;QUE, FOI PERGUNTADO A VÍTIMA SE O AUTOR POSSUI ARMA DE FOGO E OU PORTE DE ARMA, RESPONDEU QUE ELA SAIBA, NÃO;QUE, FOI PERGUNTADO A VÍTIMA SE DESEJA CASA ABRIGO, RESPONDEU QUE NÃO; QUE A VÍTIMA FOI INFORMADA SOBRE OS SERVIÇOS DA REDE DE PROTEÇÃO A MULHER;QUE,NADA MAIS DISSE, NEM LHE FOI PERGUNTADO, DEU POR ENCERRADO O PRESENTE BOLETIM DE OCORRÊNCIA COM INFORMAÇÕES NARRADAS PELA VÍTIMA. QUE DEPOIS DE LIDO E ACHADO CONFORME, VAI ASSINADO POR TODOS. A VÍTIMA JÁ FAZ ACOMPANHAMENTO NA CASA DO CIDADÃO.A VÍTIMA TAMBÉM DESEJA RESSALTAR QUE O AUTOR ESTA TRABALHANDO, SEM HAVER NECESSIDADE DELE PROCURAR IR ATÉ A CASA DE SUA MÃE. OBS INSERIDA PELO O responsável pelo preenchimento da ocorrência informou que não existem objetos a serem cadastrados nesta ocorrência. DOS ENVOLVIDOS Ordem Nome Completo 1º {name1} POLICIA Versão Tipo de envolvimento POLICIA CIVIL DO {state} / DEAM VITIMA {city} Data/hora {date} DADOS BÁSICOS: BRASIL, CASADO, FILHO DE {name2}, CPF: {cpf}, OUTRO DOCMENTO: RG: {rg} CIVIL, Nº: - , CNPJ: {cnpj} , NASCIDO EM {birth}, {age} ANOS, NATURAL DE {address3}, PROFISSÃO: , TRABALHA: ALTURA APROX .: ENDEREÇO: {address1}// EM FRENTE A {address2} E TENDO COMO TELEFONE(S) PARA CONTATO: TEL. CELULAR: {phone}, TEL. RESIDENCIAL: TEL. COMERCIAL: EMAIL: {email} ENDEREÇO: {address}, PRÓXIMO DELEGACIA PATRIMONIAL E TENDO COMO TELEFONE(S) PARA CONTATO: TEL. CELULAR: {phone}, TEL. RESIDENCIAL: {phone1}- TEL. COMERCIAL: {phone2} - EMAIL: {email} DADOS COMPLEMENTARES: PROFISSÃO: - , EMPRESA: - , RENDA: - SALÁRIOS MÍNIMOS, SEXO: FEMININO, ORIENTAÇÃO SEXUAL: HETEROSSEXUAL, CUTIS: BRANCA, ESCOLARIDADE: SUPERIOR COMPLETO, RELACIONA-SE COM: - APELIDO: RELIGIÃO: POSSUI LESÃO: NÃO - , FOI 1 AGREDIDO/TORTURADO: SIM - IP da estação {ipaddress} Verificador {number} Responsável por gerar {name8}",
    "· BU {number} Data de emissão {date} 13:56 Página 1/4 SECRETARIA DE SEGURANÇA PÚBLICA E DEFESA SOCIAL DO ESPÍRITO SANTO BOLETIM UNIFICADO (BU) {number} Registrado em {date} às 13:13 DO REGISTRO Unidade Registro DEAM VITÓRIA Método da lavratura REGISTRO PRESENCIAL Endereço da unidade de registro {street}, {cep} Telefone(s) para contato da unidade Nº Ciodes NÃO INFORMADO Observação DESCUMPRIMENTO DE MEDIDAS PROTETIVAS DOS FATOS Data/hora do fato Tipo de local {date} às 13:30 RESIDÊNCIA Evento SEM EVENTO Endereço do fato {street1}, {city}/{state}, POLICIA Versão POLICIA CIVIL DO ES Unidade Policial CIVI DEAM VITÓRIA Incidente/Nature A17 CRIMES CONTRA A PESSOA: DESCUMPRIMENTO DA MEDIDA PROTETIVA IP da estação {ipaddress} Verificador {number} Responsável por gerar {name}.",
    "No dia {date}, {street}, {number}, {city}/{state}, ocorreu um incidente de natureza criminosa em que o indivíduo {name} agrediu o cidadão {name1}, portador do CPF {cpf}, provocando-lhe lesões corporais que exigiram imediato atendimento médico, conforme boletim de ocorrência lavrado junto à autoridade competente; o ocorrido gerou ampla repercussão na comunidade local, culminando na instauração de ação penal por parte do Ministério Público, e a vítima, {name}, já havia prestado depoimento e disponibilizado seus dados de contato, telefone {phone2} e e‑mail {email}, para fins de elucidação dos fatos, enquanto o agressor, {name1}, permanece à vista das autoridades. A juíza responsável por julgar o processo, {name2}, em sua decisão preliminar, ressaltou a gravidade do ato, destacando a necessidade de aplicação de medidas cautelares em favor da vítima e a obrigação do acusado de reparar os danos causados, em conformidade com os dispositivos penais vigentes e as diretrizes de proteção às vítimas de violência.",
    "No dia 10 de maio de 2025, por volta das 14h15, no endereço {street}, {number}, {city}/{state}, foi registrado o presente boletim de ocorrência, narrando um episódio de agressão física contra a vítima {name2}, portador do CPF {cpf}, que alegou ter sido alvo de agressões verbais e físicas por parte do acusado {name2}, cuja identidade permanece a ser corroborada por diligências posteriores, estando disponível o contato telefônico {phone2} e e‑mail {email} para eventual complementação de informações. O fato, considerado de natureza criminosa, foi devidamente registrado pelo servidor policial responsável, sob a apreciação e responsabilidade do juízo competente, representado pelo {name2}, que deverá conduzir a devida tramitação processual em consonância com a legislação vigente, assegurando ao denunciado o pleno exercício do contraditório e da ampla defesa.",
    "No dia {date}, às 14h30, no {street}, {number}, {city}/{state}, foi registrado o boletim de ocorrência referente a assalto agravado cometido pelo indivíduo {name}, {age} ANOS, contra o cidadão {name1}, portador do CPF {cpf}, que, ao entrar na residência do local, foi surpreendido pelo agressor que, sob ameaça de violência física, requisitou a posse de um celular de marca e modelo desconhecido, culminando em ato de violência física leve que deixou o acusado com contusões leves e o vitimado com ferimento superficial na região do ombro. A ocorrência foi narrada pelo próprio {name}, que, com total convicção, descreveu a agressividade do acusado, bem como os detalhes do local e da situação. A delegacia competente, sob a responsabilidade do juiz a cargo do processo, a Sra. {name2}, prontamente registrou a denúncia, fixou os dados pessoais do agressor e da vítima, anotou o número de telefone de contato {phone2} para eventuais diligências e registrou o e‑mail {email} como meio de comunicação adicional, assegurando, assim, a continuidade das investigações e o adequado encaminhamento do caso aos órgãos competentes, em conformidade com o Código de Processo Penal e demais normas vigentes.",
    "Em 12 de março de 2024, no Tribunal de Justiça do Estado de Goiás, o juízo de primeira instância, presidido pela desembargadora {name2}, recebeu a denúncia de agressão de natureza física praticada pelo acusado {name2} contra a vítima {name2}, portadora do CPF {cpf}, que relatou ter sido agredida no endereço {street}, {number}, {city}/{state}, durante a qual sofreu lesões corporais confirmadas por laudos médicos anexados aos autos; a própria vítima, em contato com o juízo por meio do telefone {phone2} e do e‑mail {email}, entregou sua versão dos fatos e solicitou providências, o que motivou a abertura do inquérito policial para apurar as circunstâncias do fato e determinar a aplicação das sanções legais cabíveis.",
    "Ao Senhor Doutor Juiz {name1}, respeitosamente, cumpre a presente petição inicial à vista do ilícito ocorrido em 14 de março de 2023, na {street}, {number}, {city}/{state}, onde a vítima {name2}, portadora do CPF {cpf}, foi alvo de agressão física e verbal por parte da agressora {name2}, resultando em lesões corporais e psicológicas de natureza grave, conforme laudo médico anexo. Tal fato, testemunhado por vários moradores da rua, gerou o direito de se pleitear reparação pelos danos morais e materiais suportados, bem como a adoção de medidas de segurança, de modo a preservar a integridade física e psicológica da vítima. Para fins de contato, a assistente jurídica da autora pode ser localizada pelo telefone {phone2} e pelo e‑mail {email}, onde estão à disposição todos os documentos que corroboram o relato apresentado, aguardando deferimento das providências cabíveis e a condenação da ré, {name2}, nos termos da legislação aplicável.",
    "Na data de {date}, às 14h32, no {street1}, {name}, {street}, {number}, {city}/{state}, registrou-se o fato de agressão física e verbal contra o Sr. {name}, {age} anos, portador do CPF {cpf}, pelos atos de violência praticados por {name2}, a fim de que sejam adotadas as providências cabíveis; a ocorrência foi recebida pela autoridade policial competente e protocolada sob o número 2025-ABR-0034, sendo que a Dra. {name3}, Juíza de Direito competente, deverá avaliar a denúncia, com contato a ser efetuado através do telefone {phone2} ou pelo e‑mail {email}, para regularização do processo e definição das medidas de segurança e reparação a serem aplicadas a este caso.",
    "Na data de {date}, {street}, {number}, {city}/{state}, foi registrado o incidente descrito nos autos, no qual o Sr. {name}, {age} anos, portador do CPF {cpf}, foi agraviado pelo Sr. {name}; o fato culminou em lesões corporais e danos materiais, os quais foram devidamente avaliados pela equipe de perícia e documentados no boletim de ocorrência. O Ministério Público, representado pelo {name2}, recebeu a denúncia por meio dos contatos telefônico {phone2} e eletrônico {email}, os quais foram utilizados para a comunicação e coordenação das providências legais subsequentes, assegurando o atendimento dos princípios do contraditório, ampla defesa e observância da legalidade estrita nas ações judiciais pertinentes.",
    "No dia {date}, às 15h30, na {street}, {number}, {city}/{state}, registrou‑se o incidente de agressão física, pelo qual a vítima {name}, portador do CPF {cpf}, {age} anos, sofreu violência doméstica de parte da agressora {name2}; o fato culminou em lesões que requerem atendimento médico e consequente registro policial, conforme determina o art. 138 do Código Penal, com respaldo à Lei Maria da Penha; o Ministério Público notificou o Ministério Público Federal de que o crime foi denunciado ao delegado de polícia e que o juiz responsável pelo caso é a ilustre Juíza {name3}, que já emitiu despacho preliminar para a instauração do inquérito policial; em nota de apoio ao esclarecimento, foi disponibilizado o telefone de contato {phone2} e o e‑mail {email} para que as partes interessadas possam se comunicar com a autoridade competente e apresentar elementos probatórios que corroborem o teor dos autos, de forma a garantir o devido processo legal e a proteção dos direitos da vítima."
]


def gen_data_row():
    # Dados fake
    data = {
        "age": str(random.randint(1,110)),
        "name": fake.name(), "name1": fake.name(), "name2": fake.name(), "name3": fake.name(),
        "name4": fake.name(), "name5": fake.name(), "name6": fake.name(), "name7": fake.name(), "name8": fake.name(),
        "cpf": fake.cpf(), "cnpj": fake.cnpj(), "rg": fake.rg(),
        "email": fake.email(),
        "ipaddress": fake.ipv4_private(),
        "user_id": f"USR{random.randint(1000,99999)}",
        "phone": fake.phone_number(), "phone1": fake.phone_number(), "phone2": fake.phone_number(), "phone3": fake.phone_number(),
        "street": fake.street_name(), "street1": fake.street_name(), "street2": fake.street_name(), "street3": fake.street_name(),
        "address": fake.address(), "address1": fake.address(), "address2": fake.address(), "address3": fake.address(),
        "company": fake.company(), "company2": fake.company(),
        "date": fake.date_of_birth(minimum_age=18, maximum_age=80).strftime("%d/%m/%Y"),
        "birth": fake.date_of_birth(minimum_age=18, maximum_age=80).strftime("%d/%m/%Y"),
        "number": str(random.randint(1, 3000)), "number3": str(random.randint(1, 3000)),
        "city": fake.city(), "state": fake.estado_sigla(), "cep": fake.postcode()
    }

    template = random.choice(TEMPLATES)
    sentence = template.format(**data)

    # Tokeniza
    tokens = tokenizer.tokenize(sentence)
    labels = ["O"] * len(tokens)

    # Marca entidades
    for key, ent_type in ENTITY_TYPES.items():
        value = data.get(key)
        if not value:
            continue
        ent_tokens = tokenizer.tokenize(value)
        # percorre a sentença e marca todas as ocorrências
        i = 0
        while i <= len(tokens) - len(ent_tokens):
            if tokens[i:i+len(ent_tokens)] == ent_tokens:
                labels[i] = "B-" + ent_type
                for j in range(1, len(ent_tokens)):
                    labels[i+j] = "I-" + ent_type
                i += len(ent_tokens)
            else:
                i += 1

    return {"tokens": tokens, "ner_tags": labels}

def main():
    rows = [gen_data_row() for _ in range(NUM_EXAMPLES)]
    random.shuffle(rows)
    n_val = int(len(rows) * VAL_FRACTION)
    val = rows[:n_val]
    train = rows[n_val:]

    with open(os.path.join(OUT_DIR, "train_ner.jsonl"), "w", encoding="utf8") as f:
        for r in train: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT_DIR, "val_ner.jsonl"), "w", encoding="utf8") as f:
        for r in val: f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Gerados {len(train)} train + {len(val)} val em {OUT_DIR}")

if __name__ == "__main__":
    main()
