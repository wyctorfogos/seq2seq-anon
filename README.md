# Crie um ambiente virtual no conda

```bash

    conda create -n ner-env -y
```

# Ativar o ambiente:

```bash
    conda activate ner-env
```

# Para gerar os pares de sentenças:
```bash
    python3 src/generate_dataset.py
```


# Para treinar  modelo:
```bash
    python3 src/train_nlp_model.py
```


# Para realizar a inferência
```bash
    python3 src/inferences.py
```

# Para subir a API como serviço de anonimização dos dados

```bash
    cp ./docker/.env.example ./docker/.env
```

Depois, altere os dados referentes às variáveis de ambiente:
```# Configurações da API
    PORT_API_ANONYMIZER_SERVICES=XXXX
    NER_MODEL_FOLDER_PATH = "MODEL_FOLDER_PATH"
    THRESHOLD_FOR_RECOGNITION = 0.75 # Limiar para mascarar o texto
    
    # Configurações do Proxy
    HTTP_PROXY="PROXY_IPADDRESS"
    HTTPS_PROXY="PROXY_IPADDRESS"
    NO_PROXY="DOMAINS_TO_NOT_USE_PROXY"
    CERT_URL="CERTIFICATION_URL"
    CERT_FILE="CERTIFICATION_FOLDER_PATH"
```

# Dentro da pasta do docker:

Para construir a imagem do docker
```bash
    docker compose build 
```

Para subir o serviço
```bash
    docker compose up -d
```

Para derrubar o serviço
```bash
    docker compose down
```