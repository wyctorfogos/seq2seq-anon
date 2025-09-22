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