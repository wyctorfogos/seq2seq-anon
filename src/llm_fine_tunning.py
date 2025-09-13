import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import os

# --- 1. Configurações ---
MODEL_ID = "google/gemma-2b-it"  # Modelo pequeno e eficiente
DATA_DIR = "data_seq2seq"       # Diretório onde estão seus dados
OUTPUT_DIR = "gemma-2b-anonimizador" # Diretório para salvar o adaptador
NUM_TRAIN_EPOCHS = 3            # Número de épocas de treinamento

# --- 2. Configuração de Quantização (BitsAndBytes) ---
# Carrega o modelo em 4-bit para economizar memória
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 # Use bfloat16 para melhor desempenho
)

# --- 3. Carregar Modelo e Tokenizador ---
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto" # Mapeia o modelo para a GPU automaticamente
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
# Modelos Gemma não têm um pad_token por padrão. Usamos o eos_token.
tokenizer.pad_token = tokenizer.eos_token 

# --- 4. Preparar o Modelo para Treinamento com LoRA ---
model.gradient_checkpointing_enable() # Outra técnica de economia de memória
model = prepare_model_for_kbit_training(model)

# --- 5. Configuração do LoRA ---
# Define quais camadas do modelo serão adaptadas
lora_config = LoraConfig(
    r=8, # Dimensão do rank (menor = menos parâmetros treináveis)
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # Camadas de atenção do Gemma
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
# Aplica o LoRA ao modelo
model = get_peft_model(model, lora_config)

# --- 6. Carregar e Preparar o Dataset ---
# Carrega os arquivos .jsonl
dataset = load_dataset("json", data_dir=DATA_DIR, split="train")

# Formata o dataset para o formato de instrução que o Gemma espera
def formatting_prompts_func(example):
    # O modelo Gemma foi treinado com um formato específico de chat/instrução.
    # Seguir esse formato melhora os resultados.
    # O prompt instrui o modelo sobre a tarefa a ser executada.
    text = f"<start_of_turn>user\nAnonimize os dados pessoais no texto a seguir, substituindo-os por '[DADO_MASCARADO]'.\nTexto original: {example['input']}<end_of_turn>\n<start_of_turn>model\n{example['output']}<end_of_turn>"
    return { "text": text }

dataset = dataset.map(formatting_prompts_func)

# --- 7. Configurar os Argumentos de Treinamento ---
training_args = TrainingArguments(
    per_device_train_batch_size=2, # Batch size pequeno para caber na VRAM
    gradient_accumulation_steps=4, # Simula um batch size maior (2*4=8) sem usar mais VRAM
    learning_rate=2e-4,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    output_dir=OUTPUT_DIR,
    logging_steps=10,
    fp16=True, # Use fp16 para acelerar o treinamento
    save_strategy="epoch",
    optim="paged_adamw_8bit" # Otimizador que economiza memória
)

# --- 8. Iniciar o Treinamento ---
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=1024, # Comprimento máximo da sequência
    tokenizer=tokenizer,
    args=training_args,
    packing=False,
)

print("🚀 Iniciando o treinamento...")
trainer.train()

# --- 9. Salvar o Adaptador LoRA ---
# Salva apenas os pesos do adaptador, que é um arquivo pequeno
trainer.save_model(OUTPUT_DIR)
print(f"✅ Adaptador LoRA salvo em '{OUTPUT_DIR}'")