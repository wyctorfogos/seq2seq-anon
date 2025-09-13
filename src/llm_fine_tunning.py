import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import os

# --- 1. Configurações ---
# Corrigido para o nome do modelo de instrução mais recente da família Qwen2
MODEL_ID = "Qwen/Qwen2-0.5B-Instruct" 
DATA_DIR = "data_seq2seq"
OUTPUT_DIR = "qwen2-0.5b-anonimizador" # Nome do diretório de saída atualizado
NUM_TRAIN_EPOCHS = 3

# --- 2. Configuração de Quantização (BitsAndBytes) ---
# Esta parte está correta e não precisa de mudanças.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# --- 3. Carregar Modelo e Tokenizador ---
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# --- IMPORTANTE: Configuração do Tokenizador para Qwen ---
# O Qwen não tem um pad_token. Usar o eos_token é uma solução comum.
# Adicionamos o padding_side para evitar problemas com alguns warnings e gerações.
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" 

# --- 4. Preparar o Modelo para Treinamento com LoRA ---
# Esta parte está correta e não precisa de mudanças.
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# --- 5. Configuração do LoRA ---
# Os target_modules para o Qwen2-0.5B são os mesmos do Gemma, o que simplifica a troca.
# Para outros modelos, você precisaria verificar os nomes das camadas.
lora_config = LoraConfig(
    r=16, # Aumentei um pouco o rank para dar mais capacidade ao modelo
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # Adicionei mais camadas para um fine-tuning mais robusto
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# --- 6. Carregar e Preparar o Dataset ---
dataset = load_dataset("json", data_dir=DATA_DIR, split="train")

# --- MUDANÇA CRÍTICA: Formatação do Prompt para o Qwen2 ---
# Esta função foi reescrita para usar o template de chat do Qwen2.
# Usar o template correto é ESSENCIAL para o sucesso do fine-tuning.
def formatting_prompts_func(example):
    # A instrução para o modelo
    # Note que não há uma role "system" explícita aqui para simplificar, 
    # a instrução direta no "user" funciona muito bem para tarefas específicas.
    prompt = (
        f"Anonimize os dados pessoais no texto a seguir, substituindo-os por '[DADO_MASCARADO]'.\n"
        f"Texto original: {example['input']}"
    )
    
    # Monta o template completo
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": example['output']}
    ]
    
    # O tokenizador do Qwen2 tem um método que faz isso automaticamente
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}

dataset = dataset.map(formatting_prompts_func)

# --- 7. Configurar os Argumentos de Treinamento ---
# Esta parte está correta, apenas ajustei o nome do diretório.
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    output_dir=OUTPUT_DIR,
    logging_steps=10,
    fp16=True, # bf16=True se sua GPU suportar (séries 30xx, 40xx, A100)
    save_strategy="epoch",
    optim="paged_adamw_8bit"
)

# --- 8. Iniciar o Treinamento ---
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=1024,
    tokenizer=tokenizer,
    args=training_args,
    packing=False,
)

print("🚀 Iniciando o treinamento com o modelo Qwen2-0.5B-Instruct...")
trainer.train()

# --- 9. Salvar o Adaptador LoRA ---
trainer.save_model(OUTPUT_DIR)
print(f"✅ Adaptador LoRA salvo em '{OUTPUT_DIR}'")