import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_DIR = "data_seq2seq"
OUTPUT_DIR = f"{str(MODEL_ID).replace("/","-").lower()}-anonimizador"
NUM_TRAIN_EPOCHS = 3

# --- 1. Quantização ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# --- 2. Modelo ---
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# --- 3. LoRA ---
lora_config = LoraConfig(
    r=16,
    lora_alpha=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# --- 4. Dataset ---
dataset = load_dataset("json", data_files={"train": f"{DATA_DIR}/train.jsonl", "validation": f"{DATA_DIR}/val.jsonl"})

def formatting_prompts_func(example):
    prompt = (
        "Anonimize os dados pessoais (nome completo, CPF, endereço, e-mail, telefone, etc.) "
        "no texto a seguir substituindo-os por tags como [NOME], [CPF], [EMAIL].\n"
        f"Texto original: {example['input']}"
    )
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": example['output']}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}

dataset = dataset.map(formatting_prompts_func)

# --- 5. Treinamento ---
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    args=training_args
    )

print("🚀 Iniciando o treinamento...")
trainer.train()
trainer.save_model(OUTPUT_DIR)
print(f"✅ Modelo salvo em {OUTPUT_DIR}")
