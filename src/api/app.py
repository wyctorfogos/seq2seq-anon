# -*- coding: utf-8 -*-

from fastapi import FastAPI, HTTPException, Request
import uvicorn
from pydantic import BaseModel
from typing import Dict, Optional
import os
import json
from models.anonymizer_data import SentenceAnonymyzer
from dotenv import load_dotenv

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv(dotenv_path="./config/.env")

# É mais seguro usar valores padrão caso as variáveis não sejam encontradas
PORT_API_ANONYMIZER_SERVICES = int(os.getenv("PORT_API_ANONYMIZER_SERVICES", 8000))
NER_MODEL_FOLDER_PATH = os.getenv("NER_MODEL_FOLDER_PATH", "./models/ner-model")
# Converte o threshold para float, que é mais apropriado para limiares
THRESHOLD_FOR_RECOGNITION = float(os.getenv("THRESHOLD_FOR_RECOGNITION", 0.85))

# Criar objeto 'APP'
app = FastAPI(title="Anonimizador de Texto NER+Regex", version="1.1")

anonymizer_model: Optional[SentenceAnonymyzer] = None

class TextRequest(BaseModel):
    text: str

@app.on_event("startup")
def startup_settings():
    """
    Carrega o modelo de IA quando a aplicação inicia.
    Isso evita recarregar o modelo a cada requisição.
    """
    global anonymizer_model
    try:
        print("-> Carregando o modelo de anonimização...")
        anonymizer_model = SentenceAnonymyzer(
            model_dir=NER_MODEL_FOLDER_PATH, 
            threshold=THRESHOLD_FOR_RECOGNITION
        )
        print("-> Modelo carregado com sucesso!")
    except Exception as e:
        # Se o modelo falhar ao carregar, a aplicação não deve iniciar.
        # Lançar um erro aqui fará com que o servidor pare.
        raise RuntimeError(f"Erro fatal ao carregar o modelo de anonimização: {e}")

@app.post("/v1.0/anonymize")
def anonymize_text(request: TextRequest) -> Dict:
    """
    Recebe um texto e retorna sua versão anonimizada.
    """
    # Verificação para garantir que o modelo foi carregado corretamente no startup
    if anonymizer_model is None:
        raise HTTPException(
            status_code=503, # Service Unavailable
            detail="Serviço indisponível: o modelo de IA não foi carregado."
        )
    
    text = request.text
    try:
        # O Pydantic já garante que `request.text` seja uma string,
        # então a verificação `isinstance` é desnecessária.       
        if not text:
             raise HTTPException(status_code=400, detail="O campo 'text' não pode estar vazio.")
        
        # Limpeza do texto e chamada do pipeline de anonimização
        cleaned_text = text.replace("\n", " ").replace("\\", "")
        
        pipeline_response = anonymizer_model.pipeline(
            input_text=cleaned_text,
            width=100 
        )
        
        return pipeline_response
    except Exception as e:
        # Captura exceções inesperadas e retorna um erro 500.
        raise HTTPException(status_code=500, detail=f"Ocorreu um erro interno: {str(e)}")

@app.get("/v1.0/")
def root():
    """Endpoint para verificar a saúde da API."""
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app=app, host="0.0.0.0", port=PORT_API_ANONYMIZER_SERVICES)

