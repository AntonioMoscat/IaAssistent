# api_server.py

import subprocess
import requests
import time
import sys
import atexit
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List

from llm_wrapper import LocalLLM
from memory.memory import load_memory, save_memory
from agent import dispatch, get_available_commands
from memory.semantic_memory import SemanticMemory
from commands.registry import dispatch_semantic_hybrid
from fastapi import WebSocket, WebSocketDisconnect
from webSocket import WebSocketServerSingleton

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Variabili globali ===
sem_mem = None
llm = None
memory = None
ollama_process = None

# === Modelli per l'API ===
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Messaggio da inviare all'assistente")

class ChatResponse(BaseModel):
    response: str
    command_type: Optional[str] = None  # Tipo di comando eseguito
    
class StatusResponse(BaseModel):
    status: str
    ollama_running: bool
    available_commands: List[str]

class CorrectionRequest(BaseModel):
    old_phrase: str = Field(..., min_length=1)
    new_phrase: str = Field(..., min_length=1)

# === Funzioni di utilità ===
def is_ollama_running() -> bool:
    """Verifica se Ollama è in esecuzione"""
    try:
        res = requests.get("http://localhost:11434", timeout=5)
        return res.status_code == 200
    except Exception as e:
        logger.warning(f"Ollama non raggiungibile: {e}")
        return False

def start_ollama():
    """Avvia Ollama in background"""
    global ollama_process
    try:
        logger.info("⚙️ Avvio Ollama in background...")
        ollama_process = subprocess.Popen(
            ["ollama", "serve"], 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL
        )
        time.sleep(3)  # Tempo di inizializzazione aumentato
        return True
    except Exception as e:
        logger.error(f"Errore nell'avvio di Ollama: {e}")
        return False

def cleanup():
    """Pulizia risorse all'uscita"""
    global ollama_process
    if ollama_process:
        logger.info("🧹 Chiusura Ollama...")
        ollama_process.terminate()
        try:
            ollama_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            ollama_process.kill()
        logger.info("✅ Ollama terminato")

# Registra la funzione di cleanup
atexit.register(cleanup)

# === Gestione lifecycle con context manager ===
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global sem_mem, llm, memory
    
    logger.info("🚀 Avvio dell'assistente virtuale...")
    
    # Inizializza componenti
    sem_mem = SemanticMemory()
    llm = LocalLLM()
    memory = load_memory()
    
    # Verifica e avvia Ollama
    if not is_ollama_running():
        if start_ollama():
            # Attendi che Ollama sia pronto
            for attempt in range(15):  # Aumentato da 10 a 15
                if is_ollama_running():
                    logger.info("✅ Ollama avviato con successo")
                    break
                time.sleep(1)
                logger.info(f"⏳ Tentativo {attempt + 1}/15 - Attendo Ollama...")
            else:
                logger.error("❌ Ollama non si è avviato entro il timeout")
                raise RuntimeError("Impossibile avviare Ollama")
        else:
            raise RuntimeError("Errore nell'avvio di Ollama")
    else:
        logger.info("✅ Ollama è già attivo")
    
    yield
    
    # Shutdown
    cleanup()

# === Creazione app FastAPI ===
app = FastAPI(
    title="Assistente Virtuale API",
    description="API per interagire con l'assistente virtuale",
    version="1.0.0",
    lifespan=lifespan
)

# Aggiungi CORS per permettere chiamate da frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In produzione, specifica domini specifici
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Endpoints ===
@app.get("/health", response_model=StatusResponse)
def get_status():
    """Ottieni lo stato dell'assistente"""
    return StatusResponse(
        status="active",
        ollama_running=is_ollama_running(),
        available_commands=get_available_commands()
    )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    ws_manager = WebSocketServerSingleton()
    await ws_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            logger.info(f"📩 Messaggio ricevuto: {data}")
            await ws_manager.broadcast(f"Echo: {data}")
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
        logger.info("🔌 Connessione WebSocket chiusa")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    """Endpoint principale per la chat"""
    command = request.message.strip()
    command_type = None
    
    try:
        # Prova dispatcher semantico ibrido
        response = dispatch_semantic_hybrid(command)
        if response:
            command_type = "semantic"
        
        # Fallback su dispatcher tradizionale
        if response is None:
            response = dispatch(command)
            if response:
                command_type = "traditional"
        
        # Fallback su LLM
        if response is None:
            context = sem_mem.search(command)
            prompt = (f"Contesto precedente: {context}\nDomanda: {command}" if context and context != command else command)
            response = llm.respond(prompt)
            command_type = "llm"
        
        # Salva in memoria in background
        background_tasks.add_task(save_interaction, command, response)
        
        return ChatResponse(response=response, command_type=command_type)
        
    except Exception as e:
        logger.error(f"Errore durante l'elaborazione: {e}")
        raise HTTPException(status_code=500, detail=f"Errore interno: {str(e)}")

@app.post("/correction")
def add_correction(request: CorrectionRequest):
    """Aggiungi una correzione alla memoria semantica"""
    try:
        sem_mem.learn(request.old_phrase, request.new_phrase)
        logger.info(f"Correzione registrata: '{request.old_phrase}' -> '{request.new_phrase}'")
        return {"message": "✅ Correzione registrata con successo"}
    except Exception as e:
        logger.error(f"Errore nella registrazione della correzione: {e}")
        raise HTTPException(status_code=500, detail=f"Errore nella correzione: {str(e)}")

@app.get("/history")
def get_history(limit: int = 10):
    """Ottieni la cronologia delle conversazioni"""
    try:
        history = memory.get("history", [])
        return {"history": history[-limit:] if limit > 0 else history}
    except Exception as e:
        logger.error(f"Errore nel recupero della cronologia: {e}")
        raise HTTPException(status_code=500, detail="Errore nel recupero della cronologia")

def save_interaction(command: str, response: str):
    """Salva l'interazione in memoria (eseguita in background)"""
    try:
        sem_mem.add(command)
        memory["history"].append({"user": command, "ai": response})
        save_memory(memory)
        logger.debug("Interazione salvata in memoria")
    except Exception as e:
        logger.error(f"Errore nel salvataggio: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)