# api_server.py

import subprocess
import requests
import time
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from llm_wrapper import LocalLLM
from memory.memory import load_memory, save_memory
from agent import dispatch
from memory.semantic_memory import SemanticMemory
from commands.registry import dispatch_semantic_hybrid

app = FastAPI()

# === AI Setup globale ===
sem_mem = SemanticMemory()
llm = LocalLLM()
memory = load_memory()

ollama_process = None

# === Modello per la richiesta API ===
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str


def is_ollama_running():
    try:
        res = requests.get("http://localhost:11434")
        return res.status_code == 200
    except:
        return False

def start_ollama():
    global ollama_process
    print("⚙️ Avvio Ollama in background...")
    ollama_process = subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)  # inizializzazione

@app.on_event("startup")
def startup():
    if not is_ollama_running():
        start_ollama()
        for _ in range(10):
            if is_ollama_running():
                print("✅ Ollama avviato.")
                return
            time.sleep(1)
        print("❌ Ollama non si è avviato.")
        sys.exit(1)
    else:
        print("✅ Ollama è già attivo.")

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    command = request.message.strip()

    if command.startswith("correggi:"):
        parts = command[len("correggi:"):].split("->")
        if len(parts) == 2:
            vecchia, nuova = parts[0].strip(), parts[1].strip()
            sem_mem.learn(vecchia, nuova)
            return {"response": "✅ Correzione registrata."}

    response = dispatch_semantic_hybrid(command)

    if response is None:
        response = dispatch(command)

    if response is None:
        context = sem_mem.search(command)
        prompt = f"Contesto precedente: {context}\nDomanda: {command}" if context and context != command else command
        response = llm.respond(prompt)

    sem_mem.add(command)
    memory["history"].append({"user": command, "ai": response})
    save_memory(memory)

    return {"response": response}
