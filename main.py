# main.py

import subprocess
import requests
import time
import sys
from llm_wrapper import LocalLLM
from memory.memory import load_memory, save_memory
from agent import dispatch
from memory.semantic_memory import SemanticMemory
from commands.registry import dispatch_semantic_hybrid


def is_ollama_running():
    try:
        res = requests.get("http://localhost:11434")
        return res.status_code == 200
    except:
        return False

def start_ollama():
    print("⚙️ Avvio Ollama in background...")
    return subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def main():
    if not is_ollama_running():
        ollama_process = start_ollama()
        for _ in range(10):
            if is_ollama_running():
                print("✅ Ollama avviato.")
                break
            time.sleep(1)
        else:
            print("❌ Ollama non si è avviato.")
            sys.exit(1)
    else:
        print("✅ Ollama è già attivo.")
        ollama_process = None

    sem_mem = SemanticMemory()
    llm = LocalLLM()
    memory = load_memory()

    print("🤖 Assistente AI Avviato (offline). Scrivi 'esci' per terminare.")

    try:
        while True:
            command = input("Tu: ").strip()
            if command.lower() == "esci":
                print("🤖 Spegnimento in corso...")
                if ollama_process:
                    ollama_process.terminate()
                    try:
                        ollama_process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        ollama_process.kill()
                    print("🤖 Ollama terminato. Ciao!")
                else:
                    print("🤖 Ollama non era gestito dal processo.")
                break

            if command.startswith("correggi:"):
                parts = command[len("correggi:"):].split("->")
                if len(parts) == 2:
                    vecchia, nuova = parts[0].strip(), parts[1].strip()
                    sem_mem.learn(vecchia, nuova)
                    print("✅ Correzione registrata.")
                    continue

            # 👇 Comando semantico ibrido (fallback se non è None)
            response = dispatch_semantic_hybrid(command)  # Cambiato qui

            # 👇 Comando da agent.py se non è un comando personalizzato
            if response is None:
                response = dispatch(command)

            # 👇 LLM locale se ancora nulla
            if response is None:
                context = sem_mem.search(command)
                prompt = f"Contesto precedente: {context}\nDomanda: {command}" if context and context != command else command
                response = llm.respond(prompt)

            print("AI:", response)

            sem_mem.add(command)
            memory["history"].append({"user": command, "ai": response})
            save_memory(memory)

    except KeyboardInterrupt:
        print("\n🤖 Interruzione manuale. Spegnimento in corso...")
        if ollama_process:
            ollama_process.terminate()
            try:
                ollama_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                ollama_process.kill()
        print("🤖 Ollama terminato. Ciao!")
        sys.exit(0)

if __name__ == "__main__":
    main()