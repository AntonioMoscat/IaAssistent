# main.py

import subprocess
import requests
import time
import sys
import signal
import logging
from llm_wrapper import LocalLLM
from memory.memory import load_memory, save_memory
from agent import dispatch, get_available_commands
from memory.semantic_memory import SemanticMemory
from commands.registry import dispatch_semantic_hybrid

# Configurazione logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Variabili globali per gestione risorse
ollama_process = None
sem_mem = None
llm = None
memory = None

def signal_handler(signum, frame):
    """Gestisce l'interruzione con Ctrl+C"""
    print("\n🤖 Interruzione rilevata. Spegnimento in corso...")
    cleanup_and_exit()

def is_ollama_running():
    """Verifica se Ollama è in esecuzione"""
    try:
        res = requests.get("http://localhost:11434", timeout=5)
        return res.status_code == 200
    except Exception as e:
        logger.debug(f"Ollama non raggiungibile: {e}")
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
        return True
    except Exception as e:
        logger.error(f"Errore nell'avvio di Ollama: {e}")
        return False

def cleanup_and_exit():
    """Pulizia risorse e uscita"""
    global ollama_process
    
    if ollama_process:
        logger.info("🧹 Terminazione Ollama...")
        ollama_process.terminate()
        try:
            ollama_process.wait(timeout=5)
            logger.info("✅ Ollama terminato correttamente")
        except subprocess.TimeoutExpired:
            logger.warning("⚠️ Ollama non risponde, forzando terminazione...")
            ollama_process.kill()
            logger.info("✅ Ollama terminato forzatamente")
    
    logger.info("👋 Ciao!")
    sys.exit(0)

def process_correction(command: str) -> bool:
    """Processa i comandi di correzione"""
    if not command.startswith("correggi:"):
        return False
    
    parts = command[len("correggi:"):].split("->")
    if len(parts) != 2:
        print("❌ Formato correzione non valido. Usa: correggi: vecchia frase -> nuova frase")
        return True
    
    vecchia, nuova = parts[0].strip(), parts[1].strip()
    if not vecchia or not nuova:
        print("❌ Frasi vuote non sono permesse")
        return True
    
    sem_mem.learn(vecchia, nuova)
    print("✅ Correzione registrata.")
    return True

def process_command(command: str) -> str:
    """Processa un comando e restituisce la risposta"""
    # Prova dispatcher semantico ibrido
    response = dispatch_semantic_hybrid(command)
    if response:
        return response
    
    # Fallback su dispatcher tradizionale
    response = dispatch(command)
    if response:
        return response
    
    # Fallback su LLM con contesto
    context = sem_mem.search(command)
    prompt = (f"Contesto precedente: {context}\nDomanda: {command}" 
             if context and context != command else command)
    return llm.respond(prompt)

def show_welcome():
    """Mostra messaggio di benvenuto"""
    print("🤖 Assistente AI Avviato (offline)")
    print("📋 Comandi disponibili:", ", ".join(get_available_commands()))
    print("🔧 Comandi speciali:")
    print("   - 'correggi: vecchia frase -> nuova frase' per correzioni")
    print("   - 'esci' per terminare")
    print("   - 'help' per aiuto")
    print("-" * 50)

def show_help():
    """Mostra messaggio di aiuto"""
    print("\n📚 GUIDA COMANDI:")
    print("🔹 Comandi base:", ", ".join(get_available_commands()))
    print("🔹 Timer: 'setta timer 5 minuti', 'imposta timer 30 secondi'")
    print("🔹 Calendario: 'apri calendario', 'mostra calendario'")
    print("🔹 Correzioni: 'correggi: sbagliato -> corretto'")
    print("🔹 Uscita: 'esci'")
    print("-" * 50)

def main():
    global ollama_process, sem_mem, llm, memory
    
    # Registra handler per Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    # Inizializza Ollama
    if not is_ollama_running():
        if not start_ollama():
            logger.error("❌ Impossibile avviare Ollama")
            sys.exit(1)
        
        # Attendi che Ollama sia pronto
        for attempt in range(15):
            if is_ollama_running():
                logger.info("✅ Ollama avviato con successo")
                break
            time.sleep(1)
            logger.info(f"⏳ Tentativo {attempt + 1}/15 - Attendo Ollama...")
        else:
            logger.error("❌ Ollama non si è avviato entro il timeout")
            cleanup_and_exit()
    else:
        logger.info("✅ Ollama è già attivo")
    
    # Inizializza componenti
    try:
        sem_mem = SemanticMemory()
        llm = LocalLLM()
        memory = load_memory()
        logger.info("✅ Componenti inizializzati")
    except Exception as e:
        logger.error(f"❌ Errore nell'inizializzazione: {e}")
        cleanup_and_exit()
    
    show_welcome()
    
    # Loop principale
    try:
        while True:
            try:
                command = input("Tu: ").strip()
                
                if not command:
                    continue
                
                if command.lower() == "esci":
                    break
                
                if command.lower() == "help":
                    show_help()
                    continue
                
                # Processa correzioni
                if process_correction(command):
                    continue
                
                # Processa comando normale
                response = process_command(command)
                print("AI:", response)
                
                # Salva in memoria
                sem_mem.add(command)
                memory["history"].append({"user": command, "ai": response})
                save_memory(memory)
                
            except EOFError:
                print("\n🤖 Input terminato")
                break
            except Exception as e:
                logger.error(f"Errore durante l'elaborazione: {e}")
                print(f"❌ Errore: {e}")
                
    except KeyboardInterrupt:
        pass  # Gestito dal signal handler
    
    cleanup_and_exit()

if __name__ == "__main__":
    main()