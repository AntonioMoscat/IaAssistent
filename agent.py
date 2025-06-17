import importlib
import logging

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TASKS = {
    "ripeti": "echo",
    "meteo": "weather_fake",
}

# Cache per evitare di reimportare moduli
_module_cache = {}

def dispatch(command: str) -> str:
    """
    Dispatcher migliorato con cache dei moduli e logging
    """
    command_lower = command.lower()
    
    for keyword, module_name in TASKS.items():
        if keyword in command_lower:
            try:
                # Usa cache se disponibile
                if module_name not in _module_cache:
                    mod = importlib.import_module(f"tasks.{module_name}")
                    _module_cache[module_name] = mod
                    logger.info(f"Modulo {module_name} caricato e messo in cache")
                else:
                    mod = _module_cache[module_name]
                
                result = mod.run(command)
                logger.info(f"Comando '{keyword}' eseguito con successo")
                return result
                
            except ImportError as e:
                logger.error(f"Errore nell'importazione del modulo {module_name}: {e}")
                return f"❌ Errore: modulo '{module_name}' non trovato"
            except Exception as e:
                logger.error(f"Errore nell'esecuzione del comando '{keyword}': {e}")
                return f"❌ Errore nell'esecuzione del comando: {e}"
    
    return None

def get_available_commands():
    """Restituisce la lista dei comandi disponibili"""
    return list(TASKS.keys())

def clear_cache():
    """Pulisce la cache dei moduli (utile per development)"""
    global _module_cache
    _module_cache.clear()
    logger.info("Cache dei moduli pulita")