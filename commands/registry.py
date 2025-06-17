# registry.py

from sentence_transformers import SentenceTransformer
import numpy as np
import logging
from functools import lru_cache

from commands.custom_commands import set_timer_from_text, apri_calendario

# Configurazione logging
logger = logging.getLogger(__name__)

# Mapping esteso per migliore riconoscimento
COMMANDS = {
    # Timer commands
    "setta timer": set_timer_from_text,
    "imposta timer": set_timer_from_text,
    "avvia timer": set_timer_from_text,
    "crea timer": set_timer_from_text,
    "timer": set_timer_from_text,
    "svegliami tra": set_timer_from_text,
    "ricordami tra": set_timer_from_text,
    
    # Calendar commands
    "apri calendario": apri_calendario,
    "mostra calendario": apri_calendario,
    "calendario": apri_calendario,
    "agenda": apri_calendario,
    "pianificazione": apri_calendario,
}

# Cache per modello e embeddings
_model = None
_embeddings = {}

def get_model():
    """Carica il modello sentence transformer (lazy loading)"""
    global _model
    if _model is None:
        try:
            _model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("✅ Modello sentence transformer caricato")
        except Exception as e:
            logger.error(f"❌ Errore caricamento modello: {e}")
            raise
    return _model

def initialize_embeddings():
    """Inizializza gli embeddings per tutti i comandi"""
    global _embeddings
    if not _embeddings:
        model = get_model()
        _embeddings = {cmd: model.encode(cmd) for cmd in COMMANDS}
        logger.info(f"✅ Embeddings inizializzati per {len(_embeddings)} comandi")

@lru_cache(maxsize=128)
def cosine_similarity_cached(input_text: str, command: str):
    """Calcola similarità coseno con cache"""
    model = get_model()
    if command not in _embeddings:
        _embeddings[command] = model.encode(command)
    
    input_emb = model.encode(input_text)
    cmd_emb = _embeddings[command]
    
    return np.dot(input_emb, cmd_emb) / (np.linalg.norm(input_emb) * np.linalg.norm(cmd_emb))

def cosine_similarity(a, b):
    """Calcola similarità coseno tra due vettori"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def dispatch_semantic(user_input: str, threshold: float = 0.65):
    """
    Dispatcher semantico puro basato su embeddings
    """
    initialize_embeddings()
    model = get_model()
    input_emb = model.encode(user_input)

    best_match = None
    best_score = -1
    best_command = None

    for cmd, emb in _embeddings.items():
        sim = cosine_similarity(input_emb, emb)
        if sim > best_score:
            best_score = sim
            best_match = cmd
            best_command = COMMANDS[cmd]

    logger.debug(f"🔍 Semantic match - Input: '{user_input}' | Best: '{best_match}' | Score: {best_score:.3f}")

    if best_score >= threshold:
        logger.info(f"✅ Comando semantico trovato: {best_match}")
        return best_command(user_input)

    return None

def dispatch_semantic_hybrid(user_input: str, embedding_threshold: float = 0.6):
    """
    Dispatcher ibrido migliorato: keyword matching + embeddings con fallback intelligente
    """
    user_lower = user_input.lower()
    
    # 1. Pattern matching avanzato per timer
    timer_patterns = [
        # Verifica presenza di parole temporali E parole comando
        (["timer", "setta", "imposta", "avvia", "crea", "sveglia", "ricorda"], 
        ["secondo", "secondi", "minuto", "minuti", "ora", "ore"]),
        # Pattern specifici
        (["tra"], ["secondo", "secondi", "minuto", "minuti", "ora", "ore"])
    ]
    
    for cmd_words, time_words in timer_patterns:
        if (any(word in user_lower for word in cmd_words) and 
            any(word in user_lower for word in time_words)):
            logger.info("🎯 Timer rilevato tramite pattern matching")
            return set_timer_from_text(user_input)
    
    # 2. Pattern matching per calendario
    calendar_patterns = ["calendario", "agenda", "pianificazione", "appuntamenti"]
    calendar_actions = ["apri", "mostra", "visualizza", "vedi"]
    
    if (any(pattern in user_lower for pattern in calendar_patterns) or
        (any(action in user_lower for action in calendar_actions) and 
        any(pattern in user_lower for pattern in calendar_patterns))):
        logger.info("🎯 Calendario rilevato tramite pattern matching")
        return apri_calendario(user_input)
    
    # 3. Fallback con embeddings (migliorato)
    try:
        initialize_embeddings()
        model = get_model()
        input_emb = model.encode(user_input)
        
        best_match = None
        best_score = -1
        best_command = None

        for cmd, emb in _embeddings.items():
            sim = cosine_similarity(input_emb, emb)
            if sim > best_score:
                best_score = sim
                best_match = cmd
                best_command = COMMANDS[cmd]

        logger.debug(f"🔍 Embedding match - Input: '{user_input}' | Best: '{best_match}' | Score: {best_score:.3f}")

        if best_score >= embedding_threshold:
            logger.info(f"✅ Comando trovato tramite embeddings: {best_match}")
            return best_command(user_input)
    
    except Exception as e:
        logger.error(f"❌ Errore nel matching semantico: {e}")
    
    return None

def get_command_suggestions(user_input: str, top_k: int = 3):
    """
    Restituisce i migliori match per suggerimenti
    """
    try:
        initialize_embeddings()
        model = get_model()
        input_emb = model.encode(user_input)
        
        similarities = []
        for cmd, emb in _embeddings.items():
            sim = cosine_similarity(input_emb, emb)
            similarities.append((cmd, sim))
        
        # Ordina per similarità decrescente
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    except Exception as e:
        logger.error(f"Errore nel calcolo suggerimenti: {e}")
        return []

def clear_cache():
    """Pulisce la cache (utile per development)"""
    global _embeddings
    _embeddings.clear()
    cosine_similarity_cached.cache_clear()
    logger.info("🧹 Cache embeddings pulita")