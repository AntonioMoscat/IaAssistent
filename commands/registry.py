# registry.py

from sentence_transformers import SentenceTransformer
import numpy as np

from commands.custom_commands import set_timer_from_text, apri_calendario

# Aggiungi più varianti per ogni comando per migliorare il matching
COMMANDS = {
    "setta timer": set_timer_from_text,
    "imposta timer": set_timer_from_text,
    "avvia timer": set_timer_from_text,
    "crea timer": set_timer_from_text,
    "timer": set_timer_from_text,
    "apri calendario": apri_calendario,
    "mostra calendario": apri_calendario,
    "calendario": apri_calendario,
}

_model = SentenceTransformer("all-MiniLM-L6-v2")
_embeddings = {cmd: _model.encode(cmd) for cmd in COMMANDS}


def dispatch_semantic(user_input: str, threshold: float = 0.65):  # Soglia ridotta da 0.75 a 0.65
    input_emb = _model.encode(user_input)

    best_match = None
    best_score = -1
    best_command = None

    for cmd, emb in _embeddings.items():
        sim = cosine_similarity(input_emb, emb)
        if sim > best_score:
            best_score = sim
            best_match = cmd
            best_command = COMMANDS[cmd]

    # Debug: stampa il matching per aiutare nel tuning
    print(f"🔍 Debug - Input: '{user_input}' | Best match: '{best_match}' | Score: {best_score:.3f}")

    if best_score >= threshold:
        return best_command(user_input)

    return None


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# Funzione alternativa con approccio ibrido (keyword + embedding)
def dispatch_semantic_hybrid(user_input: str, embedding_threshold: float = 0.6):
    """
    Approccio ibrido: prima cerca parole chiave, poi usa embeddings
    """
    user_lower = user_input.lower()
    
    # 1. Matching per parole chiave specifiche per i timer
    timer_keywords = ["timer", "setta", "imposta", "avvia", "crea"]
    time_keywords = ["secondo", "secondi", "minuto", "minuti"]
    
    if any(kw in user_lower for kw in timer_keywords) and any(kw in user_lower for kw in time_keywords):
        print(f"🎯 Match diretto per timer rilevato")
        return set_timer_from_text(user_input)
    
    # 2. Matching per calendario
    calendar_keywords = ["calendario", "apri calendario", "mostra calendario"]
    if any(kw in user_lower for kw in calendar_keywords):
        print(f"🎯 Match diretto per calendario rilevato")
        return apri_calendario(user_input)
    
    # 3. Fallback con embeddings (soglia più bassa)
    input_emb = _model.encode(user_input)
    best_match = None
    best_score = -1
    best_command = None

    for cmd, emb in _embeddings.items():
        sim = cosine_similarity(input_emb, emb)
        if sim > best_score:
            best_score = sim
            best_match = cmd
            best_command = COMMANDS[cmd]

    print(f"🔍 Embedding match - Input: '{user_input}' | Best: '{best_match}' | Score: {best_score:.3f}")

    if best_score >= embedding_threshold:
        return best_command(user_input)

    return None