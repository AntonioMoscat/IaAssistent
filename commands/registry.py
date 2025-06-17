from sentence_transformers import SentenceTransformer
import numpy as np

from commands.custom_commands import set_timer_from_text, apri_calendario

COMMANDS = {
    "setta timer": set_timer_from_text,
    "apri calendario": apri_calendario,
}

_model = SentenceTransformer("all-MiniLM-L6-v2")
_embeddings = {cmd: _model.encode(cmd) for cmd in COMMANDS}


def dispatch_semantic(user_input: str, threshold: float = 0.75):
    input_emb = _model.encode(user_input)

    best_match = None
    best_score = -1

    for cmd, emb in _embeddings.items():
        sim = cosine_similarity(input_emb, emb)
        if sim > best_score:
            best_score = sim
            best_match = cmd

    if best_score >= threshold:
        return COMMANDS[best_match](user_input)

    return None


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
