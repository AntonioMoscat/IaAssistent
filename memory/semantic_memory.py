import faiss
import os
import pickle
import numpy as np
import unicodedata
from sentence_transformers import SentenceTransformer

INDEX_PATH   = "memory/memory_store/faiss.index"
MAPPING_PATH = "memory/memory_store/id_map.pkl"

model = SentenceTransformer("all-MiniLM-L6-v2")

def normalize(text: str) -> str:
    """Normalizza in minuscolo + unicode NFKC, rimuove spazi extra."""
    return unicodedata.normalize("NFKC", text.lower().strip())

class SemanticMemory:
    def __init__(self):
        self.index  = None            # faiss.IndexIDMap
        self.id_map = {}              # {int: str}
        self.dim    = 384             # Dimensionalità embedding
        self._load()

    # ---------- caricamento / salvataggio ----------
    def _load(self):
        if os.path.exists(INDEX_PATH):
            self.index = faiss.read_index(INDEX_PATH)
            # Se per qualche motivo non fosse IDMap, creane uno nuovo vuoto:
            if not isinstance(self.index, faiss.IndexIDMap):
                print("⚠️  Vecchio indice non compatibile: sarà ricreato da zero.")
                base        = faiss.IndexFlatL2(self.dim)
                self.index  = faiss.IndexIDMap(base)
                self.id_map = {}
                self.save()
            else:
                with open(MAPPING_PATH, "rb") as f:
                    self.id_map = pickle.load(f)
        else:
            base       = faiss.IndexFlatL2(self.dim)
            self.index = faiss.IndexIDMap(base)
            self.id_map = {}

    def save(self):
        # Salviamo *tutto* l’indice IDMap (ID + vettori)
        faiss.write_index(self.index, INDEX_PATH)
        with open(MAPPING_PATH, "wb") as f:
            pickle.dump(self.id_map, f)

    # ---------- operazioni di memoria ----------
    def add(self, text: str):
        text_norm = normalize(text)
        emb       = model.encode([text_norm]).astype("float32")
        idx       = len(self.id_map)
        self.index.add_with_ids(emb, np.array([idx]))
        self.id_map[idx] = text_norm
        self.save()

    def search(self, query: str, top_k: int = 1) -> str:
        query_norm = normalize(query)
        emb        = model.encode([query_norm]).astype("float32")
        D, I       = self.index.search(emb, top_k)

        if I[0][0] == -1:
            return ""

        distance = D[0][0]
        idx      = I[0][0]

        # ritorna il contesto solo se abbastanza “vicino”
        return self.id_map[idx] if distance > 0.8 else ""

    def learn(self, old_input: str, corrected_input: str):
        old_norm = normalize(old_input)
        emb      = model.encode([old_norm]).astype("float32")
        D, I     = self.index.search(emb, 1)
        idx      = I[0][0]

        # rimuovi il vecchio, se esiste
        if idx != -1 and idx in self.id_map:
            self.index.remove_ids(np.array([idx]))
            del self.id_map[idx]

        # aggiungi la versione corretta
        self.add(corrected_input)
