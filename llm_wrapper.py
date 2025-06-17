import requests

class LocalLLM:
    def __init__(self, model="mistral"):
        self.model = model
        self.url = "http://localhost:11434/api/generate"

    def respond(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        response = requests.post(self.url, json=payload)
        if response.status_code == 200:
            return response.json().get("response", "").strip()
        else:
            return f"Errore nella generazione: {response.status_code}"
