import requests

MODEL = "qwen2.5:7b-instruct"
URL = "http://localhost:11434/api/generate"

payload = {
    "model": MODEL,
    "prompt": "Reply with exactly: OK",
    "stream": False
}

r = requests.post(URL, json=payload, timeout=120)
r.raise_for_status()
print(r.json()["response"])
