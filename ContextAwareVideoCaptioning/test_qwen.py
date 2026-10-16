import requests
import base64

image_path = r"C:\Users\CHINMAY KAPILA\Downloads\Resume.jpeg"

with open(image_path, "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode("utf-8")

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "qwen3-vl:8b",
        "prompt": "Describe this image in 2 concise sentences. Do not explain your reasoning.",
        "images": [image_base64],
        "stream": False
    }
)

print(response.json()["response"])