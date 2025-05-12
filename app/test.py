import requests

response = requests.post(
    "http://localhost:8080/generate",
    headers={"Content-Type": "application/json"},
    json={
        "inputs": "explain Dockerized TGI Inference",
        "parameters": {
            "max_new_tokens": 512,
            "top_k": 50,
            "top_p": 0.9,
            "temperature": 0.7
        }
    }
)

print(response.json())