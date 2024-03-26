import json
import requests

headers = {"Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiZTViZThlODctNDk4NC00OTczLTk1ZTktNmQ3NDhjNzc2MDk0IiwidHlwZSI6ImFwaV90b2tlbiJ9.w3-pnK2DS_PQkKfYEP8Mw5AcLkqKm_7SsgBZsHDxsyA"}

url = "https://api.edenai.run/v2/text/chat"
payload = {
    "providers": "openai",
    "text": "Hello i need your help ! ",
    "chatbot_global_action": "Act as an assistant",
    "previous_history": [],
    "temperature": 0.0,
    "max_tokens": 150,
    "fallback_providers": ""
}

response = requests.post(url, json=payload, headers=headers)

result = json.loads(response.text)
print(result['openai']['generated_text'])
