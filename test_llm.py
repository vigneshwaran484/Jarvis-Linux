import requests

def ask_llm(messages):
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "llama3.2",
            "messages": messages,
            "stream": False
        }
    )
    return response.json()["message"]["content"]

# Test it
messages = [
    {"role": "user", "content": "Say hello and tell me what you are in one line"}
]

reply = ask_llm(messages)
print(reply)