import ollama

response = ollama.chat(
    model = 'llama3.1:8b',
    messages = [{'role': 'user', 'content': "What should I wear for my trip to Tasmania tomorrow?"}]
)

print(response['message']['content'])
