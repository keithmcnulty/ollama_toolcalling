import ollama
import requests

def get_current_weather(city:str):
    base_url = f"https://wttr.in/{city}?format=j1"
    response = requests.get(base_url)
    data = response.json()
    return f"The current temperature in {city} is {data['current_condition'][0]['temp_C']} degrees C"

response = ollama.chat(
    model = 'llama3.1:70b',
    messages = [{'role': 'user', 'content': 'What is the weather like in Dublin?'}],
    tools = [{'type': 'function',
              'function': {
                  'name': 'get_current_weather',
                  'description': 'Gets the current weather in a city',
                  'parameters': {
                      'type': 'object',
                      'properties': {
                          'city': {
                              'type': 'string',
                              'description':  'The name of the city'
                          },
                      },
                      'required': ['city']
                    }
                }
    }]
)

tool_calls = response['message']['tool_calls']

tool_name = tool_calls[0]['function']['name']
arguments = tool_calls[0]['function']['arguments']
city = arguments['city']

result = get_current_weather(city)
print(result)