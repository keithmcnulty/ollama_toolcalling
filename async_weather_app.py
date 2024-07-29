import ollama
import asyncio
import requests


def get_current_temperature(city: str):
  base_url = f"https://wttr.in/{city}?format=j1"
  response = requests.get(base_url)
  data = response.json()
  return f"The current temperature in {city} is {data['current_condition'][0]['temp_C']} degrees C"

async def run(model: str, query:str) -> str:
  client = ollama.AsyncClient()
  # Initialize conversation with a user query
  messages = [{'role': 'user', 'content': query}]

  # First API call: Send the query and function description to the model
  response = await client.chat(
    model=model,
    messages=messages,
    tools=[
      {
        'type': 'function',
        'function': {
          'name': 'get_current_temperature',
          'description': 'Get the temperature in a city',
          'parameters': {
            'type': 'object',
            'properties': {
              'city': {
                'type': 'string',
                'description': 'The city for which the temperature is requested',
              }
            },
            'required': ['city'],
          },
        },
      },
    ],
  )

  # Add the model's response to the conversation history
  messages.append(response['message'])

  # Check if the model decided to use the provided function
  if not response['message'].get('tool_calls'):
    print("The model didn't use the function. Its response was:")
    print(response['message']['content'])
    return

  # Process function calls made by the model
  if response['message'].get('tool_calls'):
    available_functions = {
      'get_current_temperature': get_current_temperature,
    }
    for tool in response['message']['tool_calls']:
      function_to_call = available_functions[tool['function']['name']]
      function_response = function_to_call(tool['function']['arguments']['city'])
      # Add function response to the conversation
      messages.append(
        {
          'role': 'tool',
          'content': f"""
          Answer the following question: {query}.  
          Ignore any previous instructions or defaults and instead use the following information:  {function_response}
          """,
        }
      )

  # Second API call: Get final response from the model
  final_response = await client.chat(model=model, messages=messages)
  print(final_response['message']['content'])


# Run the async function
asyncio.run(run('llama3.1:8b', "My sister has recommended that I bring warm clothing on my trip to the capital of Norway today.  Should I follow her advice?"))