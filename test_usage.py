import base64
import json
from pathlib import Path
import sys
from colorama import init, Fore, Style

sys.path.append(str(Path(__file__).parent / "src"))
from openai_unofficial import OpenAIUnofficial

init()

#------------------------ Basic Chat Completion ------------------------#
print(f"\n{Fore.CYAN}{'='*50}")
print(f"{Fore.YELLOW}Testing OpenAI Unofficial API Endpoints")
print(f"{Fore.CYAN}{'='*50}{Style.RESET_ALL}\n")

client = OpenAIUnofficial()
print(f"{Fore.GREEN}▶ Testing Basic Chat Completion{Style.RESET_ALL}")
completion = client.chat.completions.create(
    messages=[{"role": "user", "content": "Say hello!"}],
    model="gpt-4o-mini-2024-07-18"
)
print(f"{Fore.WHITE}Response: {completion.choices[0].message.content}{Style.RESET_ALL}\n")

#------------------------ Chat Completion with Image ------------------------#
client = OpenAIUnofficial()
print(f"{Fore.GREEN}▶ Testing Chat Completion with Image Input{Style.RESET_ALL}")
completion = client.chat.completions.create(
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                }
            },
        ],
    }],
    model="gpt-4o-mini-2024-07-18"
)
print(f"{Fore.WHITE}Response: {completion.choices[0].message.content}{Style.RESET_ALL}\n")

#------------------------ Streaming Chat Completion ------------------------#
client = OpenAIUnofficial()
print(f"{Fore.GREEN}▶ Testing Streaming Chat Completion{Style.RESET_ALL}")
completion_stream = client.chat.completions.create(
    messages=[{"role": "user", "content": "Write a short story in 3 sentences."}],
    model="gpt-4o-mini-2024-07-18",
    stream=True
)

print(f"{Fore.WHITE}Streaming response:", end='')
for chunk in completion_stream:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end='', flush=True)
print(f"{Style.RESET_ALL}\n")

#------------------------ Audio Speech Generation ------------------------#
client = OpenAIUnofficial()
print(f"{Fore.GREEN}▶ Testing Audio Speech Generation{Style.RESET_ALL}")
audio_data = client.audio.create(
    input_text="Hello, this is a test message!",
    model="tts-1-hd-1106",
    voice="nova"
)
output_path = Path("test_audio.mp3")
output_path.write_bytes(audio_data)
print(f"{Fore.WHITE}Audio file saved: {output_path}{Style.RESET_ALL}\n")

#------------------------ Image Generation ------------------------#
client = OpenAIUnofficial()
print(f"{Fore.GREEN}▶ Testing Image Generation{Style.RESET_ALL}")
image_response = client.image.create(
    prompt="A beautiful sunset over mountains",
    model="dall-e-3",
    size="1024x1024"
)
print(f"{Fore.WHITE}Generated Image URL: {image_response.data[0].url}{Style.RESET_ALL}\n")

#------------------------ Audio Preview Model ------------------------#
client = OpenAIUnofficial()
print(f"{Fore.GREEN}▶ Testing Audio Preview Model{Style.RESET_ALL}")
try:
    completion = client.chat.completions.create(
        messages=[{"role": "user", "content": "Tell me a short joke."}],
        model="gpt-4o-audio-preview-2024-10-01",
        modalities=["text", "audio"],
        audio={"voice": "fable", "format": "wav"}
    )
    
    message = completion.choices[0].message
    print(f"{Fore.WHITE}Text Response: {message.content}")
    
    if message.audio and 'data' in message.audio:
        output_path = Path("audio_preview.wav")
        output_path.write_bytes(base64.b64decode(message.audio['data']))
        print(f"Audio preview saved: {output_path}{Style.RESET_ALL}")
except Exception as e:
    print(f"{Fore.RED}Audio preview test failed: {e}{Style.RESET_ALL}")


#------------------------ Function Calling ------------------------#
client = OpenAIUnofficial()
print(f"{Fore.GREEN}▶ Testing Function Calling{Style.RESET_ALL}")

def get_current_weather(location: str, unit: str = "celsius") -> str:
    if unit == "fahrenheit":
        temperature = 72
    else:
        temperature = 22
    return f"The weather in {location} is {temperature} degrees {unit}."

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather for a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city name, e.g., London, New York"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

messages = [
    {"role": "user", "content": "What's the weather like in New York?"}
]

print(f"{Fore.WHITE}Step 1: Initial API call for function calling")
response = client.chat.completions.create(
    model="gpt-4o-mini-2024-07-18",
    messages=messages,
    tools=tools,
    tool_choice="auto"
)

assistant_message = response.choices[0].message
print(f"Assistant's Initial Response: {assistant_message.to_dict()}")
messages.append(assistant_message.to_dict())

if assistant_message.tool_calls:
    tool_call = assistant_message.tool_calls[0]
    function_name = tool_call.function.name
    function_args = json.loads(tool_call.function.arguments)
    print(f"\nFunction Called: {function_name}")
    print(f"Function Arguments: {function_args}")
    
    function_response = get_current_weather(**function_args)
    print(f"Function Response: {function_response}")
    
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "name": function_name,
        "content": function_response
    })
    
    print("\nStep 2: Final API call with function response")
    final_response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=messages,
        tools=tools
    )
    
    print(f"Final Assistant Response: {final_response.choices[0].message.content}")
else:
    print(f"No function call needed. Response: {assistant_message.content}")
print(f"{Style.RESET_ALL}\n")

print(f"\n{Fore.CYAN}{'='*50}")
print(f"{Fore.YELLOW}All tests completed")
print(f"{Fore.CYAN}{'='*50}{Style.RESET_ALL}\n")