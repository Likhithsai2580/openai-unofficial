import base64
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

print(f"\n{Fore.CYAN}{'='*50}")
print(f"{Fore.YELLOW}All tests completed")
print(f"{Fore.CYAN}{'='*50}{Style.RESET_ALL}\n")