# OpenAI Unofficial Python SDK

[![PyPI](https://img.shields.io/pypi/v/openai-unofficial.svg)](https://pypi.org/project/openai-unofficial/)
[![License](https://img.shields.io/pypi/l/openai-unofficial.svg)](https://github.com/SreejanPersonal/openai-unofficial/blob/main/LICENSE)
[![Python Versions](https://img.shields.io/pypi/pyversions/openai-unofficial.svg)](https://pypi.org/project/openai-unofficial/)
[![Downloads](https://static.pepy.tech/badge/openai-unofficial)](https://pepy.tech/project/openai-unofficial)

An Free & Unlimited unofficial Python SDK for the OpenAI API, providing seamless integration and easy-to-use methods for interacting with OpenAI's latest powerful AI models, including GPT-4o (Including gpt-4o-audio-preview & gpt-4o-realtime-preview Models), GPT-4, GPT-3.5 Turbo, DALL·E 3, Whisper & Text-to-Speech (TTS) models for Free

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
  - [List Available Models](#list-available-models)
  - [Basic Chat Completion](#basic-chat-completion)
  - [Chat Completion with Image Input](#chat-completion-with-image-input)
  - [Streaming Chat Completion using Real-Time Model](#streaming-chat-completion-using-real-time-model)
  - [Audio Generation with TTS Model](#audio-generation-with-tts-model)
  - [Chat Completion with Audio Preview Model](#chat-completion-with-audio-preview-model)
  - [Image Generation](#image-generation)
  - [Audio Speech Recognition with Whisper Model](#audio-speech-recognition-with-whisper-model)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Comprehensive Model Support**: Integrate with the latest OpenAI models, including GPT-4, GPT-4o, GPT-3.5 Turbo, DALL·E 3, Whisper, Text-to-Speech (TTS) models, and the newest audio preview and real-time models.
- **Chat Completions**: Generate chat-like responses using a variety of models.
- **Streaming Responses**: Support for streaming chat completions, including real-time models for instantaneous outputs.
- **Audio Generation**: Generate high-quality speech audio with various voice options using TTS models.
- **Audio and Text Responses**: Utilize models like `gpt-4o-audio-preview` to receive both audio and text responses.
- **Image Generation**: Create stunning images using DALL·E models with customizable parameters.
- **Audio Transcription**: Convert speech to text using Whisper models.
- **Easy to Use**: Simple and intuitive methods to interact with various endpoints.
- **Extensible**: Designed to be easily extendable for future OpenAI models and endpoints.

---

## Installation

Install the package via pip:

```bash
pip install openai-unofficial
```

---

## Quick Start

```python
from openai_unofficial import OpenAIUnofficial

# Initialize the client
client = OpenAIUnofficial()

# Basic chat completion
response = client.chat.completions.create(
    messages=[{"role": "user", "content": "Say hello!"}],
    model="gpt-4o"
)
print(response.choices[0].message.content)
```

---

## Usage Examples

### List Available Models

```python
from openai_unofficial import OpenAIUnofficial

client = OpenAIUnofficial()
models = client.list_models()
print("Available Models:")
for model in models['data']:
    print(f"- {model['id']}")
```

### Basic Chat Completion

```python
from openai_unofficial import OpenAIUnofficial

client = OpenAIUnofficial()
response = client.chat.completions.create(
    messages=[{"role": "user", "content": "Tell me a joke."}],
    model="gpt-4o"
)
print("ChatBot:", response.choices[0].message.content)
```

### Chat Completion with Image Input

```python
from openai_unofficial import OpenAIUnofficial

client = OpenAIUnofficial()
response = client.chat.completions.create(
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
print("Response:", response.choices[0].message.content)
```

### Streaming Chat Completion

```python
from openai_unofficial import OpenAIUnofficial

client = OpenAIUnofficial()
completion_stream = client.chat.completions.create(
    messages=[{"role": "user", "content": "Write a short story in 3 sentences."}],
    model="gpt-4o-mini-2024-07-18",
    stream=True
)
for chunk in completion_stream:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end='', flush=True)
```

### Audio Generation with TTS Model

```python
from openai_unofficial import OpenAIUnofficial

client = OpenAIUnofficial()
audio_data = client.audio.create(
    input_text="This is a test of the TTS capabilities!",
    model="tts-1-hd",
    voice="nova"
)
with open("tts_output.mp3", "wb") as f:
    f.write(audio_data)
print("TTS Audio saved as tts_output.mp3")
```

### Chat Completion with Audio Preview Model

```python
from openai_unofficial import OpenAIUnofficial

client = OpenAIUnofficial()
response = client.chat.completions.create(
    messages=[{"role": "user", "content": "Tell me a fun fact."}],
    model="gpt-4o-audio-preview",
    modalities=["text", "audio"],
    audio={"voice": "fable", "format": "wav"}
)

message = response.choices[0].message
print("Text Response:", message.content)

if message.audio and 'data' in message.audio:
    from base64 import b64decode
    with open("audio_preview.wav", "wb") as f:
        f.write(b64decode(message.audio['data']))
    print("Audio saved as audio_preview.wav")
```

### Image Generation

```python
from openai_unofficial import OpenAIUnofficial

client = OpenAIUnofficial()
response = client.image.create(
    prompt="A futuristic cityscape at sunset",
    model="dall-e-3",
    size="1024x1024"
)
print("Image URL:", response.data[0].url)
```

### Audio Speech Recognition with Whisper Model

```python
from openai_unofficial import OpenAIUnofficial

client = OpenAIUnofficial()
with open("speech.mp3", "rb") as audio_file:
    transcription = client.audio.transcribe(
        file=audio_file,
        model="whisper-1"
    )
print("Transcription:", transcription.text)
```

---

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository.
2. **Create a new branch**: `git checkout -b feature/my-feature`.
3. **Commit your changes**: `git commit -am 'Add new feature'`.
4. **Push to the branch**: `git push origin feature/my-feature`.
5. **Open a pull request**.

Please ensure your code adheres to the project's coding standards and passes all tests.

---

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/SreejanPersonal/openai-unofficial/blob/main/LICENSE) file for details.

---

**Note**: This SDK is unofficial and not affiliated with OpenAI.

---

If you encounter any issues or have suggestions, please open an issue on [GitHub](https://github.com/SreejanPersonal/openai-unofficial/issues).

---

## Supported Models

Here's a partial list of models that the SDK currently supports. For Complete list, check out the `/models` endpoint:

- **Chat Models**:
  - `gpt-4`
  - `gpt-4-turbo`
  - `gpt-4o`
  - `gpt-4o-mini`
  - `gpt-3.5-turbo`
  - `gpt-3.5-turbo-16k`
  - `gpt-3.5-turbo-instruct`
  - `gpt-4o-realtime-preview`
  - `gpt-4o-audio-preview`

- **Image Generation Models**:
  - `dall-e-2`
  - `dall-e-3`

- **Text-to-Speech (TTS) Models**:
  - `tts-1`
  - `tts-1-hd`
  - `tts-1-1106`
  - `tts-1-hd-1106`

- **Audio Models**:
  - `whisper-1`

- **Embedding Models**:
  - `text-embedding-ada-002`
  - `text-embedding-3-small`
  - `text-embedding-3-large`

---