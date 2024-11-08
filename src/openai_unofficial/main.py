import requests
import json
from typing import Optional, List, Union, Dict, Any, Iterator, TypeVar
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import logging
from urllib.parse import urljoin

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T = TypeVar('T')

class APIError(Exception):
    """Base exception for API errors."""
    def __init__(self, message: str, status_code: Optional[int] = None, response: Optional[Dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response

class ModelType(Enum):
    CHAT = "chat"
    AUDIO = "audio"
    IMAGE = "image"

class ResponseFormat(Enum):
    URL = "url"
    B64_JSON = "b64_json"

@dataclass
class APIConfig:
    base_url: str
    timeout: int = 30
    max_retries: int = 3

class BaseAPIHandler:
    def __init__(self, config: APIConfig):
        self.config = config
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(max_retries=self.config.max_retries)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session

    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        url = urljoin(self.config.base_url + '/', endpoint)
        try:
            response = self.session.request(
                method=method,
                url=url,
                timeout=self.config.timeout,
                **kwargs
            )
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            raise APIError(f"Request failed: {str(e)}", 
                         getattr(e.response, 'status_code', None),
                         getattr(e.response, 'json', lambda: None)())

class BaseModel(ABC):
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        pass

class FunctionCall(BaseModel):
    def __init__(self, data: Dict[str, Any]):
        self.name = data.get('name')
        self.arguments = data.get('arguments')
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'arguments': self.arguments
        }

class ToolCall(BaseModel):
    def __init__(self, data: Dict[str, Any]):
        self.id = data.get('id')
        self.type = data.get('type')
        self.function = FunctionCall(data.get('function', {})) if data.get('function') else None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'type': self.type,
            'function': self.function.to_dict() if self.function else None
        }

class ChatMessage(BaseModel):
    def __init__(self, data: Dict[str, Any]):
        self.role = data.get('role')
        self.content = data.get('content')
        self.function_call = FunctionCall(data.get('function_call', {})) if data.get('function_call') else None
        self.tool_calls = [ToolCall(tc) for tc in data.get('tool_calls', [])] if data.get('tool_calls') else []
        self.audio = data.get('audio')
        # For messages of role 'tool', include 'tool_call_id' and 'name'
        self.tool_call_id = data.get('tool_call_id')
        self.name = data.get('name')
        
    def to_dict(self) -> Dict[str, Any]:
        message_dict = {'role': self.role}
        if self.content is not None:
            message_dict['content'] = self.content
        if self.function_call is not None:
            message_dict['function_call'] = self.function_call.to_dict()
        if self.tool_calls:
            message_dict['tool_calls'] = [tool_call.to_dict() for tool_call in self.tool_calls]
        if self.audio is not None:
            message_dict['audio'] = self.audio
        if self.role == 'tool':
            if self.tool_call_id:
                message_dict['tool_call_id'] = self.tool_call_id
            if self.name:
                message_dict['name'] = self.name
        return message_dict

class ChatCompletionChoice(BaseModel):
    def __init__(self, data: Dict[str, Any]):
        self.index = data.get('index')
        self.message = ChatMessage(data.get('message', {}))
        self.finish_reason = data.get('finish_reason')
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'index': self.index,
            'message': self.message.to_dict(),
            'finish_reason': self.finish_reason
        }

class ChatCompletionResponse(BaseModel):
    def __init__(self, data: Dict[str, Any]):
        self.id = data.get('id')
        self.object = data.get('object')
        self.created = data.get('created')
        self.model = data.get('model')
        self.choices = [ChatCompletionChoice(choice) for choice in data.get('choices', [])]
        self.usage = data.get('usage')
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'object': self.object,
            'created': self.created,
            'model': self.model,
            'choices': [choice.to_dict() for choice in self.choices],
            'usage': self.usage
        }

class ChatCompletionChunk(BaseModel):
    def __init__(self, data: Dict[str, Any]):
        self.id = data.get('id')
        self.object = data.get('object')
        self.created = data.get('created')
        self.model = data.get('model')
        self.choices = [ChatCompletionChunkChoice(choice) for choice in data.get('choices', [])]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'object': self.object,
            'created': self.created,
            'model': self.model,
            'choices': [choice.to_dict() for choice in self.choices]
        }

class ChatCompletionChunkChoice(BaseModel):
    def __init__(self, data: Dict[str, Any]):
        self.index = data.get('index')
        self.delta = ChatMessage(data.get('delta', {}))
        self.finish_reason = data.get('finish_reason')
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'index': self.index,
            'delta': self.delta.to_dict(),
            'finish_reason': self.finish_reason
        }

class ImageGenerationResponse(BaseModel):
    def __init__(self, data: Dict[str, Any]):
        self.created = data.get('created')
        self.data = [ImageData(item) for item in data.get('data', [])]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'created': self.created,
            'data': [item.to_dict() for item in self.data]
        }

class ImageData(BaseModel):
    def __init__(self, data: Dict[str, Any]):
        self.url = data.get('url')
        self.b64_json = data.get('b64_json')
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'url': self.url,
            'b64_json': self.b64_json
        }

class ChatCompletions:
    def __init__(self, api_handler: BaseAPIHandler):
        self.api_handler = api_handler

    def create(
        self,
        messages: List[Dict[str, Any]],
        model: str = "gpt-4o-mini-2024-07-18",
        temperature: float = 0.7,
        top_p: float = 1.0,
        stream: bool = False,
        presence_penalty: float = 0,
        frequency_penalty: float = 0,
        modalities: List[str] = None,
        audio: Dict[str, str] = None,
        tools: List[Dict[str, Any]] = None,
        tool_choice: str = None,
        **kwargs
    ) -> Union[ChatCompletionResponse, Iterator[ChatCompletionChunk]]:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            **kwargs
        }

        if modalities:
            payload["modalities"] = modalities
        if audio:
            payload["audio"] = audio
        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice

        if stream:
            response = self.api_handler._make_request(
                'POST',
                'chat/completions',
                json=payload,
                stream=True
            )
            return self._handle_streaming_response(response)
        else:
            response = self.api_handler._make_request(
                'POST',
                'chat/completions',
                json=payload
            )
            return ChatCompletionResponse(response.json())

    def _handle_streaming_response(self, response: requests.Response) -> Iterator[ChatCompletionChunk]:
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8').strip()
                if line_str == "[DONE]":
                    break
                try:
                    if line_str.startswith('data: '):
                        line_str = line_str[len('data: '):]
                    data = json.loads(line_str)
                    yield ChatCompletionChunk(data)
                except Exception as e:
                    continue

class Audio:
    def __init__(self, api_handler: BaseAPIHandler):
        self.api_handler = api_handler

    def create(
        self,
        input_text: str,
        model: str = "tts-1-hd-1106",
        voice: str = "nova",
        **kwargs
    ) -> bytes:
        payload = {
            "model": model,
            "voice": voice,
            "input": input_text,
            **kwargs
        }

        response = self.api_handler._make_request(
            'POST',
            'audio/speech',
            json=payload,
            stream=True
        )

        return b''.join(chunk for chunk in response.iter_content(chunk_size=8192) if chunk)

class Image:
    def __init__(self, api_handler: BaseAPIHandler):
        self.api_handler = api_handler

    def create(
        self,
        prompt: str,
        model: str = "dall-e-3",
        n: int = 1,
        size: str = "1024x1024",
        response_format: str = "url",
        quality: str = "hd",
        **kwargs
    ) -> ImageGenerationResponse:
        payload = {
            "model": model,
            "prompt": prompt,
            "n": n,
            "size": size,
            "response_format": response_format,
            "quality": quality,
            **kwargs
        }

        response = self.api_handler._make_request(
            'POST',
            'images/generations',
            json=payload
        )
        return ImageGenerationResponse(json.loads(response.json()))

class OpenAIUnofficial:
    def __init__(self, base_url: str = "https://devsdocode-openai.hf.space"):
        self.config = APIConfig(base_url.rstrip('/'))
        self.api_handler = BaseAPIHandler(self.config)
        self.chat = Chat(self.api_handler)
        self.audio = Audio(self.api_handler)
        self.image = Image(self.api_handler)

    def list_models(self) -> Dict[str, Any]:
        response = self.api_handler._make_request('GET', 'models')
        return response.json()

    def get_api_info(self) -> Dict[str, Any]:
        response = self.api_handler._make_request('GET', 'about')
        return response.json()

class Chat:
    def __init__(self, api_handler: BaseAPIHandler):
        self.completions = ChatCompletions(api_handler)