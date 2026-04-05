from typing import Any, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class OllamaMessage(BaseModel):
    role: str
    content: str
    images: Optional[list[str]] = None


class OllamaChatRequest(BaseModel):
    model: str
    messages: list[OllamaMessage]
    stream: bool = True
    options: Optional[dict] = None


class OllamaGenerateRequest(BaseModel):
    model: str
    prompt: str
    stream: bool = True
    options: Optional[dict] = None


class OllamaChatResponse(BaseModel):
    model: str
    created_at: str
    message: dict
    done: bool


class OllamaGenerateResponse(BaseModel):
    model: str
    created_at: str
    response: str
    done: bool


class OllamaEmbeddingsRequest(BaseModel):
    model: str
    input: list[str]
    truncate: bool = True


class OllamaShowRequest(BaseModel):
    model: str = Field(validation_alias=AliasChoices("model", "name"))


class OpenAIImageUrl(BaseModel):
    url: str


class OpenAIContentPart(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str
    text: Optional[str] = None
    image_url: Optional[OpenAIImageUrl] = None


class OpenAIFunctionTool(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    description: Optional[str] = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class OpenAITool(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str
    function: OpenAIFunctionTool


class OpenAIFunctionCall(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    arguments: str


class OpenAIToolCall(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    type: str
    function: OpenAIFunctionCall


class OpenAIChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: str | list[OpenAIContentPart] | None = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[list[OpenAIToolCall]] = None


class OpenAIChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[OpenAIChatMessage]
    stream: bool = False
    max_tokens: Optional[int] = None
    tools: Optional[list[OpenAITool]] = None
    tool_choice: Optional[str | dict[str, Any]] = None
