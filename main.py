import json
import os
from datetime import datetime, timezone
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

from models import (
    OllamaChatRequest,
    OllamaChatResponse,
    OllamaEmbeddingsRequest,
    OllamaGenerateRequest,
    OllamaGenerateResponse,
    OllamaMessage,
    OllamaShowRequest,
    OpenAIChatCompletionRequest,
)
from openai_compat import (
    build_chat_completion_id,
    build_image_block,
    build_openai_chat_completion,
    build_openai_stream_chunk,
    build_openai_system_prompt,
    build_openai_tool_call_delta,
    build_openai_tool_calls,
    convert_openai_messages_to_anthropic,
    convert_openai_tool_choice,
    convert_openai_tools_to_anthropic,
    get_non_system_openai_messages,
    map_anthropic_finish_reason,
    normalize_role,
    response_text,
)

load_dotenv(Path(__file__).parent / ".env")

app = FastAPI(title="Ollama-MiniMax Proxy")

client = anthropic.Anthropic(
    base_url=os.getenv("ANTHROPIC_BASE_URL", "https://api.minimax.io/anthropic"),
    api_key=os.getenv("ANTHROPIC_AUTH_TOKEN"),
    timeout=int(os.getenv("API_TIMEOUT_MS", "3000000")) / 1000,
)

OLLAMA_VERSION = "0.6.4"
UPSTREAM_MODEL_NAME = os.getenv("ANTHROPIC_MODEL", "MiniMax-M2.7")
MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", UPSTREAM_MODEL_NAME)
MODEL_ARCHITECTURE = os.getenv("OLLAMA_MODEL_ARCHITECTURE", "minimax")
MODEL_BASENAME = os.getenv("OLLAMA_MODEL_BASENAME", MODEL_NAME)
MODEL_CONTEXT_LENGTH = int(os.getenv("OLLAMA_MODEL_CONTEXT_LENGTH", "204800"))
MODEL_SIZE = int(os.getenv("OLLAMA_MODEL_SIZE", "4000000000"))
MODEL_FAMILY = os.getenv("OLLAMA_MODEL_FAMILY", MODEL_ARCHITECTURE)
MODEL_FORMAT = os.getenv("OLLAMA_MODEL_FORMAT", "proxy")
MODEL_PARAMETER_SIZE = os.getenv("OLLAMA_MODEL_PARAMETER_SIZE", "unknown")
MODIFIED_AT = os.getenv("OLLAMA_MODIFIED_AT", "2026-01-01T00:00:00Z")
DEFAULT_SYSTEM_PROMPT = os.getenv(
    "DEFAULT_SYSTEM_PROMPT", "You are a helpful assistant."
)


def parse_capabilities(raw: str) -> list[str]:
    capabilities = [item.strip() for item in raw.split(",") if item.strip()]
    return capabilities or ["completion"]


MODEL_CAPABILITIES = parse_capabilities(
    os.getenv("OLLAMA_MODEL_CAPABILITIES", "completion,tools,vision")
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_now_timestamp() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def convert_ollama_messages_to_anthropic(
    ollama_messages: list[OllamaMessage],
) -> list[dict]:
    messages = []
    for msg in ollama_messages:
        content: list[dict] = [{"type": "text", "text": msg.content}]
        if msg.images:
            for img in msg.images:
                content.append(build_image_block(img))
        messages.append({"role": normalize_role(msg.role), "content": content})
    return messages


def build_system_prompt(ollama_messages: list[OllamaMessage]) -> str:
    system_msg = next((m for m in ollama_messages if m.role == "system"), None)
    return system_msg.content if system_msg else ""


def get_non_system_messages(ollama_messages: list[OllamaMessage]) -> list[OllamaMessage]:
    return [m for m in ollama_messages if m.role != "system"]


def create_message_response(
    system: str,
    messages: list[dict],
    *,
    max_tokens: int = 4096,
    stream: bool = False,
    tools: list[dict] | None = None,
    tool_choice: dict | None = None,
):
    payload = {
        "model": UPSTREAM_MODEL_NAME,
        "max_tokens": max_tokens,
        "system": system or DEFAULT_SYSTEM_PROMPT,
        "messages": messages,
    }
    if tools:
        payload["tools"] = tools
    if tool_choice:
        payload["tool_choice"] = tool_choice
    if stream:
        payload["stream"] = True
    return client.messages.create(**payload)


def build_ollama_listing(model_name: str) -> dict:
    return {
        "name": model_name,
        "model": model_name,
        "size": MODEL_SIZE,
        "modified_at": MODIFIED_AT,
    }


def build_model_info() -> dict:
    return {
        "general.architecture": MODEL_ARCHITECTURE,
        "general.basename": MODEL_BASENAME,
        f"{MODEL_ARCHITECTURE}.context_length": MODEL_CONTEXT_LENGTH,
    }


def build_ollama_show_response(model_name: str) -> dict:
    return {
        **build_ollama_listing(model_name),
        "remote_model": UPSTREAM_MODEL_NAME,
        "parameters": f"num_ctx {MODEL_CONTEXT_LENGTH}",
        "template": "{{ .Prompt }}",
        "details": {
            "parent_model": "",
            "format": MODEL_FORMAT,
            "family": MODEL_FAMILY,
            "families": [MODEL_FAMILY],
            "parameter_size": MODEL_PARAMETER_SIZE,
            "quantization_level": "proxy",
        },
        "model_info": build_model_info(),
        "capabilities": MODEL_CAPABILITIES,
    }


def build_openai_model_listing() -> dict:
    return {
        "id": MODEL_NAME,
        "object": "model",
        "created": 0,
        "owned_by": "ollama-proxy",
    }


def sse_bytes(payload: dict | str) -> bytes:
    data = payload if isinstance(payload, str) else json.dumps(payload, separators=(",", ":"))
    return f"data: {data}\n\n".encode("utf-8")


@app.get("/api/version")
async def version():
    return {"version": OLLAMA_VERSION}


@app.get("/api/models")
async def list_models():
    return {"models": [build_ollama_listing(MODEL_NAME)]}


@app.get("/api/tags")
async def list_tags():
    return {"models": [build_ollama_listing(MODEL_NAME)]}


@app.post("/api/embeddings")
async def embeddings(req: OllamaEmbeddingsRequest):
    return {"embeddings": [[0.0] * 1024 for _ in req.input]}


@app.post("/api/show")
async def show(req: OllamaShowRequest):
    return build_ollama_show_response(req.model)


@app.get("/api/ps")
async def ps():
    return {"models": []}


@app.get("/v1/models")
async def list_openai_models():
    return {"object": "list", "data": [build_openai_model_listing()]}


@app.post("/api/chat")
async def chat(req: OllamaChatRequest):
    system = build_system_prompt(req.messages)
    messages = convert_ollama_messages_to_anthropic(
        get_non_system_messages(req.messages)
    )

    if req.stream:
        return StreamingResponse(
            stream_chat_response(system, messages, req.model),
            media_type="application/x-ndjson",
        )
    else:
        return await non_streaming_chat(system, messages, req.model)


async def stream_chat_response(system: str, messages: list[dict], model: str):
    response = create_message_response(system, messages, stream=True)

    full_text = ""
    for chunk in response:
        if chunk.type == "content_block_delta":
            if hasattr(chunk.delta, "text") and chunk.delta.text:
                full_text += chunk.delta.text
                yield JSONResponse(
                    content=OllamaChatResponse(
                        model=model,
                        created_at=utc_now_iso(),
                        message={"role": "assistant", "content": chunk.delta.text},
                        done=False,
                    ).model_dump()
                ).body + b"\n"

    yield JSONResponse(
        content=OllamaChatResponse(
            model=model,
            created_at=utc_now_iso(),
            message={"role": "assistant", "content": ""},
            done=True,
        ).model_dump()
    ).body + b"\n"


async def non_streaming_chat(system: str, messages: list[dict], model: str):
    response = create_message_response(system, messages)
    text = response_text(response)

    return OllamaChatResponse(
        model=model,
        created_at=utc_now_iso(),
        message={"role": "assistant", "content": text},
        done=True,
    )


@app.post("/api/generate")
async def generate(req: OllamaGenerateRequest):
    messages = [OllamaMessage(role="user", content=req.prompt)]

    if req.stream:
        return StreamingResponse(
            stream_generate_response(messages, req.model),
            media_type="application/x-ndjson",
        )
    else:
        return await non_streaming_generate(messages, req.model)


async def stream_generate_response(messages: list[OllamaMessage], model: str):
    anthropic_messages = convert_ollama_messages_to_anthropic(messages)
    response = create_message_response("", anthropic_messages, stream=True)

    for chunk in response:
        if chunk.type == "content_block_delta":
            if hasattr(chunk.delta, "text") and chunk.delta.text:
                yield JSONResponse(
                    content=OllamaGenerateResponse(
                        model=model,
                        created_at=utc_now_iso(),
                        response=chunk.delta.text,
                        done=False,
                    ).model_dump()
                ).body + b"\n"

    yield JSONResponse(
        content=OllamaGenerateResponse(
            model=model,
            created_at=utc_now_iso(),
            response="",
            done=True,
        ).model_dump()
    ).body + b"\n"


async def non_streaming_generate(messages: list[OllamaMessage], model: str):
    anthropic_messages = convert_ollama_messages_to_anthropic(messages)
    response = create_message_response("", anthropic_messages)
    text = response_text(response)

    return OllamaGenerateResponse(
        model=model,
        created_at=utc_now_iso(),
        response=text,
        done=True,
    )


@app.post("/v1/chat/completions")
async def openai_chat_completions(req: OpenAIChatCompletionRequest):
    system = build_openai_system_prompt(req.messages)
    messages = convert_openai_messages_to_anthropic(
        get_non_system_openai_messages(req.messages)
    )
    tools = convert_openai_tools_to_anthropic(req.tools, req.tool_choice)
    tool_choice = convert_openai_tool_choice(req.tool_choice)

    if req.stream:
        return StreamingResponse(
            stream_openai_chat_completions(
                system,
                messages,
                req.model,
                req.max_tokens,
                tools,
                tool_choice,
            ),
            media_type="text/event-stream",
        )

    return await non_streaming_openai_chat_completions(
        system,
        messages,
        req.model,
        req.max_tokens,
        tools,
        tool_choice,
    )


async def stream_openai_chat_completions(
    system: str,
    messages: list[dict],
    model: str,
    max_tokens: int | None,
    tools: list[dict] | None,
    tool_choice: dict | None,
):
    response = create_message_response(
        system,
        messages,
        max_tokens=max_tokens or 4096,
        stream=True,
        tools=tools,
        tool_choice=tool_choice,
    )
    completion_id = build_chat_completion_id()
    created = utc_now_timestamp()
    sent_role = False
    saw_tool_calls = False

    for chunk in response:
        if chunk.type == "content_block_start":
            block = getattr(chunk, "content_block", None)
            if getattr(block, "type", None) != "tool_use":
                continue
            saw_tool_calls = True
            yield sse_bytes(
                build_openai_stream_chunk(
                    model,
                    completion_id,
                    created,
                    include_role=not sent_role,
                    tool_calls=[
                        build_openai_tool_call_delta(
                            getattr(chunk, "index", 0),
                            tool_id=block.id,
                            name=block.name,
                        )
                    ],
                )
            )
            sent_role = True
            continue

        if chunk.type != "content_block_delta":
            continue
        delta_type = getattr(chunk.delta, "type", None)
        if delta_type == "text_delta" and getattr(chunk.delta, "text", None):
            yield sse_bytes(
                build_openai_stream_chunk(
                    model,
                    completion_id,
                    created,
                    text=chunk.delta.text,
                    include_role=not sent_role,
                )
            )
            sent_role = True
            continue

        if delta_type != "input_json_delta":
            continue
        if not getattr(chunk.delta, "partial_json", None):
            continue

        saw_tool_calls = True
        yield sse_bytes(
            build_openai_stream_chunk(
                model,
                completion_id,
                created,
                include_role=not sent_role,
                tool_calls=[
                    build_openai_tool_call_delta(
                        getattr(chunk, "index", 0),
                        arguments=chunk.delta.partial_json,
                    )
                ],
            )
        )
        sent_role = True

    yield sse_bytes(
        build_openai_stream_chunk(
            model,
            completion_id,
            created,
            finish_reason="tool_calls" if saw_tool_calls else "stop",
        )
    )
    yield sse_bytes("[DONE]")


async def non_streaming_openai_chat_completions(
    system: str,
    messages: list[dict],
    model: str,
    max_tokens: int | None,
    tools: list[dict] | None,
    tool_choice: dict | None,
):
    response = create_message_response(
        system,
        messages,
        max_tokens=max_tokens or 4096,
        tools=tools,
        tool_choice=tool_choice,
    )
    completion_id = build_chat_completion_id()
    created = utc_now_timestamp()
    text = response_text(response)
    tool_calls = build_openai_tool_calls(response)
    finish_reason = map_anthropic_finish_reason(
        getattr(response, "stop_reason", None),
        tool_calls,
    )
    return build_openai_chat_completion(
        model,
        completion_id,
        created,
        text=text,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
    )
