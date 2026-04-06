import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

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
    extract_response_thinking_blocks,
    format_assistant_content,
    get_non_system_openai_messages,
    map_anthropic_finish_reason,
    normalize_thinking_block,
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
HIGH_MODEL_NAME = os.getenv("OLLAMA_HIGH_MODEL_NAME", f"{MODEL_NAME} High")
MODEL_ARCHITECTURE = os.getenv("OLLAMA_MODEL_ARCHITECTURE", "minimax")
MODEL_BASENAME = os.getenv("OLLAMA_MODEL_BASENAME", MODEL_NAME)
HIGH_MODEL_BASENAME = os.getenv("OLLAMA_HIGH_MODEL_BASENAME", HIGH_MODEL_NAME)
MODEL_CONTEXT_LENGTH = int(os.getenv("OLLAMA_MODEL_CONTEXT_LENGTH", "204800"))
MODEL_SIZE = int(os.getenv("OLLAMA_MODEL_SIZE", "4000000000"))
MODEL_FAMILY = os.getenv("OLLAMA_MODEL_FAMILY", MODEL_ARCHITECTURE)
MODEL_FORMAT = os.getenv("OLLAMA_MODEL_FORMAT", "proxy")
MODEL_PARAMETER_SIZE = os.getenv("OLLAMA_MODEL_PARAMETER_SIZE", "unknown")
MODIFIED_AT = os.getenv("OLLAMA_MODIFIED_AT", "2026-01-01T00:00:00Z")
MODEL_THINKING_TYPE = os.getenv("OLLAMA_DEFAULT_THINKING_TYPE", "enabled").strip() or "enabled"
MODEL_THINKING_DISPLAY = os.getenv("ANTHROPIC_THINKING_DISPLAY", "").strip()
MODEL_THINKING_BUDGET = int(os.getenv("OLLAMA_MODEL_THINKING_BUDGET_TOKENS", "8192"))
HIGH_MODEL_THINKING_BUDGET = int(
    os.getenv("OLLAMA_HIGH_MODEL_THINKING_BUDGET_TOKENS", "24576")
)
DEFAULT_SYSTEM_PROMPT = os.getenv(
    "DEFAULT_SYSTEM_PROMPT", "You are a helpful assistant."
)


def build_env_default_thinking_config() -> dict | None:
    thinking_type = os.getenv("ANTHROPIC_THINKING_TYPE", "").strip()
    if not thinking_type:
        return None

    config: dict[str, str | int] = {"type": thinking_type}
    budget_tokens = os.getenv("ANTHROPIC_THINKING_BUDGET_TOKENS", "").strip()
    if budget_tokens:
        config["budget_tokens"] = int(budget_tokens)

    display = os.getenv("ANTHROPIC_THINKING_DISPLAY", "").strip()
    if display:
        config["display"] = display

    return config


def build_model_default_thinking_config(budget_tokens: int) -> dict:
    config: dict[str, str | int] = {
        "type": MODEL_THINKING_TYPE,
        "budget_tokens": budget_tokens,
    }
    if MODEL_THINKING_DISPLAY:
        config["display"] = MODEL_THINKING_DISPLAY
    return config


def build_proxy_models() -> list[dict[str, object]]:
    models = [
        {
            "name": MODEL_NAME,
            "basename": MODEL_BASENAME,
            "remote_model": UPSTREAM_MODEL_NAME,
            "thinking": build_model_default_thinking_config(MODEL_THINKING_BUDGET),
        },
        {
            "name": HIGH_MODEL_NAME,
            "basename": HIGH_MODEL_BASENAME,
            "remote_model": UPSTREAM_MODEL_NAME,
            "thinking": build_model_default_thinking_config(HIGH_MODEL_THINKING_BUDGET),
        },
    ]
    return list({model["name"]: model for model in models}.values())


ENV_DEFAULT_THINKING_CONFIG = build_env_default_thinking_config()
PROXY_MODELS = build_proxy_models()
PROXY_MODEL_INDEX = {model["name"]: model for model in PROXY_MODELS}


def parse_capabilities(raw: str) -> list[str]:
    capabilities = [item.strip() for item in raw.split(",") if item.strip()]
    return capabilities or ["completion"]


MODEL_CAPABILITIES = parse_capabilities(
    os.getenv("OLLAMA_MODEL_CAPABILITIES", "completion,tools,vision")
)
MAX_EMBEDDINGS_BATCH = int(os.getenv("MAX_EMBEDDINGS_BATCH", "256"))


def resolve_model(model_name: str) -> dict[str, object]:
    model = PROXY_MODEL_INDEX.get(model_name)
    if model is not None:
        return model

    default_model = PROXY_MODEL_INDEX[MODEL_NAME]
    return {
        **default_model,
        "name": model_name,
        "thinking": dict(default_model["thinking"]),
    }


def resolve_thinking_config(model_name: str, thinking: dict | None) -> dict | None:
    if thinking is not None:
        return thinking

    model = PROXY_MODEL_INDEX.get(model_name)
    if model is not None:
        return dict(model["thinking"])

    return ENV_DEFAULT_THINKING_CONFIG


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_now_timestamp() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def convert_ollama_messages_to_anthropic(
    ollama_messages: list[OllamaMessage],
) -> list[dict]:
    messages = []
    for msg in ollama_messages:
        content: list[dict] = []
        if msg.role == "assistant":
            for block in msg.thinking_blocks or []:
                normalized_block = normalize_thinking_block(block)
                if normalized_block is not None:
                    content.append(normalized_block)

        if msg.content or not content:
            content.append({"type": "text", "text": msg.content})

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
    model_name: str,
    system: str,
    messages: list[dict],
    *,
    max_tokens: int = 4096,
    stream: bool = False,
    tools: list[dict] | None = None,
    tool_choice: dict | None = None,
    thinking: dict | None = None,
):
    model = resolve_model(model_name)
    payload = {
        "model": model["remote_model"],
        "max_tokens": max_tokens,
        "system": system or DEFAULT_SYSTEM_PROMPT,
        "messages": messages,
    }
    if tools:
        payload["tools"] = tools
    if tool_choice:
        payload["tool_choice"] = tool_choice
    if thinking:
        payload["thinking"] = thinking
    if stream:
        payload["stream"] = True
    return client.messages.create(**payload)


def build_ollama_message(text: str | None, thinking_blocks: list[dict[str, str]]) -> dict:
    message = {"role": "assistant", "content": text or ""}
    if thinking_blocks:
        message["thinking_blocks"] = thinking_blocks
    return message


def build_ollama_listing(model_name: str) -> dict:
    return {
        "name": model_name,
        "model": model_name,
        "size": MODEL_SIZE,
        "modified_at": MODIFIED_AT,
    }


def build_model_info(model_basename: str) -> dict:
    return {
        "general.architecture": MODEL_ARCHITECTURE,
        "general.basename": model_basename,
        f"{MODEL_ARCHITECTURE}.context_length": MODEL_CONTEXT_LENGTH,
    }


def build_ollama_show_response(model_name: str) -> dict:
    model = resolve_model(model_name)
    return {
        **build_ollama_listing(model_name),
        "remote_model": model["remote_model"],
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
        "model_info": build_model_info(model["basename"]),
        "capabilities": MODEL_CAPABILITIES,
    }


def build_openai_model_listing(model_name: str) -> dict:
    return {
        "id": model_name,
        "object": "model",
        "created": 0,
        "owned_by": "ollama-proxy",
    }


def sse_bytes(payload: dict | str) -> bytes:
    data = payload if isinstance(payload, str) else json.dumps(payload, separators=(",", ":"))
    return f"data: {data}\n\n".encode("utf-8")


def ndjson_bytes(payload: dict) -> bytes:
    return (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")


def normalize_upstream_status(exc: Exception) -> int:
    error_name = exc.__class__.__name__
    if error_name in {"AuthenticationError", "PermissionDeniedError"}:
        return 401
    if error_name == "RateLimitError":
        return 429
    if error_name in {"APITimeoutError", "TimeoutError"}:
        return 504
    return 502


def build_error_payload(message: str, error_type: str, code: str) -> dict:
    return {"error": {"message": message, "type": error_type, "code": code}}


def upstream_error_response(exc: Exception) -> JSONResponse:
    message = str(exc) or "Upstream request failed"
    return JSONResponse(
        status_code=normalize_upstream_status(exc),
        content=build_error_payload(
            message=message,
            error_type="upstream_error",
            code=exc.__class__.__name__,
        ),
    )


def validation_error_response(message: str) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content=build_error_payload(
            message=message,
            error_type="validation_error",
            code="invalid_input",
        ),
    )


async def create_message_response_async(
    model_name: str,
    system: str,
    messages: list[dict],
    *,
    max_tokens: int = 4096,
    stream: bool = False,
    tools: list[dict] | None = None,
    tool_choice: dict | None = None,
    thinking: dict | None = None,
):
    return await run_in_threadpool(
        create_message_response,
        model_name,
        system,
        messages,
        max_tokens=max_tokens,
        stream=stream,
        tools=tools,
        tool_choice=tool_choice,
        thinking=thinking,
    )


async def iter_sync_chunks(response: Iterable):
    iterator = iter(response)
    sentinel = object()
    while True:
        chunk = await run_in_threadpool(next, iterator, sentinel)
        if chunk is sentinel:
            break
        yield chunk


@app.get("/api/version")
async def version():
    return {"version": OLLAMA_VERSION}


@app.get("/api/models")
async def list_models():
    return {"models": [build_ollama_listing(model["name"]) for model in PROXY_MODELS]}


@app.get("/api/tags")
async def list_tags():
    return {"models": [build_ollama_listing(model["name"]) for model in PROXY_MODELS]}


@app.post("/api/embeddings")
async def embeddings(req: OllamaEmbeddingsRequest):
    if len(req.input) > MAX_EMBEDDINGS_BATCH:
        return validation_error_response(
            f"input batch too large: max {MAX_EMBEDDINGS_BATCH} items"
        )
    return {"embeddings": [[0.0] * 1024 for _ in req.input]}


@app.post("/api/show")
async def show(req: OllamaShowRequest):
    return build_ollama_show_response(req.model)


@app.get("/api/ps")
async def ps():
    return {"models": []}


@app.get("/v1/models")
async def list_openai_models():
    return {
        "object": "list",
        "data": [build_openai_model_listing(model["name"]) for model in PROXY_MODELS],
    }


@app.post("/api/chat")
async def chat(req: OllamaChatRequest):
    system = build_system_prompt(req.messages)
    messages = convert_ollama_messages_to_anthropic(
        get_non_system_messages(req.messages)
    )
    thinking = resolve_thinking_config(req.model, req.thinking)

    try:
        if req.stream:
            response = await create_message_response_async(
                req.model,
                system,
                messages,
                stream=True,
                thinking=thinking,
            )
            return StreamingResponse(
                stream_chat_response(response, req.model),
                media_type="application/x-ndjson",
            )
        return await non_streaming_chat(system, messages, req.model, thinking)
    except Exception as exc:
        return upstream_error_response(exc)


async def stream_chat_response(
    response,
    model: str,
):
    showing_thinking = False
    try:
        async for chunk in iter_sync_chunks(response):
            if chunk.type != "content_block_delta":
                continue

            delta_type = getattr(chunk.delta, "type", None)
            if delta_type == "thinking_delta" and getattr(chunk.delta, "thinking", None):
                text = chunk.delta.thinking
                if not showing_thinking:
                    text = f"<think>\n{text}"
                    showing_thinking = True
                yield ndjson_bytes(
                    OllamaChatResponse(
                        model=model,
                        created_at=utc_now_iso(),
                        message={"role": "assistant", "content": text},
                        done=False,
                    ).model_dump()
                )
                continue

            if delta_type == "signature_delta":
                if showing_thinking:
                    yield ndjson_bytes(
                        OllamaChatResponse(
                            model=model,
                            created_at=utc_now_iso(),
                            message={"role": "assistant", "content": "\n</think>\n\n"},
                            done=False,
                        ).model_dump()
                    )
                    showing_thinking = False
                continue

            if delta_type == "text_delta" and getattr(chunk.delta, "text", None):
                if showing_thinking:
                    yield ndjson_bytes(
                        OllamaChatResponse(
                            model=model,
                            created_at=utc_now_iso(),
                            message={"role": "assistant", "content": "\n</think>\n\n"},
                            done=False,
                        ).model_dump()
                    )
                    showing_thinking = False

                yield ndjson_bytes(
                    OllamaChatResponse(
                        model=model,
                        created_at=utc_now_iso(),
                        message={"role": "assistant", "content": chunk.delta.text},
                        done=False,
                    ).model_dump()
                )

        if showing_thinking:
            yield ndjson_bytes(
                OllamaChatResponse(
                    model=model,
                    created_at=utc_now_iso(),
                    message={"role": "assistant", "content": "\n</think>\n\n"},
                    done=False,
                ).model_dump()
            )
    except Exception as exc:
        yield ndjson_bytes(build_error_payload(str(exc) or "Upstream request failed", "upstream_error", exc.__class__.__name__))
        return

    yield ndjson_bytes(
        OllamaChatResponse(
            model=model,
            created_at=utc_now_iso(),
            message={"role": "assistant", "content": ""},
            done=True,
        ).model_dump()
    )


async def non_streaming_chat(
    system: str,
    messages: list[dict],
    model: str,
    thinking: dict | None,
):
    response = await create_message_response_async(
        model,
        system,
        messages,
        thinking=thinking,
    )
    thinking_blocks = extract_response_thinking_blocks(response)
    text = format_assistant_content(response_text(response), thinking_blocks)

    return OllamaChatResponse(
        model=model,
        created_at=utc_now_iso(),
        message=build_ollama_message(text, thinking_blocks),
        done=True,
    )


@app.post("/api/generate")
async def generate(req: OllamaGenerateRequest):
    messages = [OllamaMessage(role="user", content=req.prompt)]
    thinking = resolve_thinking_config(req.model, req.thinking)

    try:
        if req.stream:
            anthropic_messages = convert_ollama_messages_to_anthropic(messages)
            response = await create_message_response_async(
                req.model,
                "",
                anthropic_messages,
                stream=True,
                thinking=thinking,
            )
            return StreamingResponse(
                stream_generate_response(response, req.model),
                media_type="application/x-ndjson",
            )
        return await non_streaming_generate(messages, req.model, thinking)
    except Exception as exc:
        return upstream_error_response(exc)


async def stream_generate_response(
    response,
    model: str,
):
    showing_thinking = False

    try:
        async for chunk in iter_sync_chunks(response):
            if chunk.type != "content_block_delta":
                continue

            delta_type = getattr(chunk.delta, "type", None)
            if delta_type == "thinking_delta" and getattr(chunk.delta, "thinking", None):
                text = chunk.delta.thinking
                if not showing_thinking:
                    text = f"<think>\n{text}"
                    showing_thinking = True
                yield ndjson_bytes(
                    OllamaGenerateResponse(
                        model=model,
                        created_at=utc_now_iso(),
                        response=text,
                        done=False,
                    ).model_dump()
                )
                continue

            if delta_type == "signature_delta":
                if showing_thinking:
                    yield ndjson_bytes(
                        OllamaGenerateResponse(
                            model=model,
                            created_at=utc_now_iso(),
                            response="\n</think>\n\n",
                            done=False,
                        ).model_dump()
                    )
                    showing_thinking = False
                continue

            if delta_type == "text_delta" and getattr(chunk.delta, "text", None):
                if showing_thinking:
                    yield ndjson_bytes(
                        OllamaGenerateResponse(
                            model=model,
                            created_at=utc_now_iso(),
                            response="\n</think>\n\n",
                            done=False,
                        ).model_dump()
                    )
                    showing_thinking = False

                yield ndjson_bytes(
                    OllamaGenerateResponse(
                        model=model,
                        created_at=utc_now_iso(),
                        response=chunk.delta.text,
                        done=False,
                    ).model_dump()
                )

        if showing_thinking:
            yield ndjson_bytes(
                OllamaGenerateResponse(
                    model=model,
                    created_at=utc_now_iso(),
                    response="\n</think>\n\n",
                    done=False,
                ).model_dump()
            )
    except Exception as exc:
        yield ndjson_bytes(build_error_payload(str(exc) or "Upstream request failed", "upstream_error", exc.__class__.__name__))
        return

    yield ndjson_bytes(
        OllamaGenerateResponse(
            model=model,
            created_at=utc_now_iso(),
            response="",
            done=True,
        ).model_dump()
    )


async def non_streaming_generate(
    messages: list[OllamaMessage],
    model: str,
    thinking: dict | None,
):
    anthropic_messages = convert_ollama_messages_to_anthropic(messages)
    response = await create_message_response_async(
        model,
        "",
        anthropic_messages,
        thinking=thinking,
    )
    thinking_blocks = extract_response_thinking_blocks(response)
    text = format_assistant_content(response_text(response), thinking_blocks)

    return OllamaGenerateResponse(
        model=model,
        created_at=utc_now_iso(),
        response=text or "",
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
    thinking = resolve_thinking_config(req.model, req.thinking)

    try:
        if req.stream:
            response = await create_message_response_async(
                req.model,
                system,
                messages,
                max_tokens=req.max_tokens or 4096,
                stream=True,
                tools=tools,
                tool_choice=tool_choice,
                thinking=thinking,
            )
            return StreamingResponse(
                stream_openai_chat_completions(response, req.model),
                media_type="text/event-stream",
            )

        return await non_streaming_openai_chat_completions(
            system,
            messages,
            req.model,
            req.max_tokens,
            tools,
            tool_choice,
            thinking,
        )
    except Exception as exc:
        return upstream_error_response(exc)


async def stream_openai_chat_completions(
    response,
    model: str,
):
    completion_id = build_chat_completion_id()
    created = utc_now_timestamp()
    sent_role = False
    saw_tool_calls = False

    try:
        async for chunk in iter_sync_chunks(response):
            if chunk.type == "content_block_start":
                block = getattr(chunk, "content_block", None)
                if getattr(block, "type", None) in ("thinking", None):
                    continue
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
            if delta_type in ("thinking_delta", "signature_delta"):
                continue

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
    except Exception as exc:
        yield sse_bytes(build_error_payload(str(exc) or "Upstream request failed", "upstream_error", exc.__class__.__name__))
        yield sse_bytes("[DONE]")
        return

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
    thinking: dict | None,
):
    response = await create_message_response_async(
        model,
        system,
        messages,
        max_tokens=max_tokens or 4096,
        tools=tools,
        tool_choice=tool_choice,
        thinking=thinking,
    )
    completion_id = build_chat_completion_id()
    created = utc_now_timestamp()
    thinking_blocks = extract_response_thinking_blocks(response)
    text = response_text(response) or None
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
        thinking_blocks=thinking_blocks,
        finish_reason=finish_reason,
    )
