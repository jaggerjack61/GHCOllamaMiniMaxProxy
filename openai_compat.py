import json
from typing import Any
from uuid import uuid4

from models import OpenAIChatMessage, OpenAIContentPart, OpenAITool, OpenAIToolCall


def normalize_role(role: str) -> str:
    return "assistant" if role == "assistant" else "user"


def build_image_block(data: str, media_type: str = "image/jpeg") -> dict:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": data,
        },
    }


def extract_openai_text(content: str | list[OpenAIContentPart] | None) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "\n".join(part.text for part in content if part.type == "text" and part.text)


def parse_data_url(url: str) -> tuple[str, str] | None:
    if not url.startswith("data:") or ";base64," not in url:
        return None

    header, data = url.split(",", 1)
    media_type = header.removeprefix("data:").split(";")[0] or "image/jpeg"
    return media_type, data


def convert_openai_content_to_anthropic(
    content: str | list[OpenAIContentPart] | None,
) -> list[dict]:
    if content is None:
        return [{"type": "text", "text": ""}]
    if isinstance(content, str):
        return [{"type": "text", "text": content}]

    blocks: list[dict] = []
    for part in content:
        if part.type == "text" and part.text:
            blocks.append({"type": "text", "text": part.text})
            continue

        if part.type == "image_url" and part.image_url:
            parsed = parse_data_url(part.image_url.url)
            if parsed:
                media_type, data = parsed
                blocks.append(build_image_block(data, media_type))

    return blocks or [{"type": "text", "text": ""}]


def build_openai_system_prompt(messages: list[OpenAIChatMessage]) -> str:
    prompts = [
        extract_openai_text(message.content)
        for message in messages
        if message.role in {"system", "developer"}
    ]
    return "\n\n".join(prompt for prompt in prompts if prompt)


def get_non_system_openai_messages(
    messages: list[OpenAIChatMessage],
) -> list[OpenAIChatMessage]:
    return [m for m in messages if m.role not in {"system", "developer"}]


def append_anthropic_message(messages: list[dict], role: str, content: list[dict]):
    if not content:
        return
    if messages and messages[-1]["role"] == role:
        messages[-1]["content"].extend(content)
        return
    messages.append({"role": role, "content": content})


def parse_tool_arguments(arguments: str | None) -> dict[str, Any]:
    if not arguments:
        return {}
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return {"raw": arguments}
    if isinstance(parsed, dict):
        return parsed
    return {"value": parsed}


def convert_assistant_tool_calls(tool_calls: list[OpenAIToolCall] | None) -> list[dict]:
    if not tool_calls:
        return []
    return [
        {
            "type": "tool_use",
            "id": tool_call.id,
            "name": tool_call.function.name,
            "input": parse_tool_arguments(tool_call.function.arguments),
        }
        for tool_call in tool_calls
    ]


def normalize_thinking_block(block: dict[str, Any] | None) -> dict[str, str] | None:
    if not isinstance(block, dict) or block.get("type") != "thinking":
        return None
    return {
        "type": "thinking",
        "thinking": block.get("thinking") or "",
        "signature": block.get("signature") or "",
    }


def get_assistant_thinking_blocks(message: OpenAIChatMessage) -> list[dict[str, str]]:
    normalized_blocks = []
    for block in message.thinking_blocks or []:
        normalized_block = normalize_thinking_block(block)
        if normalized_block is not None:
            normalized_blocks.append(normalized_block)
    return normalized_blocks


def extract_response_thinking_blocks(response) -> list[dict[str, str]]:
    blocks = []
    for block in response.content:
        if getattr(block, "type", None) != "thinking":
            continue
        blocks.append(
            {
                "type": "thinking",
                "thinking": getattr(block, "thinking", "") or "",
                "signature": getattr(block, "signature", "") or "",
            }
        )
    return blocks


def format_assistant_content(
    text: str | None,
    thinking_blocks: list[dict[str, str]] | None,
) -> str | None:
    visible_thinking = "\n\n".join(
        block["thinking"].strip()
        for block in thinking_blocks or []
        if block.get("thinking", "").strip()
    )
    if not visible_thinking:
        return text
    if text:
        return f"<think>\n{visible_thinking}\n</think>\n\n{text}"
    return f"<think>\n{visible_thinking}\n</think>"


def convert_openai_message_to_anthropic(message: OpenAIChatMessage) -> tuple[str, list[dict]]:
    if message.role == "tool":
        return "user", [
            {
                "type": "tool_result",
                "tool_use_id": message.tool_call_id or "",
                "content": extract_openai_text(message.content),
            }
        ]

    if message.role == "assistant":
        content: list[dict] = get_assistant_thinking_blocks(message)
        text = extract_openai_text(message.content)
        if text:
            content.append({"type": "text", "text": text})
        content.extend(convert_assistant_tool_calls(message.tool_calls))
        return "assistant", content or [{"type": "text", "text": ""}]

    return normalize_role(message.role), convert_openai_content_to_anthropic(message.content)


def convert_openai_messages_to_anthropic(messages: list[OpenAIChatMessage]) -> list[dict]:
    converted: list[dict] = []
    for message in messages:
        role, content = convert_openai_message_to_anthropic(message)
        append_anthropic_message(converted, role, content)
    return converted


def convert_openai_tools_to_anthropic(
    tools: list[OpenAITool] | None,
    tool_choice: str | dict[str, Any] | None = None,
) -> list[dict] | None:
    if not tools or tool_choice == "none":
        return None

    converted = []
    for tool in tools:
        if tool.type != "function":
            continue
        converted.append(
            {
                "name": tool.function.name,
                "description": tool.function.description or "",
                "input_schema": tool.function.parameters or {"type": "object", "properties": {}},
            }
        )
    return converted or None


def convert_openai_tool_choice(tool_choice: str | dict[str, Any] | None) -> dict | None:
    if tool_choice in {None, "auto", "none"}:
        return None
    if tool_choice == "required":
        return {"type": "any"}
    if not isinstance(tool_choice, dict):
        return None

    function = tool_choice.get("function") or {}
    name = function.get("name")
    if not name:
        return None
    return {"type": "tool", "name": name}


def response_text(response) -> str:
    text = ""
    for block in response.content:
        if block.type == "text":
            text += block.text
    return text


def compact_json(value: dict[str, Any]) -> str:
    return json.dumps(value, separators=(",", ":"))


def convert_tool_use_block_to_openai_tool_call(block) -> dict:
    return {
        "id": block.id,
        "type": "function",
        "function": {
            "name": block.name,
            "arguments": compact_json(block.input),
        },
    }


def build_openai_tool_calls(response) -> list[dict] | None:
    tool_calls = [
        convert_tool_use_block_to_openai_tool_call(block)
        for block in response.content
        if block.type == "tool_use"
    ]
    return tool_calls or None


def map_anthropic_finish_reason(
    stop_reason: str | None,
    tool_calls: list[dict] | None,
) -> str:
    if tool_calls or stop_reason == "tool_use":
        return "tool_calls"
    if stop_reason == "max_tokens":
        return "length"
    return "stop"


def build_chat_completion_id() -> str:
    return f"chatcmpl-{uuid4().hex}"


def build_openai_chat_completion(
    model: str,
    completion_id: str,
    created: int,
    *,
    text: str | None,
    tool_calls: list[dict] | None,
    thinking_blocks: list[dict[str, str]] | None,
    finish_reason: str,
) -> dict:
    message = {"role": "assistant", "content": text if text else None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    if thinking_blocks:
        message["thinking_blocks"] = thinking_blocks

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def build_openai_tool_call_delta(
    index: int,
    *,
    tool_id: str | None = None,
    name: str | None = None,
    arguments: str | None = None,
) -> dict:
    delta = {"index": index}
    if tool_id is not None:
        delta["id"] = tool_id
        delta["type"] = "function"

    function: dict[str, str] = {}
    if name is not None:
        function["name"] = name
    if arguments is not None:
        function["arguments"] = arguments
    if function:
        delta["function"] = function
    return delta


def build_openai_stream_chunk(
    model: str,
    completion_id: str,
    created: int,
    *,
    text: str | None = None,
    include_role: bool = False,
    finish_reason: str | None = None,
    tool_calls: list[dict] | None = None,
) -> dict:
    delta: dict[str, Any] = {}
    if include_role:
        delta["role"] = "assistant"
    if text is not None:
        delta["content"] = text
    if tool_calls:
        delta["tool_calls"] = tool_calls

    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }