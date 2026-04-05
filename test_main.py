import importlib
import os
import unittest

from fastapi.testclient import TestClient


os.environ.setdefault("ANTHROPIC_AUTH_TOKEN", "test-token")
os.environ.setdefault("ANTHROPIC_MODEL", "MiniMax-M2.7")

main = importlib.import_module("main")


class FakeTextBlock:
    type = "text"

    def __init__(self, text: str):
        self.text = text


class FakeToolUseBlock:
    type = "tool_use"

    def __init__(self, tool_id: str, name: str, tool_input: dict):
        self.id = tool_id
        self.name = name
        self.input = tool_input


class FakeTextDelta:
    type = "text_delta"

    def __init__(self, text: str):
        self.text = text


class FakeInputJsonDelta:
    type = "input_json_delta"

    def __init__(self, partial_json: str):
        self.partial_json = partial_json


class FakeStreamChunk:
    type = "content_block_delta"

    def __init__(self, delta, index: int = 0):
        self.delta = delta
        self.index = index


class FakeContentBlockStartChunk:
    type = "content_block_start"

    def __init__(self, content_block, index: int = 0):
        self.content_block = content_block
        self.index = index


class FakeMessageResponse:
    def __init__(self, content, stop_reason: str = "end_turn"):
        self.content = content
        self.stop_reason = stop_reason


class FakeMessagesAPI:
    def __init__(self):
        self.calls: list[dict] = []
        self.response = FakeMessageResponse([FakeTextBlock("Hello world")])
        self.stream_response = [
            FakeStreamChunk(FakeTextDelta("Hello")),
            FakeStreamChunk(FakeTextDelta(" world")),
        ]

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return self.stream_response
        return self.response


class FakeAnthropicClient:
    def __init__(self):
        self.messages = FakeMessagesAPI()


class ProxyCompatibilityTests(unittest.TestCase):
    def setUp(self):
        main.client = FakeAnthropicClient()
        self.client = TestClient(main.app)

    def test_import_does_not_expose_logging_hooks(self):
        reloaded_main = importlib.reload(main)

        self.assertFalse(hasattr(reloaded_main, "file_handler"))
        self.assertFalse(hasattr(reloaded_main, "log_request"))
        self.assertFalse(hasattr(reloaded_main, "log_response"))

    def test_show_accepts_model_field_and_returns_copilot_metadata(self):
        response = self.client.post("/api/show", json={"model": main.MODEL_NAME})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["model"], main.MODEL_NAME)
        self.assertIn("tools", payload["capabilities"])
        self.assertIn("vision", payload["capabilities"])
        self.assertIn("general.architecture", payload["model_info"])
        self.assertEqual(
            payload["model_info"][
                f"{payload['model_info']['general.architecture']}.context_length"
            ],
            204800,
        )

    def test_chat_completions_returns_openai_response_shape(self):
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": main.MODEL_NAME,
                "stream": False,
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Say hello."},
                ],
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["object"], "chat.completion")
        self.assertEqual(payload["choices"][0]["message"]["role"], "assistant")
        self.assertEqual(payload["choices"][0]["message"]["content"], "Hello world")
        self.assertEqual(payload["choices"][0]["finish_reason"], "stop")

    def test_chat_completions_passes_tools_and_returns_tool_calls(self):
        main.client.messages.response = FakeMessageResponse(
            [
                FakeToolUseBlock(
                    "toolu_read_file",
                    "read_file",
                    {"filePath": "main.py"},
                )
            ],
            stop_reason="tool_use",
        )

        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": main.MODEL_NAME,
                "stream": False,
                "messages": [{"role": "user", "content": "Read main.py"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "description": "Read a file",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "filePath": {"type": "string"}
                                },
                                "required": ["filePath"],
                            },
                        },
                    }
                ],
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        tool_call = payload["choices"][0]["message"]["tool_calls"][0]
        self.assertEqual(tool_call["id"], "toolu_read_file")
        self.assertEqual(tool_call["function"]["name"], "read_file")
        self.assertEqual(
            tool_call["function"]["arguments"],
            '{"filePath":"main.py"}',
        )
        self.assertEqual(payload["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(
            main.client.messages.calls[0]["tools"],
            [
                {
                    "name": "read_file",
                    "description": "Read a file",
                    "input_schema": {
                        "type": "object",
                        "properties": {"filePath": {"type": "string"}},
                        "required": ["filePath"],
                    },
                }
            ],
        )

    def test_chat_completions_translates_tool_history_to_anthropic(self):
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": main.MODEL_NAME,
                "stream": False,
                "messages": [
                    {"role": "user", "content": "Read main.py"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_read_file",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"filePath":"main.py"}',
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_read_file",
                        "content": "import json",
                    },
                ],
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            main.client.messages.calls[0]["messages"],
            [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Read main.py"}],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call_read_file",
                            "name": "read_file",
                            "input": {"filePath": "main.py"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_read_file",
                            "content": "import json",
                        }
                    ],
                },
            ],
        )

    def test_chat_completions_streams_sse_chunks(self):
        with self.client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": main.MODEL_NAME,
                "stream": True,
                "messages": [{"role": "user", "content": "Say hello."}],
            },
        ) as response:
            lines = [line for line in response.iter_lines() if line]

        self.assertEqual(response.status_code, 200)
        self.assertTrue(any('"chat.completion.chunk"' in line for line in lines))
        self.assertTrue(any('"content":"Hello"' in line for line in lines))
        self.assertEqual(lines[-1], "data: [DONE]")

    def test_chat_completions_streams_tool_call_chunks(self):
        main.client.messages.stream_response = [
            FakeContentBlockStartChunk(
                FakeToolUseBlock("toolu_read_file", "read_file", {}),
                index=0,
            ),
            FakeStreamChunk(
                FakeInputJsonDelta('{"filePath":"main.py"}'),
                index=0,
            ),
        ]

        with self.client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": main.MODEL_NAME,
                "stream": True,
                "messages": [{"role": "user", "content": "Read main.py"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "filePath": {"type": "string"}
                                },
                            },
                        },
                    }
                ],
            },
        ) as response:
            lines = [line for line in response.iter_lines() if line]

        self.assertEqual(response.status_code, 200)
        self.assertTrue(any('"tool_calls"' in line for line in lines))
        self.assertTrue(any('"name":"read_file"' in line for line in lines))
        self.assertTrue(any('"arguments":"{\\"filePath\\":\\"main.py\\"}"' in line for line in lines))
        self.assertTrue(any('"finish_reason":"tool_calls"' in line for line in lines))
        self.assertEqual(lines[-1], "data: [DONE]")


if __name__ == "__main__":
    unittest.main()