# Github Copilot Ollama-MiniMax Proxy

Github Copilot Ollama-MiniMax Proxy is a FastAPI service built primarily to let GitHub Copilot talk to MiniMax through Copilot's Ollama integration path. It makes MiniMax's Anthropic-compatible API look like a local Ollama server on the default Ollama port, while also exposing the OpenAI-compatible chat endpoint that Copilot and similar clients use after model discovery.

## Features

- Designed around GitHub Copilot's Ollama-provider flow
- Advertises `MiniMax-M2.7` and `MiniMax-M2.7 High` during discovery
- Ollama-compatible model discovery, chat, generate, and show endpoints
- OpenAI-compatible `/v1/models` and `/v1/chat/completions`
- Streaming responses for both Ollama NDJSON and OpenAI SSE clients
- Tool and function calling translation between OpenAI tool calls and Anthropic tool use blocks
- Vision input support for Ollama `images` and OpenAI `image_url` data URLs
- Model-specific default thinking budgets of 8K and 24K, with Anthropic thinking passthrough and visible `<think>...</think>` output when enabled

## Primary Use Case

This project exists mainly to bridge GitHub Copilot to MiniMax without requiring a separate Ollama model runtime.

- Copilot can discover the built-in `MiniMax-M2.7` and `MiniMax-M2.7 High` variants through Ollama-style metadata endpoints such as `/api/show` and `/api/tags`
- After discovery, Copilot can send chat traffic to `/v1/chat/completions` using an OpenAI-compatible payload
- Running this proxy on port `11434` lets it behave like a local Ollama instance for tools that expect Ollama defaults

## How It Works

1. A client sends an Ollama-compatible or OpenAI-compatible request to this proxy.
2. The proxy converts that request into a MiniMax Anthropic `messages.create` call.
3. The upstream response is translated back into the format expected by the client.

## Prerequisites

- Python
- A MiniMax API key for the Anthropic-compatible endpoint

## Quick Start

1. Create a virtual environment:

```bash
python -m venv .venv
```

2. Activate it.

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS and Linux:

```bash
source .venv/bin/activate
```

3. Install the runtime dependencies:

```bash
pip install -r requirements.txt
```

4. Copy `.env.example` to `.env`, then set `ANTHROPIC_AUTH_TOKEN`.

PowerShell:

```powershell
Copy-Item .env.example .env
```

macOS and Linux:

```bash
cp .env.example .env
```

5. Start the server on Ollama's default port:

```bash
uvicorn main:app --host 0.0.0.0 --port 11434
```

6. Verify the proxy is running:

```bash
curl http://127.0.0.1:11434/api/version
```

## Using With GitHub Copilot

This is the main intended workflow for the project.

1. Start the proxy locally on `http://127.0.0.1:11434`.
2. In VS Code, open GitHub Copilot's model or provider settings.
3. Select the Ollama-based local provider path.
4. If your Copilot version asks for a server URL, use `http://127.0.0.1:11434`.
5. Refresh or rediscover local models, then select either `MiniMax-M2.7` or `MiniMax-M2.7 High`.

Notes:

- The exact VS Code labels can vary between Copilot releases.
- The default discovery names are `MiniMax-M2.7` with an 8K thinking budget and `MiniMax-M2.7 High` with a 24K thinking budget.
- If you want Copilot to display different names, set `OLLAMA_MODEL_NAME` and `OLLAMA_HIGH_MODEL_NAME` in `.env`.
- This proxy is built to satisfy the Ollama discovery endpoints and the OpenAI-style chat completions path Copilot uses after discovery.

## Docker Compose

1. Copy `.env.example` to `.env` and set `ANTHROPIC_AUTH_TOKEN`.
2. Start the service:

```bash
docker compose up --build
```

3. The proxy will be available at `http://127.0.0.1:11434`.

## Configuration

The application loads `.env` from the repository root.

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `ANTHROPIC_AUTH_TOKEN` | Yes | None | MiniMax API key used for upstream requests |
| `ANTHROPIC_BASE_URL` | No | `https://api.minimax.io/anthropic` | Anthropic-compatible MiniMax base URL |
| `ANTHROPIC_MODEL` | No | `MiniMax-M2.7` | Upstream model name sent to MiniMax |
| `API_TIMEOUT_MS` | No | `3000000` | Upstream request timeout in milliseconds |
| `ANTHROPIC_THINKING_TYPE` | No | None | Fallback Anthropic thinking mode used when the client does not send a `thinking` object and the request uses a non-discovered model name |
| `ANTHROPIC_THINKING_BUDGET_TOKENS` | No | None | Fallback thinking budget passed upstream when `ANTHROPIC_THINKING_TYPE` is set |
| `ANTHROPIC_THINKING_DISPLAY` | No | None | Optional Anthropic thinking display mode such as `summarized` or `omitted` |
| `DEFAULT_SYSTEM_PROMPT` | No | `You are a helpful assistant.` | Fallback system prompt when none is provided by the client |
| `OLLAMA_MODEL_NAME` | No | Inherits `ANTHROPIC_MODEL` | Standard model name exposed to Ollama and OpenAI clients |
| `OLLAMA_HIGH_MODEL_NAME` | No | `OLLAMA_MODEL_NAME High` | High-thinking model name exposed to Ollama and OpenAI clients |
| `OLLAMA_MODEL_ARCHITECTURE` | No | `minimax` | Architecture reported by `/api/show` |
| `OLLAMA_MODEL_BASENAME` | No | Inherits `OLLAMA_MODEL_NAME` | Basename reported by `/api/show` |
| `OLLAMA_HIGH_MODEL_BASENAME` | No | Inherits `OLLAMA_HIGH_MODEL_NAME` | Basename reported by `/api/show` for the High variant |
| `OLLAMA_MODEL_CONTEXT_LENGTH` | No | `204800` | Context window reported by `/api/show` |
| `OLLAMA_MODEL_SIZE` | No | `4000000000` | Model size reported in Ollama listings |
| `OLLAMA_MODEL_FAMILY` | No | Inherits `OLLAMA_MODEL_ARCHITECTURE` | Family reported by `/api/show` |
| `OLLAMA_MODEL_FORMAT` | No | `proxy` | Format reported by `/api/show` |
| `OLLAMA_MODEL_PARAMETER_SIZE` | No | `unknown` | Parameter size reported by `/api/show` |
| `OLLAMA_MODIFIED_AT` | No | `2026-01-01T00:00:00Z` | Modified timestamp reported in Ollama listings |
| `OLLAMA_MODEL_CAPABILITIES` | No | `completion,tools,vision` | Comma-separated capabilities surfaced by `/api/show` |
| `OLLAMA_DEFAULT_THINKING_TYPE` | No | `enabled` | Thinking mode applied to the built-in discovered model variants when the client omits `thinking` |
| `OLLAMA_MODEL_THINKING_BUDGET_TOKENS` | No | `8192` | Default thinking budget for `OLLAMA_MODEL_NAME` |
| `OLLAMA_HIGH_MODEL_THINKING_BUDGET_TOKENS` | No | `24576` | Default thinking budget for `OLLAMA_HIGH_MODEL_NAME` |

## Supported Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/version` | Return the proxy version |
| `GET` | `/api/models` | List exposed Ollama models |
| `GET` | `/api/tags` | Return Ollama tags for available models |
| `POST` | `/api/show` | Return Ollama model metadata |
| `GET` | `/api/ps` | Return running models |
| `POST` | `/api/chat` | Ollama chat endpoint |
| `POST` | `/api/generate` | Ollama prompt-based generation endpoint |
| `POST` | `/api/embeddings` | Compatibility endpoint returning placeholder vectors |
| `GET` | `/v1/models` | OpenAI-compatible model listing |
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat completions |

## Compatibility Notes

- `POST /api/show` accepts either `model` or `name` in the request body.
- GitHub Copilot's Ollama provider discovers models via `/api/show` and `/api/tags`, then sends chat traffic to `/v1/chat/completions`.
- OpenAI vision input is supported for `image_url` values that use base64 data URLs.
- `/api/embeddings` currently returns 1024-dimensional zero vectors for compatibility rather than real embeddings.
- Clients can send a top-level `thinking` object on `/api/chat`, `/api/generate`, or `/v1/chat/completions`; if omitted, the proxy uses the discovered model's built-in default thinking budget and only falls back to `ANTHROPIC_THINKING_*` for unknown model names.
- Ollama streaming (`/api/chat`, `/api/generate`) can emit visible thinking as `<think>...</think>` fragments when thinking is enabled.
- OpenAI streaming (`/v1/chat/completions` with `stream=true`) does not emit thinking deltas in SSE chunks.
- OpenAI non-streaming includes raw `thinking_blocks` on assistant messages for clients that round-trip custom fields.

## Usage Examples

### Ollama Chat

```bash
curl http://127.0.0.1:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiniMax-M2.7 High",
    "stream": false,
    "messages": [
      {"role": "user", "content": "Say hello."}
    ]
  }'
```

### OpenAI Chat Completions

```bash
curl http://127.0.0.1:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiniMax-M2.7 High",
    "stream": false,
    "messages": [
      {"role": "system", "content": "You are helpful."},
      {"role": "user", "content": "Say hello."}
    ]
  }'
```

## Development

Run the test suite:

```bash
python -m unittest -v test_main.py
```

Install or update dependencies with:

```bash
pip install -r requirements.txt
```

To run the app in Docker during development:

```bash
docker compose up --build
```

## Project Structure

```text
.
├── main.py
├── models.py
├── openai_compat.py
├── test_main.py
├── requirements.txt
├── README.md
├── .env.example
├── Dockerfile
└── docker-compose.yml
```

## License

This repository includes a [LICENSE](LICENSE) file.