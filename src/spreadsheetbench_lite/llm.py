from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI

from .prompt import SYSTEM_PROMPT


@dataclass(frozen=True)
class LLMConfig:
    model: str
    base_url: str
    api_key: str = "EMPTY"
    temperature: float = 0.2
    max_tokens: int = 4096
    disable_thinking: bool = True


@dataclass(frozen=True)
class LLMResult:
    content: str
    reasoning_content: str
    finish_reason: str | None
    usage: dict | None
    raw_response: dict


def chat_completion(prompt: str, config: LLMConfig) -> LLMResult:
    """Call a local or remote OpenAI-compatible chat endpoint."""

    client = OpenAI(api_key=config.api_key, base_url=config.base_url)
    user_prompt = prompt
    extra_body = None
    if config.disable_thinking:
        # Qwen reasoning models can spend the whole budget in hidden reasoning,
        # leaving message.content empty. Use both the text switch and the vLLM
        # chat-template switch; unsupported servers usually ignore extra_body.
        user_prompt = "/no_think\n\n" + prompt
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

    last_context_error: Exception | None = None
    for max_tokens in _max_token_attempts(config.max_tokens):
        try:
            response = client.chat.completions.create(
                model=config.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=config.temperature,
                max_tokens=max_tokens,
                extra_body=extra_body,
            )
            break
        except Exception as exc:
            if not _looks_like_context_window_error(exc):
                raise
            last_context_error = exc
    else:
        raise last_context_error or RuntimeError("chat completion failed")
    choice = response.choices[0]
    message = choice.message
    content = message.content or ""
    # vLLM/Qwen may expose hidden thinking as either `reasoning_content`
    # or `reasoning` depending on the OpenAI-compatible server version.
    reasoning_content = (
        getattr(message, "reasoning_content", None)
        or getattr(message, "reasoning", None)
        or ""
    )
    if not content.strip() and ("```" in reasoning_content or "def solve" in reasoning_content):
        content = reasoning_content

    usage = response.usage.model_dump(mode="json") if response.usage else None
    return LLMResult(
        content=content,
        reasoning_content=reasoning_content,
        finish_reason=choice.finish_reason,
        usage=usage,
        raw_response=response.model_dump(mode="json"),
    )


def _max_token_attempts(max_tokens: int) -> list[int]:
    """Try large budgets first, then back off if prompt+completion exceeds the model window."""

    candidates = [max_tokens, 28672, 24576, 20480, 16384, 12288, 8192, 4096, 2048, 1024]
    attempts: list[int] = []
    for value in candidates:
        if value <= max_tokens and value not in attempts:
            attempts.append(value)
    return attempts or [max_tokens]


def _looks_like_context_window_error(exc: Exception) -> bool:
    text = str(exc).lower()
    markers = (
        "context length",
        "maximum context",
        "max_model_len",
        "max model len",
        "max seq len",
        "sequence length",
        "too many tokens",
        "prompt is too long",
    )
    return any(marker in text for marker in markers)
