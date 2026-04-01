"""Shared utilities for building full_logprobs via post-rollout forward pass.

After a teacher model completes a rollout, call collect_full_logprobs() to:
1. Format the conversation with a chat template
2. Forward pass with echo=True to get all tokens + logprobs
3. Mark assistant positions with real logprobs, rest with None

Output format:
    {
        "full": "complete conversation text",
        "token_ids": [151644, 872, 198, ...],
        "logprobs":  [None, None, ..., -0.5, -0.1, ..., None, ...],
    }
    - None = user/system token (skip in KL)
    - float = assistant token (participate in KL)

Student-side KL:
    Use the same echo=True approach, or /v1/chat/completions with logprobs=True.
"""

from typing import Any, Dict, List, Optional, Tuple

import httpx

# Per-role chat templates
CHAT_TEMPLATES = {
    "qwen": {
        "system": "<|im_start|>system\n{content}<|im_end|>\n",
        "user": "<|im_start|>user\n{content}<|im_end|>\n",
        "assistant": "<|im_start|>assistant\n{content}<|im_end|>\n",
    },
    "llama": {
        "system": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>",
        "user": "<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>",
        "assistant": "<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>",
    },
    "deepseek": {
        "system": "<|System|>{content}",
        "user": "<|User|>{content}",
        "assistant": "<|Assistant|>{content}",
    },
}


def format_conversation(
    conversation: List[Dict[str, Any]],
    chat_template: str = "qwen",
) -> Tuple[str, List[Tuple[int, int]]]:
    """Format conversation and return (full_text, assistant_char_ranges).

    Returns:
        full_text: The formatted conversation string.
        assistant_ranges: List of (start, end) char offsets for assistant segments.
    """
    tmpl = CHAT_TEMPLATES.get(chat_template, CHAT_TEMPLATES["qwen"])
    parts = []
    ranges = []
    offset = 0

    for msg in conversation:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                c.get("text", "") for c in content if c.get("type") == "text"
            )
        if role not in tmpl:
            continue

        formatted = tmpl[role].format(content=content)
        parts.append(formatted)

        if role == "assistant":
            ranges.append((offset, offset + len(formatted)))

        offset += len(formatted)

    return "".join(parts), ranges


async def collect_full_logprobs(
    conversation: List[Dict[str, Any]],
    model: str,
    base_url: str,
    api_key: str,
    chat_template: str = "qwen",
    timeout: float = 300.0,
) -> Optional[Dict[str, Any]]:
    """Build full_logprobs by doing a forward pass with echo=True.

    Uses /v1/completions with echo=True + logprobs=1 to get logprobs
    for ALL tokens (prompt + completion) in a single call.

    Args:
        conversation: Full conversation [{role, content}, ...].
        model: Model name (same model that generated the rollout).
        base_url: API base URL.
        api_key: API key.
        chat_template: Chat template name.
        timeout: Request timeout.

    Returns:
        {
            "full": str,
            "token_ids": [int, ...],
            "logprobs": [float|None, ...],  # None = non-assistant token
        }
        or None on failure.
    """
    full_text, assistant_ranges = format_conversation(conversation, chat_template)

    if not full_text or not assistant_ranges:
        return None

    # Forward pass with echo=True: returns logprobs for prompt + completion tokens
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout), verify=False) as client:
        resp = await client.post(
            f"{base_url.rstrip('/')}/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "prompt": full_text,
                "max_tokens": 1,
                "logprobs": 1,
                "echo": True,
                "stream": False,
            },
        )

        if resp.status_code != 200:
            return None

        data = resp.json()

    choices = data.get("choices", [])
    if not choices:
        return None

    lp = choices[0].get("logprobs")
    if not lp or not lp.get("tokens"):
        return None

    raw_tokens = lp["tokens"]
    raw_logprobs = lp.get("token_logprobs", [])
    raw_token_ids = lp.get("text_offset", [])  # char offsets, not token ids

    # Map each token to char position, check if within assistant range
    token_ids = []
    logprobs_masked = []
    char_pos = 0

    for i, tok in enumerate(raw_tokens):
        tok_logprob = raw_logprobs[i] if i < len(raw_logprobs) else None

        # Determine if this token falls within any assistant range
        is_assistant = False
        for a_start, a_end in assistant_ranges:
            if char_pos >= a_start and char_pos < a_end:
                is_assistant = True
                break

        # Use text_offset for token_id (vLLM returns text_offset in logprobs)
        # Store the position as a proxy for token_id
        token_ids.append(raw_token_ids[i] if i < len(raw_token_ids) else char_pos)

        if is_assistant and tok_logprob is not None:
            logprobs_masked.append(tok_logprob)
        else:
            logprobs_masked.append(None)

        char_pos += len(tok)

    return {
        "full": full_text,
        "token_ids": token_ids,
        "logprobs": logprobs_masked,
    }
