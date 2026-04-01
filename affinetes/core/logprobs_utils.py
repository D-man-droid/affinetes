"""Shared utilities for building full_logprobs via post-rollout forward pass.

After a teacher model completes a rollout, call collect_full_logprobs() to:
1. Format the conversation with a chat template
2. Forward pass with prompt_logprobs to get all token_ids + logprobs
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
    student_resp = student.completions.create(
        prompt=full_logprobs["full"],
        prompt_logprobs=1,
        max_tokens=0,
    )
    # Same tokenizer → same token positions → direct comparison
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
    """Build full_logprobs by doing a forward pass on the formatted conversation.

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

    # Forward pass: get prompt_logprobs for ALL tokens
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
                "max_tokens": 0,
                "prompt_logprobs": 1,
                "stream": False,
            },
        )

        if resp.status_code != 200:
            return None

        data = resp.json()

    choices = data.get("choices", [])
    if not choices:
        return None

    raw_prompt_logprobs = choices[0].get("prompt_logprobs")
    if not raw_prompt_logprobs:
        return None

    # Parse prompt_logprobs into token_ids + per-token logprobs + char positions
    # vLLM format: list of (None | {token_id_str: {"logprob": float, "decoded_token": str}})
    token_ids = []
    all_logprobs = []
    token_texts = []

    for entry in raw_prompt_logprobs:
        if entry is None:
            # BOS token
            token_ids.append(0)
            all_logprobs.append(0.0)
            token_texts.append("")
        elif isinstance(entry, dict):
            for tok_id_str, info in entry.items():
                token_ids.append(int(tok_id_str))
                all_logprobs.append(info.get("logprob", 0.0))
                token_texts.append(info.get("decoded_token", ""))
                break

    # Map each token to its char position in full_text, then check if it's assistant
    char_pos = 0
    logprobs_masked = []

    for i, tok_text in enumerate(token_texts):
        tok_len = len(tok_text)
        # Check if this token falls within any assistant range
        is_assistant = False
        for a_start, a_end in assistant_ranges:
            if char_pos >= a_start and char_pos < a_end:
                is_assistant = True
                break

        if is_assistant:
            logprobs_masked.append(all_logprobs[i])
        else:
            logprobs_masked.append(None)

        char_pos += tok_len

    return {
        "full": full_text,
        "token_ids": token_ids,
        "logprobs": logprobs_masked,
    }
