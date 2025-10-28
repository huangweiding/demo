import re
from dataclasses import dataclass
from typing import List


@dataclass
class RewardConfig:
    w_format: float = 0.25
    w_relevance: float = 0.45
    w_length: float = 0.20
    w_refusal: float = 0.10
    min_tokens: int = 32
    max_tokens: int = 256


def _simple_tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def check_format(response: str) -> float:
    score = 0.0
    if len(response.strip()) == 0:
        return 0.0
    if any(x in response for x in ["<|system|>", "<|user|>"]):
        return 0.0
    if any(response.strip().startswith(x) for x in ["当然", "好的", "好的，", "当然，", "Sure", "Certainly", "Here", "Hi", "你好"]):
        score += 0.5
    if response.strip()[-1:] in ["。", ".", "!", "！", "?", "？"]:
        score += 0.5
    return score


def relevance_overlap(response: str, last_user: str, full_prompt: str) -> float:
    resp_tokens = set(_simple_tokenize(response))
    user_tokens = set(_simple_tokenize(last_user))
    ctx_tokens = set(_simple_tokenize(full_prompt))
    # Overlap with last user has more weight
    if len(resp_tokens) == 0:
        return 0.0
    overlap_user = len(resp_tokens & user_tokens) / max(1, len(user_tokens))
    overlap_ctx = len(resp_tokens & ctx_tokens) / max(1, len(ctx_tokens))
    return 0.7 * overlap_user + 0.3 * overlap_ctx


def length_score(response: str, min_tokens: int, max_tokens: int) -> float:
    tokens = _simple_tokenize(response)
    n = len(tokens)
    if n < min_tokens:
        return n / float(min_tokens)
    if n > max_tokens:
        # linearly decrease after max
        over = min(n - max_tokens, max_tokens)
        return max(0.0, 1.0 - over / float(max_tokens))
    return 1.0


def refusal_penalty(response: str) -> float:
    lower = response.lower()
    refusals = [
        "i can't", "i cannot", "i won't", "无法", "不能", "不可以", "抱歉我不能", "对不起我不能",
        "as an ai", "作为一个ai", "作为一个人工智能",
    ]
    return -1.0 if any(x in lower for x in refusals) else 0.0


def score_response(prompt: str, response: str, last_user: str, config: RewardConfig) -> float:
    return (
        config.w_format * check_format(response) +
        config.w_relevance * relevance_overlap(response, last_user, prompt) +
        config.w_length * length_score(response, config.min_tokens, config.max_tokens) +
        config.w_refusal * refusal_penalty(response)
    )


