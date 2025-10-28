from typing import Dict, List


SYSTEM_PREFIX = "You are a helpful assistant."


def build_prompt(turns: List[Dict[str, str]]) -> str:
    parts: List[str] = [f"<|system|> {SYSTEM_PREFIX}"]
    for t in turns:
        role = t["role"].lower()
        if role == "user":
            parts.append(f"<|user|> {t['content']}")
        else:
            parts.append(f"<|assistant|> {t['content']}")
    parts.append("<|assistant|>")
    return "\n".join(parts)


