from __future__ import annotations


DEFAULT_SYSTEM_PROMPT = "Convert the assistant response into the correct GenUI JSON. Return only valid JSON."


def build_prompt(response_text: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    return f"{system_prompt}\n\n### response_text\n{response_text.strip()}\n\n### genui_json\n"
