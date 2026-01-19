import re
from rapidfuzz import fuzz

INTENT_THRESHOLD = 70
ENTITY_THRESHOLD = 80


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z ]", "", text)
    return text.strip()


def score(a: str, b: str) -> int:
    return max(
        fuzz.ratio(a, b),
        fuzz.token_set_ratio(a, b)
    )


def parse_command(text: str, cfg: dict):
    lang = cfg.get('language', 'en')

    best_intent = None
    best_score = 0

    for intent, phrases in cfg['voice_commands'].items():
        for p in phrases:
            s = score(text, p)
            if s > best_score:
                best_score = s
                best_intent = intent

    if best_score < INTENT_THRESHOLD:
        return None

    tool_match = None
    for tool, names in cfg['tools_dict'].items():
        for n in names:
            if score(text, n) > ENTITY_THRESHOLD:
                tool_match = tool

    return {
        'intent': best_intent,
        'tool': tool_match,
        'confidence': best_score
    }

