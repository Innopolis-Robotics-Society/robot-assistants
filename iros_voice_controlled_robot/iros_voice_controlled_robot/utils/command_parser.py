import re
from rapidfuzz import fuzz

INTENT_THRESHOLD = 70
ENTITY_THRESHOLD = 80

def normalize(text):
    text = text.lower()
    text = re.sub(r'[^a-z ]', '', text)
    return text.strip()

def match(text, cfg):
    text = normalize(text)

    best_intent = None
    best_score = 0

    for name, intent in cfg['intents'].items():
        for p in intent['phrases']:
            s = fuzz.token_set_ratio(text, p)
            if s > best_score:
                best_score = s
                best_intent = name

    if best_score < INTENT_THRESHOLD:
        return None

    tool_match = None
    for tool, names in cfg['tools'].items():
        for n in names:
            if fuzz.token_set_ratio(text, n) > ENTITY_THRESHOLD:
                tool_match = tool

    return {
        'intent': best_intent,
        'tool': tool_match,
        'confidence': best_score
    }

