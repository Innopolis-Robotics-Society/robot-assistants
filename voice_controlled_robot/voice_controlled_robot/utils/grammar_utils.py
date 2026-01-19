import yaml
from itertools import product

def generate_grammar(cfg):
    grammar = []

    wake = cfg.get('wake_word', '').strip()

    # 1. Полные команды: [wake] + intent + tool
    for intent in cfg['intents'].values():
        for tool in cfg['tools'].values():
            for p, t in product(intent['phrases'], tool):
                cmd = f"{p} {t}"
                grammar.append(cmd)

                if wake:
                    grammar.append(f"{wake} {cmd}")

    # 2. Интенты без инструмента
    for intent in cfg['intents'].values():
        for p in intent['phrases']:
            grammar.append(p)

            if wake:
                grammar.append(f"{wake} {p}")

    # 3. Сам wake-word как отдельная фраза
    if wake:
        grammar.append(wake)

    # Убираем дубликаты
    grammar = list(sorted(set(grammar)))

    return grammar

