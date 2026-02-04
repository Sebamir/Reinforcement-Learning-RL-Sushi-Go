import numpy as np

# Mapeo de cartas a ID numérico para la IA
CARD_MAP = {
    'none': 0,
    'tempura': 1,
    'sashimi': 2,
    'dumpling': 3,
    'maki_1': 4,
    'maki_2': 5,
    'maki_3': 6,
    'pudding': 7,
    'nigiri_salmon':8,
    'nigiri_squid':9,
    'nigiri_egg':10,
    'wasabi':11,
}

# Inverso para nosotros poder leer los resultados
ID_TO_CARD = {v: k for k, v in CARD_MAP.items()}

# Configuración básica del mazo (simplificado para entrenamiento rápido)
DECK_COMPOSITION = {
    'tempura': 14,
    'sashimi': 14,
    'dumpling': 14,
    'maki_1': 6,
    'maki_2': 12,
    'maki_3': 8,
    'pudding': 10,
    'nigiri_salmon':10,
    'nigiri_squid':5,   
    'nigiri_egg':5,
    'wasabi':6,
}

def calculate_score(played_cards_ids):
    """
    Recibe una lista de IDs de cartas y devuelve el puntaje total.
    """
    cards = [ID_TO_CARD[cid] for cid in played_cards_ids]
    score = 0
    
    # Conteos
    c = {card: cards.count(card) for card in CARD_MAP.keys()}
    
    # 1. Tempura: 5 puntos por pareja
    score += (c['tempura'] // 2) * 5
    
    # 2. Sashimi: 10 puntos por trío
    score += (c['sashimi'] // 3) * 10
    
    # 3. Dumplings: Escala progresiva
    dumpling_scores = [0, 1, 3, 6, 10, 15]
    num_d = min(c['dumpling'], 5)
    score += dumpling_scores[num_d]
    
    # 4. Maki Rolls (Simplificado para 1 solo agente por ahora)
    # En la versión completa, esto dependerá de los oponentes
    total_maki = c['maki_1'] * 1 + c['maki_2'] * 2 + c['maki_3'] * 3

    
    # 5. Nigiri
    if c['nigiri_salmon']:
        score += 3 * c['nigiri_salmon']
    if c['nigiri_squid']:
        score += 2 * c['nigiri_squid']
    if c['nigiri_egg']:
        score += 1 * c['nigiri_egg']

    # 6. Wasabi
    if c['wasabi']:
        nigiri_total = (c['nigiri_salmon'] + c['nigiri_squid'] + c['nigiri_egg'])
        wasabi_used = min(c['wasabi'], nigiri_total)
        # Cada wasabi usado triplica el valor del nigiri correspondiente
        # Asumimos que se usan en orden de mayor a menor valor
        for _ in range(wasabi_used):
            if c['nigiri_salmon'] > 0:
                score += 6  # 3 * 2 (salmon)
                c['nigiri_salmon'] -= 1
            elif c['nigiri_squid'] > 0:
                score += 4  # 2 * 2 (squid)
                c['nigiri_squid'] -= 1
            elif c['nigiri_egg'] > 0:
                score += 2  # 1 * 2 (egg)
                c['nigiri_egg'] -= 1
    return score, total_maki