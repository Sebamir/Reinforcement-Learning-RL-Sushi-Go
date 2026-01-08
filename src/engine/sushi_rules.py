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
    'pudding': 7
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
    'pudding': 10
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
    score += (total_maki // 3) * 2 # Estimación de puntos maki
    
    return score