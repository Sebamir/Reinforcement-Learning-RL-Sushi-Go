"""
Sistema de scoring unificado para Sushi Go.
ÚNICA FUENTE DE VERDAD para todos los cálculos de puntos.
"""

import numpy as np
from collections import Counter

# Mapeo de cartas a ID numérico
CARD_MAP = {
    'none': 0,
    'tempura': 1,
    'sashimi': 2,
    'dumpling': 3,
    'maki_1': 4,
    'maki_2': 5,
    'maki_3': 6,
    'pudding': 7,
    'nigiri_salmon': 8,
    'nigiri_squid': 9,
    'nigiri_egg': 10,
    'wasabi': 11,
}

# Inverso para leer resultados
ID_TO_CARD = {v: k for k, v in CARD_MAP.items()}

# Composición del mazo
DECK_COMPOSITION = {
    'tempura': 14,
    'sashimi': 14,
    'dumpling': 14,
    'maki_1': 6,
    'maki_2': 12,
    'maki_3': 8,
    'pudding': 10,
    'nigiri_salmon': 10,
    'nigiri_squid': 5,   
    'nigiri_egg': 5,
    'wasabi': 6,
}


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def count_maki_icons(played_cards_ids):
    """
    Cuenta el total de iconos de maki de un jugador.
    
    Args:
        played_cards_ids: Lista de IDs de cartas jugadas
        
    Returns:
        int: Total de iconos maki
    """
    cards = [ID_TO_CARD[cid] for cid in played_cards_ids]
    c = Counter(cards)
    return c['maki_1'] * 1 + c['maki_2'] * 2 + c['maki_3'] * 3


def calculate_maki_points(all_players_cards, player_idx):
    """
    Calcula puntos de maki de forma COMPETITIVA.
    
    Reglas:
    - Ganador único: 6 puntos
    - Empate múltiple: 6 / número_ganadores (redondeado hacia abajo)
    - Perdedor: 0 puntos
    - Nadie jugó maki: 0 puntos para todos
    
    Args:
        all_players_cards: Lista de listas de IDs (todas las cartas de todos)
        player_idx: Índice del jugador para el que calcular
        
    Returns:
        int: Puntos de maki para este jugador
    """
    # Contar maki de todos
    all_maki_counts = [count_maki_icons(cards) for cards in all_players_cards]
    
    max_maki = max(all_maki_counts)
    
    if max_maki == 0:
        return 0  # Nadie jugó maki
    
    # Encontrar ganadores
    winners = [i for i, count in enumerate(all_maki_counts) if count == max_maki]
    
    if player_idx in winners:
        return 6 // len(winners)
    else:
        return 0


def calculate_nigiri_wasabi_points(played_cards_ids):
    """
    Calcula puntos de nigiri + wasabi respetando ORDEN TEMPORAL.
    
    Reglas:
    - Wasabi solo afecta al SIGUIENTE nigiri que se juegue
    - Un wasabi "espera" hasta que se juegue un nigiri
    - Si se juega nigiri sin wasabi activo, vale su valor base
    
    Args:
        played_cards_ids: Lista de IDs EN ORDEN de jugada
        
    Returns:
        tuple: (puntos_totales, wasabi_usados)
    """
    nigiri_points = 0
    wasabi_active = 0
    wasabi_used = 0
    
    for card_id in played_cards_ids:
        card_name = ID_TO_CARD[card_id]
        
        if card_name == 'wasabi':
            wasabi_active += 1
            
        elif card_name.startswith('nigiri_'):
            # Determinar valor base
            if card_name == 'nigiri_salmon':
                base_value = 3
            elif card_name == 'nigiri_squid':
                base_value = 2
            elif card_name == 'nigiri_egg':
                base_value = 1
            
            # Aplicar wasabi si está activo
            if wasabi_active > 0:
                nigiri_points += base_value * 3
                wasabi_active -= 1
                wasabi_used += 1
            else:
                nigiri_points += base_value
    
    return nigiri_points, wasabi_used


# ============================================================================
# FUNCIONES PRINCIPALES DE SCORING
# ============================================================================

def calculate_score_simple(played_cards_ids):
    """
    Calcula score SIN maki (para scoring individual no competitivo).
    
    Útil para:
    - Recompensas inmediatas durante entrenamiento
    - Visualización de progreso individual
    
    Args:
        played_cards_ids: Lista de IDs de cartas jugadas
        
    Returns:
        int: Puntaje total (sin maki)
    """
    cards = [ID_TO_CARD[cid] for cid in played_cards_ids]
    c = Counter(cards)
    score = 0
    
    # Tempura: 5 puntos por pareja
    score += (c['tempura'] // 2) * 5
    
    # Sashimi: 10 puntos por trío
    score += (c['sashimi'] // 3) * 10
    
    # Dumpling: escala progresiva
    dumpling_scores = {0: 0, 1: 1, 2: 3, 3: 6, 4: 10, 5: 15}
    num_d = min(c['dumpling'], 5)
    score += dumpling_scores[num_d]
    
    # Nigiri + Wasabi (orden temporal)
    nigiri_points, _ = calculate_nigiri_wasabi_points(played_cards_ids)
    score += nigiri_points
    
    # Pudding: 0 en partida de 1 ronda
    # (En juego completo de 3 rondas se contaría al final)
    
    return score


def calculate_score_competitive(all_players_cards, player_idx):
    """
    Calcula score COMPLETO incluyendo maki competitivo.
    
    Esta es la función principal para calcular el ganador.
    
    Args:
        all_players_cards: Lista de listas de IDs (todos los jugadores)
        player_idx: Índice del jugador para el que calcular
        
    Returns:
        int: Puntaje total final
    """
    played_cards_ids = all_players_cards[player_idx]
    
    # Score base (sin maki)
    score = calculate_score_simple(played_cards_ids)
    
    # Maki competitivo
    maki_points = calculate_maki_points(all_players_cards, player_idx)
    score += maki_points
    
    return score


def calculate_all_scores(all_players_cards):
    """
    Calcula scores de TODOS los jugadores de una vez.
    
    Args:
        all_players_cards: Lista de listas de IDs
        
    Returns:
        list: Puntajes finales de cada jugador
    """
    return [
        calculate_score_competitive(all_players_cards, i) 
        for i in range(len(all_players_cards))
    ]


# ============================================================================
# FUNCIÓN DE BREAKDOWN DETALLADO (para UI)
# ============================================================================

def calculate_detailed_breakdown(played_cards_ids, all_players_cards=None, player_idx=None):
    """
    Calcula desglose detallado para mostrar en UI.
    
    Args:
        played_cards_ids: Cartas del jugador
        all_players_cards: Todas las cartas (necesario para maki)
        player_idx: Índice del jugador (necesario para maki)
        
    Returns:
        dict: Desglose por categoría con puntos y detalles
    """
    cards = [ID_TO_CARD[cid] for cid in played_cards_ids]
    c = Counter(cards)
    
    breakdown = {
        'tempura': {'count': c['tempura'], 'points': 0, 'detail': ''},
        'sashimi': {'count': c['sashimi'], 'points': 0, 'detail': ''},
        'dumpling': {'count': c['dumpling'], 'points': 0, 'detail': ''},
        'maki': {'count': 0, 'points': 0, 'detail': ''},
        'nigiri': {'count': 0, 'points': 0, 'detail': ''},
        'pudding': {'count': c['pudding'], 'points': 0, 'detail': ''},
        'wasabi': {'count': c['wasabi'], 'points': 0, 'detail': ''}
    }
    
    # Tempura
    pairs = c['tempura'] // 2
    breakdown['tempura']['points'] = pairs * 5
    breakdown['tempura']['detail'] = f"{pairs} parejas × 5pts" if pairs > 0 else "Incompleto"
    
    # Sashimi
    trios = c['sashimi'] // 3
    breakdown['sashimi']['points'] = trios * 10
    breakdown['sashimi']['detail'] = f"{trios} tríos × 10pts" if trios > 0 else "Incompleto"
    
    # Dumpling
    dumpling_scores = {0: 0, 1: 1, 2: 3, 3: 6, 4: 10, 5: 15}
    num_d = min(c['dumpling'], 5)
    breakdown['dumpling']['points'] = dumpling_scores[num_d]
    breakdown['dumpling']['detail'] = f"{num_d} dumplings" if num_d > 0 else "Ninguno"
    
    # Maki (competitivo)
    total_maki = count_maki_icons(played_cards_ids)
    breakdown['maki']['count'] = total_maki
    
    if all_players_cards is not None and player_idx is not None:
        maki_points = calculate_maki_points(all_players_cards, player_idx)
        breakdown['maki']['points'] = maki_points
        
        # Generar detalle descriptivo
        all_maki_counts = [count_maki_icons(cards) for cards in all_players_cards]
        max_maki = max(all_maki_counts)
        winners = [i for i, m in enumerate(all_maki_counts) if m == max_maki]
        
        if max_maki == 0:
            breakdown['maki']['detail'] = "Nadie jugó maki"
        elif player_idx in winners:
            if len(winners) == 1:
                breakdown['maki']['detail'] = f"{total_maki} maki (máximo) → 6pts"
            else:
                breakdown['maki']['detail'] = f"{total_maki} maki (empate {len(winners)}) → {maki_points}pts"
        else:
            breakdown['maki']['detail'] = f"{total_maki} maki (no ganó)"
    else:
        breakdown['maki']['detail'] = f"{total_maki} maki (sin comparar)"
    
    # Nigiri + Wasabi
    nigiri_points, wasabi_used = calculate_nigiri_wasabi_points(played_cards_ids)
    
    breakdown['nigiri']['count'] = c['nigiri_salmon'] + c['nigiri_squid'] + c['nigiri_egg']
    breakdown['nigiri']['points'] = nigiri_points
    breakdown['nigiri']['detail'] = (
        f"{c['nigiri_salmon']}🍣 + {c['nigiri_squid']}🦑 + {c['nigiri_egg']}🥚"
    )
    
    breakdown['wasabi']['detail'] = f"{wasabi_used}/{c['wasabi']} usados" if c['wasabi'] > 0 else "Ninguno"
    
    # Pudding
    breakdown['pudding']['detail'] = f"{c['pudding']} puddins (no cuenta en 1 ronda)" if c['pudding'] > 0 else "Ninguno"
    
    return breakdown


# ============================================================================
# FUNCIÓN LEGACY (compatibilidad con código antiguo)
# ============================================================================

def calculate_score(played_cards_ids):
    """
    Función legacy para compatibilidad.
    
    DEPRECATED: Usar calculate_score_simple() o calculate_score_competitive()
    
    Returns:
        tuple: (score_sin_maki, total_maki_icons)
    """
    score = calculate_score_simple(played_cards_ids)
    total_maki = count_maki_icons(played_cards_ids)
    return score, total_maki


# ============================================================================
# TESTS UNITARIOS
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("TESTS DEL SISTEMA DE SCORING UNIFICADO")
    print("="*70)
    
    # Test 1: Maki competitivo
    print("\n📊 Test 1: Maki Competitivo")
    p0 = [CARD_MAP['maki_3'], CARD_MAP['maki_2']]  # 5 maki
    p1 = [CARD_MAP['maki_1'], CARD_MAP['maki_1']]  # 2 maki
    
    all_cards = [p0, p1]
    
    maki_p0 = calculate_maki_points(all_cards, 0)
    maki_p1 = calculate_maki_points(all_cards, 1)
    
    print(f"  Jugador 0: {count_maki_icons(p0)} maki → {maki_p0} pts")
    print(f"  Jugador 1: {count_maki_icons(p1)} maki → {maki_p1} pts")
    assert maki_p0 == 6 and maki_p1 == 0, "❌ Fallo en maki competitivo"
    print("  ✅ PASS")
    
    # Test 2: Maki empate
    print("\n📊 Test 2: Maki Empate")
    p0 = [CARD_MAP['maki_3']]  # 3 maki
    p1 = [CARD_MAP['maki_2'], CARD_MAP['maki_1']]  # 3 maki
    
    all_cards = [p0, p1]
    
    maki_p0 = calculate_maki_points(all_cards, 0)
    maki_p1 = calculate_maki_points(all_cards, 1)
    
    print(f"  Jugador 0: {count_maki_icons(p0)} maki → {maki_p0} pts")
    print(f"  Jugador 1: {count_maki_icons(p1)} maki → {maki_p1} pts")
    assert maki_p0 == 3 and maki_p1 == 3, "❌ Fallo en empate maki"
    print("  ✅ PASS")
    
    # Test 3: Nigiri orden temporal
    print("\n🍱 Test 3: Nigiri + Wasabi (orden temporal)")
    
    # wasabi → egg
    cards1 = [CARD_MAP['wasabi'], CARD_MAP['nigiri_egg']]
    pts1, used1 = calculate_nigiri_wasabi_points(cards1)
    print(f"  wasabi → egg: {pts1} pts ({used1} wasabi usados)")
    assert pts1 == 3 and used1 == 1, "❌ Fallo wasabi → egg"
    
    # egg → wasabi
    cards2 = [CARD_MAP['nigiri_egg'], CARD_MAP['wasabi']]
    pts2, used2 = calculate_nigiri_wasabi_points(cards2)
    print(f"  egg → wasabi: {pts2} pts ({used2} wasabi usados)")
    assert pts2 == 1 and used2 == 0, "❌ Fallo egg → wasabi"
    print("  ✅ PASS")
    
    # Test 4: Score completo
    print("\n🎯 Test 4: Score Completo")
    p0 = [
        CARD_MAP['tempura'], CARD_MAP['tempura'],  # 5 pts
        CARD_MAP['wasabi'], CARD_MAP['nigiri_salmon'],  # 9 pts
        CARD_MAP['maki_3']  # 3 maki
    ]
    p1 = [
        CARD_MAP['sashimi'], CARD_MAP['sashimi'],  # 0 pts (incompleto)
        CARD_MAP['maki_1'], CARD_MAP['maki_1']  # 2 maki
    ]
    
    all_cards = [p0, p1]
    
    score_p0 = calculate_score_competitive(all_cards, 0)
    score_p1 = calculate_score_competitive(all_cards, 1)
    
    print(f"  Jugador 0: {score_p0} pts (tempura 5 + nigiri 9 + maki 6)")
    print(f"  Jugador 1: {score_p1} pts (sashimi 0 + maki 0)")
    assert score_p0 == 20 and score_p1 == 0, "❌ Fallo en score completo"
    print("  ✅ PASS")
    
    # Test 5: Breakdown detallado
    print("\n📋 Test 5: Breakdown Detallado")
    breakdown = calculate_detailed_breakdown(p0, all_cards, 0)
    
    print(f"  Tempura: {breakdown['tempura']['points']} pts - {breakdown['tempura']['detail']}")
    print(f"  Nigiri: {breakdown['nigiri']['points']} pts - {breakdown['nigiri']['detail']}")
    print(f"  Maki: {breakdown['maki']['points']} pts - {breakdown['maki']['detail']}")
    print(f"  Wasabi: {breakdown['wasabi']['detail']}")
    print("  ✅ PASS")
    
    print("\n" + "="*70)
    print("✅ TODOS LOS TESTS PASARON")
    print("="*70)
    print("""
Resumen del sistema unificado:

1. calculate_score_simple(cards)
   → Score individual SIN maki competitivo
   → Usar para recompensas inmediatas

2. calculate_score_competitive(all_cards, player_idx)
   → Score completo CON maki competitivo
   → Usar para determinar ganador final

3. calculate_all_scores(all_cards)
   → Scores de todos los jugadores a la vez
   → Útil para comparaciones

4. calculate_detailed_breakdown(cards, all_cards, idx)
   → Desglose detallado para UI
   → Muestra puntos por categoría con descripciones

5. calculate_score(cards) [LEGACY]
   → Compatibilidad con código antiguo
   → Devuelve (score, maki_count)
    """)