import gradio as gr
import numpy as np
from stable_baselines3 import PPO
from src.env.SushiGoSelfPlayEnv import SushiGoSelfPlayEnv
from src.engine.sushi_rules import (
    ID_TO_CARD, CARD_MAP,
    calculate_score_competitive,
    calculate_all_scores,
    calculate_detailed_breakdown
)

# ============================================================================
# CONFIGURACIÓN INICIAL
# ============================================================================

model_path = r"experiments\selfplay_2p_optimized\final_model.zip"
model = PPO.load(model_path)

# Estado global del juego
class GameState:
    def __init__(self):
        self.env = None
        self.current_obs = None
        self.game_started = False
        self.waiting_for_human = False
        self.game_history = []
        
    def reset(self):
        self.env = SushiGoSelfPlayEnv(num_players=2)
        self.current_obs, info = self.env.reset()
        self.game_started = True
        self.waiting_for_human = True
        self.game_history = []
        return info

# ============================================================================
# FUNCIONES DE SCORING
# ============================================================================

def calculate_detailed_breakdown(played_cards_ids, all_players_cards=None, player_idx=None):
    """
    Calcula desglose detallado de puntos para mostrar.
    
    Args:
        played_cards_ids: Cartas del jugador a analizar
        all_players_cards: Lista de cartas de TODOS los jugadores (para maki)
        player_idx: Índice del jugador actual (para maki)
    
    Returns:
        dict con categorías y puntos de cada una
    """
    from collections import Counter
    
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
    breakdown['dumpling']['points'] = dumpling_scores.get(num_d, 15)
    breakdown['dumpling']['detail'] = f"{num_d} dumplings" if num_d > 0 else "Ninguno"
    
    # ========================================================================
    # MAKI - COMPETITIVO (comparar con otros jugadores)
    # ========================================================================
    total_maki = c['maki_1'] * 1 + c['maki_2'] * 2 + c['maki_3'] * 3
    breakdown['maki']['count'] = total_maki
    
    if all_players_cards is not None and player_idx is not None:
        # Calcular maki de todos los jugadores
        all_maki_counts = []
        for p_cards in all_players_cards:
            p_cards_names = [ID_TO_CARD[cid] for cid in p_cards]
            p_counter = Counter(p_cards_names)
            p_maki = (p_counter['maki_1'] * 1 + 
                     p_counter['maki_2'] * 2 + 
                     p_counter['maki_3'] * 3)
            all_maki_counts.append(p_maki)
        
        # Encontrar el máximo
        max_maki = max(all_maki_counts)
        
        if max_maki == 0:
            # Nadie jugó maki
            breakdown['maki']['points'] = 0
            breakdown['maki']['detail'] = "Nadie jugó maki"
        else:
            # Contar cuántos jugadores tienen el máximo
            winners = [i for i, count in enumerate(all_maki_counts) if count == max_maki]
            
            if player_idx in winners:
                # Este jugador tiene el máximo (o empató)
                points_per_winner = 6 // len(winners)
                breakdown['maki']['points'] = points_per_winner
                
                if len(winners) == 1:
                    breakdown['maki']['detail'] = f"{total_maki} maki (máximo) → 6pts"
                else:
                    breakdown['maki']['detail'] = f"{total_maki} maki (empate {len(winners)}) → {points_per_winner}pts"
            else:
                # Este jugador NO tiene el máximo
                breakdown['maki']['points'] = 0
                breakdown['maki']['detail'] = f"{total_maki} maki (no ganó)"
    else:
        # Fallback si no hay info de otros jugadores (no debería pasar)
        breakdown['maki']['points'] = 0
        breakdown['maki']['detail'] = f"{total_maki} maki icons"
    
    # ========================================================================
    # NIGIRI + WASABI (orden temporal importa)
    # ========================================================================
    nigiri_points = 0
    wasabi_active = 0
    wasabi_used = 0
    
    nigiri_breakdown = {'salmon': 0, 'squid': 0, 'egg': 0}
    
    # Recorrer cartas en orden jugado
    for card_id in played_cards_ids:
        card_name = ID_TO_CARD[card_id]
        
        if card_name == 'wasabi':
            wasabi_active += 1
        elif card_name.startswith('nigiri_'):
            # Determinar tipo y valor base
            if card_name == 'nigiri_salmon':
                base_value = 3
                nigiri_type = 'salmon'
            elif card_name == 'nigiri_squid':
                base_value = 2
                nigiri_type = 'squid'
            elif card_name == 'nigiri_egg':
                base_value = 1
                nigiri_type = 'egg'
            
            # Aplicar wasabi si hay disponible
            if wasabi_active > 0:
                nigiri_points += base_value * 3
                wasabi_active -= 1
                wasabi_used += 1
            else:
                nigiri_points += base_value
            
            nigiri_breakdown[nigiri_type] += 1
    
    breakdown['nigiri']['count'] = sum(nigiri_breakdown.values())
    breakdown['nigiri']['points'] = nigiri_points
    breakdown['nigiri']['detail'] = (
        f"{nigiri_breakdown['salmon']}🍣 + {nigiri_breakdown['squid']}🦑 + {nigiri_breakdown['egg']}🥚"
    )
    
    breakdown['wasabi']['detail'] = f"{wasabi_used}/{c['wasabi']} usados" if c['wasabi'] > 0 else "Ninguno"
    
    # Pudding (no cuenta en partida de 1 ronda)
    breakdown['pudding']['detail'] = f"{c['pudding']} puddins (no cuenta en 1 ronda)" if c['pudding'] > 0 else "Ninguno"
    
    return breakdown

def format_score_display(env):
    """Formatea los scores actuales en tiempo real."""
    human_score, _ = calculate_score(env.played_cards[0])
    ai_score, _ = calculate_score(env.played_cards[1])
    
    score_html = f"""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 20px; border-radius: 15px; margin: 10px 0;'>
        <h2 style='color: white; text-align: center; margin: 0;'>📊 PUNTUACIÓN EN VIVO</h2>
        <div style='display: flex; justify-content: space-around; margin-top: 15px;'>
            <div style='background: rgba(255,255,255,0.2); padding: 15px; 
                        border-radius: 10px; text-align: center; flex: 1; margin: 0 10px;'>
                <div style='font-size: 14px; color: #e0e0e0;'>🧑 TÚ</div>
                <div style='font-size: 48px; font-weight: bold; color: white;'>{human_score}</div>
                <div style='font-size: 12px; color: #e0e0e0;'>puntos</div>
            </div>
            <div style='background: rgba(255,255,255,0.2); padding: 15px; 
                        border-radius: 10px; text-align: center; flex: 1; margin: 0 10px;'>
                <div style='font-size: 14px; color: #e0e0e0;'>🤖 IA</div>
                <div style='font-size: 48px; font-weight: bold; color: white;'>{ai_score}</div>
                <div style='font-size: 12px; color: #e0e0e0;'>puntos</div>
            </div>
        </div>
        <div style='text-align: center; margin-top: 15px; color: white; font-size: 18px;'>
            {'🎉 Vas ganando!' if human_score > ai_score else 
             '😬 La IA va ganando' if ai_score > human_score else 
             '🤝 Empate'}
            {f' (+{abs(human_score - ai_score)} pts)' if human_score != ai_score else ''}
        </div>
    </div>
    """
    
    return score_html

def format_breakdown_table(breakdown, player_name):
    """Formatea el desglose de puntos en tabla HTML."""
    
    html = f"""
    <div style='margin: 15px 0;'>
        <h3 style='color: #667eea; font-size: 20px; margin-bottom: 10px;'>{player_name}</h3>
        <table style='width: 100%; border-collapse: collapse; background: white; 
                      border-radius: 10px; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.15);'>
            <thead>
                <tr style='background: #667eea; color: white;'>
                    <th style='padding: 14px; text-align: left; font-size: 15px; font-weight: 600;'>Categoría</th>
                    <th style='padding: 14px; text-align: center; font-size: 15px; font-weight: 600;'>Cantidad</th>
                    <th style='padding: 14px; text-align: center; font-size: 15px; font-weight: 600;'>Detalle</th>
                    <th style='padding: 14px; text-align: right; font-size: 15px; font-weight: 600;'>Puntos</th>
                </tr>
            </thead>
            <tbody>
    """
    
    total = 0
    icons = {
        'tempura': '🍤',
        'sashimi': '🍣',
        'dumpling': '🥟',
        'maki': '🍙',
        'nigiri': '🍱',
        'pudding': '🍮',
        'wasabi': '🟢'
    }
    
    row_count = 0
    for category, data in breakdown.items():
        if data['count'] > 0 or data['points'] > 0:
            icon = icons.get(category, '🎴')
            row_color = 'background: #f8f9fa;' if row_count % 2 == 0 else 'background: white;'
            
            html += f"""
                <tr style='{row_color}'>
                    <td style='padding: 12px; color: #1a1a1a; font-size: 15px; font-weight: 500;'>
                        {icon} {category.title()}
                    </td>
                    <td style='padding: 12px; text-align: center; color: #2c3e50; font-size: 15px; font-weight: 600;'>
                        {data['count']}
                    </td>
                    <td style='padding: 12px; text-align: center; font-size: 13px; color: #34495e;'>
                        {data['detail']}
                    </td>
                    <td style='padding: 12px; text-align: right; font-weight: bold; color: #667eea; font-size: 16px;'>
                        {data['points']}
                    </td>
                </tr>
            """
            total += data['points']
            row_count += 1
    
    html += f"""
                <tr style='background: #667eea; color: white; font-weight: bold;'>
                    <td colspan='3' style='padding: 14px; text-align: right; font-size: 16px;'>TOTAL:</td>
                    <td style='padding: 14px; text-align: right; font-size: 22px;'>{total}</td>
                </tr>
            </tbody>
        </table>
    </div>
    """
    
    return html

def format_final_breakdown(env):
    """Genera desglose final completo con tablas detalladas."""
    
    # Usar sistema unificado de sushi_rules.py
    human_breakdown = calculate_detailed_breakdown(
        env.played_cards[0],
        all_players_cards=env.played_cards,
        player_idx=0
    )
    ai_breakdown = calculate_detailed_breakdown(
        env.played_cards[1],
        all_players_cards=env.played_cards,
        player_idx=1
    )
    
    # Scores finales
    scores = calculate_all_scores(env.played_cards)
    human_score = scores[0]
    ai_score = scores[1]
    
    # Header con resultado
    winner_html = ""
    if human_score > ai_score:
        winner_html = f"""
        <div style='background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                    padding: 30px; border-radius: 15px; text-align: center; margin: 20px 0;'>
            <h1 style='color: white; margin: 0; font-size: 48px;'>🎉 ¡VICTORIA! 🎉</h1>
            <p style='color: white; font-size: 24px; margin: 10px 0;'>
                Ganaste por <strong>{human_score - ai_score}</strong> puntos
            </p>
            <p style='color: rgba(255,255,255,0.9); font-size: 18px;'>
                {human_score} pts vs {ai_score} pts
            </p>
        </div>
        """
    elif ai_score > human_score:
        winner_html = f"""
        <div style='background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%); 
                    padding: 30px; border-radius: 15px; text-align: center; margin: 20px 0;'>
            <h1 style='color: white; margin: 0; font-size: 48px;'>😢 DERROTA 😢</h1>
            <p style='color: white; font-size: 24px; margin: 10px 0;'>
                La IA ganó por <strong>{ai_score - human_score}</strong> puntos
            </p>
            <p style='color: rgba(255,255,255,0.9); font-size: 18px;'>
                {human_score} pts vs {ai_score} pts
            </p>
        </div>
        """
    else:
        winner_html = f"""
        <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    padding: 30px; border-radius: 15px; text-align: center; margin: 20px 0;'>
            <h1 style='color: white; margin: 0; font-size: 48px;'>🤝 ¡EMPATE! 🤝</h1>
            <p style='color: white; font-size: 24px; margin: 10px 0;'>
                Ambos con <strong>{human_score}</strong> puntos
            </p>
        </div>
        """
    
    # Tablas de desglose
    breakdown_html = f"""
    <div style='background: #f8f9fa; padding: 20px; border-radius: 15px; margin: 20px 0;'>
        <h2 style='text-align: center; color: #667eea; margin-bottom: 20px;'>
            📊 DESGLOSE DETALLADO DE PUNTOS
        </h2>
        {format_breakdown_table(human_breakdown, '🧑 TU PUNTUACIÓN')}
        {format_breakdown_table(ai_breakdown, '🤖 PUNTUACIÓN DE LA IA')}
    </div>
    """
    
    # Comparativa de cartas jugadas
    human_cards = [ID_TO_CARD[c] for c in env.played_cards[0]]
    ai_cards = [ID_TO_CARD[c] for c in env.played_cards[1]]
    
    cards_html = f"""
    <div style='background: white; padding: 20px; border-radius: 15px; 
                box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin: 20px 0;'>
        <h3 style='color: #667eea;'>🎴 Cartas Jugadas</h3>
        <div style='margin: 10px 0;'>
            <strong>🧑 Tus cartas:</strong><br/>
            <span style='font-family: monospace; color: #555;'>
                {' | '.join(human_cards)}
            </span>
        </div>
        <div style='margin: 10px 0;'>
            <strong>🤖 Cartas IA:</strong><br/>
            <span style='font-family: monospace; color: #555;'>
                {' | '.join(ai_cards)}
            </span>
        </div>
    </div>
    """
    
    return winner_html + breakdown_html + cards_html

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def get_human_hand(env):
    """Obtiene la mano actual del jugador humano (jugador 0)."""
    return [ID_TO_CARD[card_id] for card_id in env.hands[0]]

def format_game_state(env):
    """Formatea el estado actual del juego para mostrar."""
    human_played = [ID_TO_CARD[card_id] for card_id in env.played_cards[0]]
    ai_played = [ID_TO_CARD[card_id] for card_id in env.played_cards[1]]
    
    state_text = f"""
    <div style='background: white; padding: 15px; border-radius: 10px; 
                box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin: 10px 0;'>
        <h3 style='color: #667eea; margin-top: 0;'>
            🎮 Turno {env.current_turn + 1} de {env.max_turns}
        </h3>
        
        <div style='margin: 15px 0;'>
            <strong style='color: #555;'>🧑 Tus cartas jugadas:</strong><br/>
            <span style='font-family: monospace; color: #333; font-size: 14px;'>
                {' | '.join(human_played) if human_played else '(ninguna)'}
            </span>
        </div>
        
        <div style='margin: 15px 0;'>
            <strong style='color: #555;'>🤖 Cartas IA jugadas:</strong><br/>
            <span style='font-family: monospace; color: #333; font-size: 14px;'>
                {' | '.join(ai_played) if ai_played else '(ninguna)'}
            </span>
        </div>
    </div>
    """
    
    return state_text

# ============================================================================
# FUNCIONES PRINCIPALES DEL JUEGO
# ============================================================================

def start_new_game(game_state):
    """Inicia una nueva partida."""
    game_state.reset()
    
    state_display = format_game_state(game_state.env)
    score_display = format_score_display(game_state.env)
    human_hand = get_human_hand(game_state.env)
    
    message = f"""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 20px; border-radius: 15px; color: white; margin-bottom: 20px;'>
        <h2 style='margin: 0;'>🍱 ¡Nueva Partida Iniciada!</h2>
        <p style='margin: 10px 0 0 0;'>
            Selecciona una carta de tu mano para comenzar
        </p>
    </div>
    {state_display}
    """
    
    return (
        message,
        score_display,
        gr.update(choices=human_hand, value=None, interactive=True),
        gr.update(interactive=True),
        game_state
    )

def play_turn(selected_card, game_state):
    """Ejecuta un turno completo: humano juega, luego IA juega."""
    
    if not game_state.game_started:
        return (
            "⚠️ Primero debes iniciar una nueva partida.",
            "",
            gr.update(),
            gr.update(interactive=False),
            game_state
        )
    
    if selected_card is None:
        current_display = format_game_state(game_state.env)
        score_display = format_score_display(game_state.env)
        return (
            current_display + "<p style='color: orange;'>⚠️ Selecciona una carta</p>",
            score_display,
            gr.update(),
            gr.update(interactive=True),
            game_state
        )
    
    env = game_state.env
    result_html = ""
    
    # ========================================================================
    # TURNO DEL HUMANO
    # ========================================================================
    
    human_hand = env.hands[0]
    human_hand_names = [ID_TO_CARD[card_id] for card_id in human_hand]
    
    try:
        action = human_hand_names.index(selected_card)
    except ValueError:
        current_display = format_game_state(env)
        score_display = format_score_display(env)
        return (
            current_display + f"<p style='color: red;'>❌ '{selected_card}' no está en tu mano</p>",
            score_display,
            gr.update(),
            gr.update(interactive=True),
            game_state
        )
    
    # Ejecutar acción
    obs, reward, terminated, _, info = env.step(action)
    
    result_html += f"""
    <div style='background: #e8f5e9; padding: 15px; border-radius: 10px; 
                border-left: 4px solid #4caf50; margin: 10px 0;'>
        <strong style='color: #2e7d32;'>🎴 Tu Jugada:</strong> {selected_card}
        <span style='color: #666; margin-left: 10px;'>
            (+{reward:.1f} pts este turno)
        </span>
    </div>
    """
    
    # Verificar si terminó
    if terminated:
        final_breakdown = format_final_breakdown(env)
        game_state.game_started = False
        
        return (
            result_html + final_breakdown,
            "",
            gr.update(choices=[], interactive=False),
            gr.update(interactive=False),
            game_state
        )
    
    # ========================================================================
    # TURNO DE LA IA
    # ========================================================================
    
    ai_action, _ = model.predict(obs, deterministic=True)
    ai_hand = env.hands[1]
    ai_card_played = ID_TO_CARD[ai_hand[int(ai_action)]]
    
    obs, ai_reward, terminated, _, info = env.step(int(ai_action))
    
    result_html += f"""
    <div style='background: #e3f2fd; padding: 15px; border-radius: 10px; 
                border-left: 4px solid #2196f3; margin: 10px 0;'>
        <strong style='color: #1565c0;'>🤖 Jugada IA:</strong> {ai_card_played}
        <span style='color: #666; margin-left: 10px;'>
            (+{ai_reward:.1f} pts este turno)
        </span>
    </div>
    """
    
    # Verificar de nuevo si terminó
    if terminated:
        final_breakdown = format_final_breakdown(env)
        game_state.game_started = False
        
        return (
            result_html + final_breakdown,
            "",
            gr.update(choices=[], interactive=False),
            gr.update(interactive=False),
            game_state
        )
    
    # ========================================================================
    # CONTINUAR JUEGO
    # ========================================================================
    
    result_html += format_game_state(env)
    score_display = format_score_display(env)
    new_human_hand = get_human_hand(env)
    
    return (
        result_html,
        score_display,
        gr.update(choices=new_human_hand, value=None, interactive=True),
        gr.update(interactive=True),
        game_state
    )

# ============================================================================
# INTERFAZ DE GRADIO
# ============================================================================

def create_interface():
    """Crea la interfaz mejorada de Gradio."""
    
    with gr.Blocks(title="🍱 Sushi Go! - Humano vs IA", theme=gr.themes.Soft()) as demo:
        
        game_state = gr.State(GameState())
        
        # Header
        gr.HTML("""
        <div style='text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 30px; border-radius: 20px; margin-bottom: 20px;'>
            <h1 style='color: white; margin: 0; font-size: 48px;'>🍱 Sushi Go!</h1>
            <p style='color: rgba(255,255,255,0.9); font-size: 18px; margin: 10px 0 0 0;'>
                Humano vs IA entrenada con Reinforcement Learning
            </p>
        </div>
        """)
        
        with gr.Row():
            # Columna izquierda: Juego
            with gr.Column(scale=2):
                output_display = gr.HTML(
                    "<div style='text-align: center; padding: 40px; color: #666;'>"
                    "👋 Presiona '🎮 Nueva Partida' para comenzar</div>"
                )
                
            # Columna derecha: Controles y scores
            with gr.Column(scale=1):
                # Puntuación en vivo
                score_display = gr.HTML("")
                
                # Controles
                gr.Markdown("### 🎯 Controles")
                start_button = gr.Button(
                    "🎮 Nueva Partida", 
                    variant="primary", 
                    size="lg"
                )
                
                gr.Markdown("### 🃏 Tu Mano")
                hand_selector = gr.Radio(
                    choices=[],
                    label="Selecciona una carta para jugar",
                    interactive=False
                )
                
                play_button = gr.Button(
                    "▶️ Jugar Carta", 
                    variant="secondary", 
                    interactive=False
                )
                
                # Reglas
                with gr.Accordion("📖 Guía Rápida", open=False):
                    gr.Markdown("""
                    **Puntuación:**
                    - 🍤 **Tempura**: 5 pts por cada 2
                    - 🍣 **Sashimi**: 10 pts por cada 3
                    - 🥟 **Dumpling**: 1,3,6,10,15 pts (1-5)
                    - 🍙 **Maki**: Mayoría gana puntos
                    - 🍱 **Nigiri**: 1-3 pts base
                    - 🟢 **Wasabi**: ×3 al próximo nigiri
                    - 🍮 **Pudding**: No cuenta (1 ronda)
                    
                    **Estrategia:**
                    - Wasabi antes de nigiri valioso
                    - Completar sets (2 tempura, 3 sashimi)
                    - Múltiples dumplings = mejor
                    """)
        
        # Event handlers
        start_button.click(
            fn=start_new_game,
            inputs=[game_state],
            outputs=[output_display, score_display, hand_selector, play_button, game_state]
        )
        
        play_button.click(
            fn=play_turn,
            inputs=[hand_selector, game_state],
            outputs=[output_display, score_display, hand_selector, play_button, game_state]
        )
        
        # Footer
        gr.HTML("""
        <div style='text-align: center; padding: 20px; margin-top: 30px; 
                    color: #999; border-top: 1px solid #eee;'>
            <p style='margin: 0;'>
                🎲 Sushi Go! · PPO (Stable-Baselines3) · Gradio Interface
            </p>
        </div>
        """)
    
    return demo

# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=False)