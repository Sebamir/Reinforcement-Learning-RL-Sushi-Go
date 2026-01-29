import gradio as gr
import numpy as np
from stable_baselines3 import PPO
from src.env.SushiGoSelfPlayEnv import SushiGoSelfPlayEnv
from src.engine.sushi_rules import ID_TO_CARD, CARD_MAP

# ============================================================================
# CONFIGURACIÓN INICIAL
# ============================================================================

model_path = "experiments/selfplay_2p_basic/best_model.zip"
model = PPO.load(model_path)

# Estado global del juego (usando Gradio State para mantener persistencia)
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
# FUNCIONES AUXILIARES
# ============================================================================

def get_human_hand(env):
    """Obtiene la mano actual del jugador humano (jugador 0)."""
    return [ID_TO_CARD[card_id] for card_id in env.hands[0]]

def get_ai_hand(env):
    """Obtiene la mano actual de la IA (jugador 1)."""
    return [ID_TO_CARD[card_id] for card_id in env.hands[1]]

def get_played_cards(env):
    """Obtiene las cartas jugadas por ambos jugadores."""
    human_played = [ID_TO_CARD[card_id] for card_id in env.played_cards[0]]
    ai_played = [ID_TO_CARD[card_id] for card_id in env.played_cards[1]]
    return human_played, ai_played

def format_game_state(env):
    """Formatea el estado actual del juego para mostrar."""
    human_hand = get_human_hand(env)
    human_played, ai_played = get_played_cards(env)
    
    state_text = f"## 🎮 Estado del Juego\n\n"
    state_text += f"**Turno:** {env.current_turn + 1}/{env.max_turns}\n\n"
    
    state_text += f"### 🧑 Tu área de juego:\n"
    state_text += f"Cartas jugadas: {', '.join(human_played) if human_played else 'Ninguna'}\n\n"
    
    state_text += f"### 🤖 Área de la IA:\n"
    state_text += f"Cartas jugadas: {', '.join(ai_played) if ai_played else 'Ninguna'}\n\n"
    
    return state_text

def format_final_scores(env):
    """Formatea los puntajes finales."""
    from src.engine.sushi_rules import calculate_score
    
    human_score = calculate_score(env.played_cards[0])
    ai_score = calculate_score(env.played_cards[1])
    
    result = f"\n\n## 🏆 ¡JUEGO TERMINADO!\n\n"
    result += f"### Puntajes Finales:\n"
    result += f"- **Tu puntaje:** {human_score} puntos\n"
    result += f"- **Puntaje IA:** {ai_score} puntos\n\n"
    
    if human_score > ai_score:
        result += f"### 🎉 ¡GANASTE! (+{human_score - ai_score} puntos)\n"
    elif ai_score > human_score:
        result += f"### 😢 La IA ganó (+{ai_score - human_score} puntos)\n"
    else:
        result += f"### 🤝 ¡EMPATE!\n"
    
    # Desglose detallado
    result += f"\n### 📊 Desglose de puntos:\n"
    result += f"**Tus cartas:** {', '.join([ID_TO_CARD[c] for c in env.played_cards[0]])}\n"
    result += f"**Cartas IA:** {', '.join([ID_TO_CARD[c] for c in env.played_cards[1]])}\n"
    
    return result

# ============================================================================
# FUNCIONES PRINCIPALES DEL JUEGO
# ============================================================================

def start_new_game(game_state):
    """Inicia una nueva partida."""
    game_state.reset()
    
    state_display = format_game_state(game_state.env)
    human_hand = get_human_hand(game_state.env)
    
    message = "## 🍱 ¡Nueva partida iniciada!\n\n"
    message += "Eres el **Jugador 0** (juegas primero). Selecciona una carta de tu mano.\n\n"
    message += state_display
    
    return (
        message,
        gr.update(choices=human_hand, value=None, interactive=True),
        gr.update(interactive=True),
        game_state
    )

def play_turn(selected_card, game_state):
    """Ejecuta un turno completo: humano juega, luego IA juega."""
    
    if not game_state.game_started:
        return (
            "⚠️ Primero debes iniciar una nueva partida.",
            gr.update(),
            gr.update(interactive=False),
            game_state
        )
    
    if selected_card is None:
        return (
            "⚠️ Por favor selecciona una carta de tu mano.",
            gr.update(),
            gr.update(interactive=True),
            game_state
        )
    
    env = game_state.env
    result_text = ""
    
    # ========================================================================
    # TURNO DEL HUMANO (Jugador 0)
    # ========================================================================
    
    human_hand = env.hands[0]
    human_hand_names = [ID_TO_CARD[card_id] for card_id in human_hand]
    
    try:
        action = human_hand_names.index(selected_card)
    except ValueError:
        return (
            f"❌ Error: '{selected_card}' no está en tu mano.",
            gr.update(),
            gr.update(interactive=True),
            game_state
        )
    
    # Ejecutar acción del humano
    obs, reward, terminated, _, info = env.step(action)
    
    result_text += f"### 🎴 Tu Jugada\n"
    result_text += f"**Jugaste:** {selected_card}\n"
    if 'last_card' in info:
        result_text += f"Recompensa inmediata: +{reward:.1f}\n\n"
    
    # Guardar en historial
    game_state.game_history.append({
        'turn': env.current_turn,
        'player': 'Humano',
        'card': selected_card,
        'reward': reward
    })
    
    # ========================================================================
    # VERIFICAR SI EL JUEGO TERMINÓ
    # ========================================================================
    
    if terminated:
        result_text += format_final_scores(env)
        game_state.game_started = False
        
        return (
            result_text,
            gr.update(choices=[], interactive=False),
            gr.update(interactive=False),
            game_state
        )
    
    # ========================================================================
    # TURNO DE LA IA (Jugador 1)
    # ========================================================================
    
    # La IA predice su mejor acción
    ai_action, _ = model.predict(obs, deterministic=True)
    ai_hand = env.hands[1]
    ai_card_played = ID_TO_CARD[ai_hand[int(ai_action)]]
    
    # Ejecutar acción de la IA
    obs, ai_reward, terminated, _, info = env.step(int(ai_action))
    
    result_text += f"### 🤖 Jugada de la IA\n"
    result_text += f"**La IA jugó:** {ai_card_played}\n"
    result_text += f"Recompensa IA: +{ai_reward:.1f}\n\n"
    
    game_state.game_history.append({
        'turn': env.current_turn,
        'player': 'IA',
        'card': ai_card_played,
        'reward': ai_reward
    })
    
    # ========================================================================
    # VERIFICAR DE NUEVO SI EL JUEGO TERMINÓ
    # ========================================================================
    
    if terminated:
        result_text += format_final_scores(env)
        game_state.game_started = False
        
        return (
            result_text,
            gr.update(choices=[], interactive=False),
            gr.update(interactive=False),
            game_state
        )
    
    # ========================================================================
    # ACTUALIZAR ESTADO PARA SIGUIENTE TURNO
    # ========================================================================
    
    result_text += "---\n\n"
    result_text += format_game_state(env)
    
    # Actualizar mano del humano para la siguiente jugada
    new_human_hand = get_human_hand(env)
    
    return (
        result_text,
        gr.update(choices=new_human_hand, value=None, interactive=True),
        gr.update(interactive=True),
        game_state
    )

# ============================================================================
# INTERFAZ DE GRADIO
# ============================================================================

def create_interface():
    """Crea la interfaz de Gradio."""
    
    with gr.Blocks(title="🍱 Sushi Go! - Humano vs IA", theme=gr.themes.Soft()) as demo:
        
        # Estado del juego (persistente entre interacciones)
        game_state = gr.State(GameState())
        
        # Header
        gr.Markdown("""
        # 🍱 Sushi Go! - Humano vs IA
        
        Juega una partida completa contra una IA entrenada con Reinforcement Learning.
        
        **Reglas básicas:**
        - Cada turno, elige una carta de tu mano
        - Después de cada ronda, las manos se pasan al siguiente jugador
        - El juego termina después de 10 turnos (mano completa)
        - Gana quien tenga más puntos al final
        """)
        
        # Área principal
        with gr.Row():
            with gr.Column(scale=2):
                output_display = gr.Markdown("👋 Presiona '🎮 Nueva Partida' para comenzar")
                
            with gr.Column(scale=1):
                gr.Markdown("### 🎯 Controles")
                
                start_button = gr.Button("🎮 Nueva Partida", variant="primary", size="lg")
                
                gr.Markdown("### 🃏 Tu Mano")
                hand_selector = gr.Radio(
                    choices=[],
                    label="Selecciona una carta",
                    interactive=False
                )
                
                play_button = gr.Button("▶️ Jugar Carta", variant="secondary", interactive=False)
                
                gr.Markdown("""
                ### 💡 Consejos
                - **Nigiri**: Más puntos con wasabi
                - **Tempura**: 5 pts por pareja
                - **Sashimi**: 10 pts por trío  
                - **Dumpling**: Más dumplings = más puntos
                - **Maki Rolls**: Compite por mayoría
                - **Pudding**: ¡Se cuenta al final!
                """)
        
        # Event handlers
        start_button.click(
            fn=start_new_game,
            inputs=[game_state],
            outputs=[output_display, hand_selector, play_button, game_state]
        )
        
        play_button.click(
            fn=play_turn,
            inputs=[hand_selector, game_state],
            outputs=[output_display, hand_selector, play_button, game_state]
        )
        
        # Footer
        gr.Markdown("""
        ---
        🎲 **Sushi Go!** - Creado con Gradio + Stable-Baselines3 (PPO)
        """)
    
    return demo

# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=False)