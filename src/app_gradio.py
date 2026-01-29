import gradio as gr
import numpy as np
from stable_baselines3 import PPO
from src.env.SushiGoSelfPlayEnv import SushiGoSelfPlayEnv
from src.engine.sushi_rules import ID_TO_CARD, CARD_MAP

# 1. Cargar el modelo y el entorno
# Cambia la ruta por la de tu mejor modelo (ej: experiments/selfplay_2p_basic/best_model.zip)
model_path = "experiments/selfplay_2p_basic/best_model.zip"
model = PPO.load(model_path)
env = SushiGoSelfPlayEnv(num_players=2)

def play_round(human_choice):
    # Reiniciamos o recuperamos el estado actual si fuera una partida larga,
    # pero para la demo, simularemos un turno basado en una observación fresca.
    obs, _ = env.reset()
    
    # Mapeo de nombres a IDs (ajusta según tu ID_TO_CARD)
    card_map = {v: k for k, v in env.ID_TO_CARD.items()}
    human_action = card_map[human_choice]
    
    # La IA predice su jugada
    ai_action, _ = model.predict(obs, deterministic=True)
    
    # Ejecutamos el turno en el entorno
    # (Aquí simplificamos: el humano es Jugador 0, la IA es Jugador 1)
    obs, reward, done, _, info = env.step(human_action)
    
    ai_card_name = env.ID_TO_CARD[int(ai_action)]
    
    res_text = f"### 🍣 Resultado del Turno\n"
    res_text += f"**Tú elegiste:** {human_choice}\n"
    res_text += f"**La IA eligió:** {ai_card_name}\n\n"
    
    if done:
        res_text += f"## 🏆 ¡Partida terminada!\n"
        res_text += f"Puntajes finales: {info['final_scores']}"
    
    return res_text

# 3. Interfaz de Gradio
with gr.Blocks(title="Sushi Go AI") as demo:
    gr.Markdown("# 🍱 Sushi Go: Humano vs IA")
    gr.Markdown("Selecciona una carta de tu mano y mira qué decide la inteligencia artificial.")
    
    with gr.Row():
        with gr.Column():
            options = [env.ID_TO_CARD[i] for i in range(len(env.ID_TO_CARD))]
            input_card = gr.Radio(choices=options, label="Tu Mano")
            btn = gr.Button("Jugar Carta", variant="primary")
            
        with gr.Column():
            output = gr.Markdown("Esperando jugada...")

    btn.click(fn=play_round, inputs=input_card, outputs=output)

if __name__ == "__main__":
    demo.launch()