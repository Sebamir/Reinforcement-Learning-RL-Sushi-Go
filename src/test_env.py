import os
import sys
import torch
import numpy as np

# Aseguramos que reconozca la carpeta 'src'
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from src.env.sushi_env import SushiGoEnv
from src.modelo.dqn_agent import DQNAgent

def test():
    # 1. Configuración
    env = SushiGoEnv(num_players=2)
    action_size = env.action_space.n
    agent = DQNAgent(action_size=action_size)
    
    # 2. Cargar el modelo guardado
    model_path = "best_sushi_model.pth"
    if os.path.exists(model_path):
        agent.model.load_state_dict(torch.load(model_path))
        agent.epsilon = 0.0  # Desactivamos el azar para ver su inteligencia pura
        print(f"--- Modelo '{model_path}' cargado con éxito ---")
    else:
        print("Error: No se encontró el archivo de modelo.")
        return

    # 3. Jugar una partida de exhibición
    state, _ = env.reset()
    total_reward = 0
    done = False
    
    print("\n--- Inicio de la Partida de Prueba ---")
    while not done:
        # El agente elige la mejor acción según lo aprendido
        action = agent.act(state)
        
        # Ejecutar acción
        next_state, reward, done, _, info = env.step(action)
        
        # Mostrar qué está pasando (puedes añadir un print en tu env.render si lo tienes)
        print(f"IA jugó carta ID: {action} | Recompensa obtenida: {reward}")
        
        state = next_state
        total_reward = reward

    print(f"\n--- Fin de la partida. Puntaje Final de la IA: {total_reward} ---")

if __name__ == "__main__":
    test()