import sys
import os
import numpy as np

# Esto asegura que Python encuentre tu carpeta 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.env.sushi_env import SushiGoEnv
from src.engine.sushi_rules import calculate_score

def run_test():
    # Simulamos una partida para 5 jugadores (7 turnos)
    env = SushiGoEnv(num_players=2)
    obs, info = env.reset()
    
    print("Observación inicial shape:", obs.shape)
    
    for step in range(10):
        env.render()
        
        # Acción aleatoria válida
        action = np.random.randint(0, info['hand_size'])
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Acción: {action}, Reward: {reward}")
        
        if terminated:
            print(f"\n¡Ronda terminada! Puntaje final: {calculate_score(env.played_cards)}")
            break

if __name__ == "__main__":
    run_test()