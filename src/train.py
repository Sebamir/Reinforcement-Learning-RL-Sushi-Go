import numpy as np
import os
import torch
from src.env.sushi_env import SushiGoEnv
from src.modelo.dqn_agent import DQNAgent
import matplotlib.pyplot as plt

def train():
    # 1. Inicializar Entorno y Agente
    env = SushiGoEnv(num_players=2) # Empezamos con 2 jugadores para aprender rápido
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(action_size=action_size)
    
    episodes = 500  # Número de partidas de práctica
    scores = []      # Para graficar el progreso
    best_score = -float('inf')

    print(f"Iniciando entrenamiento: {episodes} episodios...")

    for e in range(episodes):
        state, info = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # El agente decide qué carta jugar
            action = agent.act(state)
            
            # El entorno aplica la jugada
            next_state, reward, done, truncated, info = env.step(action)
            
            # El agente guarda la experiencia
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            total_reward = reward # En tu lógica, reward final es el puntaje total
            
            # El agente estudia (Replay)
            agent.replay()
            
        scores.append(total_reward)
        
        # Guardar el mejor modelo
        if total_reward > best_score:
            best_score = total_reward
            torch.save(agent.model.state_dict(), "best_sushi_model.pth")

        # Reporte cada 10 episodios
        if e % 10 == 0:
            avg_score = np.mean(scores[-10:])
            print(f"Episodio: {e}/{episodes} | Score Promedio: {avg_score:.1f} | Epsilon: {agent.epsilon:.2f}")

    return scores

if __name__ == "__main__":
    history = train()
    
    # 1. Crear la carpeta 'Results' si no existe
    results_dir = "Results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Carpeta '{results_dir}' creada.")

    # 2. Graficar resultados
    plt.figure(figsize=(10, 6)) # Opcional: define un tamaño más legible
    plt.plot(history)
    plt.title("Evolución del Aprendizaje (Sushi Go)")
    plt.xlabel("Episodio")
    plt.ylabel("Puntaje Total")
    
    # 3. Guardar el gráfico antes de mostrarlo
    # Usamos una ruta relativa que funcionará en cualquier sistema
    save_path = os.path.join(results_dir, "training_curve.png")
    plt.savefig(save_path)
    print(f"Gráfico guardado en: {save_path}")
    
    # 4. Mostrarlo (opcional)
    plt.show()