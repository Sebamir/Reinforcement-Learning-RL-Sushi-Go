import torch
import torch.optim as optim
import torch.nn as nn
import random
import numpy as np
from collections import deque
from src.modelo.modelo import SushiDQN

class DQNAgent:
    def __init__(self, action_size, max_hand=10, num_card_types=8):
        self.action_size = action_size
        
        # Parámetros de RL
        self.gamma = 0.95          # Descuento: cuánto valoramos recompensas futuras
        self.epsilon = 1.0         # Exploración inicial (100% azar)
        self.epsilon_min = 0.01    # Exploración mínima
        self.epsilon_decay = 0.995 # Cuánto se reduce el azar en cada episodio
        self.batch_size = 64       # Cuántas experiencias procesar a la vez
        
        # Memoria: guarda las últimas 2000 jugadas
        self.memory = deque(maxlen=2000)
        
        # Modelo y Optimizador
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SushiDQN(action_size, max_hand, num_card_types).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss() # Error Cuadrático Medio para comparar Q-values

    def remember(self, state, action, reward, next_state, done):
        """Guarda la experiencia en la memoria."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Decide qué carta jugar (Exploración vs Explotación)."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values[0]).item()

    def replay(self):
        """Entrena la red con una muestra de la memoria."""
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            
            # Calculamos el Q-target (lo que debería haber predicho)
            target = reward
            if not done:
                # Ecuación de Bellman: recompensa actual + valor futuro descontado
                target = reward + self.gamma * torch.max(self.model(next_state)[0]).item()
            
            # Obtenemos la predicción actual
            target_f = self.model(state)
            
            # Actualizamos solo el valor de la acción tomada
            target_f[0][action] = target
            
            # Paso de entrenamiento
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(state), target_f)
            loss.backward()
            self.optimizer.step()

        # Reducimos la exploración
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay