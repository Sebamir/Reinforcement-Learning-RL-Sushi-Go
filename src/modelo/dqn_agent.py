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
        if len(self.memory) < self.batch_size:
            return

        # 1. Tomamos una muestra aleatoria
        minibatch = random.sample(self.memory, self.batch_size)
        
        # 2. Convertimos todo a Tensors de golpe (más rápido)
        states = torch.FloatTensor(np.array([x[0] for x in minibatch])).to(self.device)
        actions = torch.LongTensor([x[1] for x in minibatch]).to(self.device)
        rewards = torch.FloatTensor([x[2] for x in minibatch]).to(self.device)
        next_states = torch.FloatTensor(np.array([x[3] for x in minibatch])).to(self.device)
        dones = torch.FloatTensor([x[4] for x in minibatch]).to(self.device)

        # 3. Predicción actual y futura
        curr_q_values = self.model(states)
        next_q_values = self.model(next_states)

        # 4. Calculamos el Target usando Bellman vectorizado
        max_next_q = torch.max(next_q_values, dim=1)[0]
        targets = rewards + (self.gamma * max_next_q * (1 - dones))

        # 5. Actualizamos solo los Q-values de las acciones que tomamos
        target_f = curr_q_values.clone()
        for i in range(self.batch_size):
            target_f[i][actions[i]] = targets[i]

        # 6. Entrenamiento de la red
        self.optimizer.zero_grad()
        loss = self.criterion(curr_q_values, target_f)
        loss.backward()
        self.optimizer.step()

        # Reducir exploración
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay