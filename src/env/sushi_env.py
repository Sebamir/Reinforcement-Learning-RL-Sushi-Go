import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import Counter
from src.engine.sushi_rules import ID_TO_CARD, CARD_MAP, DECK_COMPOSITION, calculate_score

class SushiGoEnv(gym.Env):
    """
    Entorno de Gymnasium para Sushi Go (versión simplificada de 1 ronda).
    
    Mejoras implementadas:
    - Mazo real que se reparte y pasa entre jugadores
    - Validación de acciones (solo cartas en mano)
    - Observación estructurada (one-hot encoding)
    - Recompensas intermedias (shaping)
    - Manejo correcto del paso de manos
    """
    
    def __init__(self, num_players=2):
        super(SushiGoEnv, self).__init__()
        
        self.num_players = num_players
        self.hand_size_map = {2: 10, 3: 9, 4: 8, 5: 7}
        self.initial_hand_size = self.hand_size_map.get(num_players, 10)
        self.max_turns = self.initial_hand_size
        
        # Espacio de acciones: índices de la mano (0 a hand_size-1)
        self.action_space = spaces.Discrete(self.initial_hand_size)
        
        # Observación estructurada:
        # - Mano actual: one-hot de cada carta (hand_size x num_cards)
        # - Cartas jugadas: conteo de cada tipo
        # - Turno actual
        # - Tamaño de mano actual
        num_card_types = len(CARD_MAP)
        obs_size = (
            self.initial_hand_size * num_card_types +  # Mano (one-hot)
            num_card_types +                            # Cartas jugadas (conteo)
            2                                           # Turno y tamaño mano
        )
        
        self.observation_space = spaces.Box(
            low=0, 
            high=max(self.initial_hand_size, 20), 
            shape=(obs_size,), 
            dtype=np.float32
        )
        
        # Estado interno
        self.deck = []
        self.hands = []  # Manos de todos los jugadores
        self.played_cards = []  # Cartas jugadas por el agente
        self.current_turn = 0
        self.agent_idx = 0  # El agente siempre es el jugador 0
        
    def _create_deck(self):
        """Crea y mezcla un mazo completo."""
        deck = []
        for card_name, count in DECK_COMPOSITION.items():
            card_id = CARD_MAP[card_name]
            deck.extend([card_id] * count)
        np.random.shuffle(deck)
        return deck
    
    def _deal_hands(self):
        """Reparte las manos iniciales a todos los jugadores."""
        hands = []
        cards_per_player = self.initial_hand_size
        
        for i in range(self.num_players):
            start_idx = i * cards_per_player
            end_idx = start_idx + cards_per_player
            hand = self.deck[start_idx:end_idx].copy()
            hands.append(hand)
        
        return hands
    
    def _get_observation(self):
        """
        Genera la observación actual para el agente.
        Estructura:
        - One-hot de cada carta en mano
        - Conteo de cartas jugadas
        - Turno actual
        - Tamaño de mano actual
        """
        obs = []
        
        # 1. One-hot encoding de la mano actual
        current_hand = self.hands[self.agent_idx]
        num_card_types = len(CARD_MAP)
        
        for card_id in range(self.initial_hand_size):
            one_hot = np.zeros(num_card_types, dtype=np.float32)
            if card_id < len(current_hand):
                one_hot[current_hand[card_id]] = 1
            obs.extend(one_hot)
        
        # 2. Conteo de cartas jugadas
        played_counts = np.zeros(num_card_types, dtype=np.float32)
        for card_id in self.played_cards:
            played_counts[card_id] += 1
        obs.extend(played_counts)
        
        # 3. Info contextual
        obs.append(self.current_turn)
        obs.append(len(current_hand))
        
        return np.array(obs, dtype=np.float32)
    
    def _pass_hands(self):
        """Pasa las manos a la izquierda (rotación)."""
        self.hands = [self.hands[-1]] + self.hands[:-1]
    
    def _calculate_reward(self, card_played):
        """
        Calcula recompensa con reward shaping.
        - Recompensa inmediata por completar sets
        - Recompensa final por puntaje total
        """
        reward = 0
        
        # Recompensa inmediata por completar sets
        c = Counter([ID_TO_CARD[cid] for cid in self.played_cards])
        
        # Tempura: +5 al completar pareja
        if c['tempura'] % 2 == 0 and c['tempura'] > 0:
            if ID_TO_CARD[card_played] == 'tempura':
                reward += 5
        
        # Sashimi: +10 al completar trío
        if c['sashimi'] % 3 == 0 and c['sashimi'] > 0:
            if ID_TO_CARD[card_played] == 'sashimi':
                reward += 10
        
        # Dumpling: recompensa incremental
        if ID_TO_CARD[card_played] == 'dumpling':
            dumpling_rewards = [0, 1, 2, 3, 4, 5]
            if c['dumpling'] <= 5:
                reward += dumpling_rewards[c['dumpling'] - 1]
        
        return reward
    
    def reset(self, seed=None, options=None):
        """Reinicia el entorno para un nuevo episodio."""
        super().reset(seed=seed)
        
        # Crear y repartir mazo
        self.deck = self._create_deck()
        self.hands = self._deal_hands()
        self.played_cards = []
        self.current_turn = 0
        
        obs = self._get_observation()
        info = {
            'hand_size': len(self.hands[self.agent_idx]),
            'turn': self.current_turn
        }
        
        return obs, info
    
    def step(self, action):
        """
        Ejecuta una acción (elegir carta de la mano).
        
        Args:
            action: Índice de la carta en la mano actual (0 a len(hand)-1)
        """
        current_hand = self.hands[self.agent_idx]
        
        # Validar acción
        if action >= len(current_hand) or action < 0:
            # Acción inválida: penalización y terminar
            obs = self._get_observation()
            return obs, -10, True, False, {'error': 'invalid_action'}
        
        # Jugar la carta
        card_played = current_hand.pop(action)
        self.played_cards.append(card_played)
        
        # Calcular recompensa inmediata
        reward = self._calculate_reward(card_played)
        
        # Los otros jugadores también juegan (simulación simple)
        for i in range(1, self.num_players):
            if len(self.hands[i]) > 0:
                # Estrategia aleatoria para oponentes
                opponent_card = self.hands[i].pop(np.random.randint(len(self.hands[i])))
        
        # Pasar manos
        self._pass_hands()
        
        # Avanzar turno
        self.current_turn += 1
        
        # Verificar si terminó la ronda
        terminated = self.current_turn >= self.max_turns
        
        if terminated:
            # Recompensa final: puntaje total
            final_score = calculate_score(self.played_cards)
            reward += final_score
        
        obs = self._get_observation()
        info = {
            'hand_size': len(self.hands[self.agent_idx]),
            'turn': self.current_turn,
            'cards_played': len(self.played_cards),
            'last_card': ID_TO_CARD[card_played]
        }
        
        return obs, reward, terminated, False, info
    
    def render(self):
        """Muestra el estado actual del juego."""
        print(f"\n=== Turno {self.current_turn}/{self.max_turns} ===")
        print(f"Mano actual: {[ID_TO_CARD[c] for c in self.hands[self.agent_idx]]}")
        print(f"Cartas jugadas: {[ID_TO_CARD[c] for c in self.played_cards]}")
        print(f"Puntaje actual: {calculate_score(self.played_cards)}")
