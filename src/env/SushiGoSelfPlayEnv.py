import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import Counter
from src.engine.sushi_rules import ID_TO_CARD, CARD_MAP, DECK_COMPOSITION, calculate_score

class SushiGoSelfPlayEnv(gym.Env):
    """
    Entorno de Sushi Go con self-play: múltiples agentes de RL compiten entre sí.
    
    Características:
    - Todos los jugadores son agentes entrenables
    - Observaciones relativas (cada agente ve desde su perspectiva)
    - Recompensas competitivas (diferencia de puntaje)
    - Soporte para entrenamiento simultáneo o alternado
    """
    
    def __init__(self, num_players=2, competitive_reward=True):
        super(SushiGoSelfPlayEnv, self).__init__()
        
        self.num_players = num_players
        self.competitive_reward = competitive_reward
        self.hand_size_map = {2: 10, 3: 9, 4: 8, 5: 7}
        self.initial_hand_size = self.hand_size_map.get(num_players, 10)
        self.max_turns = self.initial_hand_size
        
        # Espacio de acciones
        self.action_space = spaces.Discrete(self.initial_hand_size)
        
        # Espacio de observación (igual para todos los agentes)
        num_card_types = len(CARD_MAP)
        obs_size = (
            self.initial_hand_size * num_card_types +  # Mano actual
            num_card_types * self.num_players +        # Cartas jugadas por cada jugador
            3                                           # Turno, tamaño mano, jugador actual
        )
        
        self.observation_space = spaces.Box(
            low=0, 
            high=max(self.initial_hand_size, 20), 
            shape=(obs_size,), 
            dtype=np.float32
        )
        
        # Estado interno
        self.deck = []
        self.hands = []
        self.played_cards = [[] for _ in range(num_players)]  # Cartas por jugador
        self.current_turn = 0
        self.current_player = 0  # Jugador actual en turno
        
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
    
    def _get_observation(self, player_idx):
        """
        Genera observación para un jugador específico.
        Cada agente ve el juego desde su perspectiva.
        """
        obs = []
        
        # 1. One-hot encoding de la mano del jugador
        current_hand = self.hands[player_idx]
        num_card_types = len(CARD_MAP)
        
        for card_id in range(self.initial_hand_size):
            one_hot = np.zeros(num_card_types, dtype=np.float32)
            if card_id < len(current_hand):
                one_hot[current_hand[card_id]] = 1
            obs.extend(one_hot)
        
        # 2. Conteo de cartas jugadas por cada jugador
        for p_idx in range(self.num_players):
            played_counts = np.zeros(num_card_types, dtype=np.float32)
            for card_id in self.played_cards[p_idx]:
                played_counts[card_id] += 1
            obs.extend(played_counts)
        
        # 3. Info contextual
        obs.append(self.current_turn)
        obs.append(len(current_hand))
        obs.append(player_idx)  # Identidad del jugador
        
        return np.array(obs, dtype=np.float32)
    
    def _pass_hands(self):
        """Pasa las manos a la izquierda."""
        self.hands = [self.hands[-1]] + self.hands[:-1]
    
    def _calculate_immediate_reward(self, player_idx, card_played):
        """Calcula recompensa inmediata por completar sets."""
        reward = 0
        c = Counter([ID_TO_CARD[cid] for cid in self.played_cards[player_idx]])
        
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
    
    def _calculate_competitive_rewards(self):
        """
        Calcula recompensas competitivas al final.
        Cada jugador recibe: su_puntaje - promedio_de_otros
        """
        scores = [calculate_score(cards) for cards in self.played_cards]
        rewards = []
        
        for i, my_score in enumerate(scores):
            other_scores = [s for j, s in enumerate(scores) if j != i]
            avg_opponent = np.mean(other_scores) if other_scores else 0
            reward = my_score - avg_opponent
            rewards.append(reward)
        
        return rewards, scores
    
    def reset(self, seed=None, options=None):
        """Reinicia el entorno."""
        super().reset(seed=seed)
        
        self.deck = self._create_deck()
        self.hands = self._deal_hands()
        self.played_cards = [[] for _ in range(self.num_players)]
        self.current_turn = 0
        self.current_player = 0
        
        # Devuelve observación del primer jugador
        obs = self._get_observation(0)
        info = {
            'player': 0,
            'hand_size': len(self.hands[0]),
            'turn': self.current_turn
        }
        
        return obs, info
    
    def step(self, action):
        """
        Ejecuta una acción para el jugador actual.
        Devuelve la observación del SIGUIENTE jugador.
        """
        player = self.current_player
        current_hand = self.hands[player]
        
        # Validar acción
        if action >= len(current_hand) or action < 0:
            obs = self._get_observation(player)
            return obs, -10, True, False, {'error': 'invalid_action', 'player': player}
        
        # Jugar la carta
        card_played = current_hand.pop(action)
        self.played_cards[player].append(card_played)
        
        # Calcular recompensa inmediata
        reward = self._calculate_immediate_reward(player, card_played)
        
        # Avanzar al siguiente jugador
        self.current_player = (self.current_player + 1) % self.num_players
        
        # Si completamos una ronda de todos los jugadores
        if self.current_player == 0:
            self._pass_hands()
            self.current_turn += 1
        
        # Verificar si terminó el juego
        terminated = self.current_turn >= self.max_turns
        
        if terminated:
            # Calcular recompensas finales
            if self.competitive_reward:
                final_rewards, scores = self._calculate_competitive_rewards()
                reward = final_rewards[player]
            else:
                scores = [calculate_score(cards) for cards in self.played_cards]
                reward = scores[player]
            
            info = {
                'player': player,
                'final_scores': scores,
                'winner': np.argmax(scores)
            }
        else:
            info = {
                'player': player,
                'next_player': self.current_player,
                'hand_size': len(self.hands[self.current_player]),
                'turn': self.current_turn,
                'last_card': ID_TO_CARD[card_played]
            }
        
        # Observación para el SIGUIENTE jugador (o el mismo si terminó)
        next_obs = self._get_observation(self.current_player if not terminated else player)
        
        return next_obs, reward, terminated, False, info
    
    def get_all_observations(self):
        """Obtiene observaciones para todos los jugadores simultáneamente."""
        return [self._get_observation(i) for i in range(self.num_players)]
    
    def render(self):
        """Muestra el estado actual del juego."""
        print(f"\n=== Turno {self.current_turn}/{self.max_turns} | Jugador {self.current_player} ===")
        
        for i in range(self.num_players):
            print(f"\nJugador {i}:")
            print(f"  Mano: {[ID_TO_CARD[c] for c in self.hands[i]]}")
            print(f"  Jugadas: {[ID_TO_CARD[c] for c in self.played_cards[i]]}")
            if self.current_turn == self.max_turns:
                print(f"  Puntaje: {calculate_score(self.played_cards[i])}")