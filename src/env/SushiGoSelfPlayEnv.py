import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import Counter
from src.engine.sushi_rules import (
    ID_TO_CARD, CARD_MAP, DECK_COMPOSITION,
    calculate_score_simple,
    calculate_score_competitive,
    calculate_all_scores,
    calculate_nigiri_wasabi_points
)

class SushiGoSelfPlayEnv(gym.Env):
    """
    Entorno de Sushi Go con self-play usando SISTEMA DE SCORING UNIFICADO.
    
    Todas las funciones de scoring están centralizadas en sushi_rules.py
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
        
        # Espacio de observación
        num_card_types = len(CARD_MAP)
        obs_size = (
            self.initial_hand_size * num_card_types +  # Mano actual
            num_card_types * self.num_players +        # Cartas jugadas
            3 +                                         # Turno, tamaño, jugador
            self.num_players                           # Wasabi disponible
        )
        
        self.observation_space = spaces.Box(
            low=-5,
            high=max(self.initial_hand_size, 20), 
            shape=(obs_size,), 
            dtype=np.float32
        )
        
        # Estado interno
        self.deck = []
        self.hands = []
        self.played_cards = [[] for _ in range(num_players)]
        self.current_turn = 0
        self.current_player = 0
        
    def _create_deck(self):
        """Crea y mezcla un mazo completo."""
        deck = []
        for card_name, count in DECK_COMPOSITION.items():
            card_id = CARD_MAP[card_name]
            deck.extend([card_id] * count)
        np.random.shuffle(deck)
        return deck
    
    def _deal_hands(self):
        """Reparte las manos iniciales."""
        hands = []
        cards_per_player = self.initial_hand_size
        
        for i in range(self.num_players):
            start_idx = i * cards_per_player
            end_idx = start_idx + cards_per_player
            hand = self.deck[start_idx:end_idx].copy()
            hands.append(hand)
        
        return hands
    
    def _calculate_available_wasabi(self, played_card_ids):
        """
        Calcula wasabi disponibles (esperando nigiri).
        
        wasabi_disponible = wasabi_jugados - nigiri_jugados
        """
        wasabi_count = sum(1 for cid in played_card_ids if cid == CARD_MAP['wasabi'])
        nigiri_count = sum(
            1 for cid in played_card_ids 
            if cid in [CARD_MAP['nigiri_salmon'], CARD_MAP['nigiri_squid'], CARD_MAP['nigiri_egg']]
        )
        return wasabi_count - nigiri_count
    
    def _get_observation(self, player_idx):
        """Genera observación para un jugador."""
        obs = []
        
        # 1. One-hot de la mano
        current_hand = self.hands[player_idx]
        num_card_types = len(CARD_MAP)
        
        for card_idx in range(self.initial_hand_size):
            one_hot = np.zeros(num_card_types, dtype=np.float32)
            if card_idx < len(current_hand):
                one_hot[current_hand[card_idx]] = 1
            obs.extend(one_hot)
        
        # 2. Conteo de cartas jugadas
        for p_idx in range(self.num_players):
            played_counts = np.zeros(num_card_types, dtype=np.float32)
            for card_id in self.played_cards[p_idx]:
                played_counts[card_id] += 1
            obs.extend(played_counts)
        
        # 3. Info contextual
        obs.append(self.current_turn)
        obs.append(len(current_hand))
        obs.append(player_idx)
        
        # 4. Wasabi disponible
        for p_idx in range(self.num_players):
            wasabi_available = self._calculate_available_wasabi(self.played_cards[p_idx])
            obs.append(wasabi_available)
        
        return np.array(obs, dtype=np.float32)
    
    def _pass_hands(self):
        """Pasa las manos a la izquierda."""
        self.hands = [self.hands[-1]] + self.hands[:-1]
    
    def _calculate_immediate_reward(self, player_idx, card_played):
        """
        Calcula recompensa INCREMENTAL usando sistema unificado.
        
        Solo recompensa por sets COMPLETADOS, no puntaje total.
        """
        reward = 0
        cards = [ID_TO_CARD[cid] for cid in self.played_cards[player_idx]]
        card_name = ID_TO_CARD[card_played]
        c = Counter(cards)
        
        # Tempura: +5 al completar pareja
        if card_name == 'tempura' and c['tempura'] % 2 == 0:
            reward += 5
        
        # Sashimi: +10 al completar trío
        if card_name == 'sashimi' and c['sashimi'] % 3 == 0:
            reward += 10
        
        # Dumpling: recompensa incremental
        if card_name == 'dumpling':
            dumpling_points = {1: 1, 2: 3, 3: 6, 4: 10, 5: 15}
            current_points = dumpling_points.get(c['dumpling'], 15)
            previous_points = dumpling_points.get(c['dumpling'] - 1, 0)
            reward += current_points - previous_points
        
        # Nigiri: usar función unificada
        if card_name.startswith('nigiri_'):
            # Calcular puntos ANTES de esta carta
            cards_before = self.played_cards[player_idx][:-1]
            pts_before, _ = calculate_nigiri_wasabi_points(cards_before)
            
            # Calcular puntos DESPUÉS de esta carta
            pts_after, _ = calculate_nigiri_wasabi_points(self.played_cards[player_idx])
            
            # Recompensa = diferencia
            reward += pts_after - pts_before
        
        # Wasabi: sin recompensa inmediata
        # Maki: sin recompensa inmediata (se calcula al final)
        
        return reward
    
    def _calculate_competitive_rewards(self):
        """
        Calcula recompensas competitivas usando sistema unificado.
        """
        # Usar función unificada para calcular scores
        final_scores = calculate_all_scores(self.played_cards)
        
        # Calcular recompensas relativas
        rewards = []
        for i, my_score in enumerate(final_scores):
            other_scores = [s for j, s in enumerate(final_scores) if j != i]
            avg_opponent = np.mean(other_scores) if other_scores else 0
            reward = my_score - avg_opponent
            rewards.append(reward)
        
        return rewards, final_scores
    
    def reset(self, seed=None, options=None):
        """Reinicia el entorno."""
        super().reset(seed=seed)
        
        self.deck = self._create_deck()
        self.hands = self._deal_hands()
        self.played_cards = [[] for _ in range(self.num_players)]
        self.current_turn = 0
        self.current_player = 0
        
        obs = self._get_observation(0)
        info = {
            'player': 0,
            'hand_size': len(self.hands[0]),
            'turn': self.current_turn
        }
        
        return obs, info
    
    def step(self, action):
        """Ejecuta una acción."""
        player = self.current_player
        current_hand = self.hands[player]
        
        # Validar acción
        if action >= len(current_hand) or action < 0:
            obs = self._get_observation(player)
            return obs, -10, True, False, {'error': 'invalid_action', 'player': player}
        
        # Jugar carta
        card_played = current_hand.pop(action)
        self.played_cards[player].append(card_played)
        
        # Recompensa inmediata
        reward = self._calculate_immediate_reward(player, card_played)
        
        # Siguiente jugador
        self.current_player = (self.current_player + 1) % self.num_players
        
        # Pasar manos si completamos ronda
        if self.current_player == 0:
            self._pass_hands()
            self.current_turn += 1
        
        # Verificar fin
        terminated = self.current_turn >= self.max_turns
        
        if terminated:
            # Usar sistema unificado para scores finales
            if self.competitive_reward:
                final_rewards, scores = self._calculate_competitive_rewards()
                reward = final_rewards[player]
            else:
                scores = calculate_all_scores(self.played_cards)
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
        
        next_obs = self._get_observation(self.current_player if not terminated else player)
        
        return next_obs, reward, terminated, False, info
    
    def get_all_observations(self):
        """Obtiene observaciones de todos los jugadores."""
        return [self._get_observation(i) for i in range(self.num_players)]
    
    def render(self):
        """Muestra el estado del juego."""
        print(f"\n=== Turno {self.current_turn}/{self.max_turns} | Jugador {self.current_player} ===")
        
        for i in range(self.num_players):
            wasabi_avail = self._calculate_available_wasabi(self.played_cards[i])
            print(f"\nJugador {i}:")
            print(f"  Mano: {[ID_TO_CARD[c] for c in self.hands[i]]}")
            print(f"  Jugadas: {[ID_TO_CARD[c] for c in self.played_cards[i]]}")
            print(f"  Wasabi disponible: {wasabi_avail}")
            
            if self.current_turn == self.max_turns:
                # Usar sistema unificado
                score = calculate_score_competitive(self.played_cards, i)
                print(f"  Puntaje final: {score}")