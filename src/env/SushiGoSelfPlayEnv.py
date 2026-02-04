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
    - Info explícita de wasabi disponible
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
            num_card_types * self.num_players +        # Cartas jugadas por cada jugador
            3 +                                         # Turno, tamaño mano, jugador actual
            self.num_players                           # Wasabi disponible por jugador
        )
        
        self.observation_space = spaces.Box(
            low=-5,  # Wasabi puede ser negativo si hay más nigiri que wasabi
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
        """Reparte las manos iniciales a todos los jugadores."""
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
        Calcula wasabi disponibles para un jugador.
        
        wasabi_disponible = wasabi_jugados - nigiri_jugados
        
        Puede ser:
        - Positivo: hay wasabi esperando nigiri
        - Cero: todos los wasabi están usados
        - Negativo: hay más nigiri que wasabi (algunos sin triplicar)
        """
        wasabi_count = sum(1 for cid in played_card_ids if cid == CARD_MAP['wasabi'])
        nigiri_count = sum(
            1 for cid in played_card_ids 
            if cid in [CARD_MAP['nigiri_salmon'], CARD_MAP['nigiri_squid'], CARD_MAP['nigiri_egg']]
        )
        return wasabi_count - nigiri_count
    
    def _get_observation(self, player_idx):
        """
        Genera observación para un jugador específico.
        """
        obs = []
        
        # 1. One-hot encoding de la mano del jugador
        current_hand = self.hands[player_idx]
        num_card_types = len(CARD_MAP)
        
        for card_idx in range(self.initial_hand_size):
            one_hot = np.zeros(num_card_types, dtype=np.float32)
            if card_idx < len(current_hand):
                one_hot[current_hand[card_idx]] = 1
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
        obs.append(player_idx)
        
        # 4. Wasabi disponible por jugador
        for p_idx in range(self.num_players):
            wasabi_available = self._calculate_available_wasabi(self.played_cards[p_idx])
            obs.append(wasabi_available)
        
        return np.array(obs, dtype=np.float32)
    
    def _pass_hands(self):
        """Pasa las manos a la izquierda."""
        self.hands = [self.hands[-1]] + self.hands[:-1]
    
    def _calculate_immediate_reward(self, player_idx, card_played):
        """
        Calcula recompensa INCREMENTAL por jugar esta carta.
        
        IMPORTANTE: Solo recompensa por sets COMPLETADOS, no por puntaje total.
        El puntaje total se calcula al final con calculate_score().
        """
        reward = 0
        cards = [ID_TO_CARD[cid] for cid in self.played_cards[player_idx]]
        card_name = ID_TO_CARD[card_played]
        c = Counter(cards)
        
        # Tempura: +5 al completar pareja (2, 4, 6, ...)
        if card_name == 'tempura' and c['tempura'] % 2 == 0:
            reward += 5
        
        # Sashimi: +10 al completar trío (3, 6, 9, ...)
        if card_name == 'sashimi' and c['sashimi'] % 3 == 0:
            reward += 10
        
        # Dumpling: recompensa incremental por cada uno
        if card_name == 'dumpling':
            # Puntos según cantidad: 1→1, 2→3, 3→6, 4→10, 5→15
            dumpling_points = {1: 1, 2: 3, 3: 6, 4: 10, 5: 15}
            current_points = dumpling_points.get(c['dumpling'], 15)
            previous_points = dumpling_points.get(c['dumpling'] - 1, 0)
            reward += current_points - previous_points
        
        # Nigiri: recompensa solo si hay wasabi disponible
        if card_name.startswith('nigiri_'):
            wasabi_available = self._calculate_available_wasabi(
                self.played_cards[player_idx][:-1]  # Antes de jugar este nigiri
            )
            
            if card_name == 'nigiri_salmon':
                base_value = 3
            elif card_name == 'nigiri_squid':
                base_value = 2
            elif card_name == 'nigiri_egg':
                base_value = 1
            
            # Si había wasabi disponible, este nigiri se triplica
            if wasabi_available > 0:
                reward += base_value * 3
            else:
                reward += base_value
        
        # Wasabi: sin recompensa inmediata (es inversión para el futuro)
        # Maki: sin recompensa inmediata (depende de oponentes al final)
        
        return reward
    
    def _calculate_competitive_rewards(self):
        """
        Calcula recompensas competitivas al final.
        Cada jugador recibe: su_puntaje - promedio_de_otros
        """
        scores, total_maki = zip(*[calculate_score(cards) for cards in self.played_cards])
        final_scores = list(scores)
        maki_counts = list(total_maki)
        max_maki = max(maki_counts)
        maki_winners = [i for i, m in enumerate(maki_counts) if m == max_maki]

        if len(maki_winners) == 1:
            final_scores[maki_winners[0]] += 6  # Premio completo
        else:
            split_prize = 6 / len(maki_winners)
            for w in maki_winners:
                final_scores[w] += split_prize

        rewards = []
        
        for i, my_score in enumerate(final_scores):
            other_scores = [s for j, s in enumerate(final_scores) if j != i]
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
        
        # Calcular recompensa inmediata (solo para sets completados)
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
            wasabi_avail = self._calculate_available_wasabi(self.played_cards[i])
            print(f"\nJugador {i}:")
            print(f"  Mano: {[ID_TO_CARD[c] for c in self.hands[i]]}")
            print(f"  Jugadas: {[ID_TO_CARD[c] for c in self.played_cards[i]]}")
            print(f"  Wasabi disponible: {wasabi_avail}")
            if self.current_turn == self.max_turns:
                print(f"  Puntaje: {calculate_score(self.played_cards[i])}")


# ===== TEST DE VALIDACIÓN =====

def test_environment():
    """Prueba el entorno con casos específicos."""
    print("="*60)
    print("TEST DEL ENTORNO")
    print("="*60)
    
    env = SushiGoSelfPlayEnv(num_players=2, competitive_reward=True)
    
    # Test 1: Verificar tamaño de observación
    print("\n1. Verificando espacios...")
    obs, info = env.reset()
    print(f"   Observation shape: {obs.shape}")
    print(f"   Expected: ({env.observation_space.shape[0]},)")
    assert obs.shape == env.observation_space.shape, "❌ Tamaño de observación incorrecto"
    print("   ✅ Espacios correctos")
    
    # Test 2: Verificar wasabi info
    print("\n2. Verificando wasabi info...")
    # Simular jugar un wasabi
    env.played_cards[0] = [CARD_MAP['wasabi']]
    obs = env._get_observation(0)
    wasabi_info_start = (
        env.initial_hand_size * len(CARD_MAP) +
        len(CARD_MAP) * env.num_players +
        3
    )
    wasabi_value = obs[wasabi_info_start]
    print(f"   Wasabi disponible (debería ser 1): {wasabi_value}")
    assert wasabi_value == 1, "❌ Wasabi info incorrecta"
    print("   ✅ Wasabi info correcta")
    
    # Test 3: Verificar recompensa incremental
    print("\n3. Verificando recompensas...")
    env.reset()
    
    # Jugar 2 tempuras
    env.played_cards[0] = [CARD_MAP['tempura']]
    reward1 = env._calculate_immediate_reward(0, CARD_MAP['tempura'])
    print(f"   Primera tempura: {reward1} pts (debería ser 0)")
    
    env.played_cards[0] = [CARD_MAP['tempura'], CARD_MAP['tempura']]
    reward2 = env._calculate_immediate_reward(0, CARD_MAP['tempura'])
    print(f"   Segunda tempura: {reward2} pts (debería ser 5)")
    
    assert reward1 == 0, "❌ Primera tempura debería dar 0"
    assert reward2 == 5, "❌ Segunda tempura debería dar 5"
    print("   ✅ Recompensas correctas")
    
    # Test 4: Verificar wasabi + nigiri
    print("\n4. Verificando wasabi + nigiri...")
    env.reset()
    
    # Jugar wasabi primero
    env.played_cards[0] = [CARD_MAP['wasabi']]
    reward_wasabi = env._calculate_immediate_reward(0, CARD_MAP['wasabi'])
    print(f"   Wasabi solo: {reward_wasabi} pts (debería ser 0)")
    
    # Luego jugar salmon
    env.played_cards[0] = [CARD_MAP['wasabi'], CARD_MAP['nigiri_salmon']]
    reward_salmon = env._calculate_immediate_reward(0, CARD_MAP['nigiri_salmon'])
    print(f"   Salmon con wasabi: {reward_salmon} pts (debería ser 9)")
    
    assert reward_wasabi == 0, "❌ Wasabi solo debería dar 0"
    assert reward_salmon == 9, "❌ Salmon con wasabi debería dar 9"
    print("   ✅ Wasabi + Nigiri correcto")
    
    print("\n" + "="*60)
    print("✅ TODOS LOS TESTS PASARON")
    print("="*60)


if __name__ == "__main__":
    test_environment()