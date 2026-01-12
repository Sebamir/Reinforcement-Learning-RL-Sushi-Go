import torch
import torch.nn as nn
import torch.nn.functional as F

from src.env.sushi_env import ID_TO_CARD

class SushiDQN(nn.Module):
    """
    Red neuronal mejorada para Sushi Go que aprovecha la estructura
    de la observación y usa técnicas modernas de deep learning.
    """
    def __init__(self, action_size, max_hand=10, num_card_types=8, hidden_size=128):
        super(SushiDQN, self).__init__()
        

        self.hand_encoding_size = max_hand* num_card_types # 80
        self.played_cards_size = num_card_types # 8 (uno por cada tipo de carta)
        self.context_size = 2 # Turno + tamaño mano actual
     
        
        # Sub-red para procesar la mano (one-hot encoding)
        self.hand_net = nn.Sequential(
            nn.Linear(self.hand_encoding_size, hidden_size),
            nn.LayerNorm(hidden_size),  # MEJORA: Normalización
            nn.ReLU(),
            nn.Dropout(0.2),            # MEJORA: Regularización
            nn.Linear(hidden_size, hidden_size // 2)
        )
        
        # Sub-red para procesar cartas jugadas
        self.played_net = nn.Sequential(
            nn.Linear(self.played_cards_size, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # Sub-red para contexto
        self.context_net = nn.Sequential(
            nn.Linear(self.context_size, 16),
            nn.ReLU()
        )
        
        # MEJORA 2: Red combinada más profunda
        combined_size = (hidden_size // 2) + 32 + 16  # 128 + 32 + 16 = 112
        
        self.combined_net = nn.Sequential(
            nn.Linear(combined_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, action_size)
        )
        
    def forward(self, x):
        # Separar la observación en componentes
        hand_encoding = x[:, :self.hand_encoding_size]
        played_cards = x[:, self.hand_encoding_size:self.hand_encoding_size + self.played_cards_size]
        context = x[:, -self.context_size:]
        
        # Procesar cada componente
        hand_features = self.hand_net(hand_encoding)
        played_features = self.played_net(played_cards)
        context_features = self.context_net(context)
        
        # Combinar features
        combined = torch.cat([hand_features, played_features, context_features], dim=1)
        
        # Calcular Q-values
        q_values = self.combined_net(combined)
        
        return q_values