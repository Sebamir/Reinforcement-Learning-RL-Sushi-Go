import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from collections import deque
import copy

class SelfPlayTrainer:
    """
    Gestor de entrenamiento self-play para Sushi Go.
    
    Estrategias soportadas:
    1. Sequential: Todos los jugadores comparten el mismo modelo
    2. League: Múltiples versiones del modelo compiten entre sí
    3. Historical: Modelo actual vs versiones pasadas
    """
    
    def __init__(self, env_class, num_players=2, strategy='sequential'):
        self.env_class = env_class
        self.num_players = num_players
        self.strategy = strategy
        
        # Modelo principal
        self.model = None
        
        # Para league/historical training
        self.past_models = deque(maxlen=5)  # Mantiene últimas 5 versiones
        self.update_interval = 10000  # Pasos entre actualizaciones
        
    def create_env(self):
        """Crea el entorno de self-play."""
        return self.env_class(num_players=self.num_players, competitive_reward=True)
    
    def init_model(self, policy='MlpPolicy', **kwargs):
        """Inicializa el modelo principal."""
        env = DummyVecEnv([self.create_env])
        
        default_params = {
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'verbose': 1
        }
        default_params.update(kwargs)
        
        self.model = PPO(policy, env, **default_params)
        return self.model
    
    def train_sequential(self, total_timesteps, save_interval=50000):
        """
        Entrenamiento secuencial: todos los jugadores usan el mismo modelo.
        El modelo aprende jugando contra sí mismo.
        """
        env = self.create_env()
        env = DummyVecEnv([lambda: env])
        
        self.model.set_env(env)
        
        print("🎮 Iniciando entrenamiento SEQUENTIAL SELF-PLAY")
        print(f"   Jugadores: {self.num_players}")
        print(f"   Timesteps totales: {total_timesteps}")
        
        # Entrenar con callbacks para guardar versiones
        class SelfPlayCallback:
            def __init__(self, trainer, save_interval):
                self.trainer = trainer
                self.save_interval = save_interval
                self.n_calls = 0
                
            def __call__(self, locals, globals):
                self.n_calls += 1
                if self.n_calls % self.save_interval == 0:
                    # Guardar versión actual
                    model_copy = copy.deepcopy(self.trainer.model)
                    self.trainer.past_models.append(model_copy)
                    print(f"📸 Snapshot guardado en step {self.n_calls}")
                return True
        
        callback = SelfPlayCallback(self, save_interval)
        self.model.learn(total_timesteps=total_timesteps, callback=callback)
        
        print("✅ Entrenamiento completado")
        return self.model
    
    def play_full_game_sequential(self, render=False):
        """
        Juega un juego completo con el modelo actual.
        Todos los jugadores usan el mismo modelo.
        """
        env = self.create_env()
        obs, info = env.reset()
        done = False
        
        actions_log = []
        
        while not done:
            if render:
                env.render()
            
            # El modelo predice para el jugador actual
            action, _ = self.model.predict(obs, deterministic=True)
            actions_log.append({
                'player': info.get('player', env.current_player),
                'action': action
            })
            
            obs, reward, done, truncated, info = env.step(action)
        
        if render:
            env.render()
            print(f"\n🏆 Ganador: Jugador {info['winner']}")
            print(f"📊 Puntajes finales: {info['final_scores']}")
        
        return info['final_scores'], info['winner'], actions_log
    
    def train_historical(self, total_timesteps, opponent_pool_size=3):
        """
        Entrenamiento contra versiones históricas.
        El modelo actual juega contra versiones pasadas de sí mismo.
        """
        print("🎮 Iniciando entrenamiento HISTORICAL SELF-PLAY")
        print(f"   Pool de oponentes: {opponent_pool_size}")
        
        steps_per_iteration = total_timesteps // 10
        
        for iteration in range(10):
            print(f"\n=== Iteración {iteration + 1}/10 ===")
            
            # Entrenar contra el pool actual
            env = self.create_env()
            env = DummyVecEnv([lambda: env])
            self.model.set_env(env)
            self.model.learn(total_timesteps=steps_per_iteration)
            
            # Agregar versión actual al pool
            if len(self.past_models) < opponent_pool_size:
                model_copy = copy.deepcopy(self.model)
                self.past_models.append(model_copy)
                print(f"   Modelo agregado al pool ({len(self.past_models)}/{opponent_pool_size})")
        
        print("✅ Entrenamiento completado")
        return self.model
    
    def evaluate_vs_random(self, num_games=100):
        """
        Evalúa el modelo contra un jugador aleatorio.
        """
        env = self.create_env()
        wins = 0
        total_score_diff = 0
        
        for game in range(num_games):
            obs, info = env.reset()
            done = False
            
            while not done:
                current_player = env.current_player
                
                if current_player == 0:
                    # Modelo entrenado
                    action, _ = self.model.predict(obs, deterministic=True)
                else:
                    # Jugador aleatorio
                    valid_actions = list(range(len(env.hands[current_player])))
                    action = np.random.choice(valid_actions)
                
                obs, reward, done, truncated, info = env.step(action)
            
            # Evaluar resultado
            scores = info['final_scores']
            if info['winner'] == 0:
                wins += 1
            
            total_score_diff += scores[0] - scores[1]
        
        win_rate = wins / num_games
        avg_score_diff = total_score_diff / num_games
        
        print(f"\n📊 Evaluación vs Random ({num_games} juegos):")
        print(f"   Win rate: {win_rate:.2%}")
        print(f"   Diferencia promedio de puntos: {avg_score_diff:.2f}")
        
        return {
            'win_rate': win_rate,
            'avg_score_diff': avg_score_diff,
            'games_played': num_games
        }
    
    def save_model(self, path):
        """Guarda el modelo principal."""
        self.model.save(path)
        print(f"💾 Modelo guardado en: {path}")
    
    def load_model(self, path):
        """Carga un modelo previamente entrenado."""
        env = DummyVecEnv([self.create_env])
        self.model = PPO.load(path, env=env)
        print(f"📂 Modelo cargado desde: {path}")
        return self.model

"""
# Ejemplo de uso
if __name__ == "__main__":
    from sushi_go_selfplay_env import SushiGoSelfPlayEnv
    
    # Crear trainer
    trainer = SelfPlayTrainer(
        env_class=SushiGoSelfPlayEnv,
        num_players=3,
        strategy='sequential'
    )
    
    # Inicializar modelo
    trainer.init_model(
        policy='MlpPolicy',
        learning_rate=3e-4,
        verbose=1
    )
    
    # Entrenar con self-play
    trainer.train_sequential(total_timesteps=100000, save_interval=25000)
    
    # Evaluar
    trainer.evaluate_vs_random(num_games=100)
    
    # Jugar algunos juegos de demostración
    print("\n🎲 Juegos de demostración:")
    for i in range(3):
        print(f"\n--- Juego {i+1} ---")
        scores, winner, _ = trainer.play_full_game_sequential(render=True)
    
    # Guardar modelo
    trainer.save_model("models/sushi_go_selfplay.zip")
    """