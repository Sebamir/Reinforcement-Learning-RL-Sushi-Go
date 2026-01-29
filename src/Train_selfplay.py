"""
Script completo de entrenamiento self-play para Sushi Go
Incluye métricas, evaluación y comparación de estrategias
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
from pathlib import Path

from src.env.SushiGoSelfPlayEnv import SushiGoSelfPlayEnv
from src.SelfPlayTrainer import SelfPlayTrainer


class SelfPlayExperiment:
    """Gestiona experimentos completos de self-play con tracking."""
    
    def __init__(self, experiment_name, num_players=2):
        self.experiment_name = experiment_name
        self.num_players = num_players
        self.results = {
            'training_history': [],
            'evaluation_history': [],
            'best_win_rate': 0,
            'metadata': {
                'num_players': num_players,
                'start_time': datetime.now().isoformat()
            }
        }
        
        # Crear directorio de experimentos
        self.exp_dir = Path(f"experiments/{experiment_name}")
        self.exp_dir.mkdir(parents=True, exist_ok=True)
    
    def run_training(self, total_timesteps=200000, eval_freq=25000):
        """
        Ejecuta entrenamiento completo con evaluaciones periódicas.
        """
        print("="*60)
        print(f"🚀 EXPERIMENTO: {self.experiment_name}")
        print(f"   Jugadores: {self.num_players}")
        print(f"   Timesteps: {total_timesteps}")
        print("="*60)
        
        # Crear trainer
        trainer = SelfPlayTrainer(
            env_class=SushiGoSelfPlayEnv,
            num_players=self.num_players,
            strategy='sequential'
        )
        
        # Inicializar modelo
        trainer.init_model()
        
        
        # Entrenamiento con evaluaciones periódicas
        n_evals = total_timesteps // eval_freq
        
        for eval_step in range(n_evals):
            current_timestep = eval_step * eval_freq
            
            print(f"\n📈 Progreso: {current_timestep}/{total_timesteps}")
            
            # Entrenar
            trainer.model.learn(total_timesteps=eval_freq, reset_num_timesteps=False)
            
            # Evaluar
            eval_results = trainer.evaluate_vs_random(num_games=50)
            eval_results['timestep'] = current_timestep + eval_freq
            
            self.results['evaluation_history'].append(eval_results)
            
            # Guardar mejor modelo
            if eval_results['win_rate'] > self.results['best_win_rate']:
                self.results['best_win_rate'] = eval_results['win_rate']
                best_model_path = self.exp_dir / "best_model.zip"
                trainer.save_model(str(best_model_path))
                print(f"   🌟 Nuevo mejor modelo! Win rate: {eval_results['win_rate']:.2%}")
        
        # Guardar modelo final
        final_model_path = self.exp_dir / "final_model.zip"
        trainer.save_model(str(final_model_path))
        
        # Evaluación final extendida
        print("\n" + "="*60)
        print("📊 EVALUACIÓN FINAL")
        print("="*60)
        
        final_eval = trainer.evaluate_vs_random(num_games=200)
        self.results['final_evaluation'] = final_eval
        
        # Guardar resultados
        self._save_results()
        self._plot_results()
        
        # Juegos de demostración
        print("\n🎲 Juegos de demostración del modelo final:")
        demo_results = []
        for i in range(5):
            scores, winner, actions = trainer.play_full_game_sequential(render=True)
            demo_results.append({
                'game': i+1,
                'scores': scores,
                'winner': winner
            })
        
        self.results['demo_games'] = demo_results
        self._save_results()
        
        return trainer
    
    def _save_results(self):
        """Guarda resultados en JSON."""
        results_path = self.exp_dir / "results.json"
        
        # Convertir numpy arrays a listas para JSON
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        serializable_results = json.loads(
            json.dumps(self.results, default=convert_to_serializable)
        )
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"💾 Resultados guardados en: {results_path}")
    
    def _plot_results(self):
        """Genera gráficos de resultados."""
        if not self.results['evaluation_history']:
            return
        
        eval_history = self.results['evaluation_history']
        timesteps = [e['timestep'] for e in eval_history]
        win_rates = [e['win_rate'] for e in eval_history]
        score_diffs = [e['avg_score_diff'] for e in eval_history]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Win rate
        ax1.plot(timesteps, win_rates, 'b-', linewidth=2, marker='o')
        ax1.axhline(y=0.5, color='r', linestyle='--', label='Random baseline')
        ax1.set_xlabel('Timesteps')
        ax1.set_ylabel('Win Rate vs Random')
        ax1.set_title('Progreso de Entrenamiento')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Score difference
        ax2.plot(timesteps, score_diffs, 'g-', linewidth=2, marker='s')
        ax2.axhline(y=0, color='r', linestyle='--', label='Empate')
        ax2.set_xlabel('Timesteps')
        ax2.set_ylabel('Diferencia Promedio de Puntos')
        ax2.set_title('Ventaja en Puntuación')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        plot_path = self.exp_dir / "training_progress.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Gráficos guardados en: {plot_path}")
        plt.close()
    
    def compare_strategies(self, strategies=['sequential', 'historical']):
        """
        Compara diferentes estrategias de self-play.
        """
        print("\n🔬 Comparando estrategias de self-play...")
        
        comparison_results = {}
        
        for strategy in strategies:
            print(f"\n--- Estrategia: {strategy.upper()} ---")
            
            trainer = SelfPlayTrainer(
                env_class=SushiGoSelfPlayEnv,
                num_players=self.num_players,
                strategy=strategy
            )
            
            trainer.init_model()
            
            if strategy == 'sequential':
                trainer.train_sequential(total_timesteps=100000)
            elif strategy == 'historical':
                trainer.train_historical(total_timesteps=100000)
            
            # Evaluar
            eval_results = trainer.evaluate_vs_random(num_games=100)
            comparison_results[strategy] = eval_results
            
            # Guardar modelo
            model_path = self.exp_dir / f"model_{strategy}.zip"
            trainer.save_model(str(model_path))
        
        # Mostrar comparación
        print("\n" + "="*60)
        print("📊 COMPARACIÓN DE ESTRATEGIAS")
        print("="*60)
        
        for strategy, results in comparison_results.items():
            print(f"\n{strategy.upper()}:")
            print(f"  Win rate: {results['win_rate']:.2%}")
            print(f"  Avg score diff: {results['avg_score_diff']:.2f}")
        
        self.results['strategy_comparison'] = comparison_results
        self._save_results()
        
        return comparison_results


def main():
    """Script principal de entrenamiento."""
    
    # Experimento 1: Entrenamiento básico (2 jugadores)
    print("\n" + "🎯"*30)
    print("EXPERIMENTO 1: Self-Play 2 Jugadores")
    print("🎯"*30 + "\n")
    
    exp1 = SelfPlayExperiment(
        experiment_name="selfplay_2p_basic",
        num_players=2
    )
    
    trainer_2p = exp1.run_training(
        total_timesteps=200000,
        eval_freq=25000
    )
    
    # Experimento 2: Entrenamiento con 3 jugadores
    print("\n" + "🎯"*30)
    print("EXPERIMENTO 2: Self-Play 3 Jugadores")
    print("🎯"*30 + "\n")
    
    exp2 = SelfPlayExperiment(
        experiment_name="selfplay_3p_basic",
        num_players=3
    )
    
    trainer_3p = exp2.run_training(
        total_timesteps=200000,
        eval_freq=25000
    )
    
    # Experimento 3: Comparación de estrategias
    print("\n" + "🎯"*30)
    print("EXPERIMENTO 3: Comparación de Estrategias")
    print("🎯"*30 + "\n")
    
    exp3 = SelfPlayExperiment(
        experiment_name="strategy_comparison",
        num_players=2
    )
    
    exp3.compare_strategies(strategies=['sequential', 'historical'])
    
    print("\n" + "="*60)
    print("✅ TODOS LOS EXPERIMENTOS COMPLETADOS")
    print("="*60)
    print(f"\nResultados guardados en: experiments/")
    print("\nModelos entrenados:")
    print("  - experiments/selfplay_2p_basic/best_model.zip")
    print("  - experiments/selfplay_3p_basic/best_model.zip")
    print("  - experiments/strategy_comparison/model_*.zip")


if __name__ == "__main__":
    main()