"""
Script optimizado de entrenamiento self-play para Sushi Go
Usa callbacks para eficiencia máxima y métricas continuas
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
from pathlib import Path
import time

from stable_baselines3.common.callbacks import BaseCallback
from src.env.SushiGoSelfPlayEnv import SushiGoSelfPlayEnv
from src.SelfPlayTrainer import SelfPlayTrainer


class EvalAndCheckpointCallback(BaseCallback):
    """
    Callback optimizado que combina evaluación y checkpointing.
    
    Ventajas sobre loop manual:
    - No destruye/recrea entornos
    - Mantiene buffer de experiencias
    - Métricas continuas sin interrupciones
    - Puede ejecutar evaluación en paralelo (futuro)
    """
    
    def __init__(self, eval_freq, trainer, experiment, num_eval_games=50, verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.trainer = trainer
        self.experiment = experiment
        self.num_eval_games = num_eval_games
        
        # Tracking
        self.last_eval_step = 0
        self.eval_count = 0
        self.start_time = time.time()
        
    def _on_step(self) -> bool:
        """Llamado después de cada step del entorno."""
        
        # Verificar si es momento de evaluar
        if self.num_timesteps - self.last_eval_step >= self.eval_freq:
            self.eval_count += 1
            elapsed = time.time() - self.start_time
            
            print(f"\n{'='*60}")
            print(f"📊 EVALUACIÓN #{self.eval_count} - Step {self.num_timesteps}")
            print(f"   Tiempo transcurrido: {elapsed/60:.1f} min")
            print(f"{'='*60}")
            
            # Evaluar vs random
            eval_start = time.time()
            eval_results = self.trainer.evaluate_vs_random(num_games=self.num_eval_games)
            eval_time = time.time() - eval_start
            
            # Añadir metadata
            eval_results['timestep'] = self.num_timesteps
            eval_results['eval_time'] = eval_time
            eval_results['total_time'] = elapsed
            
            self.experiment.results['evaluation_history'].append(eval_results)
            
            # Mostrar resultados
            print(f"\n   Win Rate:      {eval_results['win_rate']:.2%}")
            print(f"   Avg Score Diff: {eval_results['avg_score_diff']:+.2f}")
            print(f"   Eval Time:     {eval_time:.1f}s")
            
            # Guardar mejor modelo
            if eval_results['win_rate'] > self.experiment.results['best_win_rate']:
                self.experiment.results['best_win_rate'] = eval_results['win_rate']
                best_model_path = self.experiment.exp_dir / "best_model.zip"
                self.trainer.save_model(str(best_model_path))
                print(f"   🌟 ¡NUEVO RÉCORD! Guardado en {best_model_path.name}")
            
            # Guardar checkpoint periódico
            if self.eval_count % 4 == 0:  # Cada 4 evaluaciones
                checkpoint_path = self.experiment.exp_dir / f"checkpoint_{self.num_timesteps}.zip"
                self.trainer.save_model(str(checkpoint_path))
                print(f"   💾 Checkpoint guardado: {checkpoint_path.name}")
            
            # Guardar resultados intermedios
            self.experiment._save_results()
            
            self.last_eval_step = self.num_timesteps
            print(f"{'='*60}\n")
        
        return True  # Continuar entrenamiento


class ProgressCallback(BaseCallback):
    """Callback para mostrar progreso durante el entrenamiento."""
    
    def __init__(self, total_timesteps, update_freq=10000, verbose=1):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.update_freq = update_freq
        self.start_time = None
        self.last_update = 0
        
    def _on_training_start(self):
        self.start_time = time.time()
        print(f"\n🚀 Iniciando entrenamiento de {self.total_timesteps:,} steps")
        print(f"   Actualizaciones cada {self.update_freq:,} steps\n")
    
    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_update >= self.update_freq:
            elapsed = time.time() - self.start_time
            progress = self.num_timesteps / self.total_timesteps
            eta = (elapsed / progress - elapsed) if progress > 0 else 0
            
            # Barra de progreso
            bar_length = 30
            filled = int(bar_length * progress)
            bar = '█' * filled + '░' * (bar_length - filled)
            
            print(f"[{bar}] {progress*100:5.1f}% | "
                  f"{self.num_timesteps:,}/{self.total_timesteps:,} steps | "
                  f"Tiempo: {elapsed/60:.1f}m | "
                  f"ETA: {eta/60:.1f}m")
            
            self.last_update = self.num_timesteps
        
        return True


class SelfPlayExperiment:
    """Gestiona experimentos completos de self-play con tracking optimizado."""
    
    def __init__(self, experiment_name, num_players=2):
        self.experiment_name = experiment_name
        self.num_players = num_players
        self.results = {
            'training_history': [],
            'evaluation_history': [],
            'best_win_rate': 0,
            'metadata': {
                'num_players': num_players,
                'start_time': datetime.now().isoformat(),
                'training_method': 'callback_optimized'
            }
        }
        
        # Crear directorio de experimentos
        self.exp_dir = Path(f"experiments/{experiment_name}")
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Experimento creado: {self.exp_dir}")
    
    def run_training(self, total_timesteps=500000, eval_freq=25000, num_eval_games=50):
        """
        Entrenamiento optimizado con callbacks.
        
        MEJORAS vs versión anterior:
        ✅ Una sola llamada a learn() - sin overhead de crear/destruir entornos
        ✅ Buffer de experiencias se mantiene - mejor aprovechamiento de datos
        ✅ Métricas continuas - gráficos sin interrupciones
        ✅ Evaluación integrada - no pausa completamente el entrenamiento
        ✅ Checkpoints automáticos - recuperación ante fallos
        """
        print("="*60)
        print(f"🚀 EXPERIMENTO: {self.experiment_name}")
        print(f"   Jugadores: {self.num_players}")
        print(f"   Timesteps: {total_timesteps:,}")
        print(f"   Eval cada: {eval_freq:,} steps")
        print(f"   Método: Callback Optimizado")
        print("="*60)
        
        start_time = time.time()
        
        # Crear trainer
        trainer = SelfPlayTrainer(
            env_class=SushiGoSelfPlayEnv,
            num_players=self.num_players,
            strategy='sequential'
        )
        
        # Inicializar modelo
        print("\n🧠 Inicializando modelo PPO...")
        trainer.init_model(
            policy='MlpPolicy',
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            verbose=0  # Silenciar output de PPO (usamos callbacks personalizados)
        )
        
        # Crear callbacks
        eval_callback = EvalAndCheckpointCallback(
            eval_freq=eval_freq,
            trainer=trainer,
            experiment=self,
            num_eval_games=num_eval_games
        )
        
        progress_callback = ProgressCallback(
            total_timesteps=total_timesteps,
            update_freq=10000
        )
        
        # Entrenar con callbacks (UNA SOLA LLAMADA)
        print("\n" + "="*60)
        print("🎮 INICIANDO ENTRENAMIENTO")
        print("="*60)
        
        trainer.model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, progress_callback],
            progress_bar=False,  # Desactivar barra de progreso de SB3
            log_interval=None    # Silenciar logs automáticos
        )
        
        total_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("✅ ENTRENAMIENTO COMPLETADO")
        print(f"   Tiempo total: {total_time/60:.1f} minutos")
        print(f"   Steps/segundo: {total_timesteps/total_time:.0f}")
        print("="*60)
        
        # Guardar modelo final
        final_model_path = self.exp_dir / "final_model.zip"
        trainer.save_model(str(final_model_path))
        print(f"\n💾 Modelo final guardado: {final_model_path}")
        
        # Evaluación final extendida
        print("\n" + "="*60)
        print("📊 EVALUACIÓN FINAL EXTENDIDA")
        print("="*60)
        
        final_eval = trainer.evaluate_vs_random(num_games=200)
        self.results['final_evaluation'] = final_eval
        self.results['metadata']['end_time'] = datetime.now().isoformat()
        self.results['metadata']['total_training_time'] = total_time
        
        print(f"\nResultados (200 juegos):")
        print(f"  Win Rate:      {final_eval['win_rate']:.2%}")
        print(f"  Avg Score Diff: {final_eval['avg_score_diff']:+.2f}")
        
        # Guardar resultados finales
        self._save_results()
        self._plot_results()
        
        # Juegos de demostración
        self._run_demo_games(trainer, num_games=5)
        
        return trainer
    
    def _run_demo_games(self, trainer, num_games=5):
        """Ejecuta y guarda juegos de demostración."""
        print("\n" + "="*60)
        print(f"🎲 JUEGOS DE DEMOSTRACIÓN ({num_games} juegos)")
        print("="*60)
        
        demo_results = []
        for i in range(num_games):
            print(f"\n--- Juego {i+1}/{num_games} ---")
            scores, winner, actions = trainer.play_full_game_sequential(render=True)
            demo_results.append({
                'game': i+1,
                'scores': scores,
                'winner': winner,
                'num_actions': len(actions)
            })
        
        self.results['demo_games'] = demo_results
        self._save_results()
    
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
        
        if self.results.get('evaluation_history'):
            print(f"💾 Resultados guardados: {results_path}")
    
    def _plot_results(self):
        """Genera gráficos de resultados mejorados."""
        if not self.results['evaluation_history']:
            return
        
        eval_history = self.results['evaluation_history']
        timesteps = [e['timestep'] for e in eval_history]
        win_rates = [e['win_rate'] for e in eval_history]
        score_diffs = [e['avg_score_diff'] for e in eval_history]
        
        # Crear figura con 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        
        # 1. Win rate
        ax1.plot(timesteps, win_rates, 'b-', linewidth=2, marker='o', markersize=6)
        ax1.axhline(y=0.5, color='r', linestyle='--', linewidth=2, label='Random baseline (50%)')
        ax1.fill_between(timesteps, 0.5, win_rates, alpha=0.3, 
                         where=[wr >= 0.5 for wr in win_rates], color='green', label='Above baseline')
        ax1.set_xlabel('Training Steps', fontsize=12)
        ax1.set_ylabel('Win Rate vs Random', fontsize=12)
        ax1.set_title('Learning Progress: Win Rate', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim(0, 1)
        
        # 2. Score difference
        ax2.plot(timesteps, score_diffs, 'g-', linewidth=2, marker='s', markersize=6)
        ax2.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Tie')
        ax2.fill_between(timesteps, 0, score_diffs, alpha=0.3,
                         where=[sd >= 0 for sd in score_diffs], color='green', label='Positive')
        ax2.set_xlabel('Training Steps', fontsize=12)
        ax2.set_ylabel('Avg Score Difference', fontsize=12)
        ax2.set_title('Scoring Advantage', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. Improvement rate
        if len(win_rates) >= 5:
            # Calcular tasa de mejora (rolling average)
            window = 3
            smoothed = np.convolve(win_rates, np.ones(window)/window, mode='valid')
            smooth_steps = timesteps[window-1:]
            
            ax3.plot(smooth_steps, smoothed, 'purple', linewidth=2, label='Smoothed Win Rate')
            ax3.plot(timesteps, win_rates, 'b--', linewidth=1, alpha=0.5, label='Raw Win Rate')
            ax3.axhline(y=0.5, color='r', linestyle='--', linewidth=2)
            ax3.set_xlabel('Training Steps', fontsize=12)
            ax3.set_ylabel('Win Rate (Smoothed)', fontsize=12)
            ax3.set_title('Training Stability', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            ax3.set_ylim(0, 1)
        
        plt.tight_layout()
        
        plot_path = self.exp_dir / "training_progress.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Gráficos guardados: {plot_path}")
        plt.close()
        
        # Gráfico adicional: Eficiencia temporal
        if 'eval_time' in eval_history[0]:
            self._plot_efficiency(eval_history)
    
    def _plot_efficiency(self, eval_history):
        """Grafica eficiencia del entrenamiento."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        timesteps = [e['timestep'] for e in eval_history]
        total_times = [e['total_time']/60 for e in eval_history]  # en minutos
        
        ax.plot(timesteps, total_times, 'orange', linewidth=2, marker='D')
        ax.set_xlabel('Training Steps', fontsize=12)
        ax.set_ylabel('Total Training Time (minutes)', fontsize=12)
        ax.set_title('Training Efficiency', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Añadir tasa (steps/minuto)
        if len(timesteps) > 1:
            steps_per_min = timesteps[-1] / total_times[-1]
            ax.text(0.02, 0.98, f'Avg: {steps_per_min:.0f} steps/min',
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.8))
        
        plt.tight_layout()
        efficiency_path = self.exp_dir / "training_efficiency.png"
        plt.savefig(efficiency_path, dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Script principal de entrenamiento optimizado."""
    
    print("\n" + "🎯"*30)
    print("SISTEMA DE ENTRENAMIENTO SELF-PLAY OPTIMIZADO")
    print("🎯"*30)
    
    # Experimento 1: Entrenamiento básico (2 jugadores)
    print("\n" + "="*60)
    print("EXPERIMENTO 1: Self-Play 2 Jugadores")
    print("="*60)
    
    exp1 = SelfPlayExperiment(
        experiment_name="selfplay_2p_optimized",
        num_players=2
    )
    
    trainer_2p = exp1.run_training(
        total_timesteps=500000,
        eval_freq=25000,
        num_eval_games=50
    )
    
    # Experimento 2: Entrenamiento con 3 jugadores
    print("\n" + "="*60)
    print("EXPERIMENTO 2: Self-Play 3 Jugadores")
    print("="*60)
    
    exp2 = SelfPlayExperiment(
        experiment_name="selfplay_3p_optimized",
        num_players=3
    )
    
    trainer_3p = exp2.run_training(
        total_timesteps=500000,
        eval_freq=25000,
        num_eval_games=50
    )
    
    # Resumen final
    print("\n" + "="*60)
    print("✅ TODOS LOS EXPERIMENTOS COMPLETADOS")
    print("="*60)
    print(f"\n📁 Resultados guardados en: experiments/")
    print("\n📊 Modelos entrenados:")
    print("  - experiments/selfplay_2p_optimized/best_model.zip")
    print("  - experiments/selfplay_2p_optimized/final_model.zip")
    print("  - experiments/selfplay_3p_optimized/best_model.zip")
    print("  - experiments/selfplay_3p_optimized/final_model.zip")
    print("\n📈 Gráficos generados:")
    print("  - training_progress.png (win rate, score diff, stability)")
    print("  - training_efficiency.png (tiempo vs steps)")
    print("\n💾 Datos:")
    print("  - results.json (métricas completas)")
    print("  - checkpoint_*.zip (puntos de recuperación)")
    
    # Comparación de rendimiento
    if exp1.results.get('final_evaluation') and exp2.results.get('final_evaluation'):
        print("\n" + "="*60)
        print("📊 COMPARACIÓN DE RENDIMIENTO")
        print("="*60)
        print(f"\n2 Jugadores:")
        print(f"  Win Rate: {exp1.results['final_evaluation']['win_rate']:.2%}")
        print(f"  Score Diff: {exp1.results['final_evaluation']['avg_score_diff']:+.2f}")
        print(f"\n3 Jugadores:")
        print(f"  Win Rate: {exp2.results['final_evaluation']['win_rate']:.2%}")
        print(f"  Score Diff: {exp2.results['final_evaluation']['avg_score_diff']:+.2f}")


if __name__ == "__main__":
    main()