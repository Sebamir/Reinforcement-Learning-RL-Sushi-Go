🍣 Reinforcement Learning - Sushi Go Master

Este repositorio contiene la implementación de un agente de Aprendizaje por Refuerzo (RL) entrenado para jugar Sushi Go de forma profesional.
A diferencia de un bot basado en reglas, este agente utiliza PPO (Proximal Policy Optimization) y Self-Play para descubrir estrategias ganadoras por sí mismo.

🚀 Características Principales

Entorno de Juego Robusto
Implementación personalizada bajo el estándar de Gymnasium.

Lógica de Dependencia Temporal
El agente entiende que el Wasabi es una inversión a futuro, buscando maximizar el combo con el Nigiri de Calamar:

3
×
3
=
9
3×3=9 pts.

Competencia Multi-Agente
Sistema de puntuación de Maki Rolls que incentiva a la IA a monitorear el progreso de los oponentes para asegurar el bono de mayoría.

Entrenamiento Optimizado
Sistema de callbacks que gestiona evaluaciones periódicas, guarda el mejor modelo histórico y genera métricas de rendimiento visuales.

🧠 El Cerebro de la IA: Detalles Técnicos
El Reto del Wasabi

El Wasabi representa el problema clásico de recompensa retardada en RL.
Jugar un Wasabi otorga 0 puntos inmediatos, pero triplica el siguiente Nigiri.

Para resolver esto:

Espacio de Observación Extendido
Se añadieron bits de estado que indican si el jugador tiene un Wasabi Activo.

Modelado de Recompensa
Se configuró la lógica para que la red neuronal detecte que el valor esperado 
(
𝐸
)
(E) de esperar un Calamar es mayor que conformarse con un Huevo de forma inmediata.

El Algoritmo PPO

Se utiliza Proximal Policy Optimization (PPO) debido a su estabilidad en entornos donde la política de juego cambia rápidamente (Self-Play).

🛠️ Instalación y Uso
1. Requisitos
pip install gymnasium stable-baselines3 shimmy gradio matplotlib numpy

2. Entrenamiento

Para iniciar un experimento de 500,000 pasos con evaluación cada 25,000:

python -m src.train_optimized

3. Jugar contra el Agente

Lanza la interfaz de Gradio para poner a prueba tus habilidades:

python -m src.app_gradio

🏥 Conexión con el Mundo Real: Aplicaciones en Salud

Este proyecto no es solo sobre sushi; es una simulación de Toma de Decisiones Secuenciales bajo Incertidumbre, un campo crítico en la salud moderna:

Sinergia Farmacológica
La lógica del Wasabi (una carta que potencia a otra) es análoga al modelado de tratamientos adyuvantes.
La IA aprende cuándo una intervención preparatoria maximiza la eficacia de una terapia principal posterior.

Triaje y Recursos Críticos
La competencia por los Makis simula la asignación de recursos limitados en un hospital.
La IA decide si “invertir” en un paciente/plato basándose en lo que el resto del sistema (oponentes) está haciendo.

Medicina Personalizada
El entrenamiento mediante Self-Play vuelve al modelo robusto ante distintos estilos de paciente (estrategias), permitiendo adaptarse a comportamientos no lineales en datos biométricos.

📈 Próximos Pasos

 Lógica de Puddings: Implementar la recompensa negativa por tener la menor cantidad de Puddings al final de la partida completa (3 rondas).

 Deep Q-Learning: Comparar el rendimiento de PPO contra DQN para este tipo de juegos de cartas.

 UI Mejorada: Renderizar imágenes de cartas reales en la interfaz de Gradio.
