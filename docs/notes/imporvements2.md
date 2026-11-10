# Implementation Roadmap: _Prior Knowledge Injection_

Es una idea brillante desde el punto de vista de la optimización, y de hecho es una técnica muy utilizada en la investigación avanzada de RL conocida como **Prior Knowledge Injection** (Inyección de Conocimiento Previo) o **Action Space Shaping**.

Sin embargo, en el contexto de ataques adversarios, es una **espada de doble filo**.

Aquí tienes un análisis técnico de por qué es una buena idea, el peligro mortal que esconde (el "Local Minima Trap"), y tres estrategias para implementarlo de forma inteligente en tu código.

## To-Do
- [ ] Implement Action Space Truncation in AudioAttackEnv
- [ ] Implement Actor Network Biasing
- [ ] Implement Auxiliary Reward Shaping
- [ ] Compare and evaluate all three strategies

---

### La Ventaja: Aceleración Masiva (Warm-Starting)

Si el agente inicializa su exploración completamente al azar, pasa el 80% del tiempo de entrenamiento descubriendo lo básico (ej. que la compresión arruina el audio). Si inicializas la red neuronal para que sus primeras acciones se agrupen alrededor de la "Firma Ganadora" (Ratio 1.0, Jitter bajo, Bitrate alto), el agente **saltará el valle de exploración** y comenzará a generar ataques letales desde el paso 1, enfocando todo su poder de cómputo en afinar el ataque en lugar de descubrirlo.

### El Peligro: La Trampa del Mínimo Local (Policy Collapse)

El objetivo del Aprendizaje por Refuerzo es que la red neuronal aprenda a mapear **Estados a Acciones**.
Si sesgas el modelo demasiado, la red neuronal se volverá "perezosa". No aprenderá *por qué* el Ratio 1.0 funciona para el archivo A y por qué el archivo B necesita algo distinto; simplemente memorizará que "Ratio 1.0 = Recompensa" y se negará a explorar otras estrategias.
Si el modelo AASIST3 se actualiza mañana y se vuelve inmune a este ataque de alta fidelidad, tu agente será incapaz de descubrir un vector de ataque secundario (por ejemplo, un ataque de ruido extremo) porque su exploración quedó ciega a esas posibilidades.

---

### Cómo Implementarlo Inteligentemente (3 Estrategias)

Si decides aplicar este sesgo en tu entorno `AudioAttackEnv` o en PPO, no lo hagas hardcodeando los valores. Usa una de estas tres arquitecturas de RL:

#### 1. Action Space Truncation (Modificación Dura)

En lugar de sesgar la red, **redefine los límites del entorno Gymnasium**. Si sabes que los bitrates por debajo de 60k arruinan el ataque y los umbrales (Threshold) no importan, recorta el espacio matemático de PPO.

* **Implementación:** En el `__init__` de tu entorno, cambia el `spaces.Box`.
```python
# Antes: Todas las acciones de -1.0 a 1.0
self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

# Ahora: Limitar Jitter/Shimmer a ser negativos, obligar al Bitrate a ser alto
self.action_space = spaces.Box(
    low=np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.5]), # Bitrate minimo alto
    high=np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]),     # Jitter/Shimmer max = 0
    dtype=np.float32
)

```


* **Veredicto:** Excelente para crear un modelo de explotación puro y rápido, pero destruye la capacidad del agente para encontrar estrategias fuera de esa caja.

#### 2. Actor Network Biasing (Modificación Suave - *Recomendada*)

Consiste en modificar los sesgos (biases) de la última capa de la red neuronal del Actor en PPO antes de empezar a entrenar.

* PPO asume que las acciones inician con una media de `0.0`. Si cambias los pesos de inicialización para que la media de la acción *Ratio* sea `1.0` y la de *Jitter* sea `-0.8`, el agente comenzará probando tus parámetros ganadores.
* **Por qué es mejor:** A medida que entrena, **sigue teniendo la libertad matemática de cambiar esa media** si encuentra una estrategia mejor, porque la entropía sigue activa.

#### 3. Reward Shaping Auxiliar (Estilo RL Puro)

No toques el espacio de acciones ni la red neuronal. En su lugar, modifica `reward_logic.py` para darle al agente pequeñas **"migas de pan"** si elige acciones cercanas a la firma ganadora, incluso si el ataque falla.

* **Implementación:**
```python
aux_reward = 0.0
# Recompensar suavemente por mantener la fidelidad (Ratio cerca de 1.0)
if dsp_params['ratio'] < 2.0:
    aux_reward += 1.5

# Recompensa principal
reward = base_reward + aux_reward + bonus

```


* **Veredicto:** Es la forma más segura. Guías al agente hacia la zona de alta fidelidad, pero dejas que la red neuronal descubra por sí misma cómo combinar esos parámetros con el audio de entrada.

### Conclusión para tus Dev-Notes
Aplicar un sesgo (Bias) es un movimiento estratégico excelente para la fase final de tu tesis (para conseguir ataques perfectos del 99%). Te sugiero utilizar la **Estrategia 1 (Recortar el Action Space)** si necesitas resultados garantizados de forma rápida, o la **Estrategia 3 (Reward Auxiliar)** si quieres demostrar en tu tesis que el agente descubrió el ataque orgánicamente pero con un "empujón" de ingeniería de recompensas.