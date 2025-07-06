[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Lj3YlzJp)
# Proyecto Final 2025-1: AI Neural Network
## **CS2013 Programación III** · Informe Final

### **Descripción**

> Ejemplo: Implementación de una red neuronal multicapa en C++ para clasificación de dígitos manuscritos.

### Contenidos

1. [Datos generales](#datos-generales)
2. [Requisitos e instalación](#requisitos-e-instalación)
3. [Investigación teórica](#1-investigación-teórica)
4. [Diseño e implementación](#2-diseño-e-implementación)
5. [Ejecución](#3-ejecución)
6. [Análisis del rendimiento](#4-análisis-del-rendimiento)
7. [Trabajo en equipo](#5-trabajo-en-equipo)
8. [Conclusiones](#6-conclusiones)
9. [Bibliografía](#7-bibliografía)
10. [Licencia](#licencia)
---

### Datos generales

* **Tema**: Redes Neuronales en AI
* **Grupo**: `group_3_custom_name`
* **Integrantes**:

  * Jhogan Haldo Pachacutec Aguilar – 202410582 (Responsable de investigación teórica)
  * Alumno B – 209900002 (Desarrollo de la arquitectura)
  * Alumno C – 209900003 (Implementación del modelo)
  * Alumno D – 209900004 (Pruebas y benchmarking)
  * Alumno E – 209900005 (Documentación y demo)

> *Nota: Reemplazar nombres y roles reales.*

---

### Requisitos e instalación

1. **Compilador**: GCC 11 o superior
2. **Dependencias**:

   * CMake 3.18+
   * Eigen 3.4
   * \[Otra librería opcional]
3. **Instalación**:

   ```bash
   git clone https://github.com/EJEMPLO/proyecto-final.git
   cd proyecto-final
   mkdir build && cd build
   cmake ..
   make
   ```

> *Ejemplo de repositorio y comandos, ajustar según proyecto.*

---

### 1. Investigación teórica

**Objetivo:** Explorar los fundamentos teóricos de las redes neuronales para sustentar la implementación modular de una red neuronal en C++, orientada a la clasificación multiclase mediante entrenamiento supervisado.

#### 1.1 Historia y evolución de las redes neuronales

El estudio de las redes neuronales artificiales (ANNs) se remonta a la década de 1940, cuando Warren McCulloch y Walter Pitts propusieron el primer modelo matemático de neurona artificial inspirado en el funcionamiento del cerebro humano. Pero fue en 1958 cuando Frank Rosenblatt desarrolló el perceptrón, uno de los primeros modelos de redes neuronales artificiales. Este consiste en una unidad que recibe múltiples entradas, las pondera, las suma y aplica una función de activación (como el escalón) para generar una salida. Este modelo fue capaz de aprender mediante un proceso de ajuste de pesos, pero solo podía resolver problemas linealmente separables [1].

El campo avanzó significativamente en 1986 con la introducción del algoritmo de retropropagación del error (*backpropagation*), desarrollado por Rumelhart, Hinton y Williams [2]. Este método permitió entrenar redes neuronales con múltiples capas ocultas al calcular los gradientes del error en relación con los pesos y propagarlos hacia atrás desde la capa de salida. La retropropagación, basada en la regla de la cadena del cálculo diferencial, permitió que las redes pudieran aprender representaciones complejas, marcando el inicio del aprendizaje profundo moderno.

Así, en la última década, el campo experimentó una transformación radical con el auge del *deep learning*, impulsado por mejoras en hardware (GPU), grandes volúmenes de datos y nuevos algoritmos. Investigaciones de LeCun, Bengio y Hinton consolidaron el uso de redes neuronales profundas para tareas como visión por computadora, reconocimiento de voz y procesamiento de lenguaje natural [3].

#### 1.2 Arquitecturas principales de redes neuronales

##### a) Perceptrón Multicapa (MLP)

Un MLP (Multi-Layer Perceptron) es una red neuronal *feedforward* compuesta por una capa de entrada, una o más capas ocultas y una capa de salida. Cada neurona en una capa está conectada a todas las neuronas de la siguiente, formando una estructura totalmente conectada [4]. Se utiliza comúnmente en tareas de clasificación y regresión.

En el proyecto desarrollado, se ha implementado una red MLP con la siguiente configuración:

- **Entrada:** 64 neuronas (8×8 píxeles).
- **Capa oculta 1:** 128 neuronas + función de activación ReLU.
- **Capa oculta 2:** 64 neuronas + ReLU.
- **Salida:** 10 neuronas + Sigmoid, para clasificación multiclase.

Esta arquitectura es coherente con el uso de datasets como el MNIST reducido, que presentan imágenes de dígitos en escala de grises y formato vectorizado.

##### b) Funciones de activación: ReLU y Sigmoid

La función ReLU (Rectified Linear Unit) se ha convertido en el estándar en redes profundas debido a su simplicidad y eficiencia computacional. Su fórmula, f(x) = max(0,x) permite una propagación más estable del gradiente, reduciendo el problema del gradiente desvanecido en redes profundas [5]. ReLU fue introducida formalmente en 2010 por Nair y Hinton, y desde entonces se ha adoptado ampliamente en tareas de visión por computadora y procesamiento de señales.

Por su parte, la función Sigmoid tiene la forma σ(x) = 1 / (1 + e^(-x)) y transforma cualquier valor real en un rango entre 0 y 1, interpretado como probabilidad. Aunque puede sufrir saturación para valores extremos (lo cual afecta la retropropagación en capas ocultas), sigue siendo especialmente útil en la capa de salida de redes para clasificación binaria o multiclase con etiquetas *one-hot*, ya que cada neurona puede modelar la probabilidad independiente de cada clase [4]. En este proyecto, la función Sigmoid se utiliza correctamente en la última capa de la red para producir una salida de 10 dimensiones (una por clase), facilitando el uso de la función de pérdida Binary Cross-Entropy, que requiere salidas en formato probabilístico.

#### 1.3 Algoritmos de entrenamiento

##### a) Backpropagation

El algoritmo de *backpropagation* permite actualizar los pesos de la red neuronal utilizando el gradiente de la función de pérdida con respecto a los pesos, propagando el error desde la capa de salida hacia las capas anteriores. Este método se basa en la regla de la cadena del cálculo diferencial y es fundamental para el aprendizaje supervisado [2].

La red implementada utiliza `train<Loss, Optimizer>` como plantilla para aplicar este mecanismo de forma genérica, permitiendo experimentar con distintas configuraciones de pérdida y optimización.

##### b) Inicialización de pesos: Xavier

Una adecuada inicialización de pesos es crucial para asegurar la estabilidad del entrenamiento. La inicialización Xavier, propuesta por Glorot y Bengio en 2010, sugiere que los pesos deben ser escalados de acuerdo con el número de neuronas de entrada y salida, para mantener la varianza del gradiente estable [6].

##### c) Funciones de pérdida: MSE y BCE

Se implementaron dos funciones de pérdida en el entrenamiento de la red:

- **MSE (Mean Squared Error):** calcula el promedio del cuadrado de las diferencias entre las salidas reales y las predichas. Aunque es ampliamente utilizada en problemas de regresión, su aplicación en clasificación puede ser limitada, ya que no modela adecuadamente la incertidumbre probabilística ni penaliza lo suficiente las predicciones incorrectas [4].

- **Binary Cross-Entropy (BCE):** mide la disimilitud entre la distribución de salida del modelo y las etiquetas reales codificadas en formato *one-hot*. Esta función evalúa el rendimiento del modelo bajo una perspectiva probabilística, siendo más sensible a errores de predicción en tareas de clasificación [4].

##### d) Optimizadores: SGD y Adam

El entrenamiento de redes neuronales requiere optimizadores eficientes. Uno de los optimizadores más ampliamente utilizados y simples es el Stochastic Gradient Descent (SGD). Este consiste en actualizar los pesos del modelo utilizando los gradientes calculados a partir de mini-lotes de datos, lo que reduce el costo computacional por iteración [4].

Por el contrario, Adaptive Moment Estimation (Adam) combina los beneficios de AdaGrad y RMSProp, ajustando la tasa de aprendizaje por parámetro mediante momentos del gradiente y su cuadrado. Este método fue introducido por Kingma y Ba en 2015 y es ampliamente utilizado por su eficiencia y robustez [7].






---

### 2. Diseño e implementación

#### 2.1 Arquitectura de la solución

* **Patrones de diseño**: ejemplo: Factory para capas, Strategy para optimizadores.
* **Estructura de carpetas (ejemplo)**:

  ```
  proyecto-final/
  ├── src/
  │   ├── layers/
  │   ├── optimizers/
  │   └── main.cpp
  ├── tests/
  └── docs/
  ```

#### 2.2 Manual de uso y casos de prueba

* **Cómo ejecutar**: `./build/neural_net_demo input.csv output.csv`
* **Casos de prueba**:

  * Test unitario de capa densa.
  * Test de función de activación ReLU.
  * Test de convergencia en dataset de ejemplo.

> *Personalizar rutas, comandos y casos reales.*

---

### 3. Ejecución

> **Demo de ejemplo**: Video/demo alojado en `docs/demo.mp4`.
> Pasos:
>
> 1. Preparar datos de entrenamiento (formato CSV).
> 2. Ejecutar comando de entrenamiento.
> 3. Evaluar resultados con script de validación.

---

### 4. Análisis del rendimiento

* **Métricas de ejemplo**:

  * Iteraciones: 1000 épocas.
  * Tiempo total de entrenamiento: 2m30s.
  * Precisión final: 92.5%.
* **Ventajas/Desventajas**:

  * * Código ligero y dependencias mínimas.
  * – Sin paralelización, rendimiento limitado.
* **Mejoras futuras**:

  * Uso de BLAS para multiplicaciones (Justificación).
  * Paralelizar entrenamiento por lotes (Justificación).

---

### 5. Trabajo en equipo

| Tarea                     | Miembro  | Rol                       |
| ------------------------- | -------- | ------------------------- |
| Investigación teórica     | Jhogan Haldo Pachacutec Aguilar | Documentar bases teóricas |
| Diseño de la arquitectura | Alumno B | UML y esquemas de clases  |
| Implementación del modelo | Alumno C | Código C++ de la NN       |
| Pruebas y benchmarking    | Alumno D | Generación de métricas    |
| Documentación y demo      | Alumno E | Tutorial y video demo     |

> *Actualizar con tareas y nombres reales.*

---

### 6. Conclusiones

* **Logros**: Implementar NN desde cero, validar en dataset de ejemplo.
* **Evaluación**: Calidad y rendimiento adecuados para propósito académico.
* **Aprendizajes**: Profundización en backpropagation y optimización.
* **Recomendaciones**: Escalar a datasets más grandes y optimizar memoria.

---

### 7. Bibliografía

[1] F. Rosenblatt, “The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain,” *Psychological Review*, vol. 65, no. 6, pp. 386–408, 1958.

[2] D. E. Rumelhart, G. E. Hinton, and R. J. Williams, “Learning Representations by Back-Propagating Errors,” *Nature*, vol. 323, pp. 533–536, 1986.

[3] Y. LeCun, Y. Bengio, and G. Hinton, “Deep Learning,” *Nature*, vol. 521, no. 7553, pp. 436–444, 2015.

[4] C. M. Bishop, *Pattern Recognition and Machine Learning*, Springer, 2006.

[5] V. Nair and G. E. Hinton, “Rectified Linear Units Improve Restricted Boltzmann Machines,” in *Proc. 27th Int. Conf. on Machine Learning (ICML)*, 2010.

[6] X. Glorot and Y. Bengio, “Understanding the Difficulty of Training Deep Feedforward Neural Networks,” in *Proc. 13th Int. Conf. on Artificial Intelligence and Statistics (AISTATS)*, 2010.

[7] D. P. Kingma and J. Ba, “Adam: A Method for Stochastic Optimization,” in *Proc. 3rd Int. Conf. on Learning Representations (ICLR)*, 2015.


---

### Licencia

Este proyecto usa la licencia **MIT**. Ver [LICENSE](LICENSE) para detalles.

---
