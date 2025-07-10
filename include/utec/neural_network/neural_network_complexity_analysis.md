# Análisis de Complejidad Algorítmica - Clase NeuralNetwork

## Introducción

Este documento presenta un análisis exhaustivo de la complejidad algorítmica de la implementación de la clase `NeuralNetwork` en C++. La implementación utiliza templates y manejo de memoria inteligente para crear una red neuronal modular con entrenamiento por lotes y predicción batch.

## Definiciones y Notación

- **N**: Número total de muestras de entrenamiento
- **B**: Tamaño del lote (batch size)
- **E**: Número de épocas
- **L**: Número de capas en la red
- **F_i**: Número de características/neuronas en la capa i
- **C**: Número de clases (para clasificación)
- **M**: Número de muestras para predicción

## Análisis de Complejidad por Operación

### 1. Gestión de Capas

#### Agregar capa
```cpp
void add_layer(std::unique_ptr<ILayer<T>> layer)
```
- **Complejidad temporal**: O(1) amortizado
- **Complejidad espacial**: O(1)
- **Análisis**: Inserción en vector con std::move, sin copia de datos

### 2. Función de Entrenamiento

#### Configuración inicial y validación
```cpp
template<template<typename...> class LossType,
         template<typename...> class OptimizerType = SGD>
void train(...)
```

**Validaciones iniciales**:
- **Complejidad temporal**: O(L)
- **Complejidad espacial**: O(1)
- **Análisis**: Verificación de capas válidas y dimensiones compatibles

#### Bucle principal de entrenamiento
**Estructura anidada**:
```
for epoch in E:
    for batch in (N/B):
        // Operaciones por lote
```

**Complejidad por época**:
- **Creación de lotes**: O(N)
- **Forward pass**: O(B × Σ(F_i × F_{i+1})) para i = 0 to L-1
- **Cálculo de loss**: O(B × C)
- **Backward pass**: O(B × Σ(F_i × F_{i+1})) para i = L-1 to 0
- **Actualización de parámetros**: O(Σ(F_i × F_{i+1}))

#### Análisis detallado del entrenamiento por lotes

**Creación de lotes**:
```cpp
// Extracción de datos por lote
for (size_t i = 0; i < actual_batch_size; ++i) {
    for (size_t j = 0; j < X.shape()[1]; ++j) {
        X_batch(i, j) = X(start_idx + i, j);
    }
}
```
- **Complejidad temporal**: O(B × F_input)
- **Complejidad espacial**: O(B × F_input)

**Forward propagation**:
```cpp
for (size_t i = 0; i < layers_.size(); ++i) {
    out = layers_[i]->forward(out);
}
```
- **Complejidad temporal**: O(B × Σ(F_i × F_{i+1}))
- **Complejidad espacial**: O(B × max(F_i))

**Cálculo de accuracy**:
```cpp
// Búsqueda de máximo por muestra
for (size_t i = 0; i < actual_batch_size; ++i) {
    // Encuentra clase predicha: O(C)
    // Encuentra clase verdadera: O(C)
}
```
- **Complejidad temporal**: O(B × C)
- **Complejidad espacial**: O(1)

**Backward propagation**:
```cpp
for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; --i) {
    grad = layers_[i]->backward(grad);
    layers_[i]->update_params(opt);
}
```
- **Complejidad temporal**: O(B × Σ(F_i × F_{i+1}))
- **Complejidad espacial**: O(B × max(F_i))

### 3. Función de Predicción

#### Predicción por lotes
```cpp
utec::algebra::Tensor<T,2> predict(const utec::algebra::Tensor<T,2>& X)
```

**Análisis de la implementación**:
- **Determinación de tamaño de salida**: O(Σ(F_i × F_{i+1}))
- **Procesamiento por lotes**: O(⌈M/100⌉) lotes de tamaño 100
- **Forward pass por lote**: O(B × Σ(F_i × F_{i+1}))

**Complejidad total de predicción**:
- **Temporal**: O(M × Σ(F_i × F_{i+1}))
- **Espacial**: O(M × F_output + B × max(F_i))

## Análisis de Complejidad Total

### Complejidad del entrenamiento completo

**Por época**:
- **Temporal**: O(N × Σ(F_i × F_{i+1}) + N × C)
- **Espacial**: O(B × max(F_i) + Σ(F_i × F_{i+1}))

**Para E épocas**:
- **Temporal**: O(E × N × Σ(F_i × F_{i+1}))
- **Espacial**: O(B × max(F_i) + Σ(F_i × F_{i+1}))

### Desglose por componentes

#### 1. Operaciones de matriz (dominantes)
- **Forward pass**: O(B × Σ(F_i × F_{i+1}))
- **Backward pass**: O(B × Σ(F_i × F_{i+1}))
- **Total por lote**: O(B × Σ(F_i × F_{i+1}))

#### 2. Operaciones de clasificación
- **Cálculo de loss**: O(B × C)
- **Cálculo de accuracy**: O(B × C)
- **Total**: O(B × C)

#### 3. Operaciones de gestión de datos
- **Creación de lotes**: O(B × F_input)
- **Copia de resultados**: O(B × F_output)

## Análisis de Eficiencia

### Estrategias de optimización implementadas

#### 1. Procesamiento por lotes
- **Ventaja**: Reduce overhead de función por muestra
- **Complejidad**: O(B × operations) vs O(N × operations)
- **Beneficio**: Mejor utilización de caché y paralelización

#### 2. Gestión de memoria inteligente
- **std::unique_ptr**: Eliminación automática, O(1) para transferencia
- **Reutilización de tensores**: Reduce allocations dinámicas

#### 3. Validación temprana
- **Complejidad**: O(L) para validación vs potencial O(E × N × operations)
- **Beneficio**: Previene computación innecesaria

### Cuellos de botella identificados

#### 1. Creación de lotes
```cpp
// Copia elemento por elemento
X_batch(i, j) = X(start_idx + i, j);
```
- **Complejidad**: O(B × F_input) por lote
- **Optimización sugerida**: Usar vistas de memoria o slicing

#### 2. Cálculo de accuracy
```cpp
// Búsqueda lineal de máximo
for (size_t j = 1; j < out.shape()[1]; ++j) {
    if (out(i, j) > max_pred) { ... }
}
```
- **Complejidad**: O(B × C) por lote
- **Optimización**: Usar std::max_element

#### 3. Manejo de excepciones
- **Overhead**: Múltiples try-catch pueden afectar rendimiento
- **Beneficio**: Robustez vs performance



## Conclusiones

### Complejidad general
- **Entrenamiento**: O(E × N × P) donde P = Σ(F_i × F_{i+1})
- **Predicción**: O(M × P)
- **Memoria**: O(B × max(F_i) + P)

### Características de la implementación
- **Modularidad**: Excelente separación de responsabilidades
- **Robustez**: Manejo extensivo de errores
- **Eficiencia**: Competitiva para el tamaño de implementación

### Escalabilidad
- **Pequeños datasets**: Rendimiento adecuado
- **Datasets grandes**: Requiere optimizaciones adicionales
- **Redes profundas**: Escala linealmente con número de capas

La implementación presenta un balance sólido entre claridad de código y eficiencia computacional, siendo apropiada para propósitos educativos y aplicaciones de tamaño moderado.