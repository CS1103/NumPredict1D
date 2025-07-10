# Análisis de Complejidad Algorítmica - Clase Tensor

## Introducción

Este documento presenta un análisis detallado de la complejidad algorítmica de la implementación de la clase `Tensor` en C++. La clase utiliza templates para soportar diferentes tipos de datos y rangos dimensionales, implementando operaciones fundamentales de álgebra tensorial con consideraciones de eficiencia computacional.

## Definiciones y Notación

- **n**: Número total de elementos en el tensor
- **R**: Rango (número de dimensiones) del tensor
- **d_i**: Tamaño de la dimensión i-ésima
- **N**: Producto de todas las dimensiones (n = d₁ × d₂ × ... × d_R)

## Análisis de Complejidad por Operación

### 1. Constructores

#### Constructor por defecto
```cpp
Tensor()
```
- **Complejidad temporal**: O(1)
- **Complejidad espacial**: O(1)
- **Análisis**: Inicializa arrays de tamaño fijo y un vector con un elemento

#### Constructor con forma (array)
```cpp
Tensor(const std::array<size_t, Rank>& shape)
```
- **Complejidad temporal**: O(R + n)
- **Complejidad espacial**: O(n)
- **Análisis**: 
  - Cálculo de strides: O(R)
  - Cálculo de tamaño total: O(R)
  - Inicialización del vector: O(n)

#### Constructor variádico con valores
```cpp
Tensor(Dims... dims, std::initializer_list<T> values)
```
- **Complejidad temporal**: O(R + n)
- **Complejidad espacial**: O(n)
- **Análisis**: Similar al anterior, pero con copia de valores desde initializer_list

### 2. Operaciones de Acceso

#### Acceso por índices multidimensionales
```cpp
T& operator()(Idxs... idxs)
```
- **Complejidad temporal**: O(R)
- **Complejidad espacial**: O(1)
- **Análisis**: Requiere calcular el índice plano usando la fórmula:
  ```
  índice_plano = Σ(i=0 to R-1) índice[i] × stride[i]
  ```

#### Acceso directo por índice plano
```cpp
T& operator[](size_t i)
```
- **Complejidad temporal**: O(1)
- **Complejidad espacial**: O(1)
- **Análisis**: Acceso directo al vector subyacente

### 3. Operaciones Aritméticas

#### Operaciones elemento a elemento (broadcasting)
```cpp
Tensor operator+(const Tensor& other) const
Tensor operator-(const Tensor& other) const
Tensor operator*(const Tensor& other) const
```
- **Complejidad temporal**: O(R + n)
- **Complejidad espacial**: O(n)
- **Análisis**:
  - Cálculo de forma broadcast: O(R)
  - Aplicación de operación: O(n)
  - Cada iteración requiere conversión de índices: O(R) por elemento
  - **Total**: O(n × R)

#### Operaciones con escalar
```cpp
Tensor operator+(const T& scalar) const
```
- **Complejidad temporal**: O(n)
- **Complejidad espacial**: O(n)
- **Análisis**: Iteración simple sobre todos los elementos

### 4. Operaciones de Álgebra Lineal

#### Multiplicación de matrices
```cpp
matrix_product(const Tensor<T, Rank>& a, const Tensor<T, Rank>& b)
```
- **Complejidad temporal**: O(m × n × p)
- **Complejidad espacial**: O(m × n)
- **Análisis**: Para matrices A(m×k) y B(k×n):
  - Tres bucles anidados
  - m × n × k operaciones de multiplicación y suma
  - Algoritmo clásico de multiplicación de matrices

#### Transposición 2D
```cpp
transpose_2d(const Tensor<T, Rank>& matrix)
```
- **Complejidad temporal**: O(m × n)
- **Complejidad espacial**: O(m × n)
- **Análisis**: Copia cada elemento a su posición transpuesta

### 5. Operaciones de Restructuración

#### Reshape
```cpp
void reshape(const std::array<size_t, Rank>& new_shape)
```
- **Complejidad temporal**: O(R + n)
- **Complejidad espacial**: O(n) en el peor caso
- **Análisis**:
  - Cálculo de nuevo tamaño: O(R)
  - Redimensionamiento del vector: O(n) si cambia el tamaño
  - Recálculo de strides: O(R)

#### Fill
```cpp
void fill(const T& value)
```
- **Complejidad temporal**: O(n)
- **Complejidad espacial**: O(1)
- **Análisis**: Asignación de valor a todos los elementos

### 6. Operaciones de Utilidad

#### Conversión de índices
```cpp
size_t linearIndex(const std::array<size_t, Rank>& idxs) const
std::array<size_t, Rank> multiIndex(size_t linear_idx) const
```
- **linearIndex**: O(R)
- **multiIndex**: O(R)
- **Análisis**: Conversión entre representaciones requiere R operaciones

#### Aplicación de funciones
```cpp
template<typename UnaryOp>
Tensor apply(UnaryOp op) const
```
- **Complejidad temporal**: O(n × f)
- **Complejidad espacial**: O(n)
- **Análisis**: Donde f es la complejidad de la operación unaria

## Análisis de Eficiencia de Memoria

### Estructura de Datos
- **Arrays de forma y strides**: O(R) - Tamaño constante para R fijo
- **Vector de datos**: O(n) - Almacenamiento principal
- **Overhead total**: O(R + n) ≈ O(n) para tensores grandes

### Localidad de Referencia
- **Acceso secuencial**: Excelente localidad (operaciones como fill, apply)
- **Acceso por índices**: Buena localidad si se accede en orden row-major
- **Broadcasting**: Puede causar patrones de acceso no locales

## Optimizaciones Implementadas

### 1. Cálculo de Strides
- **Ventaja**: Conversión O(R) de índices multidimensionales a lineales
- **Alternativa**: Cálculo on-the-fly sería O(R) por cada acceso

### 2. Almacenamiento Contiguo
- **Ventaja**: Mejor localidad de caché y compatibilidad con bibliotecas externas
- **Complejidad**: Mantiene O(n) de memoria con acceso O(1)

### 3. Template Specialization
- **Ventaja**: Optimizaciones en tiempo de compilación
- **Complejidad**: No afecta la complejidad asintótica, pero mejora constantes

## Casos de Uso y Complejidad

### Tensor 1D (Vector)
- Acceso: O(1)
- Operaciones: O(n)
- Equivalente a std::vector con operaciones matemáticas

### Tensor 2D (Matrix)
- Acceso: O(1)
- Multiplicación: O(n³)
- Transposición: O(n²)

### Tensor 3D+
- Acceso: O(R)
- Operaciones elemento a elemento: O(n)
- Broadcasting: O(n × R)

## Conclusiones y Recomendaciones

### Fortalezas
1. **Acceso eficiente**: O(1) para acceso directo, O(R) para multidimensional
2. **Operaciones escalares**: Complejidad lineal óptima O(n)
3. **Memoria contigua**: Buena localidad de caché

### Áreas de Mejora
1. **Broadcasting**: Podría optimizarse para casos especiales
2. **Multiplicación de matrices**: Implementar algoritmos más eficientes (Strassen, etc.)
3. **Paralelización**: Las operaciones elemento a elemento son paralelizables

### Complejidad General
- **Mejor caso**: O(1) para acceso directo
- **Caso típico**: O(n) para la mayoría de operaciones
- **Peor caso**: O(n³) para multiplicación de matrices grandes

La implementación presenta un balance adecuado entre simplicidad y eficiencia, con complejidades que escalan apropiadamente con el tamaño y dimensionalidad de los tensores.