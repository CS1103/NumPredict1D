#pragma once
#include "../test_base.h"
#include "../../include/utec/neural_network/neural_network.h"
#include "../../include/utec/factories/nn_factory.h"
#include "../../include/utec/algebra/tensor.h"
#include "../../include/utec/loss_functions/nn_loss.h"
#include "../../include/utec/optimizers/nn_optimizer.h"
#include <chrono>
#include <iomanip>

using utec::neural_network::LayerFactory;
using utec::neural_network::NeuralNetwork;
using utec::neural_network::MSELoss;
using utec::neural_network::BCELoss;
using utec::neural_network::Adam;
using utec::neural_network::SGD;
using utec::algebra::Tensor;

namespace tests {
class TestConvergence : public TestBase {
public:
    void run_tests() override {
        test_simple_xor_convergence();
        test_linear_regression_convergence();
        test_binary_classification_convergence();
        print_summary("TESTS DE CONVERGENCIA");
    }
private:
    // Funcion auxiliar para calcular perdida MSE
    float calculate_mse_loss(const Tensor<float, 2>& predictions, const Tensor<float, 2>& targets) {
        float loss = 0.0f;
        size_t n_samples = predictions.shape()[0];
        for (size_t i = 0; i < n_samples; ++i) {
            float diff = predictions(i, 0) - targets(i, 0);
            loss += diff * diff;
        }
        return loss / n_samples;
    }
    // Funcion auxiliar para calcular precision
    float calculate_accuracy(const Tensor<float, 2>& predictions, const Tensor<float, 2>& targets) {
        size_t correct = 0;
        size_t total = predictions.shape()[0];
        for (size_t i = 0; i < total; ++i) {
            // Para clasificacion binaria
            if (predictions.shape()[1] == 1) {
                float pred = predictions(i, 0) > 0.5f ? 1.0f : 0.0f;
                if (pred == targets(i, 0)) {
                    correct++;
                }
            } else {
                // Para clasificacion multiclase
                size_t predicted_class = 0;
                float max_pred = predictions(i, 0);
                for (size_t j = 1; j < predictions.shape()[1]; ++j) {
                    if (predictions(i, j) > max_pred) {
                        max_pred = predictions(i, j);
                        predicted_class = j;
                    }
                }
                size_t actual_class = 0;
                float max_actual = targets(i, 0);
                for (size_t j = 1; j < targets.shape()[1]; ++j) {
                    if (targets(i, j) > max_actual) {
                        max_actual = targets(i, j);
                        actual_class = j;
                    }
                }
                if (predicted_class == actual_class) {
                    correct++;
                }
            }
        }
        return static_cast<float>(correct) / total;
    }
    void test_simple_xor_convergence() {
        print_test_header("TEST DE CONVERGENCIA EN PROBLEMA XOR");
        bool all_passed = true;
        try {
            std::cout << "Creando dataset XOR...\n";
            // Dataset XOR: 4 muestras con 2 caracteristicas cada una
            Tensor<float, 2> X_train(4, 2);
            Tensor<float, 2> Y_train(4, 1);
            // Patron XOR clasico
            X_train(0, 0) = 0.0f; X_train(0, 1) = 0.0f; Y_train(0, 0) = 0.0f; // 0 XOR 0 = 0
            X_train(1, 0) = 0.0f; X_train(1, 1) = 1.0f; Y_train(1, 0) = 1.0f; // 0 XOR 1 = 1
            X_train(2, 0) = 1.0f; X_train(2, 1) = 0.0f; Y_train(2, 0) = 1.0f; // 1 XOR 0 = 1
            X_train(3, 0) = 1.0f; X_train(3, 1) = 1.0f; Y_train(3, 0) = 0.0f; // 1 XOR 1 = 0
            std::cout << "Dataset XOR creado (4 muestras, 2 caracteristicas)\n";
            // Crear red neuronal
            NeuralNetwork<float> nn;
            nn.add_layer(LayerFactory<float>::create_dense(2, 8));  // Mas neuronas para XOR
            nn.add_layer(LayerFactory<float>::create_relu());
            nn.add_layer(LayerFactory<float>::create_dense(8, 4));
            nn.add_layer(LayerFactory<float>::create_relu());
            nn.add_layer(LayerFactory<float>::create_dense(4, 1));
            nn.add_layer(LayerFactory<float>::create_sigmoid());
            std::cout << "Red neuronal creada (2->8->4->1)\n";
            // Configuracion de entrenamiento
            const int epochs = 1000;
            const int batch_size = 4;  // Usar todas las muestras por lote
            const float learning_rate = 0.01f;
            std::cout << "Configuracion: " << epochs << " epocas, lr=" << learning_rate << ", batch_size=" << batch_size << "\n";
            // Prediccion inicial
            auto initial_pred = nn.predict(X_train);
            float initial_loss = calculate_mse_loss(initial_pred, Y_train);
            float initial_accuracy = calculate_accuracy(initial_pred, Y_train);
            std::cout << "Estado inicial:\n";
            std::cout << "  Perdida: " << std::fixed << std::setprecision(6) << initial_loss << "\n";
            std::cout << "  Precision: " << std::setprecision(2) << initial_accuracy * 100 << "%\n";
            // Entrenar la red usando el metodo correcto
            auto start_time = std::chrono::high_resolution_clock::now();
            // Usar MSELoss + Adam para XOR (funciona bien para este problema)
            nn.train<MSELoss, Adam>(X_train, Y_train, epochs, batch_size, 0, learning_rate);
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            // Evaluacion final
            auto final_pred = nn.predict(X_train);
            float final_loss = calculate_mse_loss(final_pred, Y_train);
            float final_accuracy = calculate_accuracy(final_pred, Y_train);
            std::cout << "\nResultados finales:\n";
            std::cout << "  Tiempo de entrenamiento: " << duration.count() << " ms\n";
            std::cout << "  Perdida final: " << final_loss << "\n";
            std::cout << "  Precision final: " << final_accuracy * 100 << "%\n";
            std::cout << "  Mejora en perdida: " << (initial_loss - final_loss) << "\n";
            std::cout << "  Mejora en precision: " << (final_accuracy - initial_accuracy) * 100 << "%\n";
            // Verificar convergencia
            assert(final_loss < initial_loss * 0.5f); // Debe mejorar al menos 50%
            assert(final_accuracy > 0.6f); // Debe lograr al menos 60% de precision
            std::cout << "La red converge correctamente en XOR\n";
            // Verificar predicciones especificas
            std::cout << "\nPredicciones XOR detalladas:\n";
            for (size_t i = 0; i < 4; ++i) {
                float pred_value = final_pred(i, 0);
                float target_value = Y_train(i, 0);
                std::cout << "  " << X_train(i, 0) << " XOR " << X_train(i, 1)
                         << " = " << std::setprecision(4) << pred_value
                         << " (esperado: " << target_value << ")\n";
            }
        } catch (const std::exception& e) {
            std::cout << "Error en test de convergencia XOR: " << e.what() << "\n";
            all_passed = false;
        }
        print_test_result("Test de convergencia XOR", all_passed);
    }
    void test_linear_regression_convergence() {
        print_test_header("TEST DE CONVERGENCIA EN REGRESION LINEAL");
        bool all_passed = true;
        try {
            std::cout << "Creando dataset de regresion lineal...\n";
            // Dataset simple: y = 2x + 1 con algo de ruido
            const int n_samples = 100;
            Tensor<float, 2> X_train(n_samples, 1);
            Tensor<float, 2> Y_train(n_samples, 1);
            // Normalizar los datos para mejor convergencia
            for (int i = 0; i < n_samples; ++i) {
                float x = static_cast<float>(i) / (n_samples - 1); // x va de 0 a 1
                X_train(i, 0) = x;
                Y_train(i, 0) = 2.0f * x + 1.0f + (rand() % 100 - 50) / 5000.0f; // y = 2x + 1 + ruido pequeno
            }
            std::cout << "Dataset de regresion creado (" << n_samples << " muestras)\n";
            // Crear red neuronal simple para regresion
            NeuralNetwork<float> nn;
            nn.add_layer(LayerFactory<float>::create_dense(1, 16));
            nn.add_layer(LayerFactory<float>::create_relu());
            nn.add_layer(LayerFactory<float>::create_dense(16, 8));
            nn.add_layer(LayerFactory<float>::create_relu());
            nn.add_layer(LayerFactory<float>::create_dense(8, 1));
            std::cout << "Red neuronal creada (1->16->8->1)\n";
            // Configuracion de entrenamiento
            const int epochs = 500;
            const int batch_size = 20;
            const float learning_rate = 0.001f;
            std::cout << "Configuracion: " << epochs << " epocas, lr=" << learning_rate << ", batch_size=" << batch_size << "\n";
            // Prediccion inicial
            auto initial_pred = nn.predict(X_train);
            float initial_loss = calculate_mse_loss(initial_pred, Y_train);
            std::cout << "Perdida inicial: " << initial_loss << "\n";
            // Entrenar usando MSELoss + Adam
            auto start_time = std::chrono::high_resolution_clock::now();
            nn.train<MSELoss, Adam>(X_train, Y_train, epochs, batch_size, 0, learning_rate);
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            // Evaluacion final
            auto final_pred = nn.predict(X_train);
            float final_loss = calculate_mse_loss(final_pred, Y_train);
            std::cout << "Tiempo de entrenamiento: " << duration.count() << " ms\n";
            std::cout << "Perdida final: " << final_loss << "\n";
            std::cout << "Mejora: " << (initial_loss - final_loss) << "\n";
            // Verificar convergencia
            assert(final_loss < initial_loss * 0.1f);
            assert(final_loss < 0.1f); // Debe tener perdida razonable
            std::cout << "La red converge correctamente en regresion lineal\n";
            // Mostrar algunas predicciones
            std::cout << "\nAlgunas predicciones:\n";
            for (int i = 0; i < std::min(5, n_samples); ++i) {
                std::cout << "  x=" << X_train(i, 0) << " -> y=" << final_pred(i, 0)
                         << " (esperado: " << Y_train(i, 0) << ")\n";
            }
        } catch (const std::exception& e) {
            std::cout << "Error en test de regresion lineal: " << e.what() << "\n";
            all_passed = false;
        }
        print_test_result("Test de convergencia regresion lineal", all_passed);
    }
    void test_binary_classification_convergence() {
        print_test_header("TEST DE CONVERGENCIA EN CLASIFICACION BINARIA");
        bool all_passed = true;
        try {
            std::cout << "Creando dataset de clasificacion binaria...\n";
            // Dataset separable linealmente
            const int n_samples = 200;
            Tensor<float, 2> X_train(n_samples, 2);
            Tensor<float, 2> Y_train(n_samples, 1);
            // Crear dos clases separables
            for (int i = 0; i < n_samples / 2; ++i) {
                // Clase 0: puntos en el cuadrante inferior izquierdo
                X_train(i, 0) = (rand() % 100) / 200.0f; // [0, 0.5]
                X_train(i, 1) = (rand() % 100) / 200.0f; // [0, 0.5]
                Y_train(i, 0) = 0.0f;
                // Clase 1: puntos en el cuadrante superior derecho
                X_train(i + n_samples/2, 0) = 0.5f + (rand() % 100) / 200.0f; // [0.5, 1.0]
                X_train(i + n_samples/2, 1) = 0.5f + (rand() % 100) / 200.0f; // [0.5, 1.0]
                Y_train(i + n_samples/2, 0) = 1.0f;
            }
            std::cout << "Dataset de clasificacion creado (" << n_samples << " muestras)\n";
            // Crear red neuronal para clasificacion
            NeuralNetwork<float> nn;
            nn.add_layer(LayerFactory<float>::create_dense(2, 16));
            nn.add_layer(LayerFactory<float>::create_relu());
            nn.add_layer(LayerFactory<float>::create_dense(16, 8));
            nn.add_layer(LayerFactory<float>::create_relu());
            nn.add_layer(LayerFactory<float>::create_dense(8, 1));
            nn.add_layer(LayerFactory<float>::create_sigmoid());
            std::cout << "Red neuronal creada (2->16->8->1)\n";
            // Configuracion de entrenamiento
            const int epochs = 300;
            const int batch_size = 32;
            const float learning_rate = 0.01f;
            std::cout << "Configuracion: " << epochs << " epocas, lr=" << learning_rate << ", batch_size=" << batch_size << "\n";
            // Prediccion inicial
            auto initial_pred = nn.predict(X_train);
            float initial_loss = calculate_mse_loss(initial_pred, Y_train);
            float initial_accuracy = calculate_accuracy(initial_pred, Y_train);
            std::cout << "Estado inicial:\n";
            std::cout << "  Perdida: " << initial_loss << "\n";
            std::cout << "  Precision: " << initial_accuracy * 100 << "%\n";
            // Entrenar usando BCELoss + Adam (mejor para clasificacion binaria)
            auto start_time = std::chrono::high_resolution_clock::now();
            nn.train<BCELoss, Adam>(X_train, Y_train, epochs, batch_size, 0, learning_rate);
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            // Evaluacion final
            auto final_pred = nn.predict(X_train);
            float final_loss = calculate_mse_loss(final_pred, Y_train);
            float final_accuracy = calculate_accuracy(final_pred, Y_train);
            std::cout << "\nResultados finales:\n";
            std::cout << "  Tiempo de entrenamiento: " << duration.count() << " ms\n";
            std::cout << "  Perdida final: " << final_loss << "\n";
            std::cout << "  Precision final: " << final_accuracy * 100 << "%\n";
            std::cout << "  Mejora en perdida: " << (initial_loss - final_loss) << "\n";
            std::cout << "  Mejora en precision: " << (final_accuracy - initial_accuracy) * 100 << "%\n";
            // Verificar convergencia
            assert(final_loss < initial_loss * 0.3f);
            assert(final_accuracy > 0.90f); // Al menos 90% de precision
            std::cout << "La red converge correctamente en clasificacion binaria\n";
            // Mostrar distribucion de predicciones
            int class_0_correct = 0, class_1_correct = 0;
            for (size_t i = 0; i < n_samples; ++i) {
                float pred = final_pred(i, 0) > 0.5f ? 1.0f : 0.0f;
                if (pred == Y_train(i, 0)) {
                    if (Y_train(i, 0) == 0.0f) class_0_correct++;
                    else class_1_correct++;
                }
            }
            std::cout << "\nPrecision por clase:\n";
            std::cout << "  Clase 0: " << (class_0_correct * 100 / (n_samples/2)) << "%\n";
            std::cout << "  Clase 1: " << (class_1_correct * 100 / (n_samples/2)) << "%\n";
        } catch (const std::exception& e) {
            std::cout << "Error en test de clasificacion binaria: " << e.what() << "\n";
            all_passed = false;
        }
        print_test_result("Test de convergencia clasificacion binaria", all_passed);
    }
};
} // namespace tests
