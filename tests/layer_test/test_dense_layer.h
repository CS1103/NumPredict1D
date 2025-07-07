#pragma once

#include "../test_base.h"
#include "../../include/utec/neural_network/neural_network.h"
#include "../../include/utec/factories/nn_factory.h"
#include "../../include/utec/algebra/tensor.h"

using utec::neural_network::LayerFactory;
using utec::algebra::Tensor;

namespace tests {

class TestDenseLayer : public TestBase {
public:
    void run_tests() override {
        test_dense_layer_creation();
        test_dense_layer_forward_pass();
        test_dense_layer_backward_pass();
        test_dense_layer_dimensions();
        print_summary("TESTS DE CAPA DENSA");
    }

private:
    void test_dense_layer_creation() {
        print_test_header("TEST CREACION DE CAPA DENSA");

        bool all_passed = true;

        try {
            // Test con diferentes dimensiones
            auto dense_layer_1 = LayerFactory<float>::create_dense(3, 2);
            auto dense_layer_2 = LayerFactory<float>::create_dense(10, 5);
            auto dense_layer_3 = LayerFactory<float>::create_dense(1, 1);

            assert(dense_layer_1 != nullptr);
            assert(dense_layer_2 != nullptr);
            assert(dense_layer_3 != nullptr);

            std::cout << "Capas densas creadas correctamente con diferentes dimensiones\n";
            std::cout << "  Capa 1: 3 -> 2\n";
            std::cout << "  Capa 2: 10 -> 5\n";
            std::cout << "  Capa 3: 1 -> 1\n";

        } catch (const std::exception& e) {
            std::cout << "Error en creacion de capa densa: " << e.what() << "\n";
            all_passed = false;
        }

        print_test_result("Creacion de capa densa", all_passed);
    }

    void test_dense_layer_forward_pass() {
        print_test_header("TEST FORWARD PASS DE CAPA DENSA");

        bool all_passed = true;

        try {
            // Crear una capa densa simple 3->2
            auto dense_layer = LayerFactory<float>::create_dense(3, 2);

            // Crear tensor de entrada
            Tensor<float, 2> input(2, 3);  // 2 muestras, 3 caracteristicas
            input(0, 0) = 1.0f; input(0, 1) = 2.0f; input(0, 2) = 3.0f;
            input(1, 0) = 4.0f; input(1, 1) = 5.0f; input(1, 2) = 6.0f;

            std::cout << "Tensor de entrada creado: [2, 3]\n";
            std::cout << "  Muestra 1: [1.0, 2.0, 3.0]\n";
            std::cout << "  Muestra 2: [4.0, 5.0, 6.0]\n";

            // Forward pass
            auto output = dense_layer->forward(input);

            // Verificar dimensiones de salida
            assert(output.shape()[0] == 2);  // 2 muestras
            assert(output.shape()[1] == 2);  // 2 neuronas de salida
            std::cout << "Forward pass ejecutado correctamente\n";
            std::cout << "  Salida shape: [" << output.shape()[0] << ", " << output.shape()[1] << "]\n";
            std::cout << "  Salida muestra 1: [" << output(0, 0) << ", " << output(0, 1) << "]\n";
            std::cout << "  Salida muestra 2: [" << output(1, 0) << ", " << output(1, 1) << "]\n";

            // Verificar que la salida no es trivial (no todos ceros)
            bool has_non_zero = false;
            for (size_t i = 0; i < output.shape()[0]; ++i) {
                for (size_t j = 0; j < output.shape()[1]; ++j) {
                    if (std::abs(output(i, j)) > 1e-6f) {
                        has_non_zero = true;
                        break;
                    }
                }
                if (has_non_zero) break;
            }
            assert(has_non_zero);
            std::cout << "Salida contiene valores no triviales\n";

        } catch (const std::exception& e) {
            std::cout << "Error en forward pass: " << e.what() << "\n";
            all_passed = false;
        }

        print_test_result("Forward pass de capa densa", all_passed);
    }

    void test_dense_layer_backward_pass() {
        print_test_header("TEST BACKWARD PASS DE CAPA DENSA");

        bool all_passed = true;

        try {
            auto dense_layer = LayerFactory<float>::create_dense(3, 2);

            // Forward pass primero
            Tensor<float, 2> input(2, 3);
            input(0, 0) = 1.0f; input(0, 1) = 2.0f; input(0, 2) = 3.0f;
            input(1, 0) = 4.0f; input(1, 1) = 5.0f; input(1, 2) = 6.0f;

            auto output = dense_layer->forward(input);

            // Backward pass
            Tensor<float, 2> grad_output(2, 2);
            grad_output(0, 0) = 0.1f; grad_output(0, 1) = 0.2f;
            grad_output(1, 0) = 0.3f; grad_output(1, 1) = 0.4f;

            auto grad_input = dense_layer->backward(grad_output);

            // Verificar dimensiones del gradiente
            assert(grad_input.shape()[0] == 2);  // 2 muestras
            assert(grad_input.shape()[1] == 3);  // 3 caracteristicas de entrada
            std::cout << "Backward pass ejecutado correctamente\n";
            std::cout << "Grad input shape: [" << grad_input.shape()[0] << ", " << grad_input.shape()[1] << "]\n";

            // Verificar que los gradientes no son triviales
            bool has_non_zero_grad = false;
            for (size_t i = 0; i < grad_input.shape()[0]; ++i) {
                for (size_t j = 0; j < grad_input.shape()[1]; ++j) {
                    if (std::abs(grad_input(i, j)) > 1e-6f) {
                        has_non_zero_grad = true;
                        break;
                    }
                }
                if (has_non_zero_grad) break;
            }
            assert(has_non_zero_grad);
            std::cout << "Gradientes contienen valores no triviales\n";

        } catch (const std::exception& e) {
            std::cout << "Error en backward pass: " << e.what() << "\n";
            all_passed = false;
        }

        print_test_result("Backward pass de capa densa", all_passed);
    }

    void test_dense_layer_dimensions() {
        print_test_header("TEST DIMENSIONES DE CAPA DENSA");

        bool all_passed = true;

        try {
            // Test con diferentes tamanos de batch
            auto dense_layer = LayerFactory<float>::create_dense(4, 3);
            
            // Batch size 1
            Tensor<float, 2> input_1(1, 4);
            input_1.fill(1.0f);
            auto output_1 = dense_layer->forward(input_1);
            assert(output_1.shape()[0] == 1);
            assert(output_1.shape()[1] == 3);
            std::cout << "Batch size 1: [1, 4] -> [1, 3]\n";
            
            // Batch size 10
            Tensor<float, 2> input_10(10, 4);
            input_10.fill(1.0f);
            auto output_10 = dense_layer->forward(input_10);
            assert(output_10.shape()[0] == 10);
            assert(output_10.shape()[1] == 3);
            std::cout << "Batch size 10: [10, 4] -> [10, 3]\n";
            
            // Batch size 100
            Tensor<float, 2> input_100(100, 4);
            input_100.fill(1.0f);
            auto output_100 = dense_layer->forward(input_100);
            assert(output_100.shape()[0] == 100);
            assert(output_100.shape()[1] == 3);
            std::cout << "Batch size 100: [100, 4] -> [100, 3]\n";
            
        } catch (const std::exception& e) {
            std::cout << "Error en test de dimensiones: " << e.what() << "\n";
            all_passed = false;
        }
        
        print_test_result("Test de dimensiones de capa densa", all_passed);
    }
};

} // namespace tests
