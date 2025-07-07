    #pragma once

    #include <algebra/tensor.h>
    #include <factories/nn_factory.h>

    #include "../test_base.h"
    #include "../../include/utec/neural_network/neural_network.h"

    using utec::neural_network::LayerFactory;
    using utec::algebra::Tensor;

    namespace tests {

    class TestActivations : public TestBase {
    public:
        void run_tests() override {
            test_relu_activation();
            test_sigmoid_activation();
            test_activation_backward_pass();
            print_summary("TESTS DE FUNCIONES DE ACTIVACION");
        }

    private:
        void test_relu_activation() {
            print_test_header("TEST DE FUNCION DE ACTIVACION ReLU");

            bool all_passed = true;

            try {
                // Crear capa ReLU
                auto relu_layer = LayerFactory<float>::create_relu();
                assert(relu_layer != nullptr);
                std::cout << "Capa ReLU creada correctamente\n";

                // Test valores positivos (deben mantenerse)
                Tensor<float, 2> input_positive(1, 4);
                input_positive(0, 0) = 1.0f;
                input_positive(0, 1) = 2.5f;
                input_positive(0, 2) = 10.0f;
                input_positive(0, 3) = 0.1f;

                auto output_positive = relu_layer->forward(input_positive);

                assert(is_close(output_positive(0, 0), 1.0f));
                assert(is_close(output_positive(0, 1), 2.5f));
                assert(is_close(output_positive(0, 2), 10.0f));
                assert(is_close(output_positive(0, 3), 0.1f));
                std::cout << "Valores positivos se mantienen correctamente\n";

                // Test valores negativos (deben convertirse a 0)
                Tensor<float, 2> input_negative(1, 4);
                input_negative(0, 0) = -1.0f;
                input_negative(0, 1) = -2.5f;
                input_negative(0, 2) = -10.0f;
                input_negative(0, 3) = -0.1f;

                auto output_negative = relu_layer->forward(input_negative);

                assert(is_close(output_negative(0, 0), 0.0f));
                assert(is_close(output_negative(0, 1), 0.0f));
                assert(is_close(output_negative(0, 2), 0.0f));
                assert(is_close(output_negative(0, 3), 0.0f));
                std::cout << "Valores negativos se convierten a 0 correctamente\n";

                // Test valores mixtos
                Tensor<float, 2> input_mixed(1, 6);
                input_mixed(0, 0) = -5.0f;
                input_mixed(0, 1) = 0.0f;
                input_mixed(0, 2) = 3.0f;
                input_mixed(0, 3) = -1.5f;
                input_mixed(0, 4) = 7.2f;
                input_mixed(0, 5) = -0.01f;

                auto output_mixed = relu_layer->forward(input_mixed);

                assert(is_close(output_mixed(0, 0), 0.0f));   // -5.0 -> 0.0
                assert(is_close(output_mixed(0, 1), 0.0f));   // 0.0 -> 0.0
                assert(is_close(output_mixed(0, 2), 3.0f));   // 3.0 -> 3.0
                assert(is_close(output_mixed(0, 3), 0.0f));   // -1.5 -> 0.0
                assert(is_close(output_mixed(0, 4), 7.2f));   // 7.2 -> 7.2
                assert(is_close(output_mixed(0, 5), 0.0f));   // -0.01 -> 0.0
                std::cout << "Valores mixtos procesados correctamente\n";

            } catch (const std::exception& e) {
                std::cout << "Error en test de ReLU: " << e.what() << "\n";
                all_passed = false;
            }

            print_test_result("Test de funcion de activacion ReLU", all_passed);
        }

        void test_sigmoid_activation() {
            print_test_header("TEST DE FUNCION DE ACTIVACION SIGMOID");

            bool all_passed = true;

            try {
                auto sigmoid_layer = LayerFactory<float>::create_sigmoid();
                assert(sigmoid_layer != nullptr);
                std::cout << "Capa Sigmoid creada correctamente\n";

                // Test valores tipicos
                Tensor<float, 2> input(1, 5);
                input(0, 0) = 0.0f;    // sigmoid(0) = 0.5
                input(0, 1) = 1.0f;    // sigmoid(1) ≈ 0.731
                input(0, 2) = -1.0f;   // sigmoid(-1) ≈ 0.269
                input(0, 3) = 5.0f;    // sigmoid(5) ≈ 0.993
                input(0, 4) = -5.0f;   // sigmoid(-5) ≈ 0.007

                auto output = sigmoid_layer->forward(input);

                // Verificar que las salidas estan en el rango [0, 1]
                for (size_t i = 0; i < output.shape()[1]; ++i) {
                    assert(output(0, i) >= 0.0f && output(0, i) <= 1.0f);
                }

                // Verificar valores especificos
                assert(is_close(output(0, 0), 0.5f, 1e-3f));
                assert(output(0, 1) > 0.7f && output(0, 1) < 0.8f);
                assert(output(0, 2) > 0.2f && output(0, 2) < 0.3f);
                assert(output(0, 3) > 0.99f);
                assert(output(0, 4) < 0.01f);

                std::cout << "Sigmoid produce valores en rango [0, 1]\n";
                std::cout << "  sigmoid(0) = " << output(0, 0) << "\n";
                std::cout << "  sigmoid(1) = " << output(0, 1) << "\n";
                std::cout << "  sigmoid(-1) = " << output(0, 2) << "\n";

            } catch (const std::exception& e) {
                std::cout << "Error en test de Sigmoid: " << e.what() << "\n";
                all_passed = false;
            }

            print_test_result("Test de funcion de activacion Sigmoid", all_passed);
        }

        void test_activation_backward_pass() {
            print_test_header("TEST BACKWARD PASS DE ACTIVACIONES");

            bool all_passed = true;

            try {
                // Test ReLU backward
                auto relu_layer = LayerFactory<float>::create_relu();

                Tensor<float, 2> input_relu(1, 6);
                input_relu(0, 0) = -5.0f;
                input_relu(0, 1) = 0.0f;
                input_relu(0, 2) = 3.0f;
                input_relu(0, 3) = -1.5f;
                input_relu(0, 4) = 7.2f;
                input_relu(0, 5) = -0.01f;

                auto output_relu = relu_layer->forward(input_relu);

                Tensor<float, 2> grad_output_relu(1, 6);
                grad_output_relu.fill(1.0f);

                auto grad_input_relu = relu_layer->backward(grad_output_relu);

                // El gradiente debe ser 0 para valores negativos y 1 para valores positivos
                assert(is_close(grad_input_relu(0, 0), 0.0f));  // input fue -5.0
                assert(is_close(grad_input_relu(0, 1), 0.0f));  // input fue 0.0
                assert(is_close(grad_input_relu(0, 2), 1.0f));  // input fue 3.0
                assert(is_close(grad_input_relu(0, 3), 0.0f));  // input fue -1.5
                assert(is_close(grad_input_relu(0, 4), 1.0f));  // input fue 7.2
                assert(is_close(grad_input_relu(0, 5), 0.0f));  // input fue -0.01
                std::cout << "Backward pass de ReLU correcto\n";

                // Test Sigmoid backward
                auto sigmoid_layer = LayerFactory<float>::create_sigmoid();

                Tensor<float, 2> input_sigmoid(1, 3);
                input_sigmoid(0, 0) = 0.0f;
                input_sigmoid(0, 1) = 1.0f;
                input_sigmoid(0, 2) = -1.0f;

                auto output_sigmoid = sigmoid_layer->forward(input_sigmoid);

                Tensor<float, 2> grad_output_sigmoid(1, 3);
                grad_output_sigmoid.fill(1.0f);

                auto grad_input_sigmoid = sigmoid_layer->backward(grad_output_sigmoid);

                // Los gradientes deben ser positivos y menores que 1
                for (size_t i = 0; i < grad_input_sigmoid.shape()[1]; ++i) {
                    assert(grad_input_sigmoid(0, i) > 0.0f);
                    assert(grad_input_sigmoid(0, i) <= 1.0f);
                }
                std::cout << "Backward pass de Sigmoid correcto\n";

            } catch (const std::exception& e) {
                std::cout << "Error en test de backward pass: " << e.what() << "\n";
                all_passed = false;
            }

            print_test_result("Test backward pass de activaciones", all_passed);
        }
    };

    } // namespace tests
