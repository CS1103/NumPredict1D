#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_FACTORY_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_FACTORY_H

#include "../neural_network/nn_interfaces.h"
#include "../neural_network/nn_dense.h"
#include "../activations/nn_activation.h"
#include "../optimizers/nn_optimizer.h"
#include "../loss_functions/nn_loss.h"
#include <memory>
#include <string>
#include <stdexcept>
#include <random>

namespace utec::neural_network {

    template<typename T>
    class LayerFactory {
    public:
        static std::unique_ptr<ILayer<T>> create_layer(const std::string& type,
                                                      size_t input_size = 0,
                                                      size_t output_size = 0) {
            if (type == "dense") {
                if (input_size == 0 || output_size == 0) {
                    throw std::invalid_argument("Dense layer requires input_size and output_size");
                }

                auto init_w = [](Tensor<T,2>& w) {
                    std::random_device rd;
                    std::mt19937 gen(rd());
                    T fan_in = static_cast<T>(w.shape()[0]);
                    T fan_out = static_cast<T>(w.shape()[1]);
                    T limit = std::sqrt(T(6.0) / (fan_in + fan_out));
                    std::uniform_real_distribution<T> dist(-limit, limit);

                    for (size_t i = 0; i < w.size(); ++i) {
                        w[i] = dist(gen);
                    }
                };

                auto init_b = [](Tensor<T,2>& b) {
                    b.fill(T(0));
                };

                return std::make_unique<Dense<T>>(input_size, output_size, init_w, init_b);
            }
            else if (type == "relu") {
                return std::make_unique<ReLU<T>>();
            }
            else if (type == "sigmoid") {
                return std::make_unique<Sigmoid<T>>();
            }
            else {
                throw std::invalid_argument("Unknown layer type: " + type);
            }
        }

        template<typename InitW, typename InitB>
        static std::unique_ptr<ILayer<T>> create_dense(size_t input_size,
                                                      size_t output_size,
                                                      InitW init_w,
                                                      InitB init_b) {
            return std::make_unique<Dense<T>>(input_size, output_size, init_w, init_b);
        }

        static std::unique_ptr<ILayer<T>> create_dense(size_t input_size, size_t output_size) {
            return create_layer("dense", input_size, output_size);
        }

        static std::unique_ptr<ILayer<T>> create_relu() {
            return std::make_unique<ReLU<T>>();
        }

        static std::unique_ptr<ILayer<T>> create_sigmoid() {
            return std::make_unique<Sigmoid<T>>();
        }
    };

    template<typename T>
    class OptimizerFactory {
    public:
        static std::unique_ptr<IOptimizer<T>> create_optimizer(const std::string& type,
                                                              T learning_rate = T(0.01)) {
            if (type == "sgd") {
                return std::make_unique<SGD<T>>(learning_rate);
            }
            else if (type == "adam") {
                return std::make_unique<Adam<T>>(learning_rate);
            }
            else {
                throw std::invalid_argument("Unknown optimizer type: " + type);
            }
        }

        static std::unique_ptr<IOptimizer<T>> create_sgd(T learning_rate = T(0.01)) {
            return std::make_unique<SGD<T>>(learning_rate);
        }

        static std::unique_ptr<IOptimizer<T>> create_adam(T learning_rate = T(0.001),
                                                         T beta1 = T(0.9),
                                                         T beta2 = T(0.999),
                                                         T epsilon = T(1e-8)) {
            return std::make_unique<Adam<T>>(learning_rate, beta1, beta2, epsilon);
        }
    };

    template<typename T>
    class LossFactory {
    public:
        static std::unique_ptr<ILoss<T,2>> create_loss(const std::string& type,
                                                      const Tensor<T,2>& y_pred,
                                                      const Tensor<T,2>& y_true) {
            if (type == "mse") {
                return std::make_unique<MSELoss<T>>(y_pred, y_true);
            }
            else if (type == "bce") {
                return std::make_unique<BCELoss<T>>(y_pred, y_true);
            }
            else {
                throw std::invalid_argument("Unknown loss type: " + type);
            }
        }

        static std::unique_ptr<ILoss<T,2>> create_mse(const Tensor<T,2>& y_pred,
                                                     const Tensor<T,2>& y_true) {
            return std::make_unique<MSELoss<T>>(y_pred, y_true);
        }

        static std::unique_ptr<ILoss<T,2>> create_bce(const Tensor<T,2>& y_pred,
                                                     const Tensor<T,2>& y_true) {
            return std::make_unique<BCELoss<T>>(y_pred, y_true);
        }
    };

    template<typename T>
    class NeuralNetworkFactory {
    public:
        static std::unique_ptr<ILayer<T>> create_layer(const std::string& type,
                                                      size_t input_size = 0,
                                                      size_t output_size = 0) {
            return LayerFactory<T>::create_layer(type, input_size, output_size);
        }
        
        static std::unique_ptr<IOptimizer<T>> create_optimizer(const std::string& type,
                                                              T learning_rate = T(0.01)) {
            return OptimizerFactory<T>::create_optimizer(type, learning_rate);
        }
        
        static std::unique_ptr<ILoss<T,2>> create_loss(const std::string& type,
                                                      const Tensor<T,2>& y_pred,
                                                      const Tensor<T,2>& y_true) {
            return LossFactory<T>::create_loss(type, y_pred, y_true);
        }
    };

}

#endif // PROG3_NN_FINAL_PROJECT_V2025_01_FACTORY_H
