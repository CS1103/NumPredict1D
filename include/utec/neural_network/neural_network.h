#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H

#include "nn_interfaces.h"
#include "activations/nn_activation.h"
#include "optimizers/nn_optimizer.h"
#include "algebra/tensor.h"
#include <memory>
#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>

namespace utec::neural_network {

    template<typename T>
    class NeuralNetwork {
        std::vector<std::unique_ptr<ILayer<T>>> layers_;

    public:
        void add_layer(std::unique_ptr<ILayer<T>> layer) {
            layers_.push_back(std::move(layer));
        }

        template<template<typename...> class LossType,
                 template<typename...> class OptimizerType = SGD>
        void train(const utec::algebra::Tensor<T,2>& X,
                   const utec::algebra::Tensor<T,2>& Y,
                   size_t epochs,
                   size_t batch_size,
                   size_t verbose,
                   T learning_rate)
        {
            if (layers_.empty()) {
                std::cout << "ERROR: No hay capas en la red!\n";
                return;
            }

            if (X.shape()[0] != Y.shape()[0]) {
                std::cout << "ERROR: Numero de muestras no coincide entre X e Y!\n";
                return;
            }

            if (batch_size == 0 || batch_size > X.shape()[0]) {
                std::cout << "ERROR: Tamanio de lote invalido!\n";
                return;
            }

            for (size_t i = 0; i < layers_.size(); ++i) {
                if (!layers_[i]) {
                    std::cout << "ERROR: Capa " << (i + 1) << " es nullptr!\n";
                    return;
                }
            }

            OptimizerType<T> opt(learning_rate);
            size_t num_samples = X.shape()[0];
            size_t num_batches = (num_samples + batch_size - 1) / batch_size;

            for (size_t epoch = 0; epoch < epochs; ++epoch) {
                auto epoch_start = std::chrono::high_resolution_clock::now();

                T total_loss = 0.0;
                size_t correct_predictions = 0;

                for (size_t batch = 0; batch < num_batches; ++batch) {
                    size_t start_idx = batch * batch_size;
                    size_t end_idx = std::min(start_idx + batch_size, num_samples);
                    size_t actual_batch_size = end_idx - start_idx;

                    try {
                        utec::algebra::Tensor<T,2> X_batch(actual_batch_size, X.shape()[1]);
                        utec::algebra::Tensor<T,2> Y_batch(actual_batch_size, Y.shape()[1]);

                        for (size_t i = 0; i < actual_batch_size; ++i) {
                            if (start_idx + i >= num_samples) {
                                return;
                            }

                            for (size_t j = 0; j < X.shape()[1]; ++j) {
                                X_batch(i, j) = X(start_idx + i, j);
                            }
                            for (size_t j = 0; j < Y.shape()[1]; ++j) {
                                Y_batch(i, j) = Y(start_idx + i, j);
                            }
                        }

                        auto out = X_batch;
                        for (size_t i = 0; i < layers_.size(); ++i) {
                            out = layers_[i]->forward(out);
                        }

                        if (out.shape()[0] != Y_batch.shape()[0] || out.shape()[1] != Y_batch.shape()[1]) {
                            return;
                        }

                        LossType<T> loss_fn(out, Y_batch);
                        auto grad = loss_fn.loss_gradient();
                        T batch_loss = loss_fn.loss();
                        total_loss += batch_loss;

                        for (size_t i = 0; i < actual_batch_size; ++i) {
                            size_t predicted_class = 0;
                            T max_pred = out(i, 0);
                            for (size_t j = 1; j < out.shape()[1]; ++j) {
                                if (out(i, j) > max_pred) {
                                    max_pred = out(i, j);
                                    predicted_class = j;
                                }
                            }

                            size_t true_class = 0;
                            T max_true = Y_batch(i, 0);
                            for (size_t j = 1; j < Y_batch.shape()[1]; ++j) {
                                if (Y_batch(i, j) > max_true) {
                                    max_true = Y_batch(i, j);
                                    true_class = j;
                                }
                            }

                            if (predicted_class == true_class) {
                                correct_predictions++;
                            }
                        }

                        if (grad.shape()[0] == 0 || grad.shape()[1] == 0) {
                            return;
                        }

                        for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; --i) {
                            if (!layers_[i]) {
                                return;
                            }

                            if (grad.shape()[0] == 0 || grad.shape()[1] == 0) {
                                return;
                            }

                            try {
                                grad = layers_[i]->backward(grad);

                                if (grad.shape()[0] == 0 || grad.shape()[1] == 0) {
                                    return;
                                }

                                layers_[i]->update_params(opt);

                            } catch (const std::exception& e) {
                                return;
                            } catch (...) {
                                return;
                            }
                        }

                    } catch (const std::exception& e) {
                        return;
                    } catch (...) {
                        return;
                    }
                }

                auto epoch_end = std::chrono::high_resolution_clock::now();
                auto epoch_time = std::chrono::duration_cast<std::chrono::milliseconds>(epoch_end - epoch_start);

                T avg_loss = total_loss / num_batches;
                T accuracy = static_cast<T>(correct_predictions) / num_samples;

                std::cout << "Epoch " << (epoch + 1) << "/" << epochs << "\n";
                std::cout << num_batches << "/" << num_batches << " "
                          << epoch_time.count() << "ms "
                          << (epoch_time.count() / num_batches) << "ms/step"
                          << " - accuracy: " << std::fixed << std::setprecision(4) << accuracy
                          << " - loss: " << std::fixed << std::setprecision(4) << avg_loss;
                std::cout << "\n";
            }

            std::cout << "Entrenamiento completado!\n";
        }

        utec::algebra::Tensor<T,2> predict(const utec::algebra::Tensor<T,2>& X) {
            if (layers_.empty()) {
                return utec::algebra::Tensor<T,2>(0, 0);
            }

            utec::algebra::Tensor<T,2> sample_input(1, X.shape()[1]);
            for (size_t j = 0; j < X.shape()[1]; ++j) {
                sample_input(0, j) = X(0, j);
            }

            auto sample_output = sample_input;
            for (size_t i = 0; i < layers_.size(); ++i) {
                sample_output = layers_[i]->forward(sample_output);
            }

            size_t output_size = sample_output.shape()[1];

            size_t batch_size = 100;
            size_t num_samples = X.shape()[0];
            size_t num_batches = (num_samples + batch_size - 1) / batch_size;

            utec::algebra::Tensor<T,2> results(num_samples, output_size);

            for (size_t batch = 0; batch < num_batches; ++batch) {
                size_t start_idx = batch * batch_size;
                size_t end_idx = std::min(start_idx + batch_size, num_samples);
                size_t actual_batch_size = end_idx - start_idx;

                utec::algebra::Tensor<T,2> X_batch(actual_batch_size, X.shape()[1]);
                for (size_t i = 0; i < actual_batch_size; ++i) {
                    for (size_t j = 0; j < X.shape()[1]; ++j) {
                        X_batch(i, j) = X(start_idx + i, j);
                    }
                }

                auto out = X_batch;
                for (size_t i = 0; i < layers_.size(); ++i) {
                    out = layers_[i]->forward(out);
                }

                for (size_t i = 0; i < actual_batch_size; ++i) {
                    for (size_t j = 0; j < out.shape()[1]; ++j) {
                        results(start_idx + i, j) = out(i, j);
                    }
                }
            }

            return results;
        }
    };

}

#endif // PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H
