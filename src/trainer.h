#ifndef TRAINER_H
#define TRAINER_H

#include "../include/utec/neural_network/neural_network.h"
#include "../include/utec/factories/nn_factory.h"
#include "../include/utec/data_processing/data_loader.h"
#include "config.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>

namespace utec::training {
    struct TrainingResult {
        std::string config_name;
        float accuracy;
        long long load_time_ms;
        long long train_time_ms;
        long long eval_time_ms;
        long long total_time_ms;
        size_t correct_predictions;
        size_t total_samples;
        TrainingResult() : accuracy(0.0f), load_time_ms(0), train_time_ms(0),
                          eval_time_ms(0), total_time_ms(0), correct_predictions(0), total_samples(0) {}
    };
    template<typename T>
    class Trainer {
    private:
        utec::neural_network::NeuralNetwork<T> nn;
        std::string data_path_train;
        std::string data_path_test;
        TrainingResult current_result;
        template<template<typename...> class LossFunction, template<typename...> class Optimizer>
        void train_with_config(const utec::config::TrainingConfig& config,
                              const utec::algebra::Tensor<T,2>& X_train,
                              const utec::algebra::Tensor<T,2>& Y_train);
    public:
        Trainer(const std::string& train_path, const std::string& test_path)
            : data_path_train(train_path), data_path_test(test_path) {
            setup_network();
        }
        void setup_network() {
            using namespace utec::neural_network;
            std::cout << "=== CONFIGURANDO ARQUITECTURA DE RED NEURONAL ===\n";
            std::cout << "Arquitectura:\n";
            std::cout << "  - Entrada: 64 neuronas (8x8 pixeles)\n";
            std::cout << "  - Capa oculta 1: 128 neuronas + ReLU\n";
            std::cout << "  - Capa oculta 2: 64 neuronas + ReLU\n";
            std::cout << "  - Capa de salida: 10 neuronas + Sigmoid\n\n";

            nn.add_layer(LayerFactory<T>::create_dense(64, 128));
            nn.add_layer(LayerFactory<T>::create_relu());
            nn.add_layer(LayerFactory<T>::create_dense(128, 64));
            nn.add_layer(LayerFactory<T>::create_relu());
            nn.add_layer(LayerFactory<T>::create_dense(64, 10));
            nn.add_layer(LayerFactory<T>::create_sigmoid());
            std::cout << "Red neuronal configurada exitosamente\n\n";
        }
        std::pair<utec::algebra::Tensor<T,2>, utec::algebra::Tensor<T,2>> load_data(bool is_train = true) {
            using namespace utec::neural_network;
            std::string path = is_train ? data_path_train : data_path_test;
            auto start = std::chrono::high_resolution_clock::now();
            auto [X, Y] = DataLoader<T>::load_csv(path);
            auto end = std::chrono::high_resolution_clock::now();
            auto load_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            if (is_train) {
                current_result.load_time_ms = load_time.count();
                std::cout << "Datos de entrenamiento cargados en " << load_time.count() << " ms\n";
                std::cout << "  - Muestras: " << X.shape()[0] << "\n";
                std::cout << "  - Caracteristicas: " << X.shape()[1] << "\n";
            } else {
                std::cout << "Datos de prueba cargados\n";
                std::cout << "  - Muestras: " << X.shape()[0] << "\n";
            }
            return {X, Y};
        }
        void evaluate(const utec::algebra::Tensor<T,2>& X_test,
                     const utec::algebra::Tensor<T,2>& Y_test) {
            std::cout << "=== EVALUANDO MODELO ===\n";
            auto start = std::chrono::high_resolution_clock::now();
            auto predictions = nn.predict(X_test);
            size_t correct = 0;
            size_t total_samples = X_test.shape()[0];

            for (size_t i = 0; i < total_samples; ++i) {
                size_t predicted = 0;
                T max_pred = predictions(i, 0);
                for (size_t j = 1; j < predictions.shape()[1]; ++j) {
                    if (predictions(i, j) > max_pred) {
                        max_pred = predictions(i, j);
                        predicted = j;
                    }
                }
                size_t actual = 0;
                T max_actual = Y_test(i, 0);
                for (size_t j = 1; j < Y_test.shape()[1]; ++j) {
                    if (Y_test(i, j) > max_actual) {
                        max_actual = Y_test(i, j);
                        actual = j;
                    }
                }
                if (predicted == actual) {
                    ++correct;
                }
            }
            auto end = std::chrono::high_resolution_clock::now();
            auto eval_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

            current_result.eval_time_ms = eval_time;
            current_result.correct_predictions = correct;
            current_result.total_samples = total_samples;
            current_result.accuracy = (float)correct / total_samples * 100.0f;
            current_result.total_time_ms = current_result.load_time_ms + current_result.train_time_ms + current_result.eval_time_ms;
            std::cout << "Evaluacion completada en " << eval_time << " ms\n";
            std::cout << "Precision: " << std::fixed << std::setprecision(2) << current_result.accuracy << "%\n";
            std::cout << "Muestras correctas: " << correct << " / " << total_samples << "\n\n";
        }
        void run_training(const utec::config::TrainingConfig& config) {
            using namespace utec::neural_network;
            std::cout << "=== INICIANDO EXPERIMENTO: " << config.name << " ===\n\n";

            auto [X_train, Y_train] = load_data(true);
            auto [X_test, Y_test] = load_data(false);

            if (config.loss_function == "BCELoss" && config.optimizer == "Adam") {
                this->template train_with_config<BCELoss, Adam>(config, X_train, Y_train);
            } else if (config.loss_function == "BCELoss" && config.optimizer == "SGD") {
                this->template train_with_config<BCELoss, SGD>(config, X_train, Y_train);
            } else if (config.loss_function == "MSELoss" && config.optimizer == "Adam") {
                this->template train_with_config<MSELoss, Adam>(config, X_train, Y_train);
            } else if (config.loss_function == "MSELoss" && config.optimizer == "SGD") {
                this->template train_with_config<MSELoss, SGD>(config, X_train, Y_train);
            } else {
                throw std::runtime_error("Configuracion no soportada: " + config.loss_function + " + " + config.optimizer);
            }

            evaluate(X_test, Y_test);
        }
        TrainingResult get_last_result() const {
            return current_result;
        }
        void reset_network() {
            nn = utec::neural_network::NeuralNetwork<T>();
            setup_network();
        }
    };
    template<typename T>
    template<template<typename...> class LossFunction, template<typename...> class Optimizer>
    void Trainer<T>::train_with_config(const utec::config::TrainingConfig& config,
                                      const utec::algebra::Tensor<T,2>& X_train,
                                      const utec::algebra::Tensor<T,2>& Y_train) {
        using namespace utec::neural_network;
        std::cout << "=== ENTRENANDO CON CONFIGURACION: " << config.name << " ===\n";
        std::cout << "Funcion de perdida: " << config.loss_function << "\n";
        std::cout << "Optimizador: " << config.optimizer << "\n";
        std::cout << "Epocas: " << config.epochs << "\n";
        std::cout << "Tamano de lote: " << config.batch_size << "\n";
        std::cout << "Tasa de aprendizaje: " << config.learning_rate << "\n";
        std::cout << "Lotes por epoca: " << (X_train.shape()[0] + config.batch_size - 1) / config.batch_size << "\n\n";
        auto start = std::chrono::high_resolution_clock::now();

        nn.template train<LossFunction, Optimizer>(X_train, Y_train,
            config.epochs, config.batch_size, 0, config.learning_rate);
        auto end = std::chrono::high_resolution_clock::now();
        current_result.train_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        current_result.config_name = config.name;
        std::cout << "Entrenamiento completado en " << current_result.train_time_ms << " ms\n";
        std::cout << "Tiempo promedio por epoca: " << current_result.train_time_ms / config.epochs << " ms\n\n";
    }
}
#endif // TRAINER_H
