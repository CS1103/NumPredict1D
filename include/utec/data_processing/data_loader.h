#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_DATA_LOADER_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_DATA_LOADER_H

#include "algebra/tensor.h"
#include <fstream>
#include <sstream>
#include <vector>

namespace utec::neural_network {

    template<typename T>
    class DataLoader {
    public:
        static std::pair<utec::algebra::Tensor<T,2>, utec::algebra::Tensor<T,2>> load_csv(const std::string& filename) {
            std::ifstream file(filename);
            if (!file.is_open()) {
                throw std::runtime_error("No se pudo abrir el archivo: " + filename);
            }

            std::vector<T> data;
            std::vector<T> labels;
            std::string line;

            while (std::getline(file, line)) {
                std::istringstream ss(line);
                std::string token;

                std::getline(ss, token, ',');
                labels.push_back(static_cast<T>(std::stoi(token)));

                for (size_t i = 0; i < 64; ++i) {
                    std::getline(ss, token, ',');
                    if (!token.empty()) {
                        data.push_back(static_cast<T>(std::stof(token) / 255.0f));
                    }
                }
            }

            size_t num_samples = labels.size();
            if (num_samples == 0) {
                throw std::runtime_error("No se encontraron datos en el archivo");
            }

            utec::algebra::Tensor<T,2> X(static_cast<size_t>(num_samples), static_cast<size_t>(64));
            utec::algebra::Tensor<T,2> Y(static_cast<size_t>(num_samples), static_cast<size_t>(10));

            Y.fill(T(0));

            for (size_t i = 0; i < num_samples; ++i) {
                for (size_t j = 0; j < 64; ++j) {
                    X(i, j) = data[i * 64 + j];
                }

                size_t label_idx = static_cast<size_t>(labels[i]);
                if (label_idx < 10) {
                    Y(i, label_idx) = T(1);
                }
            }

            return {X, Y};
        }
    };

}

#endif // PROG3_NN_FINAL_PROJECT_V2025_01_DATA_LOADER_H
