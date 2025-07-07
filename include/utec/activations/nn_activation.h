#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H

#include "neural_network/nn_interfaces.h"
#include "algebra/tensor.h"
#include <cmath>

namespace utec::neural_network {

    template<typename T, size_t Rank>
    using Tensor = utec::algebra::Tensor<T, Rank>;

    template<typename T>
    struct ReLU final : ILayer<T> {
        Tensor<T,2> last_input_;

        Tensor<T,2> forward(const Tensor<T,2>& x) override {
            last_input_ = x;
            auto s = x.shape();
            Tensor<T,2> out(s[0], s[1]);

            for (size_t i = 0; i < s[0]; ++i)
                for (size_t j = 0; j < s[1]; ++j)
                    out(i,j) = x(i,j) > T(0) ? x(i,j) : T(0);
            return out;
        }

        Tensor<T,2> backward(const Tensor<T,2>& grad) override {
            auto s = grad.shape();
            Tensor<T,2> out(s[0], s[1]);

            for (size_t i = 0; i < s[0]; ++i)
                for (size_t j = 0; j < s[1]; ++j)
                    out(i,j) = grad(i,j);

            for (size_t i = 0; i < s[0]; ++i)
                for (size_t j = 0; j < s[1]; ++j)
                    if (last_input_(i,j) <= T(0))
                        out(i,j) = T(0);
            return out;
        }
    };

    template<typename T>
    struct Sigmoid final : ILayer<T> {
        Tensor<T,2> last_output_;

        Tensor<T,2> forward(const Tensor<T,2>& x) override {
            auto s = x.shape();
            Tensor<T,2> out(s[0], s[1]);

            for (size_t i = 0; i < s[0]; ++i)
                for (size_t j = 0; j < s[1]; ++j) {
                    T v = x(i,j);
                    if (v > T(500))  v = T(500);
                    if (v < T(-500)) v = T(-500);
                    out(i,j) = T(1)/(T(1) + std::exp(-v));
                }
            last_output_ = out;
            return out;
        }

        Tensor<T,2> backward(const Tensor<T,2>& grad) override {
            auto s = grad.shape();
            Tensor<T,2> out(s[0], s[1]);

            for (size_t i = 0; i < s[0]; ++i)
                for (size_t j = 0; j < s[1]; ++j) {
                    T y = last_output_(i,j);
                    out(i,j) = grad(i,j) * y * (T(1) - y);
                }
            return out;
        }
    };

}

#endif // PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H
