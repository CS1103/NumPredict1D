#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H

#include "neural_network/nn_interfaces.h"
#include <cmath>
#include <algorithm>

namespace utec::algebra {

    template<typename T, size_t Rank>
    size_t calculateTotalSize(const Tensor<T, Rank>& tensor) {
        size_t total = 1;
        for (size_t dim : tensor.shape()) {
            total *= dim;
        }
        return total;
    }

}

namespace utec::neural_network {

    template<typename T>
    struct MSELoss final : ILoss<T,2> {
        Tensor<T,2> y_pred_, y_true_;
        MSELoss(const Tensor<T,2>& yp, const Tensor<T,2>& yt)
          : y_pred_{yp}, y_true_{yt} {}

        T loss() const override {
            T sum = T(0);
            auto n = utec::algebra::calculateTotalSize(y_pred_);
            for (size_t i = 0; i < n; ++i) {
                T d = y_pred_[i] - y_true_[i];
                sum += d*d;
            }
            return sum / static_cast<T>(n);
        }

        Tensor<T,2> loss_gradient() const override {
            auto grad = Tensor<T,2>(y_pred_.shape());
            T inv = T(2) / static_cast<T>(utec::algebra::calculateTotalSize(y_pred_));
            for (size_t i = 0; i < utec::algebra::calculateTotalSize(grad); ++i)
                grad[i] = inv * (y_pred_[i] - y_true_[i]);
            return grad;
        }
    };

    template<typename T>
    struct BCELoss final : ILoss<T,2> {
        Tensor<T,2> y_pred_, y_true_;
        BCELoss(const Tensor<T,2>& yp, const Tensor<T,2>& yt)
          : y_pred_{yp}, y_true_{yt} {}

        T loss() const override {
            T sum = T(0);
            auto n = utec::algebra::calculateTotalSize(y_pred_);
            for (size_t i = 0; i < n; ++i) {
                T p = std::min(std::max(y_pred_[i], T(1e-8)), T(1)-T(1e-8));
                T t = y_true_[i];
                sum -= t*std::log(p) + (T(1)-t)*std::log(T(1)-p);
            }
            return sum / static_cast<T>(n);
        }

        Tensor<T,2> loss_gradient() const override {
            auto grad = Tensor<T,2>(y_pred_.shape());
            T inv = T(1) / static_cast<T>(utec::algebra::calculateTotalSize(y_pred_));
            for (size_t i = 0; i < utec::algebra::calculateTotalSize(grad); ++i)
                grad[i] = inv * ((y_pred_[i] - y_true_[i]) / (y_pred_[i]*(T(1)-y_pred_[i])));
            return grad;
        }
    };

}

#endif // PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H
