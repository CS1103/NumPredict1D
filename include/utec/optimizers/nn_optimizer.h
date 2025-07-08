#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H

#include "neural_network/nn_interfaces.h"
#include <cmath>
#include <unordered_map>
#include <memory>

namespace utec::neural_network {
    template<typename T>
    struct SGD final : IOptimizer<T> {
        T lr_;
        explicit SGD(T lr = T(0.01)) : lr_{lr} {}

        void update(Tensor<T,2>& params,
                    const Tensor<T,2>& grads) override
        {
            auto n = params.size();
            for (size_t i = 0; i < n; ++i)
                params[i] -= lr_ * grads[i];
        }
    };

    template<typename T>
    struct AdamState {
        Tensor<T,2> m_;
        Tensor<T,2> v_;
        size_t t_;

        AdamState() : t_(0) {}

        void initialize(const std::array<size_t,2>& shape) {
            m_ = Tensor<T,2>(shape);
            v_ = Tensor<T,2>(shape);
            m_.fill(T(0));
            v_.fill(T(0));
            t_ = 0;
        }

        bool is_initialized() const {
            return t_ > 0 || (m_.size() > 0 && v_.size() > 0);
        }
    };

    template<typename T>
    struct Adam final : IOptimizer<T> {
        T lr_, beta1_, beta2_, eps_;
        std::unordered_map<void*, std::unique_ptr<AdamState<T>>> states_;

        explicit Adam(T learning_rate = T(0.001),
                      T beta1 = T(0.9),
                      T beta2 = T(0.999),
                      T epsilon = T(1e-8))
          : lr_{learning_rate}
          , beta1_{beta1}
          , beta2_{beta2}
          , eps_{epsilon}
        {}

        void update(Tensor<T,2>& params,
                    const Tensor<T,2>& grads) override
        {
            void* key = static_cast<void*>(&params);

            auto it = states_.find(key);
            if (it == states_.end()) {
                auto state = std::make_unique<AdamState<T>>();
                state->initialize(params.shape());
                states_[key] = std::move(state);
                it = states_.find(key);
            }

            AdamState<T>* state = it->second.get();

            if (!state->is_initialized() ||
                state->m_.shape() != params.shape() ||
                state->v_.shape() != params.shape()) {
                state->initialize(params.shape());
            }

            ++(state->t_);

            auto N = params.size();

            if (grads.size() != N) {
                throw std::runtime_error("Adam: Tamaño de gradientes no coincide con parámetros");
            }

            for (size_t i = 0; i < N; ++i) {
                if (std::isnan(grads[i]) || std::isinf(grads[i])) {
                    throw std::runtime_error("Adam: Gradiente inválido detectado");
                }

                state->m_[i] = beta1_ * state->m_[i] + (T(1) - beta1_) * grads[i];
                state->v_[i] = beta2_ * state->v_[i] + (T(1) - beta2_) * grads[i] * grads[i];

                T m_hat = state->m_[i] / (T(1) - std::pow(beta1_, T(state->t_)));
                T v_hat = state->v_[i] / (T(1) - std::pow(beta2_, T(state->t_)));

                T denominator = std::sqrt(v_hat) + eps_;
                if (denominator <= T(0)) {
                    throw std::runtime_error("Adam: Denominador inválido en actualización");
                }

                T update_value = lr_ * m_hat / denominator;

                if (std::isnan(update_value) || std::isinf(update_value)) {
                    throw std::runtime_error("Adam: Actualización inválida calculada");
                }

                params[i] -= update_value;
            }
        }
    };
}

#endif // PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H
