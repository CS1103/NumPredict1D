#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H

#include "nn_interfaces.h"
#include "algebra/tensor.h"

namespace utec::neural_network {

    template<typename T>
    class Dense final : public ILayer<T> {
        size_t in_f_, out_f_;
        Tensor<T,2> W_, b_, last_x_, dW_, db_;

    public:
        template<typename InitW, typename InitB>
        Dense(size_t in_f, size_t out_f, InitW init_w, InitB init_b)
          : in_f_{in_f}, out_f_{out_f},
            W_(in_f, out_f), b_(1, out_f),
            dW_(in_f, out_f), db_(1, out_f)
        {
            init_w(W_);
            init_b(b_);
        }

        Tensor<T,2> forward(const Tensor<T,2>& x) override {
            last_x_ = x;
            auto y = utec::algebra::matrix_product(x, W_);
            auto s = y.shape();
            for (size_t i = 0; i < s[0]; ++i)
                for (size_t j = 0; j < s[1]; ++j)
                    y(i,j) += b_(0,j);
            return y;
        }

        Tensor<T,2> backward(const Tensor<T,2>& grad) override {
            Tensor<T,2> x_t = utec::algebra::transpose_2d(last_x_);
            dW_ = utec::algebra::matrix_product(x_t, grad);

            db_.fill(T(0));
            auto s = grad.shape();
            for (size_t i = 0; i < s[0]; ++i)
                for (size_t j = 0; j < s[1]; ++j)
                    db_(0,j) += grad(i,j);

            Tensor<T,2> w_t = utec::algebra::transpose_2d(W_);
            return utec::algebra::matrix_product(grad, w_t);
        }

        void update_params(IOptimizer<T>& opt) override {
            opt.update(W_,  dW_);
            opt.update(b_,  db_);
        }
    };

}

#endif // PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H
