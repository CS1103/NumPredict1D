#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_LAYER_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_LAYER_H

#include "algebra/tensor.h"

namespace utec::neural_network {

  template<typename T, size_t DIMS>
  using Tensor = utec::algebra::Tensor<T, DIMS>;

  template<typename T>
  struct IOptimizer {
    virtual ~IOptimizer() = default;
    virtual void update(Tensor<T,2>& params, const Tensor<T,2>& gradients) = 0;
    virtual void step() {}
  };

  template<typename T>
  struct ILayer {
    virtual ~ILayer() = default;
    virtual Tensor<T,2> forward(const Tensor<T,2>& x) = 0;
    virtual Tensor<T,2> backward(const Tensor<T,2>& gradients) = 0;
    virtual void update_params(IOptimizer<T>& optimizer) {}
  };

  template<typename T, size_t DIMS>
  struct ILoss {
    virtual ~ILoss() = default;
    virtual T loss() const = 0;
    virtual Tensor<T,DIMS> loss_gradient() const = 0;
  };

}

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_LAYER_H
