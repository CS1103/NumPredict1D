#ifndef PROG3_TENSOR_FINAL_PROJECT_V2025_01_TENSOR_H
#define PROG3_TENSOR_FINAL_PROJECT_V2025_01_TENSOR_H

#include <array>
#include <vector>
#include <functional>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <iostream>

namespace utec {
    namespace algebra {
        template <typename T, size_t Rank>
        class Tensor {
        private:
            std::array<size_t, Rank> shapes;
            std::array<size_t, Rank> strides;
            std::vector<T> data;

            void compute_strides() {
                if (Rank == 0) return;
                strides[Rank - 1] = 1;
                if (Rank > 1) {
                    for (size_t i = Rank - 1; i > 0; --i)
                        strides[i - 1] = strides[i] * shapes[i];
                }
            }

            size_t total_size() const {
                if (shapes[0] == 0) return 0;
                return std::accumulate(shapes.begin(), shapes.end(), size_t{1}, std::multiplies<>());
            }

            size_t get_flat_index(const std::array<size_t, Rank>& idxs) const {
                size_t flat = 0;
                for (size_t i = 0; i < Rank; ++i) {
                    if (idxs[i] >= shapes[i]) throw std::out_of_range("Index out of bounds");
                    flat += idxs[i] * strides[i];
                }
                return flat;
            }

            size_t calculateTotalSize(const std::array<size_t, Rank>& shape) const {
                size_t total = 1;
                for (size_t dim : shape) total *= dim;
                return total;
            }

            std::array<size_t, Rank> calculateBroadcastShape(const Tensor& other) const {
                std::array<size_t, Rank> result_shape;
                for (size_t i = 0; i < Rank; ++i) {
                    if (shapes[i] == other.shapes[i]) {
                        result_shape[i] = shapes[i];
                    } else if (shapes[i] == 1) {
                        result_shape[i] = other.shapes[i];
                    } else if (other.shapes[i] == 1) {
                        result_shape[i] = shapes[i];
                    } else {
                        throw std::invalid_argument("Shapes do not match and they are not compatible for broadcasting");
                    }
                }
                return result_shape;
            }

            template<typename BinaryOp>
            Tensor applyBinaryOperation(const Tensor& other, BinaryOp op) const {
                auto result_shape = calculateBroadcastShape(other);
                Tensor result(result_shape);

                size_t total_size = result.data.size();
                for (size_t flat_idx = 0; flat_idx < total_size; ++flat_idx) {
                    auto idxs = result.multiIndex(flat_idx);

                    std::array<size_t, Rank> idx_a, idx_b;
                    for (size_t i = 0; i < Rank; ++i) {
                        idx_a[i] = (shapes[i] == 1) ? 0 : idxs[i];
                        idx_b[i] = (other.shapes[i] == 1) ? 0 : idxs[i];
                    }

                    result.data[flat_idx] = op((*this)(idx_a), other(idx_b));
                }

                return result;
            }

            template<typename UnaryOp>
            Tensor applyScalarOperation(const T& scalar, UnaryOp op) const {
                Tensor result = *this;
                for (auto& val : result.data) {
                    val = op(val, scalar);
                }
                return result;
            }

        public:
            Tensor() {
                shapes.fill(1);
                compute_strides();
                data.resize(1, T{});
            }

            Tensor(const std::array<size_t, Rank>& shape) : shapes(shape) {
                compute_strides();
                size_t total_elements = total_size();
                data.resize(total_elements, T{});
            }

            template <typename... Dims>
            Tensor(Dims... dims) {
                if (sizeof...(dims) != Rank) {
                    throw std::invalid_argument("Number of dimensions do not match with " + std::to_string(Rank));
                }
                shapes = {static_cast<size_t>(dims)...};
                compute_strides();
                size_t total_elements = total_size();
                data.resize(total_elements, T{});
            }

            template <typename... Dims>
            Tensor(Dims... dims, std::initializer_list<T> values) {
                if (sizeof...(dims) != Rank) {
                    throw std::invalid_argument("Number of dimensions do not match with " + std::to_string(Rank));
                }
                shapes = {static_cast<size_t>(dims)...};
                compute_strides();
                size_t total_elements = total_size();
                if (values.size() != total_elements) {
                    throw std::invalid_argument("Number of values does not match algebra size");
                }
                data = std::vector<T>(values);
            }

            Tensor& operator=(std::initializer_list<T> values) {
                if (values.size() != total_size()) {
                    throw std::invalid_argument("Data size does not match algebra size");
                }
                std::copy(values.begin(), values.end(), data.begin());
                return *this;
            }

            Tensor& operator=(const Tensor& other) {
                if (this != &other) {
                    shapes = other.shapes;
                    strides = other.strides;
                    data = other.data;
                }
                return *this;
            }

            template <typename... Idxs>
            T& operator()(Idxs... idxs) {
                static_assert(sizeof...(Idxs) == Rank, "Número de índices incorrecto");
                std::array<size_t, Rank> idx_array = {static_cast<size_t>(idxs)...};
                return data[get_flat_index(idx_array)];
            }

            template <typename... Idxs>
            const T& operator()(Idxs... idxs) const {
                static_assert(sizeof...(Idxs) == Rank, "Número de índices incorrecto");
                std::array<size_t, Rank> idx_array = {static_cast<size_t>(idxs)...};
                return data[get_flat_index(idx_array)];
            }

            T& operator[](size_t i) { return data[i]; }
            const T& operator[](size_t i) const { return data[i]; }

            T& operator()(const std::array<size_t, Rank>& idxs) {
                return data[get_flat_index(idxs)];
            }

            const T& operator()(const std::array<size_t, Rank>& idxs) const {
                return data[get_flat_index(idxs)];
            }

            const std::array<size_t, Rank>& shape() const noexcept {
                return shapes;
            }

            size_t num_elements() const {
                return data.size();
            }

            size_t size() const {
                return num_elements();
            }

            void reshape(const std::array<size_t, Rank>& new_shape) {
                size_t new_total = total_size();
                size_t old_total = data.size();

                if (new_total != old_total) {
                    data.resize(new_total, T{});
                }

                shapes = new_shape;
                compute_strides();
            }

            template<typename... Dims>
            void reshape(Dims... dims) {
                auto new_shape = dimsToArray(dims...);
                reshape(new_shape);
            }

            void fill(const T& value) noexcept {
                std::fill(data.begin(), data.end(), value);
            }

            Tensor operator+(const Tensor& other) const {
                return applyBinaryOperation(other, [](const T& a, const T& b) { return a + b; });
            }

            Tensor operator-(const Tensor& other) const {
                return applyBinaryOperation(other, [](const T& a, const T& b) { return a - b; });
            }

            Tensor operator*(const Tensor& other) const {
                return applyBinaryOperation(other, [](const T& a, const T& b) { return a * b; });
            }

            Tensor operator+(const T& scalar) const {
                return applyScalarOperation(scalar, [](const T& val, const T& s) { return val + s; });
            }

            Tensor operator-(const T& scalar) const {
                return applyScalarOperation(scalar, [](const T& val, const T& s) { return val - s; });
            }

            Tensor operator*(const T& scalar) const {
                return applyScalarOperation(scalar, [](const T& val, const T& s) { return val * s; });
            }

            Tensor operator/(const T& scalar) const {
                if (scalar == 0) throw std::invalid_argument("División por cero");
                return applyScalarOperation(scalar, [](const T& val, const T& s) { return val / s; });
            }

            template<typename UnaryOp>
            Tensor apply(UnaryOp op) const {
                Tensor result = *this;
                for (auto& val : result.data) {
                    val = op(val);
                }
                return result;
            }

            size_t linearIndex(const std::array<size_t, Rank>& idxs) const {
                return get_flat_index(idxs);
            }

            std::array<size_t, Rank> multiIndex(size_t linear_idx) const {
                std::array<size_t, Rank> idxs{};
                for (size_t i = Rank; i-- > 0;) {
                    idxs[i] = linear_idx % shapes[i];
                    linear_idx /= shapes[i];
                }
                return idxs;
            }

            auto begin() { return data.begin(); }
            auto end() { return data.end(); }
            auto cbegin() const { return data.cbegin(); }
            auto cend() const { return data.cend(); }

            template<typename... Dims>
            static std::array<size_t, Rank> dimsToArray(Dims... dims) {
                std::vector<size_t> temp_dims = {static_cast<size_t>(dims)...};
                if (temp_dims.size() != Rank) {
                    throw std::invalid_argument("Number of dimensions do not match with " + std::to_string(Rank));
                }

                std::array<size_t, Rank> result;
                for (size_t i = 0; i < Rank; ++i) {
                    result[i] = temp_dims[i];
                }
                return result;
            }
        };

        template <typename T, size_t Rank>
        Tensor<T, Rank> matrix_product(const Tensor<T, Rank>& a, const Tensor<T, Rank>& b) {
            static_assert(Rank >= 2, "matrix_product requiere tensores de al menos 2 dimensiones");

            const auto& shape_a = a.shape();
            const auto& shape_b = b.shape();

            if (shape_a[1] != shape_b[0]) {
                throw std::invalid_argument("Matrix dimensions are incompatible for multiplication");
            }

            std::array<size_t, Rank> result_shape = {shape_a[0], shape_b[1]};
            Tensor<T, Rank> result(result_shape);
            result.fill(T{});

            for (size_t i = 0; i < shape_a[0]; ++i) {
                for (size_t j = 0; j < shape_b[1]; ++j) {
                    for (size_t k = 0; k < shape_a[1]; ++k) {
                        result(i, j) += a(i, k) * b(k, j);
                    }
                }
            }

            return result;
        }

        template <typename T, size_t Rank>
        Tensor<T, Rank> transpose_2d(const Tensor<T, Rank>& matrix) {
            if (Rank < 2) {
                std::cout << "Cannot transpose 1D algebra: need at least 2 dimensions" << std::endl;
                return matrix;
            }

            const auto& shape = matrix.shape();
            std::array<size_t, Rank> new_shape = shape;
            std::swap(new_shape[Rank - 1], new_shape[Rank - 2]);

            Tensor<T, Rank> result(new_shape);

            for (size_t i = 0; i < shape[0]; ++i) {
                for (size_t j = 0; j < shape[1]; ++j) {
                    result(j, i) = matrix(i, j);
                }
            }

            return result;
        }

        template<typename T, size_t Rank>
        Tensor<T, Rank> operator+(const T& scalar, const Tensor<T, Rank>& tensor) {
            return tensor + scalar;
        }

        template<typename T, size_t Rank>
        Tensor<T, Rank> operator-(const T& scalar, const Tensor<T, Rank>& tensor) {
            Tensor<T, Rank> result(tensor.shape());
            auto tensor_it = tensor.cbegin();
            for (auto& val : result) {
                val = scalar - *tensor_it++;
            }
            return result;
        }

        template<typename T, size_t Rank>
        Tensor<T, Rank> operator*(const T& scalar, const Tensor<T, Rank>& tensor) {
            return tensor * scalar;
        }

        template <typename T, size_t Rank>
        std::ostream& operator<<(std::ostream& os, const Tensor<T, Rank>& tensor) {
            const auto& shape = tensor.shape();

            if (Rank == 1) {
                for (size_t i = 0; i < shape[0]; ++i) {
                    os << tensor[i];
                    if (i + 1 < shape[0]) os << " ";
                }
                return os;
            }

            std::function<void(size_t, size_t&, size_t)> print;
            print = [&](size_t dim, size_t& index, size_t indent) {
                std::string indentStr(indent, ' ');
                if (dim == Rank - 1) {
                    os << indentStr;
                    for (size_t j = 0; j < shape[dim]; ++j) {
                        os << tensor[index++];
                        if (j + 1 < shape[dim]) os << " ";
                    }
                    os << "\n";
                } else {
                    os << indentStr << "{\n";
                    for (size_t i = 0; i < shape[dim]; ++i) {
                        print(dim + 1, index, indent + 2);
                    }
                    os << indentStr << "}";
                    if (dim != 0) os << "\n";
                }
            };

            os << "{\n";
            size_t index = 0;
            for (size_t i = 0; i < shape[0]; ++i) {
                print(1, index, 2);
                if (i + 1 < shape[0]) os << "\n";
            }
            os << "\n}";
            return os;
        }

        template<typename T, size_t Rank, typename UnaryOp>
        Tensor<T, Rank> apply(const Tensor<T, Rank>& tensor, UnaryOp op) {
            return tensor.apply(op);
        }
    }
}

#endif //PROG3_TENSOR_FINAL_PROJECT_V2025_01_TENSOR_H
