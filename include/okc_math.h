#ifndef SEDONA_OKC_MATH_H
#define SEDONA_OKC_MATH_H

#include <Eigen/Eigen>

namespace sedona {
    template<class T, size_t pages, size_t rows, size_t cols>
    using Tensor = Eigen::Vector <Eigen::Matrix<T, rows, cols>, pages>;

    template<class T, size_t pages, size_t rows, size_t cols>
    Tensor<T, pages, rows, cols> Tensor_Zero() {
        Tensor<T, pages, rows, cols> ret;
        for (size_t i = 0; i < pages; ++i)
            ret[i] = Eigen::Matrix<T, rows, cols>::Zero();
        return ret;
    }

    template<class T>
    Eigen::Matrix3 <T> skew(const Eigen::Vector3 <T> &v) {
        Eigen::Matrix3 <T> ret;
        ret << 0, -v[2], v[1], v[2], 0, -v[0], -v[1], v[0], 0;
        return ret;
    }

    template<class T, int N>
    Tensor<T, N, 3, 3> skew3(const Eigen::Matrix<T, 3, N> &m) {
        Tensor<T, N, 3, 3> ret;
        for (size_t i = 0; i < N; ++i) {

            const Eigen::Vector3 <T> v = m.col(i);

            ret[i] << 0, -v[2], v[1], v[2], 0, -v[0], -v[1], v[0], 0;
        }
        return ret;
    }

// Tensor * Tensor
    template<class T, int DEPTH, int N_ROWS, int N_COLS, int M_COLS>
    inline Tensor<T, DEPTH, N_ROWS, M_COLS>
    operator*(Tensor<T, DEPTH, N_ROWS, N_COLS> n,
              Tensor<T, DEPTH, N_COLS, M_COLS> m) {
        Tensor<T, DEPTH, N_ROWS, M_COLS> result;
        for (size_t i = 0; i < DEPTH; ++i)
            result[i] = n[i] * m[i];
        return result;
    }

// Matrix * Tensor
    template<class T, int DEPTH, int N_ROWS, int N_COLS, int M_COLS>
    inline Tensor<T, DEPTH, N_ROWS, M_COLS>
    operator*(Eigen::Matrix <T, N_ROWS, N_COLS> n,
              Tensor<T, DEPTH, N_COLS, M_COLS> m) {
        Tensor<T, DEPTH, N_ROWS, M_COLS> result;
        for (size_t i = 0; i < DEPTH; ++i)
            result[i] = n * m[i];
        return result;
    }

// Tensor * Matrix
    template<class T, int DEPTH, int N_ROWS, int N_COLS, int M_COLS>
    inline Tensor<T, DEPTH, N_ROWS, M_COLS>
    operator*(Tensor<T, DEPTH, N_ROWS, N_COLS> n,
              Eigen::Matrix <T, N_COLS, M_COLS> m) {
        Tensor<T, DEPTH, N_ROWS, M_COLS> result;
        for (size_t i = 0; i < DEPTH; ++i)
            result[i] = n[i] * m;
        return result;
    }

// Tensor * Vector
    template<class T, int DEPTH, int N_ROWS, int N_COLS>
    inline Eigen::Matrix <T, N_ROWS, DEPTH> tvp(Tensor<T, DEPTH, N_ROWS, N_COLS> n,
                                                Eigen::Vector <T, N_COLS> m) {
        Eigen::Matrix <T, N_ROWS, DEPTH> result;
        for (size_t i = 0; i < DEPTH; ++i)
            result.col(i) = n[i] * m;
        return result;
    }
}

#endif // SEDONA_OKC_MATH_H
