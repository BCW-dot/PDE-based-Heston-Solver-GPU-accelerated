#ifndef VECTOR_OPS_HPP
#define VECTOR_OPS_HPP

#include <Kokkos_Core.hpp>

namespace VectorOps {
    // Vector addition: result = a + b
    template<class ViewType>
    void add(const ViewType& a, const ViewType& b, ViewType& result) {
        Kokkos::parallel_for("vector_add", 
            Kokkos::RangePolicy<>(0, a.extent(0)), 
            KOKKOS_LAMBDA(const int i) {
                result(i) = a(i) + b(i);
            }
        );
        Kokkos::fence();
    }

    // Scalar-vector multiplication: result = scalar * vec
    template<class ViewType>
    void scale(const double scalar, const ViewType& vec, ViewType& result) {
        Kokkos::parallel_for("vector_scale", 
            Kokkos::RangePolicy<>(0, vec.extent(0)), 
            KOKKOS_LAMBDA(const int i) {
                result(i) = scalar * vec(i);
            }
        );
        Kokkos::fence();
    }

    // Combined operation: result = a + scalar * b
    template<class ViewType>
    void axpy(const ViewType& a, const double scalar, const ViewType& b, ViewType& result) {
        Kokkos::parallel_for("vector_axpy", 
            Kokkos::RangePolicy<>(0, a.extent(0)), 
            KOKKOS_LAMBDA(const int i) {
                result(i) = a(i) + scalar * b(i);
            }
        );
        Kokkos::fence();
    }

    
    // Exponential scaling of vector: result = vec * exp(scalar * n)
    template<class ViewType>
    void exp_scale(const ViewType& vec, const double scalar, const int n, ViewType& result) {
        const double exp_factor = std::exp(scalar * n);
        Kokkos::parallel_for("vector_exp_scale", 
            Kokkos::RangePolicy<>(0, vec.extent(0)), 
            KOKKOS_LAMBDA(const int i) {
                result(i) = vec(i) * exp_factor;
            }
        );
        Kokkos::fence();
    }

    // New combined kernels for perfomance:
    // 1. Compute rhs_1 in one kernel:
    //    rhs_1(i) = Y_0(i) + theta*dt*(exp_factor*b1(i) - F_1(i))
    template<class ViewType>
    void build_rhs_1(const ViewType& Y_0, const ViewType& b1, const ViewType& F1_res, 
                     double theta, double delta_t, double exp_factor, ViewType& rhs_1) {
        Kokkos::parallel_for("build_rhs_1", Kokkos::RangePolicy<>(0, Y_0.extent(0)),
            KOKKOS_LAMBDA(const int i) {
                rhs_1(i) = Y_0(i) + theta * delta_t * (exp_factor * b1(i) - F1_res(i));
            });
        Kokkos::fence();
    }

    // 2. Compute rhs_2 in one kernel:
    //    rhs_2(i) = Y_1(i) + theta*dt*(exp_factor*b2(i) - F_2(i))
    template<class ViewType>
    void build_rhs_2(const ViewType& Y_1, const ViewType& b2, const ViewType& F2_res,
                     double theta, double delta_t, double exp_factor, ViewType& rhs_2) {
        Kokkos::parallel_for("build_rhs_2", Kokkos::RangePolicy<>(0, Y_1.extent(0)),
            KOKKOS_LAMBDA(const int i) {
                rhs_2(i) = Y_1(i) + theta * delta_t * (exp_factor * b2(i) - F2_res(i));
            });
        Kokkos::fence();
    }

    // 3. Step 1: Y_0 = U + dt * F(n-1,...)
    //    Instead of axpy after F, we can do:
    //    Y_0(i) = U(i) + delta_t * F_res(i)
    template<class ViewType>
    void update_Y0(const ViewType& U, const ViewType& F_res, double delta_t, ViewType& Y_0) {
        Kokkos::parallel_for("update_Y0", Kokkos::RangePolicy<>(0, U.extent(0)),
            KOKKOS_LAMBDA(const int i) {
                Y_0(i) = U(i) + delta_t * F_res(i);
            });
        Kokkos::fence();
    }
}

#endif // VECTOR_OPS_HPP