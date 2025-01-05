#ifndef F_FUNCTIONS_HPP
#define F_FUNCTIONS_HPP

#include <Kokkos_Core.hpp>
#include "vector_ops.hpp"
#include "hes_mat_fac.hpp"
#include "hes_A2_mat.hpp"

//basic F functions impleemntation. Below is an optimized version

namespace FFunctions {
    // F(n, omega, A, b, r_f, delta_t)
    template<class ViewType>
    void F(const int n, 
          const ViewType& omega,
          heston_A0Storage_gpu& A0,     
          heston_A1Storage_gpu& A1,    
          heston_A2Storage_gpu& A2,     
          const ViewType& b,
          const double r_f,
          const double delta_t,
          ViewType& result) {
        
        // Temporary storage for matrix multiplications
        ViewType temp_a0("temp_a0", omega.extent(0));
        ViewType temp_a1("temp_a1", omega.extent(0));
        ViewType temp_a2("temp_a2", omega.extent(0));
        ViewType temp_b("temp_b", omega.extent(0));
        
        // Compute A*omega using individual matrices
        //A0.multiply_seq(omega, temp_a0);
        A0.multiply_parallel_s_and_v(omega, temp_a0);

        //A1.multiply(omega, temp_a1);
        A1.multiply_parallel_s_and_v(omega, temp_a1);

        //A2.multiply(omega, temp_a2);
        A2.multiply_parallel_s_and_v(omega, temp_a2);
        
    
        // Sum up matrix multiplications: temp_a0 = A*omega
        VectorOps::add(temp_a0, temp_a1, temp_a0);
        VectorOps::add(temp_a0, temp_a2, temp_a0);
        
        // Scale boundary vector with exponential
        VectorOps::exp_scale(b, r_f * delta_t, n, temp_b);
        
        // Final sum
        VectorOps::add(temp_a0, temp_b, result);
    }

    // F_0(n, omega, A_0, b_0, r_f, delta_t)
    template<class ViewType>
    void F_0(const int n,
            const ViewType& omega,
            heston_A0Storage_gpu& A0,    // Removed const
            const ViewType& b0,
            const double r_f,
            const double delta_t,
            ViewType& result) {
        
        // Temporary storage
        ViewType temp_b("temp_b", omega.extent(0));
        
        // Matrix multiplication
        //A0.multiply_seq(omega, result);
        A0.multiply_parallel_s_and_v(omega, result);
        
        // Add scaled boundary term
        VectorOps::exp_scale(b0, r_f * delta_t, n, temp_b);
        VectorOps::add(result, temp_b, result);
    }

    // F_1(n, omega, A_1, b_1, r_f, delta_t)
    template<class ViewType>
    void F_1(const int n,
            const ViewType& omega,
            heston_A1Storage_gpu& A1,    // Removed const
            const ViewType& b1,
            const double r_f,
            const double delta_t,
            ViewType& result) {
        
        // Temporary storage
        ViewType temp_b("temp_b", omega.extent(0));
        
        // Matrix multiplication
        //A1.multiply(omega, result);
        A1.multiply_parallel_s_and_v(omega, result);
        
        // Add scaled boundary term
        VectorOps::exp_scale(b1, r_f * delta_t, n, temp_b);
        VectorOps::add(result, temp_b, result);
    }

    // F_2(n, omega, A_2, b_2, r_f, delta_t)
    template<class ViewType>
    void F_2(const int n,
            const ViewType& omega,
            heston_A2Storage_gpu& A2,    
            const ViewType& b2,
            const double r_f,
            const double delta_t,
            ViewType& result) {
        
        // Temporary storage
        ViewType temp_b("temp_b", omega.extent(0));
        
        // Matrix multiplication
        //A2.multiply(omega, result);
        A2.multiply_parallel_s_and_v(omega, result);
        
        // Add scaled boundary term
        VectorOps::exp_scale(b2, r_f * delta_t, n, temp_b);
        VectorOps::add(result, temp_b, result);
    }
}





#endif // F_FUNCTIONS_HPP