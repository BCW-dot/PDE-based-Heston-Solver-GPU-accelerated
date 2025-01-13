#ifndef DO_SCHEME_HPP
#define DO_SCHEME_HPP

#include <Kokkos_Core.hpp>
#include "vector_ops.hpp"
#include "f_functions.hpp"
#include "hes_mat_fac.hpp"
#include "hes_A2_mat.hpp"
#include "BoundaryConditions.hpp"

#include <thread>     // For std::this_thread::sleep_for


//basic DO scheme implemntation. Below is an optimized version
/*
template<class ViewType>
void DO_scheme(const int m,                    // Total size (m1+1)*(m2+1)
              const int N,                     // Number of time steps
              const ViewType& U_0,             // Initial condition
              const double delta_t,            // Time step size
              const double theta,              // Weight parameter
              heston_A0Storage_gpu& A0,        // A0 matrix (removed const)
              heston_A1Storage_gpu& A1,        // A1 matrix (removed const)
              heston_A2Storage_gpu& A2,        // A2 matrix (removed const)
              const BoundaryConditions& bounds,// Boundary conditions
              const double r_f,                // Foreign interest rate
              ViewType& U) {                   // Result vector
    
    // Copy initial condition to U
    Kokkos::deep_copy(U, U_0);

    // Temporary vectors for computations
    ViewType Y_0("Y_0", m);
    ViewType rhs_1("rhs_1", m);

    ViewType Y_1("Y_1", m);
    ViewType rhs_2("rhs_2", m);
    
    ViewType temp("temp", m);
    ViewType exp_b1("exp_b1", m);
    ViewType exp_b2("exp_b2", m);

    // Get boundary vectors
    auto b = bounds.get_b();
    auto b1 = bounds.get_b1();
    auto b2 = bounds.get_b2();

    // Main time stepping loop
    for (int n = 1; n <= N; n++) {
        // Step 1: Y_0 = U + dt * F(n-1, U, A, b, r_f, dt)
        FFunctions::F(n-1, U, A0, A1, A2, b, r_f, delta_t, temp);
        VectorOps::axpy(U, delta_t, temp, Y_0); //U + delta_t * temp = Y0

        // Step 2: Compute rhs_1 = Y_0 + theta*dt*(b1*exp(r_f*dt*n) - F_1(n-1, U, A1, b1, r_f, dt))
        VectorOps::exp_scale(b1, r_f * delta_t, n, exp_b1);
        FFunctions::F_1(n-1, U, A1, b1, r_f, delta_t, temp);
        
        VectorOps::scale(-1.0, temp, temp);  // -F_1
        VectorOps::add(exp_b1, temp, temp);  // b1*exp(...) - F_1
        VectorOps::scale(theta * delta_t, temp, temp);
        VectorOps::add(Y_0, temp, rhs_1);

        // Step 3: Solve (I - theta*dt*A1)Y_1 = rhs_1
        //A1.solve_implicit(Y_1, rhs_1);
        A1.solve_implicit_parallel_v(Y_1, rhs_1);

        // Step 4: Compute rhs_2 = Y_1 + theta*dt*(b2*exp(r_f*dt*n) - F_2(n-1, U, A2, b2, r_f, dt))
        VectorOps::exp_scale(b2, r_f * delta_t, n, exp_b2);
        FFunctions::F_2(n-1, U, A2, b2, r_f, delta_t, temp);
        
        VectorOps::scale(-1.0, temp, temp);  // -F_2
        VectorOps::add(exp_b2, temp, temp);  // b2*exp(...) - F_2
        VectorOps::scale(theta * delta_t, temp, temp);
        VectorOps::add(Y_1, temp, rhs_2);

        // Step 5: Solve (I - theta*dt*A2)U = rhs_2
        A2.solve_implicit(U, rhs_2);
    }
}
*/

template<class ViewType>
void DO_scheme(const int m,                    // Total size (m1+1)*(m2+1)
                        const int N,                     // Number of time steps
                        const ViewType& U_0,             // Initial condition
                        const double delta_t,            // Time step size
                        const double theta,              // Weight parameter
                        heston_A0Storage_gpu& A0,        // A0 matrix
                        heston_A1Storage_gpu& A1,        // A1 matrix 
                        heston_A2Storage_gpu& A2,        // A2 matrix
                        const BoundaryConditions& bounds,// Boundary conditions
                        const double r_f,                // Foreign interest rate
                        ViewType& U) {                   // Result vector
    
    // Initialize result with initial condition
    Kokkos::deep_copy(U, U_0);

    // Create persistent workspace vectors to avoid reallocations
    ViewType Y_0("Y_0", m);
    ViewType Y_1("Y_1", m);
    ViewType A0_result("A0_result", m);
    ViewType A1_result("A1_result", m);
    ViewType A2_result("A2_result", m);

    ViewType temp_a0("temp_a0", m);
    ViewType temp_a1("temp_a1", m);
    ViewType temp_a2("temp_a2", m);
    
    // Get boundary vectors
    auto b = bounds.get_b();
    auto b1 = bounds.get_b1();
    auto b2 = bounds.get_b2();

    using timer = std::chrono::high_resolution_clock;

    auto t_start = timer::now();
    // Main time stepping loop
    for (int n = 1; n <= N; n++) {
        ViewType A0_result("A0_result", m);

        //A0.multiply_seq(U, A0_result);
        A0.multiply_parallel_s_and_v(U, A0_result);

        //A1.multiply(U, A1_result);
        A1.multiply_parallel_s_and_v(U, A1_result);

        //A2.multiply(U, A2_result);
        A2.multiply_parallel_s_and_v(U, A2_result);
        
        Kokkos::parallel_for("Y0_computation", m, KOKKOS_LAMBDA(const int i) {
            double exp_factor = std::exp(r_f * delta_t * (n-1));
            Y_0(i) = U(i) + delta_t * (A0_result(i) + A1_result(i) + A2_result(i) + b(i) * exp_factor);
        });

        // Step 2 & 3: Combined RHS computation and implicit solve for A1
        A1.multiply_parallel_s_and_v(U, A1_result);  // Reuse A1_result
        
        //i think here was a mistake in the computation of the rhs
        /*
        Kokkos::parallel_for("A1_rhs_computation", m, KOKKOS_LAMBDA(const int i) {
            double exp_factor = std::exp(r_f * delta_t * (n-1));
            double rhs = Y_0(i) + theta * delta_t * (b1(i) * exp_factor - A1_result(i));
            Y_0(i) = rhs;  // Reuse Y_0 to store RHS
        });
        */

        
        Kokkos::parallel_for("A1_rhs_computation", m, KOKKOS_LAMBDA(const int i) {
            double exp_factor_n = std::exp(r_f * delta_t * n);
            double exp_factor_nm1 = std::exp(r_f * delta_t * (n-1));
            double rhs = Y_0(i) + theta * delta_t * (b1(i) * exp_factor_n - (A1_result(i) + b1(i) * exp_factor_nm1));
            Y_0(i) = rhs;  // Reuse Y_0 to store RHS
        });
        
        
        A1.solve_implicit_parallel_v(Y_1, Y_0);  // Y_0 contains RHS, result in Y_1

        // Step 4 & 5: Combined RHS computation and implicit solve for A2
        A2.multiply_parallel_s_and_v(U, A2_result);  // Reuse A2_result
        
        //same for A2
        /*
        Kokkos::parallel_for("A2_rhs_computation", m, KOKKOS_LAMBDA(const int i) {
            double exp_factor = std::exp(r_f * delta_t * (n-1));
            double rhs = Y_1(i) + theta * delta_t * (b2(i) * exp_factor - A2_result(i));
            Y_1(i) = rhs;  // Reuse Y_1 to store RHS
        });
        */
        
        Kokkos::parallel_for("A2_rhs_computation", m, KOKKOS_LAMBDA(const int i) {
            double exp_factor_n = std::exp(r_f * delta_t * n);
            double exp_factor_nm1 = std::exp(r_f * delta_t * (n-1));
            double rhs = Y_1(i) + theta * delta_t * (b2(i) * exp_factor_n - (A2_result(i) + b2(i) * exp_factor_nm1));
            Y_1(i) = rhs;  // Reuse Y_1 to store RHS
        });
        

        A2.solve_implicit(U, Y_1);  // Y_1 contains RHS, result in U
    }

    auto t_end = timer::now();

    std::cout << "DO time: "
              << std::chrono::duration<double>(t_end - t_start).count()
              << " seconds" << std::endl;
}

template<class ViewType>
void DO_scheme_shuffle(const int m,                    
              const int m1,                    // Added for shuffling
              const int m2,                    // Added for shuffling
              const int N,                     
              const ViewType& U_0,             
              const double delta_t,            
              const double theta,              
              heston_A0Storage_gpu& A0,        
              heston_A1Storage_gpu& A1,        
              heston_A2Storage_gpu& A2,        // Keep original
              heston_A2_shuffled& A2_shuf,     // Add shuffled version
              const BoundaryConditions& bounds,
              const double r_f,                
              ViewType& U) {                   
    
    // Initialize result with initial condition
    Kokkos::deep_copy(U, U_0);

    // Create persistent workspace vectors to avoid reallocations
    ViewType Y_0("Y_0", m);
    ViewType Y_1("Y_1", m);
    ViewType A0_result("A0_result", m);
    ViewType A1_result("A1_result", m);
    ViewType A2_result("A2_result", m);

    ViewType Y_1_shuffled("Y_1_shuffled", m);
    ViewType U_next_shuffled("U_next_shuffled", m);
    
    // Add shuffled workspace
    ViewType U_shuffled("U_shuffled", m);
    ViewType A2_result_shuffled("A2_result_shuffled", m);
    ViewType A2_result_unshuf("A2_result_unshuf", m);  

    // Get boundary vectors
    auto b = bounds.get_b();
    auto b1 = bounds.get_b1();
    auto b2 = bounds.get_b2();

    using timer = std::chrono::high_resolution_clock;
    auto t_start = timer::now();

    // Main time stepping loop
    for (int n = 1; n <= N; n++) {
        ViewType A0_result("A0_result", m);

        // Step 1: Let's first just verify we get same result with both A2s
        A0.multiply_parallel_s_and_v(U, A0_result);
        A1.multiply_parallel_s_and_v(U, A1_result);
        
        // Add shuffled A2 multiplication (but don't use result yet)
        shuffle_vector(U, U_shuffled, m1, m2);
        A2_shuf.multiply(U_shuffled, A2_result_shuffled);  
        unshuffle_vector(A2_result_shuffled, A2_result_unshuf, m1, m2);
        
        // Use original result for now
        Kokkos::parallel_for("Y0_computation", m, KOKKOS_LAMBDA(const int i) {
            double exp_factor = std::exp(r_f * delta_t * (n-1));
            Y_0(i) = U(i) + delta_t * (A0_result(i) + A1_result(i) + A2_result_unshuf(i) + b(i) * exp_factor);
        });

        // Rest of function exactly as original
        A1.multiply_parallel_s_and_v(U, A1_result);
        
        Kokkos::parallel_for("A1_rhs_computation", m, KOKKOS_LAMBDA(const int i) {
            double exp_factor = std::exp(r_f * delta_t * (n-1));
            double rhs = Y_0(i) + theta * delta_t * (b1(i) * exp_factor - A1_result(i));
            Y_0(i) = rhs;
        });
        
        A1.solve_implicit_parallel_v(Y_1, Y_0);

        //A2.multiply_parallel_s_and_v(U, A2_result);
        shuffle_vector(U, U_shuffled, m1, m2);
        A2_shuf.multiply(U_shuffled, A2_result_shuffled);  
        unshuffle_vector(A2_result_shuffled, A2_result_unshuf, m1, m2);
        
        Kokkos::parallel_for("A2_rhs_computation", m, KOKKOS_LAMBDA(const int i) {
            double exp_factor = std::exp(r_f * delta_t * (n-1));
            double rhs = Y_1(i) + theta * delta_t * (b2(i) * exp_factor - A2_result_unshuf(i));
            Y_1(i) = rhs;
        });

        //A2.solve_implicit(U, Y_1);
        // Shuffle input
        shuffle_vector(Y_1, Y_1_shuffled, m1, m2);
        
        // Solve with shuffled A2
        A2_shuf.solve_implicit(U_next_shuffled, Y_1_shuffled);
        
        // Unshuffle result back to U
        unshuffle_vector(U_next_shuffled, U, m1, m2);
    }

    auto t_end = timer::now();
    //std::cout << "DO time: "
              //<< std::chrono::duration<double>(t_end - t_start).count()
              //<< " HERE seconds" << std::endl;
}


//First test of a different scheme
template<class ViewType>
void CS_scheme(const int m,                    // Total size (m1+1)*(m2+1)
              const int N,                     // Number of time steps
              const ViewType& U_0,             // Initial condition
              const double delta_t,            // Time step size
              const double theta,              // Weight parameter
              heston_A0Storage_gpu& A0,        // A0 matrix
              heston_A1Storage_gpu& A1,        // A1 matrix
              heston_A2Storage_gpu& A2,        // A2 matrix
              const BoundaryConditions& bounds,// Boundary conditions
              const double r_f,                // Foreign interest rate
              ViewType& U) {                   // Result vector
    
    // Copy initial condition to U
    Kokkos::deep_copy(U, U_0);

    // Temporary vectors for computations
    ViewType Y_0("Y_0", m);
    ViewType Y_1("Y_1", m);
    ViewType Y_2("Y_2", m);
    ViewType Y_0_tilde("Y_0_tilde", m);
    ViewType Y_1_tilde("Y_1_tilde", m);
    
    ViewType rhs_1("rhs_1", m);
    ViewType rhs_2("rhs_2", m);
    
    ViewType temp("temp", m);

    // Get boundary vectors
    auto b = bounds.get_b();
    auto b0 = bounds.get_b0();
    auto b1 = bounds.get_b1();
    auto b2 = bounds.get_b2();

    // Main time stepping loop
    for (int n = 1; n <= N; n++) {
        // Step 1: Y_0 = U + dt * F(n-1, U, A, b)
        FFunctions::F(n-1, U, A0, A1, A2, b, r_f, delta_t, temp);
        VectorOps::axpy(U, delta_t, temp, Y_0);

        // Step 2: Compute rhs_1 and solve for Y_1
        FFunctions::F_1(n-1, U, A1, b1, r_f, delta_t, temp);
        ViewType exp_b1("exp_b1", m);
        VectorOps::exp_scale(b1, r_f * delta_t, n, exp_b1);
        VectorOps::add(exp_b1, temp, temp);
        VectorOps::scale(-1.0, temp, temp);
        VectorOps::axpy(Y_0, theta * delta_t, temp, rhs_1);
        A1.solve_implicit(Y_1, rhs_1);

        // Step 3: Compute rhs_2 and solve for Y_2
        FFunctions::F_2(n-1, U, A2, b2, r_f, delta_t, temp);
        ViewType exp_b2("exp_b2", m);
        VectorOps::exp_scale(b2, r_f * delta_t, n, exp_b2);
        VectorOps::add(exp_b2, temp, temp);
        VectorOps::scale(-1.0, temp, temp);
        VectorOps::axpy(Y_1, theta * delta_t, temp, rhs_2);
        A2.solve_implicit(Y_2, rhs_2);

        // Step 4: Compute Y_0_tilde with corrector term for A0
        FFunctions::F_0(n, Y_2, A0, b0, r_f, delta_t, temp);
        ViewType temp_prev("temp_prev", m);
        FFunctions::F_0(n-1, U, A0, b0, r_f, delta_t, temp_prev);
        VectorOps::scale(-1.0, temp_prev, temp_prev);
        VectorOps::add(temp, temp_prev, temp);
        VectorOps::scale(0.5 * delta_t, temp, temp);
        VectorOps::add(Y_0, temp, Y_0_tilde);

        // Step 5: Compute final rhs_1 and solve for Y_1_tilde
        FFunctions::F_1(n-1, U, A1, b1, r_f, delta_t, temp);
        VectorOps::add(exp_b1, temp, temp);
        VectorOps::scale(-1.0, temp, temp);
        VectorOps::axpy(Y_0_tilde, theta * delta_t, temp, rhs_1);
        A1.solve_implicit(Y_1_tilde, rhs_1);

        // Step 6: Compute final rhs_2 and solve for U
        FFunctions::F_2(n-1, U, A2, b2, r_f, delta_t, temp);
        VectorOps::add(exp_b2, temp, temp);
        VectorOps::scale(-1.0, temp, temp);
        VectorOps::axpy(Y_1_tilde, theta * delta_t, temp, rhs_2);
        A2.solve_implicit(U, rhs_2);
    }
}

template<class ViewType>
void CS_scheme_shuffled(const int m,                    
                       const int m1,                    // Added for shuffling
                       const int m2,                    // Added for shuffling
                       const int N,                     
                       const ViewType& U_0,             
                       const double delta_t,            
                       const double theta,              
                       heston_A0Storage_gpu& A0,        
                       heston_A1Storage_gpu& A1,        
                       heston_A2_shuffled& A2_shuf,     // Using shuffled A2
                       const BoundaryConditions& bounds,
                       const double r_f,                
                       ViewType& U) {                   

    // Initialize result with initial condition
    Kokkos::deep_copy(U, U_0);

    // Create persistent workspace vectors to avoid reallocations
    ViewType Y_0("Y_0", m);
    ViewType Y_1("Y_1", m);
    ViewType Y_2("Y_2", m);
    ViewType Y_0_tilde("Y_0_tilde", m);
    ViewType Y_1_tilde("Y_1_tilde", m);

    // Additional vectors for shuffled operations
    ViewType U_shuffled("U_shuffled", m);
    ViewType Y_2_shuffled("Y_2_shuffled", m);
    ViewType Y_1_shuffled("Y_1_shuffled", m);
    ViewType Y_1_tilde_shuffled("Y_1_tilde_shuffled", m);
    ViewType next_U_shuffled("next_U_shuffled", m);

    // Results from matrix operations
    ViewType A0_result("A0_result", m);
    ViewType A1_result("A1_result", m);
    ViewType A2_result("A2_result", m);
    ViewType A2_result_shuffled("A2_result_shuffled", m);

    // Get boundary vectors
    auto b = bounds.get_b();
    auto b0 = bounds.get_b0();
    auto b1 = bounds.get_b1();
    auto b2 = bounds.get_b2();

    using timer = std::chrono::high_resolution_clock;
    auto t_start = timer::now();

    // Main time stepping loop
    for (int n = 1; n <= N; n++) {
        ViewType A0_result("A0_result", m);
        // Step 1: Y_0 = U + dt * F(n-1, U, A, b)
        A0.multiply_parallel_s_and_v(U, A0_result);
        A1.multiply_parallel_s_and_v(U, A1_result);
        
        // For A2, use shuffled multiply
        shuffle_vector(U, U_shuffled, m1, m2);
        A2_shuf.multiply(U_shuffled, A2_result_shuffled);
        unshuffle_vector(A2_result_shuffled, A2_result, m1, m2);

        // Combine results for Y_0
        Kokkos::parallel_for("Y0_computation", m, KOKKOS_LAMBDA(const int i) {
            double exp_factor = std::exp(r_f * delta_t * (n-1));
            Y_0(i) = U(i) + delta_t * (A0_result(i) + A1_result(i) + A2_result(i) + b(i) * exp_factor);
        });

        // Step 2: Compute rhs_1 and solve for Y_1
        A1.multiply_parallel_s_and_v(U, A1_result);
        
        Kokkos::parallel_for("A1_rhs_computation", m, KOKKOS_LAMBDA(const int i) {
            double exp_factor = std::exp(r_f * delta_t * n);
            double rhs = Y_0(i) + theta * delta_t * (b1(i) * exp_factor - A1_result(i));
            Y_0(i) = rhs;  // Reuse Y_0 to store RHS
        });
        
        A1.solve_implicit_parallel_v(Y_1, Y_0);

        // Step 3: Compute rhs_2 and solve for Y_2
        // Use shuffled A2 multiply
        shuffle_vector(U, U_shuffled, m1, m2);
        A2_shuf.multiply(U_shuffled, A2_result_shuffled);
        unshuffle_vector(A2_result_shuffled, A2_result, m1, m2);
        
        Kokkos::parallel_for("A2_rhs_computation", m, KOKKOS_LAMBDA(const int i) {
            double exp_factor = std::exp(r_f * delta_t * n);
            double rhs = Y_1(i) + theta * delta_t * (b2(i) * exp_factor - A2_result(i));
            Y_1(i) = rhs;  // Reuse Y_1 to store RHS
        });

        shuffle_vector(Y_1, Y_1_shuffled, m1, m2);
        A2_shuf.solve_implicit(Y_2_shuffled, Y_1_shuffled);
        unshuffle_vector(Y_2_shuffled, Y_2, m1, m2);

        // Step 4: Compute Y_0_tilde with corrector term for A0
        A0.multiply_parallel_s_and_v(Y_2, A0_result);  // F_0(n, Y_2)
        A0.multiply_parallel_s_and_v(U, A1_result);    // F_0(n-1, U) (reusing A1_result as temp)
        
        Kokkos::parallel_for("Y0_tilde_computation", m, KOKKOS_LAMBDA(const int i) {
            double exp_factor_n = std::exp(r_f * delta_t * n);
            double exp_factor_nm1 = std::exp(r_f * delta_t * (n-1));
            Y_0_tilde(i) = Y_0(i) + 0.5 * delta_t * (
                (A0_result(i) + b0(i) * exp_factor_n) -
                (A1_result(i) + b0(i) * exp_factor_nm1)
            );
        });

        // Step 5: Compute final rhs_1 and solve for Y_1_tilde
        A1.multiply_parallel_s_and_v(U, A1_result);
        
        Kokkos::parallel_for("A1_final_rhs", m, KOKKOS_LAMBDA(const int i) {
            double exp_factor = std::exp(r_f * delta_t * n);
            double rhs = Y_0_tilde(i) + theta * delta_t * (b1(i) * exp_factor - A1_result(i));
            Y_0_tilde(i) = rhs;  // Reuse Y_0_tilde for RHS
        });
        
        A1.solve_implicit_parallel_v(Y_1_tilde, Y_0_tilde);

        // Step 6: Final A2 solve with shuffling
        shuffle_vector(U, U_shuffled, m1, m2);
        A2_shuf.multiply(U_shuffled, A2_result_shuffled);
        unshuffle_vector(A2_result_shuffled, A2_result, m1, m2);
        
        Kokkos::parallel_for("A2_final_rhs", m, KOKKOS_LAMBDA(const int i) {
            double exp_factor = std::exp(r_f * delta_t * n);
            double rhs = Y_1_tilde(i) + theta * delta_t * (b2(i) * exp_factor - A2_result(i));
            Y_1_tilde(i) = rhs;  // Reuse Y_1_tilde for RHS
        });

        shuffle_vector(Y_1_tilde, Y_1_tilde_shuffled, m1, m2);
        A2_shuf.solve_implicit(next_U_shuffled, Y_1_tilde_shuffled);
        unshuffle_vector(next_U_shuffled, U, m1, m2);
    }

    auto t_end = timer::now();
    std::cout << "CS Shuffled time: "
              << std::chrono::duration<double>(t_end - t_start).count()
              << " seconds" << std::endl;
}


void test_DO_scheme();

#endif // DO_SCHEME_HPP