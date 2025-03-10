#ifndef DO_SCHEME_HPP
#define DO_SCHEME_HPP

#include <Kokkos_Core.hpp>


#include "hes_mat_fac.hpp"
#include "hes_A2_mat.hpp"
#include "BoundaryConditions.hpp"

#include <thread>     // For std::this_thread::sleep_for


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
        
        Kokkos::parallel_for("Y0_computation", m, KOKKOS_LAMBDA(const int i) {
            double exp_factor = std::exp(r_f * delta_t * (n-1));
            Y_0(i) = U(i) + delta_t * (A0_result(i) + A1_result(i) + A2_result_unshuf(i) + b(i) * exp_factor);
        });

        // Rest of function exactly as original
        A1.multiply_parallel_s_and_v(U, A1_result);
        
        Kokkos::parallel_for("A1_rhs_computation", m, KOKKOS_LAMBDA(const int i) {
            double exp_factor_n = std::exp(r_f * delta_t * n);
            double exp_factor_nm1 = std::exp(r_f * delta_t * (n-1));
            double rhs = Y_0(i) + theta * delta_t * (b1(i) * exp_factor_n - (A1_result(i) + b1(i) * exp_factor_nm1));
            Y_0(i) = rhs;  // Reuse Y_0 to store RHS
        });
        
        A1.solve_implicit_parallel_v(Y_1, Y_0);

        shuffle_vector(U, U_shuffled, m1, m2);
        A2_shuf.multiply(U_shuffled, A2_result_shuffled);  
        unshuffle_vector(A2_result_shuffled, A2_result_unshuf, m1, m2);
        
        Kokkos::parallel_for("A2_rhs_computation", m, KOKKOS_LAMBDA(const int i) {
            double exp_factor_n = std::exp(r_f * delta_t * n);
            double exp_factor_nm1 = std::exp(r_f * delta_t * (n-1));
            double rhs = Y_1(i) + theta * delta_t * (b2(i) * exp_factor_n - (A2_result_unshuf(i) + b2(i) * exp_factor_nm1));
            Y_1(i) = rhs;  // Reuse Y_1 to store RHS
        });

        // Shuffle input
        shuffle_vector(Y_1, Y_1_shuffled, m1, m2);
        // Solve with shuffled A2
        A2_shuf.solve_implicit(U_next_shuffled, Y_1_shuffled);
        // Unshuffle result back to U
        unshuffle_vector(U_next_shuffled, U, m1, m2);
    }

    auto t_end = timer::now();
    std::cout << "DO time: "
              << std::chrono::duration<double>(t_end - t_start).count()
              << " seconds" << std::endl;
}


template<class ViewType>
void DO_scheme_american_shuffle(
    const int m,                    
    const int m1,                    
    const int m2,                    
    const int N,                     
    const ViewType& U_0,             
    const double delta_t,            
    const double theta,              
    heston_A0Storage_gpu& A0,        
    heston_A1Storage_gpu& A1,                
    heston_A2_shuffled& A2_shuf,     
    const BoundaryConditions& bounds,
    const double r_f,                
    ViewType& U) {                   
    
    // Initialize result with initial condition
    Kokkos::deep_copy(U, U_0);

    // Create persistent workspace vectors
    ViewType Y_0("Y_0", m);
    ViewType Y_1("Y_1", m);
    ViewType A0_result("A0_result", m);
    ViewType A1_result("A1_result", m);

    ViewType Y_1_shuffled("Y_1_shuffled", m);
    ViewType U_next_shuffled("U_next_shuffled", m);
    ViewType U_shuffled("U_shuffled", m);
    ViewType A2_result_shuffled("A2_result_shuffled", m);
    ViewType A2_result_unshuf("A2_result_unshuf", m);

    // Create lambda workspace for American option
    ViewType lambda_bar("lambda_bar", m);
    Kokkos::deep_copy(lambda_bar, 0.0);  // Initialize lambda to zero

    // Get boundary vectors
    auto b = bounds.get_b();
    auto b1 = bounds.get_b1();
    auto b2 = bounds.get_b2();

    using timer = std::chrono::high_resolution_clock;
    auto t_start = timer::now();

    // Main time stepping loop
    for (int n = 1; n <= N; n++) {
        //need to zero out A0
        ViewType A0_result("A0_result", m);

        // Step 1: Matrix multiplications
        A0.multiply_parallel_s_and_v(U, A0_result);
        A1.multiply_parallel_s_and_v(U, A1_result);
        
        shuffle_vector(U, U_shuffled, m1, m2);
        A2_shuf.multiply(U_shuffled, A2_result_shuffled);  
        unshuffle_vector(A2_result_shuffled, A2_result_unshuf, m1, m2);
        
        // Y_0 computation now includes lambda contribution
        Kokkos::parallel_for("Y0_computation", m, KOKKOS_LAMBDA(const int i) {
            double exp_factor = std::exp(r_f * delta_t * (n-1));
            Y_0(i) = U(i) + delta_t * (A0_result(i) + A1_result(i) + 
                        A2_result_unshuf(i) + b(i) * exp_factor + 
                        lambda_bar(i));  // Add lambda contribution
        });

        // A1 step
        A1.multiply_parallel_s_and_v(U, A1_result);
        
        Kokkos::parallel_for("A1_rhs_computation", m, KOKKOS_LAMBDA(const int i) {
            double exp_factor_n = std::exp(r_f * delta_t * n);
            double exp_factor_nm1 = std::exp(r_f * delta_t * (n-1));
            double rhs = Y_0(i) + theta * delta_t * (b1(i) * exp_factor_n - 
                        (A1_result(i) + b1(i) * exp_factor_nm1));
            Y_0(i) = rhs;
        });
        
        A1.solve_implicit_parallel_v(Y_1, Y_0);

        // A2 step
        shuffle_vector(U, U_shuffled, m1, m2);
        A2_shuf.multiply(U_shuffled, A2_result_shuffled);  
        unshuffle_vector(A2_result_shuffled, A2_result_unshuf, m1, m2);
        
        Kokkos::parallel_for("A2_rhs_computation", m, KOKKOS_LAMBDA(const int i) {
            double exp_factor_n = std::exp(r_f * delta_t * n);
            double exp_factor_nm1 = std::exp(r_f * delta_t * (n-1));
            double rhs = Y_1(i) + theta * delta_t * (b2(i) * exp_factor_n - 
                        (A2_result_unshuf(i) + b2(i) * exp_factor_nm1));
            Y_1(i) = rhs;
        });

        shuffle_vector(Y_1, Y_1_shuffled, m1, m2);
        A2_shuf.solve_implicit(U_next_shuffled, Y_1_shuffled);
        unshuffle_vector(U_next_shuffled, U, m1, m2);

        // American option early exercise check and lambda update
        Kokkos::parallel_for("early_exercise_check", m, KOKKOS_LAMBDA(const int i) {
            // Apply early exercise condition
            const double U_bar = U(i);
            U(i) = Kokkos::max(U_bar - delta_t * lambda_bar(i), U_0(i));

            // Update lambda multiplier
            lambda_bar(i) = Kokkos::max(0.0, lambda_bar(i) + (U_0(i) - U_bar) / delta_t);

            // Set lambda to zero at S_max for all variance levels
            // Every (m1+1)th entry starting at index m1 is an S_max entry
            if(i % (m1 + 1) == m1) {  // This hits exactly S_max entries at all variance levels
                lambda_bar(i) = 0.0;
            }
        });
        
        
        //This is slower than the above version, but maybe "easier" to understand
        //they do the same of course 
        // First do early exercise check for all points
        /*
        Kokkos::parallel_for("early_exercise_check", m, KOKKOS_LAMBDA(const int i) {
            const double U_bar = U(i);
            U(i) = Kokkos::max(U_bar - delta_t * lambda_bar(i), U_0(i));
            lambda_bar(i) = Kokkos::max(0.0, lambda_bar(i) + (U_0(i) - U_bar) / delta_t);
        });

        // Then zero out lambda at S_max for all variance levels
        // This loop only hits the S_max entries
        Kokkos::parallel_for("zero_lambda_at_smax", m2 + 1, KOKKOS_LAMBDA(const int j) {
            const int smax_index = m1 + j * (m1 + 1);  // Index of S_max at variance level j
            lambda_bar(smax_index) = 0.0;
        });
        */
    }

    auto t_end = timer::now();
    std::cout << "DO American time: "
              << std::chrono::duration<double>(t_end - t_start).count()
              << " seconds" << std::endl;
}


template<class ViewType>
void DO_scheme_dividend_shuffled(
    const int m, 
    const int m1, 
    const int m2,
    const int N,
    const ViewType& U_0,
    const double delta_t,
    const double theta,
    const std::vector<double>& dividend_dates,    // Host vector
    const std::vector<double>& dividend_amounts,  // Host vector 
    const std::vector<double>& dividend_percentages, // Host vector
    ViewType& device_Vec_s, //device side stock values
    heston_A0Storage_gpu& A0,
    heston_A1Storage_gpu& A1,
    heston_A2_shuffled& A2_shuf,
    const BoundaryConditions& bounds,
    const double r_f,
    ViewType& U) {

    // Initialize result with initial condition
    Kokkos::deep_copy(U, U_0);

    // Create persistent workspace vectors
    ViewType Y_0("Y_0", m);
    ViewType Y_1("Y_1", m);
    ViewType A0_result("A0_result", m);
    ViewType A1_result("A1_result", m);

    ViewType Y_1_shuffled("Y_1_shuffled", m);
    ViewType U_next_shuffled("U_next_shuffled", m);
    ViewType U_shuffled("U_shuffled", m);
    ViewType A2_result_shuffled("A2_result_shuffled", m);
    ViewType A2_result_unshuf("A2_result_unshuf", m);


    // Get boundary vectors
    auto b = bounds.get_b();
    auto b1 = bounds.get_b1();
    auto b2 = bounds.get_b2();

    // Create mutable copies of dividend data that we can modify
    std::vector<double> div_dates = dividend_dates;
    std::vector<double> div_amounts = dividend_amounts;
    std::vector<double> div_percentages = dividend_percentages;

    using timer = std::chrono::high_resolution_clock;
    auto t_start = timer::now();

    // Main time stepping loop
    for(int n = 1; n <= N; n++) {
        double t = n * delta_t;

        // Check for dividend payment
        while(!div_dates.empty() && t <= div_dates[0] && div_dates[0] < (n+1) * delta_t) {
            double div_date = div_dates[0];
            double div_amount = div_amounts[0];
            double div_percentage = div_percentages[0];

            // Remove processed dividend
            div_dates.erase(div_dates.begin());
            div_amounts.erase(div_amounts.begin());
            div_percentages.erase(div_percentages.begin());

            std::cout << "Processing dividend at t = " << div_date 
                      << ", amount = " << div_amount 
                      << ", percentage = " << div_percentage << std::endl;

            // Create temporary storage for new values
            ViewType U_new("U_new", U.extent(0));
            Kokkos::deep_copy(U_new, U);

            // Process dividend on device
            Kokkos::parallel_for("dividend_adjustment", 
                Kokkos::RangePolicy<>(0, m2+1),
                KOKKOS_LAMBDA(const int j) {
                    const int offset = j * (m1 + 1);
                    
                    // For each stock price level
                    for(int i = 0; i <= m1; i++) {
                        double old_s = device_Vec_s(i);
                        double new_s = old_s * (1.0 - div_percentage) - div_amount;

                        if(new_s > 0) {
                            // Find interpolation points
                            int idx = 0;
                            for(int k = 0; k <= m1; k++) {
                                if(device_Vec_s(k) > new_s) {
                                    idx = k;
                                    break;
                                }
                            }

                            if(idx > 0 && idx < m1 + 1) {
                                // Interpolate
                                double s_low = device_Vec_s(idx-1);
                                double s_high = device_Vec_s(idx);
                                double weight = (new_s - s_low) / (s_high - s_low);
                                
                                double val_low = U(offset + idx-1);
                                double val_high = U(offset + idx);
                                U_new(offset + i) = (1.0 - weight) * val_low + weight * val_high;
                            }
                            else if(idx == 0) {
                                // Left extrapolation
                                U_new(offset + i) = U(offset);
                            }
                            else {
                                // Right extrapolation
                                U_new(offset + i) = U(offset + m1);
                            }
                        }
                        else {
                            U_new(offset + i) = 0.0;
                        }
                    }
                });
            
            // Update U with new values
            Kokkos::deep_copy(U, U_new);
        }

        ViewType A0_result("A0_result", m);

        // Step 1: Let's first just verify we get same result with both A2s
        A0.multiply_parallel_s_and_v(U, A0_result);
        A1.multiply_parallel_s_and_v(U, A1_result);
        
        // Add shuffled A2 multiplication (but don't use result yet)
        shuffle_vector(U, U_shuffled, m1, m2);
        A2_shuf.multiply(U_shuffled, A2_result_shuffled);  
        unshuffle_vector(A2_result_shuffled, A2_result_unshuf, m1, m2);
        
        Kokkos::parallel_for("Y0_computation", m, KOKKOS_LAMBDA(const int i) {
            double exp_factor = std::exp(r_f * delta_t * (n-1));
            Y_0(i) = U(i) + delta_t * (A0_result(i) + A1_result(i) + A2_result_unshuf(i) + b(i) * exp_factor);
        });

        // Rest of function exactly as original
        A1.multiply_parallel_s_and_v(U, A1_result);
        
        Kokkos::parallel_for("A1_rhs_computation", m, KOKKOS_LAMBDA(const int i) {
            double exp_factor_n = std::exp(r_f * delta_t * n);
            double exp_factor_nm1 = std::exp(r_f * delta_t * (n-1));
            double rhs = Y_0(i) + theta * delta_t * (b1(i) * exp_factor_n - (A1_result(i) + b1(i) * exp_factor_nm1));
            Y_0(i) = rhs;  // Reuse Y_0 to store RHS
        });
        
        A1.solve_implicit_parallel_v(Y_1, Y_0);

        shuffle_vector(U, U_shuffled, m1, m2);
        A2_shuf.multiply(U_shuffled, A2_result_shuffled);  
        unshuffle_vector(A2_result_shuffled, A2_result_unshuf, m1, m2);
        
        Kokkos::parallel_for("A2_rhs_computation", m, KOKKOS_LAMBDA(const int i) {
            double exp_factor_n = std::exp(r_f * delta_t * n);
            double exp_factor_nm1 = std::exp(r_f * delta_t * (n-1));
            double rhs = Y_1(i) + theta * delta_t * (b2(i) * exp_factor_n - (A2_result_unshuf(i) + b2(i) * exp_factor_nm1));
            Y_1(i) = rhs;  // Reuse Y_1 to store RHS
        });

        // Shuffle input
        shuffle_vector(Y_1, Y_1_shuffled, m1, m2);
        // Solve with shuffled A2
        A2_shuf.solve_implicit(U_next_shuffled, Y_1_shuffled);
        // Unshuffle result back to U
        unshuffle_vector(U_next_shuffled, U, m1, m2);
        
    }

    auto t_end = timer::now();
    std::cout << "DO with Dividends time: "
              << std::chrono::duration<double>(t_end - t_start).count()
              << " seconds" << std::endl;
}


template<class ViewType>
void DO_scheme_american_dividend_shuffled(
    const int m,                    
    const int m1,                    
    const int m2,                    
    const int N,                     
    const ViewType& U_0,             
    const double delta_t,            
    const double theta,
    const std::vector<double>& dividend_dates,    // Host vector
    const std::vector<double>& dividend_amounts,  // Host vector 
    const std::vector<double>& dividend_percentages, // Host vector    
    ViewType& device_Vec_s, //device side stock values          
    heston_A0Storage_gpu& A0,        
    heston_A1Storage_gpu& A1,                
    heston_A2_shuffled& A2_shuf,     
    const BoundaryConditions& bounds,
    const double r_f,                
    ViewType& U) {                   
    
    // Initialize result with initial condition
    Kokkos::deep_copy(U, U_0);

    // Create persistent workspace vectors
    ViewType Y_0("Y_0", m);
    ViewType Y_1("Y_1", m);
    ViewType A0_result("A0_result", m);
    ViewType A1_result("A1_result", m);

    ViewType Y_1_shuffled("Y_1_shuffled", m);
    ViewType U_next_shuffled("U_next_shuffled", m);
    ViewType U_shuffled("U_shuffled", m);
    ViewType A2_result_shuffled("A2_result_shuffled", m);
    ViewType A2_result_unshuf("A2_result_unshuf", m);

    // Create lambda workspace for American option
    ViewType lambda_bar("lambda_bar", m);
    Kokkos::deep_copy(lambda_bar, 0.0);  // Initialize lambda to zero

    // Create mutable copies of dividend data that we can modify
    std::vector<double> div_dates = dividend_dates;
    std::vector<double> div_amounts = dividend_amounts;
    std::vector<double> div_percentages = dividend_percentages;

    // Get boundary vectors
    auto b = bounds.get_b();
    auto b1 = bounds.get_b1();
    auto b2 = bounds.get_b2();

    using timer = std::chrono::high_resolution_clock;
    auto t_start = timer::now();

    // Main time stepping loop
    for (int n = 1; n <= N; n++) {
        double t = n * delta_t;

        // Check for dividend payment
        while(!div_dates.empty() && t <= div_dates[0] && div_dates[0] < (n+1) * delta_t) {
            double div_date = div_dates[0];
            double div_amount = div_amounts[0];
            double div_percentage = div_percentages[0];

            // Remove processed dividend
            div_dates.erase(div_dates.begin());
            div_amounts.erase(div_amounts.begin());
            div_percentages.erase(div_percentages.begin());

            std::cout << "Processing dividend at t = " << div_date 
                      << ", amount = " << div_amount 
                      << ", percentage = " << div_percentage << std::endl;

            // Create temporary storage for new values
            ViewType U_new("U_new", U.extent(0));
            Kokkos::deep_copy(U_new, U);

            // Process dividend on device
            Kokkos::parallel_for("dividend_adjustment", 
                Kokkos::RangePolicy<>(0, m2+1),
                KOKKOS_LAMBDA(const int j) {
                    const int offset = j * (m1 + 1);
                    
                    // For each stock price level
                    for(int i = 0; i <= m1; i++) {
                        double old_s = device_Vec_s(i);
                        double new_s = old_s * (1.0 - div_percentage) - div_amount;

                        if(new_s > 0) {
                            // Find interpolation points
                            int idx = 0;
                            for(int k = 0; k <= m1; k++) {
                                if(device_Vec_s(k) > new_s) {
                                    idx = k;
                                    break;
                                }
                            }

                            if(idx > 0 && idx < m1 + 1) {
                                // Interpolate
                                double s_low = device_Vec_s(idx-1);
                                double s_high = device_Vec_s(idx);
                                double weight = (new_s - s_low) / (s_high - s_low);
                                
                                double val_low = U(offset + idx-1);
                                double val_high = U(offset + idx);
                                U_new(offset + i) = (1.0 - weight) * val_low + weight * val_high;
                            }
                            else if(idx == 0) {
                                // Left extrapolation
                                U_new(offset + i) = U(offset);
                            }
                            else {
                                // Right extrapolation
                                U_new(offset + i) = U(offset + m1);
                            }
                        }
                        else {
                            U_new(offset + i) = 0.0;
                        }
                    }
                });
            
            // Update U with new values
            Kokkos::deep_copy(U, U_new);
        }

        //need to zero out A0
        ViewType A0_result("A0_result", m);

        // Step 1: Matrix multiplications
        A0.multiply_parallel_s_and_v(U, A0_result);
        A1.multiply_parallel_s_and_v(U, A1_result);
        
        shuffle_vector(U, U_shuffled, m1, m2);
        A2_shuf.multiply(U_shuffled, A2_result_shuffled);  
        unshuffle_vector(A2_result_shuffled, A2_result_unshuf, m1, m2);
        
        // Y_0 computation now includes lambda contribution
        Kokkos::parallel_for("Y0_computation", m, KOKKOS_LAMBDA(const int i) {
            double exp_factor = std::exp(r_f * delta_t * (n-1));
            Y_0(i) = U(i) + delta_t * (A0_result(i) + A1_result(i) + 
                        A2_result_unshuf(i) + b(i) * exp_factor + 
                        lambda_bar(i));  // Add lambda contribution
        });

        // A1 step
        A1.multiply_parallel_s_and_v(U, A1_result);
        
        Kokkos::parallel_for("A1_rhs_computation", m, KOKKOS_LAMBDA(const int i) {
            double exp_factor_n = std::exp(r_f * delta_t * n);
            double exp_factor_nm1 = std::exp(r_f * delta_t * (n-1));
            double rhs = Y_0(i) + theta * delta_t * (b1(i) * exp_factor_n - 
                        (A1_result(i) + b1(i) * exp_factor_nm1));
            Y_0(i) = rhs;
        });
        
        A1.solve_implicit_parallel_v(Y_1, Y_0);

        // A2 step
        shuffle_vector(U, U_shuffled, m1, m2);
        A2_shuf.multiply(U_shuffled, A2_result_shuffled);  
        unshuffle_vector(A2_result_shuffled, A2_result_unshuf, m1, m2);
        
        Kokkos::parallel_for("A2_rhs_computation", m, KOKKOS_LAMBDA(const int i) {
            double exp_factor_n = std::exp(r_f * delta_t * n);
            double exp_factor_nm1 = std::exp(r_f * delta_t * (n-1));
            double rhs = Y_1(i) + theta * delta_t * (b2(i) * exp_factor_n - 
                        (A2_result_unshuf(i) + b2(i) * exp_factor_nm1));
            Y_1(i) = rhs;
        });

        shuffle_vector(Y_1, Y_1_shuffled, m1, m2);
        A2_shuf.solve_implicit(U_next_shuffled, Y_1_shuffled);
        unshuffle_vector(U_next_shuffled, U, m1, m2);

        // American option early exercise check and lambda update
        Kokkos::parallel_for("early_exercise_check", m, KOKKOS_LAMBDA(const int i) {
            // Apply early exercise condition
            const double U_bar = U(i);
            U(i) = Kokkos::max(U_bar - delta_t * lambda_bar(i), U_0(i));

            // Update lambda multiplier
            lambda_bar(i) = Kokkos::max(0.0, lambda_bar(i) + (U_0(i) - U_bar) / delta_t);

            //This helps with stability
            // Set lambda to zero at S_max for all variance levels
            // Every (m1+1)th entry starting at index m1 is an S_max entry
            if(i % (m1 + 1) == m1) {  // This hits exactly S_max entries at all variance levels
                lambda_bar(i) = 0.0;
            }
        });
    
    }

    auto t_end = timer::now();
    std::cout << "DO American with Dividends time: "
              << std::chrono::duration<double>(t_end - t_start).count()
              << " seconds" << std::endl;
}




//First impleemntation of a different scheme
//The CS perfoms one more corrector step with the A0 matrix after the first two implicit sweeps
//then we perfom another implicit sweep. This should offset the non-treatment of the A0 matrix in 
//an implicit manner. 
/*
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

        //Here is a mistake. We need to zero out first when using A0, since there is a bug in the 
        //explicict implementation
        // Step 4: Compute Y_0_tilde with corrector term for A0
        ViewType temp("temp", m);
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
*/

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
    
    // Initialize result with initial condition
    Kokkos::deep_copy(U, U_0);

    // Create persistent workspace vectors to avoid reallocations
    ViewType Y_0("Y_0", m);
    ViewType Y_1("Y_1", m);
    ViewType Y_2("Y_2", m);
    ViewType Y_0_tilde("Y_0_tilde", m);
    ViewType Y_1_tilde("Y_1_tilde", m);

    ViewType A0_result("A0_result", m);
    ViewType A1_result("A1_result", m);
    ViewType A2_result("A2_result", m);
    ViewType A0_Y2_result("A0_Y2_result", m);
    
    // Get boundary vectors
    auto b = bounds.get_b();
    auto b0 = bounds.get_b0();
    auto b1 = bounds.get_b1();
    auto b2 = bounds.get_b2();

    using timer = std::chrono::high_resolution_clock;
    auto t_start = timer::now();

    // Main time stepping loop
    for (int n = 1; n <= N; n++) {
        // Step 1: Calculate Y_0
        ViewType A0_result("A0_result", m);
        A0.multiply_parallel_s_and_v(U, A0_result);
        A1.multiply_parallel_s_and_v(U, A1_result);
        A2.multiply_parallel_s_and_v(U, A2_result);
        
        Kokkos::deep_copy(Y_0, U);  // Y_0 starts as U
        Kokkos::parallel_for("Y0_computation", m, KOKKOS_LAMBDA(const int i) {
            double exp_factor = std::exp(r_f * delta_t * (n-1));
            Y_0(i) += delta_t * (A0_result(i) + A1_result(i) + A2_result(i) + b(i) * exp_factor);
        });

        // Step 2: First Y_1 solve
        A1.multiply_parallel_s_and_v(U, A1_result);
        Kokkos::parallel_for("First_Y1_rhs", m, KOKKOS_LAMBDA(const int i) {
            double exp_factor_now = std::exp(r_f * delta_t * n);
            double exp_factor_prev = std::exp(r_f * delta_t * (n-1));
            Y_1(i) = Y_0(i) + theta * delta_t * (b1(i) * exp_factor_now - (A1_result(i) + b1(i) * exp_factor_prev));
        });
        A1.solve_implicit_parallel_v(Y_1, Y_1);

        // Step 3: Y_2 solve
        A2.multiply_parallel_s_and_v(U, A2_result);
        Kokkos::parallel_for("Y2_rhs", m, KOKKOS_LAMBDA(const int i) {
            double exp_factor_now = std::exp(r_f * delta_t * n);
            double exp_factor_prev = std::exp(r_f * delta_t * (n-1));
            Y_2(i) = Y_1(i) + theta * delta_t * (b2(i) * exp_factor_now - (A2_result(i) + b2(i) * exp_factor_prev));
        });
        A2.solve_implicit(Y_2, Y_2);

        // Step 4: Calculate Y_0_tilde with the corrector term
        ViewType A0_Y2_result("A0_Y2_result", m);
        A0.multiply_parallel_s_and_v(Y_2, A0_Y2_result);
        Kokkos::parallel_for("Y0_tilde", m, KOKKOS_LAMBDA(const int i) {
            double exp_factor_now = std::exp(r_f * delta_t * n);
            double exp_factor_prev = std::exp(r_f * delta_t * (n-1));
            Y_0_tilde(i) = Y_0(i) + 0.5 * delta_t * (
                (A0_Y2_result(i) + b0(i) * exp_factor_now) - 
                (A0_result(i) + b0(i) * exp_factor_prev)
            );
        });

        // Step 5: Second Y_1 solve
        Kokkos::parallel_for("Second_Y1_rhs", m, KOKKOS_LAMBDA(const int i) {
            double exp_factor_now = std::exp(r_f * delta_t * n);
            double exp_factor_prev = std::exp(r_f * delta_t * (n-1));
            Y_1_tilde(i) = Y_0_tilde(i) + theta * delta_t * (b1(i) * exp_factor_now - (A1_result(i) + b1(i) * exp_factor_prev));
        });
        A1.solve_implicit_parallel_v(Y_1_tilde, Y_1_tilde);

        // Step 6: Final solve for U
        Kokkos::parallel_for("Final_U_rhs", m, KOKKOS_LAMBDA(const int i) {
            double exp_factor_now = std::exp(r_f * delta_t * n);
            double exp_factor_prev = std::exp(r_f * delta_t * (n-1));
            U(i) = Y_1_tilde(i) + theta * delta_t * (b2(i) * exp_factor_now - (A2_result(i) + b2(i) * exp_factor_prev));
        });
        A2.solve_implicit(U, U);
    }

    auto t_end = timer::now();
    std::cout << "CS time: "
              << std::chrono::duration<double>(t_end - t_start).count()
              << " seconds" << std::endl;
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

    ViewType A0_Y2_result("A0_Y2_result", m);

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
        Kokkos::parallel_for("First_Y1_rhs", m, KOKKOS_LAMBDA(const int i) {
            double exp_factor_now = std::exp(r_f * delta_t * n);
            double exp_factor_prev = std::exp(r_f * delta_t * (n-1));
            Y_1(i) = Y_0(i) + theta * delta_t * (b1(i) * exp_factor_now - (A1_result(i) + b1(i) * exp_factor_prev));
        });
        A1.solve_implicit_parallel_v(Y_1, Y_1);

        // Step 3: Compute rhs_2 and solve for Y_2
        // Use shuffled A2 multiply
        shuffle_vector(U, U_shuffled, m1, m2);
        A2_shuf.multiply(U_shuffled, A2_result_shuffled);
        unshuffle_vector(A2_result_shuffled, A2_result, m1, m2);
        
        Kokkos::parallel_for("Y2_rhs", m, KOKKOS_LAMBDA(const int i) {
            double exp_factor_now = std::exp(r_f * delta_t * n);
            double exp_factor_prev = std::exp(r_f * delta_t * (n-1));
            Y_2(i) = Y_1(i) + theta * delta_t * (b2(i) * exp_factor_now - (A2_result(i) + b2(i) * exp_factor_prev));
        });

        shuffle_vector(Y_2, Y_2_shuffled, m1, m2);
        A2_shuf.solve_implicit(Y_2_shuffled, Y_2_shuffled);
        unshuffle_vector(Y_2_shuffled, Y_2, m1, m2);

        // Step 4: Calculate Y_0_tilde with the corrector term
        ViewType A0_Y2_result("A0_Y2_result", m);
        A0.multiply_parallel_s_and_v(Y_2, A0_Y2_result);
        Kokkos::parallel_for("Y0_tilde", m, KOKKOS_LAMBDA(const int i) {
            double exp_factor_now = std::exp(r_f * delta_t * n);
            double exp_factor_prev = std::exp(r_f * delta_t * (n-1));
            Y_0_tilde(i) = Y_0(i) + 0.5 * delta_t * (
                (A0_Y2_result(i) + b0(i) * exp_factor_now) - 
                (A0_result(i) + b0(i) * exp_factor_prev)
            );
        });
        
        // Step 5: Second Y_1 solve
        Kokkos::parallel_for("Second_Y1_rhs", m, KOKKOS_LAMBDA(const int i) {
            double exp_factor_now = std::exp(r_f * delta_t * n);
            double exp_factor_prev = std::exp(r_f * delta_t * (n-1));
            Y_1_tilde(i) = Y_0_tilde(i) + theta * delta_t * (b1(i) * exp_factor_now - (A1_result(i) + b1(i) * exp_factor_prev));
        });
        A1.solve_implicit_parallel_v(Y_1_tilde, Y_1_tilde);

        // Step 6: Final A2 solve with shuffling
        Kokkos::parallel_for("Final_U_rhs", m, KOKKOS_LAMBDA(const int i) {
            double exp_factor_now = std::exp(r_f * delta_t * n);
            double exp_factor_prev = std::exp(r_f * delta_t * (n-1));
            U(i) = Y_1_tilde(i) + theta * delta_t * (b2(i) * exp_factor_now - (A2_result(i) + b2(i) * exp_factor_prev));
        });

        shuffle_vector(U, Y_1_tilde_shuffled, m1, m2);
        A2_shuf.solve_implicit(next_U_shuffled, Y_1_tilde_shuffled);
        unshuffle_vector(next_U_shuffled, U, m1, m2);
    }

    auto t_end = timer::now();
    std::cout << "CS Shuffled time: "
              << std::chrono::duration<double>(t_end - t_start).count()
              << " seconds" << std::endl;
}


/*

These are hardcoded methods used for vizulization. 

*/
//This method is used to plot the dividend impact on the price surface where we plot
//it against time and stock price. 
template<class ViewType>
void DO_scheme_dividend_shuffled_with_surface_tracking(
    const int m, 
    const int m1, 
    const int m2,
    const int N,
    const ViewType& U_0,
    const double delta_t,
    const double theta,
    const std::vector<double>& dividend_dates,    // Host vector
    const std::vector<double>& dividend_amounts,  // Host vector 
    const std::vector<double>& dividend_percentages, // Host vector
    ViewType& device_Vec_s, //device side stock values
    heston_A0Storage_gpu& A0,
    heston_A1Storage_gpu& A1,
    heston_A2_shuffled& A2_shuf,
    const BoundaryConditions& bounds,
    const double r_f,
    ViewType& U,
    Kokkos::View<double**>& price_surface, //for plotting
    int index_v
    ) {

    // Initialize result with initial condition
    Kokkos::deep_copy(U, U_0);

    // Create persistent workspace vectors
    ViewType Y_0("Y_0", m);
    ViewType Y_1("Y_1", m);
    ViewType A0_result("A0_result", m);
    ViewType A1_result("A1_result", m);

    ViewType Y_1_shuffled("Y_1_shuffled", m);
    ViewType U_next_shuffled("U_next_shuffled", m);
    ViewType U_shuffled("U_shuffled", m);
    ViewType A2_result_shuffled("A2_result_shuffled", m);
    ViewType A2_result_unshuf("A2_result_unshuf", m);


    // Get boundary vectors
    auto b = bounds.get_b();
    auto b1 = bounds.get_b1();
    auto b2 = bounds.get_b2();

    // Create mutable copies of dividend data that we can modify
    std::vector<double> div_dates = dividend_dates;
    std::vector<double> div_amounts = dividend_amounts;
    std::vector<double> div_percentages = dividend_percentages;

    //only for ploting (vizual test)
    Kokkos::parallel_for("store_initial", m1+1, KOKKOS_LAMBDA(const int i) {
        price_surface(0, i) = U_0(i + index_v*(m1+1));
    });
    Kokkos::fence();

    using timer = std::chrono::high_resolution_clock;
    auto t_start = timer::now();

    // Main time stepping loop
    for(int n = 1; n <= N; n++) {
        
        double t = n * delta_t;

        // Check for dividend payment
        while(!div_dates.empty() && t <= div_dates[0] && div_dates[0] < (n+1) * delta_t) {
            double div_date = div_dates[0];
            double div_amount = div_amounts[0];
            double div_percentage = div_percentages[0];

            // Remove processed dividend
            div_dates.erase(div_dates.begin());
            div_amounts.erase(div_amounts.begin());
            div_percentages.erase(div_percentages.begin());

            std::cout << "Processing dividend at t = " << div_date 
                      << ", amount = " << div_amount 
                      << ", percentage = " << div_percentage << std::endl;

            // Create temporary storage for new values
            ViewType U_new("U_new", U.extent(0));
            Kokkos::deep_copy(U_new, U);

            // Process dividend on device
            Kokkos::parallel_for("dividend_adjustment", 
                Kokkos::RangePolicy<>(0, m2+1),
                KOKKOS_LAMBDA(const int j) {
                    const int offset = j * (m1 + 1);
                    
                    // For each stock price level
                    for(int i = 0; i <= m1; i++) {
                        double old_s = device_Vec_s(i);
                        double new_s = old_s * (1.0 - div_percentage) - div_amount;

                        if(new_s > 0) {
                            // Find interpolation points
                            int idx = 0;
                            for(int k = 0; k <= m1; k++) {
                                if(device_Vec_s(k) > new_s) {
                                    idx = k;
                                    break;
                                }
                            }

                            if(idx > 0 && idx < m1 + 1) {
                                // Interpolate
                                double s_low = device_Vec_s(idx-1);
                                double s_high = device_Vec_s(idx);
                                double weight = (new_s - s_low) / (s_high - s_low);
                                
                                double val_low = U(offset + idx-1);
                                double val_high = U(offset + idx);

                                U_new(offset + i) = (1.0 - weight) * val_low + weight * val_high;
                            }
                            else if(idx == 0) {
                                // Left extrapolation
                                U_new(offset + i) = U(offset);
                            }
                            else {
                                // Right extrapolation
                                U_new(offset + i) = U(offset + m1);
                            }
                        }
                        else {
                            U_new(offset + i) = 0.0;
                        }
                    }
                });
            
            // Update U with new values
            Kokkos::deep_copy(U, U_new);
        }
        
        

        ViewType A0_result("A0_result", m);

        // Step 1: Let's first just verify we get same result with both A2s
        A0.multiply_parallel_s_and_v(U, A0_result);
        A1.multiply_parallel_s_and_v(U, A1_result);
        
        // Add shuffled A2 multiplication (but don't use result yet)
        shuffle_vector(U, U_shuffled, m1, m2);
        A2_shuf.multiply(U_shuffled, A2_result_shuffled);  
        unshuffle_vector(A2_result_shuffled, A2_result_unshuf, m1, m2);
        
        Kokkos::parallel_for("Y0_computation", m, KOKKOS_LAMBDA(const int i) {
            double exp_factor = std::exp(r_f * delta_t * (n-1));
            Y_0(i) = U(i) + delta_t * (A0_result(i) + A1_result(i) + A2_result_unshuf(i) + b(i) * exp_factor);
        });

        // Rest of function exactly as original
        A1.multiply_parallel_s_and_v(U, A1_result);
        
        Kokkos::parallel_for("A1_rhs_computation", m, KOKKOS_LAMBDA(const int i) {
            double exp_factor_n = std::exp(r_f * delta_t * n);
            double exp_factor_nm1 = std::exp(r_f * delta_t * (n-1));
            double rhs = Y_0(i) + theta * delta_t * (b1(i) * exp_factor_n - (A1_result(i) + b1(i) * exp_factor_nm1));
            Y_0(i) = rhs;  // Reuse Y_0 to store RHS
        });
        
        A1.solve_implicit_parallel_v(Y_1, Y_0);

        shuffle_vector(U, U_shuffled, m1, m2);
        A2_shuf.multiply(U_shuffled, A2_result_shuffled);  
        unshuffle_vector(A2_result_shuffled, A2_result_unshuf, m1, m2);
        
        Kokkos::parallel_for("A2_rhs_computation", m, KOKKOS_LAMBDA(const int i) {
            double exp_factor_n = std::exp(r_f * delta_t * n);
            double exp_factor_nm1 = std::exp(r_f * delta_t * (n-1));
            double rhs = Y_1(i) + theta * delta_t * (b2(i) * exp_factor_n - (A2_result_unshuf(i) + b2(i) * exp_factor_nm1));
            Y_1(i) = rhs;  // Reuse Y_1 to store RHS
        });

        // Shuffle input
        shuffle_vector(Y_1, Y_1_shuffled, m1, m2);
        // Solve with shuffled A2
        A2_shuf.solve_implicit(U_next_shuffled, Y_1_shuffled);
        // Unshuffle result back to U
        unshuffle_vector(U_next_shuffled, U, m1, m2);

        //Only for ploting. Should not be done in production code
        // Store the solution for current timestep n
        
        Kokkos::parallel_for("store_timestep", m1+1, KOKKOS_LAMBDA(const int i) {
            price_surface(n, i) = U(i + index_v*(m1+1));
        });
        Kokkos::fence();
        
    }

    auto t_end = timer::now();
    std::cout << "DO time: "
              << std::chrono::duration<double>(t_end - t_start).count()
              << " seconds" << std::endl;
}

//This method is used to plot the lambda surface appearing in the american option computation
//against stokc and time
template<class ViewType>
void DO_scheme_american_shuffle_with_lambda_tracking(
    const int m, const int m1, const int m2, const int N,                     
    const ViewType& U_0, const double delta_t, const double theta,              
    heston_A0Storage_gpu& A0, 
    heston_A1Storage_gpu& A1,        
    heston_A2_shuffled& A2_shuf,     
    const BoundaryConditions& bounds, const double r_f,
    const double V_0,  // Add V_0 to help find variance level                
    ViewType& U,
    std::vector<std::vector<double>>& lambda_evolution) {                   
    
    // Initialize result with initial condition
    Kokkos::deep_copy(U, U_0);

    // Create persistent workspace vectors
    ViewType Y_0("Y_0", m);
    ViewType Y_1("Y_1", m);
    ViewType A0_result("A0_result", m);
    ViewType A1_result("A1_result", m);

    ViewType Y_1_shuffled("Y_1_shuffled", m);
    ViewType U_next_shuffled("U_next_shuffled", m);
    ViewType U_shuffled("U_shuffled", m);
    ViewType A2_result_shuffled("A2_result_shuffled", m);
    ViewType A2_result_unshuf("A2_result_unshuf", m);

    // Create lambda workspace for American option
    ViewType lambda_bar("lambda_bar", m);
    Kokkos::deep_copy(lambda_bar, 0.0);  // Initialize lambda to zero

    // Get boundary vectors
    auto b = bounds.get_b();
    auto b1 = bounds.get_b1();
    auto b2 = bounds.get_b2();

    // Create storage for lambda evolution at fixed variance
    const int v_index = std::floor(V_0 * (m2 + 1) / 5.0);  // Assuming V_max = 5.0

    // Store initial lambda values
    auto h_lambda = Kokkos::create_mirror_view(lambda_bar);
    Kokkos::deep_copy(h_lambda, lambda_bar);
    for(int i = 0; i <= m1; i++) {
        lambda_evolution[0][i] = h_lambda[i + v_index * (m1 + 1)];
    }

    using timer = std::chrono::high_resolution_clock;
    auto t_start = timer::now();

    // Main time stepping loop
    for (int n = 1; n <= N; n++) {
        //need to zero out A0
        ViewType A0_result("A0_result", m);

        // Step 1: Matrix multiplications
        A0.multiply_parallel_s_and_v(U, A0_result);
        A1.multiply_parallel_s_and_v(U, A1_result);
        
        shuffle_vector(U, U_shuffled, m1, m2);
        A2_shuf.multiply(U_shuffled, A2_result_shuffled);  
        unshuffle_vector(A2_result_shuffled, A2_result_unshuf, m1, m2);
        
        // Y_0 computation now includes lambda contribution
        Kokkos::parallel_for("Y0_computation", m, KOKKOS_LAMBDA(const int i) {
            double exp_factor = std::exp(r_f * delta_t * (n-1));
            Y_0(i) = U(i) + delta_t * (A0_result(i) + A1_result(i) + 
                        A2_result_unshuf(i) + b(i) * exp_factor + 
                        lambda_bar(i));  // Add lambda contribution
        });

        // A1 step
        A1.multiply_parallel_s_and_v(U, A1_result);
        
        Kokkos::parallel_for("A1_rhs_computation", m, KOKKOS_LAMBDA(const int i) {
            double exp_factor_n = std::exp(r_f * delta_t * n);
            double exp_factor_nm1 = std::exp(r_f * delta_t * (n-1));
            double rhs = Y_0(i) + theta * delta_t * (b1(i) * exp_factor_n - 
                        (A1_result(i) + b1(i) * exp_factor_nm1));
            Y_0(i) = rhs;
        });
        
        A1.solve_implicit_parallel_v(Y_1, Y_0);

        // A2 step
        shuffle_vector(U, U_shuffled, m1, m2);
        A2_shuf.multiply(U_shuffled, A2_result_shuffled);  
        unshuffle_vector(A2_result_shuffled, A2_result_unshuf, m1, m2);
        
        Kokkos::parallel_for("A2_rhs_computation", m, KOKKOS_LAMBDA(const int i) {
            double exp_factor_n = std::exp(r_f * delta_t * n);
            double exp_factor_nm1 = std::exp(r_f * delta_t * (n-1));
            double rhs = Y_1(i) + theta * delta_t * (b2(i) * exp_factor_n - 
                        (A2_result_unshuf(i) + b2(i) * exp_factor_nm1));
            Y_1(i) = rhs;
        });

        shuffle_vector(Y_1, Y_1_shuffled, m1, m2);
        A2_shuf.solve_implicit(U_next_shuffled, Y_1_shuffled);
        unshuffle_vector(U_next_shuffled, U, m1, m2);

        // American option early exercise check and lambda update
        Kokkos::parallel_for("early_exercise_check", m, KOKKOS_LAMBDA(const int i) {
            // Apply early exercise condition
            const double U_bar = U(i);
            U(i) = Kokkos::max(U_bar - delta_t * lambda_bar(i), U_0(i));

            // Update lambda multiplier
            lambda_bar(i) = Kokkos::max(0.0, lambda_bar(i) + (U_0(i) - U_bar) / delta_t);

            // Set lambda to zero at S_max for all variance levels
            // Every (m1+1)th entry starting at index m1 is an S_max entry
            if(i % (m1 + 1) == m1) {  // This hits exactly S_max entries at all variance levels
                lambda_bar(i) = 0.0;
            }
        });

        Kokkos::deep_copy(h_lambda, lambda_bar);
        for(int i = 0; i <= m1; i++) {
            lambda_evolution[n][i] = h_lambda[i + v_index * (m1 + 1)];
        }
        
    }

    auto t_end = timer::now();
    std::cout << "DO American time: "
              << std::chrono::duration<double>(t_end - t_start).count()
              << " seconds" << std::endl;
}

//This is just like the above, only that we have dividends as well, we are still tracking lambda tho
template<class ViewType>
void DO_scheme_american_dividend_shuffle_with_lambda_tracking(
    const int m,                    
    const int m1,                    
    const int m2,                    
    const int N,                     
    const ViewType& U_0,             
    const double delta_t,            
    const double theta,
    const std::vector<double>& dividend_dates,    // Host vector
    const std::vector<double>& dividend_amounts,  // Host vector 
    const std::vector<double>& dividend_percentages, // Host vector    
    ViewType& device_Vec_s, //device side stock values          
    heston_A0Storage_gpu& A0,        
    heston_A1Storage_gpu& A1,                
    heston_A2_shuffled& A2_shuf,     
    const BoundaryConditions& bounds,
    const double r_f,
    const double V_0,                
    ViewType& U,
    std::vector<std::vector<double>>& lambda_evolution) {                   
    
    // Initialize result with initial condition
    Kokkos::deep_copy(U, U_0);

    // Create persistent workspace vectors
    ViewType Y_0("Y_0", m);
    ViewType Y_1("Y_1", m);
    ViewType A0_result("A0_result", m);
    ViewType A1_result("A1_result", m);

    ViewType Y_1_shuffled("Y_1_shuffled", m);
    ViewType U_next_shuffled("U_next_shuffled", m);
    ViewType U_shuffled("U_shuffled", m);
    ViewType A2_result_shuffled("A2_result_shuffled", m);
    ViewType A2_result_unshuf("A2_result_unshuf", m);

    // Create lambda workspace for American option
    ViewType lambda_bar("lambda_bar", m);
    Kokkos::deep_copy(lambda_bar, 0.0);  // Initialize lambda to zero

    // Create mutable copies of dividend data that we can modify
    std::vector<double> div_dates = dividend_dates;
    std::vector<double> div_amounts = dividend_amounts;
    std::vector<double> div_percentages = dividend_percentages;

    // Get boundary vectors
    auto b = bounds.get_b();
    auto b1 = bounds.get_b1();
    auto b2 = bounds.get_b2();

    const int v_index = std::floor(V_0 * (m2 + 1) / 5.0);  // Assuming V_max = 5.0

    // Store initial lambda values
    auto h_lambda = Kokkos::create_mirror_view(lambda_bar);
    Kokkos::deep_copy(h_lambda, lambda_bar);
    for(int i = 0; i <= m1; i++) {
        lambda_evolution[0][i] = h_lambda[i + v_index * (m1 + 1)];
    }

    using timer = std::chrono::high_resolution_clock;
    auto t_start = timer::now();

    // Main time stepping loop
    for (int n = 1; n <= N; n++) {
        double t = n * delta_t;

        // Check for dividend payment
        while(!div_dates.empty() && t <= div_dates[0] && div_dates[0] < (n+1) * delta_t) {
            double div_date = div_dates[0];
            double div_amount = div_amounts[0];
            double div_percentage = div_percentages[0];

            // Remove processed dividend
            div_dates.erase(div_dates.begin());
            div_amounts.erase(div_amounts.begin());
            div_percentages.erase(div_percentages.begin());

            std::cout << "Processing dividend at t = " << div_date 
                      << ", amount = " << div_amount 
                      << ", percentage = " << div_percentage << std::endl;

            // Create temporary storage for new values
            ViewType U_new("U_new", U.extent(0));
            Kokkos::deep_copy(U_new, U);

            // Process dividend on device
            Kokkos::parallel_for("dividend_adjustment", 
                Kokkos::RangePolicy<>(0, m2+1),
                KOKKOS_LAMBDA(const int j) {
                    const int offset = j * (m1 + 1);
                    
                    // For each stock price level
                    for(int i = 0; i <= m1; i++) {
                        double old_s = device_Vec_s(i);
                        double new_s = old_s * (1.0 - div_percentage) - div_amount;

                        if(new_s > 0) {
                            // Find interpolation points
                            int idx = 0;
                            for(int k = 0; k <= m1; k++) {
                                if(device_Vec_s(k) > new_s) {
                                    idx = k;
                                    break;
                                }
                            }

                            if(idx > 0 && idx < m1 + 1) {
                                // Interpolate
                                double s_low = device_Vec_s(idx-1);
                                double s_high = device_Vec_s(idx);
                                double weight = (new_s - s_low) / (s_high - s_low);
                                
                                double val_low = U(offset + idx-1);
                                double val_high = U(offset + idx);
                                U_new(offset + i) = (1.0 - weight) * val_low + weight * val_high;
                            }
                            else if(idx == 0) {
                                // Left extrapolation
                                U_new(offset + i) = U(offset);
                            }
                            else {
                                // Right extrapolation
                                U_new(offset + i) = U(offset + m1);
                            }
                        }
                        else {
                            U_new(offset + i) = 0.0;
                        }
                    }
                });
            
            // Update U with new values
            Kokkos::deep_copy(U, U_new);
        }

        //need to zero out A0
        ViewType A0_result("A0_result", m);

        // Step 1: Matrix multiplications
        A0.multiply_parallel_s_and_v(U, A0_result);
        A1.multiply_parallel_s_and_v(U, A1_result);
        
        shuffle_vector(U, U_shuffled, m1, m2);
        A2_shuf.multiply(U_shuffled, A2_result_shuffled);  
        unshuffle_vector(A2_result_shuffled, A2_result_unshuf, m1, m2);
        
        // Y_0 computation now includes lambda contribution
        Kokkos::parallel_for("Y0_computation", m, KOKKOS_LAMBDA(const int i) {
            double exp_factor = std::exp(r_f * delta_t * (n-1));
            Y_0(i) = U(i) + delta_t * (A0_result(i) + A1_result(i) + 
                        A2_result_unshuf(i) + b(i) * exp_factor + 
                        lambda_bar(i));  // Add lambda contribution
        });

        // A1 step
        A1.multiply_parallel_s_and_v(U, A1_result);
        
        Kokkos::parallel_for("A1_rhs_computation", m, KOKKOS_LAMBDA(const int i) {
            double exp_factor_n = std::exp(r_f * delta_t * n);
            double exp_factor_nm1 = std::exp(r_f * delta_t * (n-1));
            double rhs = Y_0(i) + theta * delta_t * (b1(i) * exp_factor_n - 
                        (A1_result(i) + b1(i) * exp_factor_nm1));
            Y_0(i) = rhs;
        });
        
        A1.solve_implicit_parallel_v(Y_1, Y_0);

        // A2 step
        shuffle_vector(U, U_shuffled, m1, m2);
        A2_shuf.multiply(U_shuffled, A2_result_shuffled);  
        unshuffle_vector(A2_result_shuffled, A2_result_unshuf, m1, m2);
        
        Kokkos::parallel_for("A2_rhs_computation", m, KOKKOS_LAMBDA(const int i) {
            double exp_factor_n = std::exp(r_f * delta_t * n);
            double exp_factor_nm1 = std::exp(r_f * delta_t * (n-1));
            double rhs = Y_1(i) + theta * delta_t * (b2(i) * exp_factor_n - 
                        (A2_result_unshuf(i) + b2(i) * exp_factor_nm1));
            Y_1(i) = rhs;
        });

        shuffle_vector(Y_1, Y_1_shuffled, m1, m2);
        A2_shuf.solve_implicit(U_next_shuffled, Y_1_shuffled);
        unshuffle_vector(U_next_shuffled, U, m1, m2);

        // American option early exercise check and lambda update
        Kokkos::parallel_for("early_exercise_check", m, KOKKOS_LAMBDA(const int i) {
            // Apply early exercise condition
            const double U_bar = U(i);
            U(i) = Kokkos::max(U_bar - delta_t * lambda_bar(i), U_0(i));

            // Update lambda multiplier
            lambda_bar(i) = Kokkos::max(0.0, lambda_bar(i) + (U_0(i) - U_bar) / delta_t);

            // Set lambda to zero at S_max for all variance levels
            // Every (m1+1)th entry starting at index m1 is an S_max entry
            if(i % (m1 + 1) == m1) {  // This hits exactly S_max entries at all variance levels
                lambda_bar(i) = 0.0;
            }
        });

        Kokkos::deep_copy(h_lambda, lambda_bar);
        for(int i = 0; i <= m1; i++) {
            lambda_evolution[n][i] = h_lambda[i + v_index * (m1 + 1)];
        }
        
    }

    auto t_end = timer::now();
    std::cout << "DO American time: "
              << std::chrono::duration<double>(t_end - t_start).count()
              << " seconds" << std::endl;
}


void test_DO_scheme();

#endif // DO_SCHEME_HPP