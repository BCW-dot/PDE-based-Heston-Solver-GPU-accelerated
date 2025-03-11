#include "hes_mat_fac.hpp"
#include <iostream>

//for std::setprec() output debugging of flatten A1
#include <iomanip>
//for accumulate
#include <numeric>

/*

A0 class constructor and build function

*/
heston_A0Storage_gpu::heston_A0Storage_gpu(int m1_in, int m2_in) 
    : m1(m1_in), m2(m2_in) {
    values = Kokkos::View<double**>("A0_values", m2 - 1, (m1 - 1) * 9);
}

void heston_A0Storage_gpu::build_matrix(const Grid& grid, double rho, double sigma) {
    auto values_host = Kokkos::create_mirror_view(values);
    Kokkos::deep_copy(values_host, 0.0);

    for(int j = 0; j < m2-1; ++j) {
        for(int i = 0; i < m1-1; ++i) {
            double c = rho * sigma * grid.Vec_s[i+1] * grid.Vec_v[j+1];

            for(int k = -1; k <= 1; ++k) {
                for(int l = -1; l <= 1; ++l) {
                    double beta_s_val = beta_s(i, k, grid.Delta_s);
                    double beta_v_val = beta_v(j, l, grid.Delta_v);
                    
                    int idx = i * 9 + (l + 1) * 3 + (k + 1);
                    values_host(j, idx) = c * beta_s_val * beta_v_val;
                }
            }
        }
    }

    Kokkos::deep_copy(values, values_host);
}


/*

A1 class constructor and build function

*/
heston_A1Storage_gpu::heston_A1Storage_gpu(int m1_in, int m2_in) : m1(m1_in), m2(m2_in) {
    main_diags = Kokkos::View<double**>("A1_main_diags", m2+1, m1+1);
    lower_diags = Kokkos::View<double**>("A1_lower_diags", m2+1, m1);
    upper_diags = Kokkos::View<double**>("A1_upper_diags", m2+1, m1);

    implicit_main_diags = Kokkos::View<double**>("A1_impl_main_diags", m2+1, m1+1);
    implicit_lower_diags = Kokkos::View<double**>("A1_impl_lower_diags", m2+1, m1);
    implicit_upper_diags = Kokkos::View<double**>("A1_impl_upper_diags", m2+1, m1);

    temp_sequ = Kokkos::View<double*>("temp_sequ", m1+1);
    temp_para = Kokkos::View<double**>("temp_para", m2 + 1, m1 + 1);
}

void heston_A1Storage_gpu::build_matrix(const Grid& grid, double rho, double sigma, double r_d, double r_f) {
    auto main_diags_host = Kokkos::create_mirror_view(main_diags);
    auto lower_diags_host = Kokkos::create_mirror_view(lower_diags);
    auto upper_diags_host = Kokkos::create_mirror_view(upper_diags);
    
    Kokkos::deep_copy(main_diags_host, 0.0);
    Kokkos::deep_copy(lower_diags_host, 0.0);
    Kokkos::deep_copy(upper_diags_host, 0.0);

    // For j in range(m2 + 1)
    for(int j = 0; j <= m2; j++) {
        for(int i = 1; i < m1; i++) {
            double a = 0.5 * grid.Vec_s[i] * grid.Vec_s[i] * grid.Vec_v[j];
            double b = (r_d - r_f) * grid.Vec_s[i];
            
            // Populate diagonals using central difference coefficients
            lower_diags_host(j, i-1) = a * delta_s(i-1, -1, grid.Delta_s) + 
                                    b * beta_s(i-1, -1, grid.Delta_s);
            main_diags_host(j, i) = a * delta_s(i-1, 0, grid.Delta_s) + 
                                b * beta_s(i-1, 0, grid.Delta_s) - 0.5 * r_d;
            upper_diags_host(j, i) = a * delta_s(i-1, 1, grid.Delta_s) + 
                                    b * beta_s(i-1, 1, grid.Delta_s);
        }
        // Add boundary term
        main_diags_host(j, m1) = -0.5 * r_d;
    }

    Kokkos::deep_copy(main_diags, main_diags_host);
    Kokkos::deep_copy(lower_diags, lower_diags_host);
    Kokkos::deep_copy(upper_diags, upper_diags_host);
}

void heston_A1Storage_gpu::build_implicit(const double theta, const double delta_t) {
    const int local_m1 = m1;
    const int local_m2 = m2;
    const auto local_main = main_diags;
    const auto local_lower = lower_diags;  
    const auto local_upper = upper_diags;
    const auto local_impl_main = implicit_main_diags;
    const auto local_impl_lower = implicit_lower_diags;
    const auto local_impl_upper = implicit_upper_diags;

    Kokkos::parallel_for("build_implicit", 1, KOKKOS_LAMBDA(const int) {
        for(int j = 0; j <= local_m2; j++) {
            for(int i = 0; i <= local_m1; i++) {
                local_impl_main(j,i) = 1.0 - theta * delta_t * local_main(j,i);
            }
            for(int i = 0; i < local_m1; i++) {
                local_impl_lower(j,i) = -theta * delta_t * local_lower(j,i);
                local_impl_upper(j,i) = -theta * delta_t * local_upper(j,i);
            }
        }
    });
    Kokkos::fence();
}


//A0 class test
void test_heston_A0() {
    {
        int m1 = 5;
        int m2 = 5;
        
        // Create grid
        Grid grid = create_test_grid(m1, m2);
        
        // Create and build A0 matrix
        heston_A0Storage_gpu A0(m1, m2);
        double rho = -0.9;
        double sigma = 0.3;
        A0.build_matrix(grid, rho, sigma);
        
        // Get host copy using the getter method
        auto values = A0.get_values();
        auto values_host = Kokkos::create_mirror_view(values);
        Kokkos::deep_copy(values_host, values);
        
        // Print matrix structure
        std::cout << "A0 Matrix Structure:" << std::endl;
        std::cout << "--------------------" << std::endl;
        
        for(int j = 0; j < m2-1; ++j) {
            std::cout << "\nVariance level j=" << j << ":" << std::endl;
            for(int i = 0; i < m1-1; ++i) {
                std::cout << "Row " << i << ": ";
                for(int val = 0; val < 9; ++val) {
                    std::cout << values_host(j, i*9 + val) << " ";
                }
                std::cout << std::endl;
            }
        }
        
        // Print dimensions
        std::cout << "\nDimensions:" << std::endl;
        std::cout << "Values shape: [" << m2-1 << "][" << (m1-1)*9 << "]" << std::endl;
        std::cout << "m1: " << A0.get_m1() << ", m2: " << A0.get_m2() << std::endl;
    }
}

//A1 class test
void test_heston_A1() {
    int m1 = 100;
    int m2 = 75;
    Grid grid = create_test_grid(m1, m2);
    
    heston_A1Storage_gpu A1(m1, m2);
    double rho = -0.9;
    double sigma = 0.3;
    double r_d = 0.025;
    double r_f = 0.0;

    const double theta = 0.8;
    const double delta_t = 1.0/40.0; 

    using timer = std::chrono::high_resolution_clock;
    
    auto t_start = timer::now();
    A1.build_matrix(grid, rho, sigma, r_d, r_f);
    A1.build_implicit(theta, delta_t);
    auto t_end = timer::now();
    
    std::cout << "Build matrix time: "
              << std::chrono::duration<double>(t_end - t_start).count()
              << " seconds" << std::endl;
    
    // Get Views using getters
    auto main = A1.get_main_diags();
    auto lower = A1.get_lower_diags();
    auto upper = A1.get_upper_diags();

    auto implicit_main = A1.get_implicit_main_diags();
    auto implicit_lower = A1.get_implicit_lower_diags();
    auto implicit_upper = A1.get_implicit_upper_diags();

    // Create mirror views (fixed typo in Kokkos::)
    auto main_host = Kokkos::create_mirror_view(main);
    auto lower_host = Kokkos::create_mirror_view(lower);
    auto upper_host = Kokkos::create_mirror_view(upper);

    auto implicit_main_host = Kokkos::create_mirror_view(implicit_main);
    auto implicit_lower_host = Kokkos::create_mirror_view(implicit_lower);
    auto implicit_upper_host = Kokkos::create_mirror_view(implicit_upper);
    
    // Copy to host
    Kokkos::deep_copy(main_host, main);
    Kokkos::deep_copy(lower_host, lower);
    Kokkos::deep_copy(upper_host, upper);

    Kokkos::deep_copy(implicit_main_host, implicit_main);
    Kokkos::deep_copy(implicit_lower_host, implicit_lower);
    Kokkos::deep_copy(implicit_upper_host, implicit_upper);
    
    // Print matrices
    /*
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "A1 Matrix Structure:\n";
    std::cout << "--------------------\n";
    
    for(int j = 0; j <= m2; j++) {
        std::cout << "\nVariance level j=" << j << ":\n";
        std::cout << "Lower diagonal: ";
        for(int i = 0; i < m1; i++) std::cout << lower_host(j,i) << " ";
        std::cout << "\nMain diagonal:  ";
        for(int i = 0; i <= m1; i++) std::cout << main_host(j,i) << " ";
        std::cout << "\nUpper diagonal: ";
        for(int i = 0; i < m1; i++) std::cout << upper_host(j,i) << " ";
        std::cout << "\n";

        std::cout << "Implicit diagonals:\n";
        std::cout << "Lower diagonal: ";
        for(int i = 0; i < m1; i++) std::cout << implicit_lower_host(j,i) << " ";
        std::cout << "\nMain diagonal:  ";
        for(int i = 0; i <= m1; i++) std::cout << implicit_main_host(j,i) << " ";
        std::cout << "\nUpper diagonal: ";
        for(int i = 0; i < m1; i++) std::cout << implicit_upper_host(j,i) << " ";
        std::cout << "\n";
        std::cout << "----------------------------------------\n";

    }
    */

    // Print dimensions
    std::cout << "\nDimensions:\n";
    std::cout << "m1: " << A1.get_m1() << ", m2: " << A1.get_m2() << "\n";
}   


//This test compare the explicit and implicit output of a simple test case vector. It is compared to the python implementation
//and checked that the outputs align. This test was written when i saw oszillatory behavior in the m1 direction when increasing
//this dimension size
//RESULT OF TEST: No discrepencies between ypthon and c++
void test_A1_structure() {
    // Test dimensions
    int m1 = 4;  // Small dimensions for readable output
    int m2 = 3;
    
    Grid grid = create_test_grid(m1, m2);
    
    // Create and build A1 matrix
    heston_A1Storage_gpu A1(m1, m2);
    double rho = -0.9;
    double sigma = 0.3;
    double r_d = 0.025;
    double r_f = 0.0;
    
    A1.build_matrix(grid, rho, sigma, r_d, r_f);
    
    const int total_size = (m1 + 1) * (m2 + 1);
    std::cout << "\nA1 Matrix Structure:";
    std::cout << "\nShape: [" << total_size << ", " << total_size << "]" << std::endl;

    std::cout << "\nA1 Matrix Values by Block:" << std::endl;
        
    auto h_main = Kokkos::create_mirror_view(A1.get_main_diags());
    auto h_lower = Kokkos::create_mirror_view(A1.get_lower_diags());
    auto h_upper = Kokkos::create_mirror_view(A1.get_upper_diags());
    
    Kokkos::deep_copy(h_main, A1.get_main_diags());
    Kokkos::deep_copy(h_lower, A1.get_lower_diags());
    Kokkos::deep_copy(h_upper, A1.get_upper_diags());
    
    for(int j = 0; j <= m2; j++) {
        std::cout << "\nBlock j=" << j << ":" << std::endl;
        
        // Print lower diagonal
        std::cout << "\n  Lower diagonal for block " << j << ":" << std::endl;
        for(int i = 0; i < m1; i++) {
            double val = h_lower(j,i);
            if(std::abs(val) > 1e-10) {
                std::cout << "    [" << i+1 << "," << i << "] = " 
                            << std::fixed << std::setprecision(6) << val << std::endl;
            }
        }
        
        // Print main diagonal
        std::cout << "\n  Main diagonal for block " << j << ":" << std::endl;
        for(int i = 0; i <= m1; i++) {
            double val = h_main(j,i);
            if(std::abs(val) > 1e-10) {
                std::cout << "    [" << i << "," << i << "] = " 
                            << std::fixed << std::setprecision(6) << val << std::endl;
            }
        }
        
        // Print upper diagonal
        std::cout << "\n  Upper diagonal for block " << j << ":" << std::endl;
        for(int i = 0; i < m1; i++) {
            double val = h_upper(j,i);
            if(std::abs(val) > 1e-10) {
                std::cout << "    [" << i << "," << i+1 << "] = " 
                            << std::fixed << std::setprecision(6) << val << std::endl;
            }
        }
    }
    
    // Create test vectors
    Kokkos::View<double*> x("x", total_size);
    Kokkos::View<double*> b("b", total_size);
    Kokkos::View<double*> result("result", total_size);
    
    auto h_x = Kokkos::create_mirror_view(x);
    auto h_b = Kokkos::create_mirror_view(b);
    
    // Simple counting vectors
    for(int i = 0; i < total_size; i++) {
        h_x(i) = i + 1.0;
        h_b(i) = total_size - i;
    }
    
    std::cout << "\nTest vector x first 10 values:" << std::endl;
    for(int i = 0; i < std::min(10, total_size); i++) {
        std::cout << "x[" << i << "] = " << std::fixed 
                    << std::setprecision(6) << h_x(i) << std::endl;
    }
    
    std::cout << "\nTest vector b first 10 values:" << std::endl;
    for(int i = 0; i < std::min(10, total_size); i++) {
        std::cout << "b[" << i << "] = " << std::fixed 
                    << std::setprecision(6) << h_b(i) << std::endl;
    }
    
    Kokkos::deep_copy(x, h_x);
    Kokkos::deep_copy(b, h_b);
    
    // Test explicit multiplication
    A1.multiply(x, result);
    
    auto h_result = Kokkos::create_mirror_view(result);
    Kokkos::deep_copy(h_result, result);
    
    std::cout << "\nExplicit multiplication first 10 results:" << std::endl;
    for(int i = 0; i < std::min(30, total_size); i++) {
        std::cout << "result[" << i << "] = " << std::fixed 
                    << std::setprecision(6) << h_result(i) << std::endl;
    }
    
    // Test implicit solve
    double theta = 0.8;
    double delta_t = 1.0/40;
    A1.build_implicit(theta, delta_t);
    A1.solve_implicit(x, b);
    
    auto h_implicit = Kokkos::create_mirror_view(x);
    Kokkos::deep_copy(h_implicit, x);
    
    std::cout << "\nImplicit solve first 10 results:" << std::endl;
    for(int i = 0; i < std::min(30, total_size); i++) {
        std::cout << "implicit_result[" << i << "] = " << std::fixed 
                    << std::setprecision(6) << h_implicit(i) << std::endl;
    }
    
    // Print structure of first block
    /*
    auto h_main = Kokkos::create_mirror_view(A1.get_main_diags());
    auto h_lower = Kokkos::create_mirror_view(A1.get_lower_diags());
    auto h_upper = Kokkos::create_mirror_view(A1.get_upper_diags());
    
    Kokkos::deep_copy(h_main, A1.get_main_diags());
    Kokkos::deep_copy(h_lower, A1.get_lower_diags());
    Kokkos::deep_copy(h_upper, A1.get_upper_diags());
    
    std::cout << "\nSparsity pattern for first block:" << std::endl;
    for(int i = 0; i <= m1; i++) {
        for(int j = 0; j <= m1; j++) {
            bool is_nonzero = false;
            if(j == i) is_nonzero = std::abs(h_main(0,i)) > 1e-10;
            if(j == i-1 && i > 0) is_nonzero = std::abs(h_lower(0,i-1)) > 1e-10;
            if(j == i+1 && i < m1) is_nonzero = std::abs(h_upper(0,i)) > 1e-10;
            std::cout << (is_nonzero ? 'X' : '.');
        }
        std::cout << std::endl;
    }
    */
}


/*

Here come the numerical tests for the A_i matrix classes. This is largely just a copy of the tests 
in the mat_fac file. We basically compare the residual between implicict and explicit steps

*/
void test_A0_multiply() {
    using timer = std::chrono::high_resolution_clock;
    
    // Test dimensions
    const int m1 = 5; 
    const int m2 = 20;
    std::cout << "Testing A0 multiply with dimensions m1=" << m1 << ", m2=" << m2 << "\n";

    // Create grid
    Grid grid = create_test_grid(m1, m2);

    // Initialize A0 matrix
    heston_A0Storage_gpu A0(m1, m2);
    double rho = -0.9;
    double sigma = 0.3;
    A0.build_matrix(grid, rho, sigma);
    
    // Total size
    const int total_size = (m1 + 1) * (m2 + 1);

    // Create test vectors
    Kokkos::View<double*> x("x", total_size);
    Kokkos::View<double*> result("result", total_size);
    
    // Initialize x with values 1,2,3,...
    Kokkos::parallel_for("init_x", total_size, KOKKOS_LAMBDA(const int idx) {
        x(idx) = static_cast<double>(idx + 1);
    });

    
    // Zero result vector
    Kokkos::deep_copy(result, 0.0);

    // Test multiply
    std::cout << "\nTesting multiply...\n";
    auto t_start = timer::now();
    A0.multiply(x, result);
    auto t_end = timer::now();

    std::cout << "Multiply time: "
              << std::chrono::duration<double>(t_end - t_start).count()
              << " seconds" << std::endl;

    // Copy results back and check
    auto h_x = Kokkos::create_mirror_view(x);
    auto h_result = Kokkos::create_mirror_view(result);
    Kokkos::deep_copy(h_x, x);
    Kokkos::deep_copy(h_result, result);

    // Print first few results
    std::cout << "\nFirst 10 results:\n";
    for(int i = 0; i < std::min(60, total_size); i++) {
        std::cout << "result[" << i << "] = " << h_result(i) << " ";
    }
}

void test_A1_multiply_and_implicit() {
    using timer = std::chrono::high_resolution_clock;
    
    // Test dimensions
    const int m1 = 100;
    const int m2 = 75;
    std::cout << "Testing A1 with dimensions m1=" << m1 << ", m2=" << m2 << "\n";

    // Create grid
    Grid grid = create_test_grid(m1, m2);
    
    // Initialize A1
    heston_A1Storage_gpu A1(m1, m2);
    double rho = -0.9;
    double sigma = 0.3;
    double r_d = 0.025;
    double r_f = 0.0;

    double theta = 0.8;
    double delta_t = 1.0/40;

    A1.build_matrix(grid, rho, sigma, r_d, r_f);

    // After building A1
    auto h_main_check = Kokkos::create_mirror_view(A1.get_main_diags());
    auto h_lower_check = Kokkos::create_mirror_view(A1.get_lower_diags());
    auto h_upper_check = Kokkos::create_mirror_view(A1.get_upper_diags());

    Kokkos::deep_copy(h_main_check, A1.get_main_diags());
    Kokkos::deep_copy(h_lower_check, A1.get_lower_diags());
    Kokkos::deep_copy(h_upper_check, A1.get_upper_diags());

    const int total_size = (m1 + 1) * (m2 + 1);
    
    // Create test vectors
    Kokkos::View<double*> x("x", total_size);
    Kokkos::View<double*> b("b", total_size);
    Kokkos::View<double*> result("result", total_size);
    Kokkos::deep_copy(result, 0.0);  // Initialize result to zero
    
    // Initialize with random values
    auto h_b = Kokkos::create_mirror_view(b);
    auto h_x = Kokkos::create_mirror_view(x);
    for (int i = 0; i < total_size; ++i) {
        h_b(i) = total_size - i;//std::rand() / (RAND_MAX + 1.0);
        h_x(i) = 1.0 + i;//std::rand() / (RAND_MAX + 1.0);
    }
    Kokkos::deep_copy(b, h_b);
    Kokkos::deep_copy(x, h_x);

    // Build implicit matrix
    A1.build_implicit(theta, delta_t);

    
    // Test multiply
    auto t_start = timer::now();
    A1.multiply_parallel_s_and_v(x, result);
    auto t_end = timer::now();
    
    std::cout << "Multiply time: "
              << std::chrono::duration<double>(t_end - t_start).count()
              << " seconds" << std::endl;

    // Check multiply result
    /*
    auto h_result = Kokkos::create_mirror_view(result);
    Kokkos::deep_copy(h_result, result);
    std::cout << "After multiply, first few results: ";
    for(int i = 0; i < total_size; i++) std::cout << h_result(i) << " ";
    std::cout << "\n";
    */


    // Test implicit solve
    t_start = timer::now();
    A1.solve_implicit_parallel_v(x, b); //x values changed here
    t_end = timer::now();
    
    std::cout << "Implicit solve time: "
              << std::chrono::duration<double>(t_end - t_start).count()
              << " seconds" << std::endl;

    // Verify solution
    Kokkos::View<double*> verify("verify", total_size);
    A1.multiply(x, verify);
    
    // Compute residual
    auto h_verify = Kokkos::create_mirror_view(verify);
    Kokkos::deep_copy(h_verify, verify);
    
    double residual = 0.0;
    Kokkos::deep_copy(h_x, x);  // Make sure we have latest x values

    std::cout << "\nDebug values for first block:\n";
    for(int i = 0; i < m1+1; i++) {
        double res = h_x(i) - theta * delta_t * h_verify(i) - h_b(i);
        residual += res * res;
    }
    residual = std::sqrt(residual);
    
    std::cout << "Residual norm: " << residual << std::endl;
}



/*

The next two tests are perfomance tests for the two different parallisation tequnices of the two matrices

*/
void test_A0_multiplication_performance() {
    using timer = std::chrono::high_resolution_clock;
    
    // Define test dimensions - try different sizes to see scaling behavior
    std::vector<std::pair<int, int>> test_dimensions = {
        {50, 25},    // Small grid
        {100, 50},   // Medium grid
        {200, 100},  // Large grid
        {300, 150}   // Very large grid
    };
    
    // Number of iterations for each test to get reliable timings
    const int NUM_ITERATIONS = 50;
    
    for (const auto& dims : test_dimensions) {
        const int m1 = dims.first;
        const int m2 = dims.second;
        const int total_size = (m1 + 1) * (m2 + 1);
        
        std::cout << "\n=========================================\n";
        std::cout << "Testing A0 multiply with dimensions m1=" << m1 << ", m2=" << m2 << "\n";
        std::cout << "Total grid size: " << total_size << " points\n";
        
        // Create grid
        Grid grid = create_test_grid(m1, m2);
        
        // Initialize A0 matrix
        heston_A0Storage_gpu A0(m1, m2);
        double rho = -0.9;
        double sigma = 0.3;
        A0.build_matrix(grid, rho, sigma);
        
        // Create test vectors
        Kokkos::View<double*> x("x", total_size);
        Kokkos::View<double*> result_par_v("result_par_v", total_size);
        Kokkos::View<double*> result_par_sv("result_par_sv", total_size);
        
        // Initialize x with values 1,2,3,...
        Kokkos::parallel_for("init_x", total_size, KOKKOS_LAMBDA(const int idx) {
            x(idx) = static_cast<double>(idx + 1);
        });
        Kokkos::fence();
        
        // Variables to track timing statistics
        std::vector<double> times_par_v(NUM_ITERATIONS);
        std::vector<double> times_par_sv(NUM_ITERATIONS);
        
        // Test 2: Variance-parallel multiply
        std::cout << "Testing variance-parallel multiply (multiply_parallel_v)...\n";
        for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
            Kokkos::deep_copy(result_par_v, 0.0);
            
            auto t_start = timer::now();
            A0.multiply(x, result_par_v);
            auto t_end = timer::now();
            
            times_par_v[iter] = std::chrono::duration<double>(t_end - t_start).count();
        }
        
        // Test 3: Stock and Variance parallel multiply
        std::cout << "Testing fully parallel multiply (multiply_parallel_s_and_v)...\n";
        for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
            Kokkos::deep_copy(result_par_sv, 0.0);
            
            auto t_start = timer::now();
            A0.multiply_parallel_s_and_v(x, result_par_sv);
            auto t_end = timer::now();
            
            times_par_sv[iter] = std::chrono::duration<double>(t_end - t_start).count();
        }
        
        // Calculate statistics
        auto calculate_stats = [](const std::vector<double>& times) {
            double sum = std::accumulate(times.begin(), times.end(), 0.0);
            double mean = sum / times.size();
            
            std::vector<double> diff(times.size());
            std::transform(times.begin(), times.end(), diff.begin(), 
                           [mean](double x) { return x - mean; });
            double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
            double stddev = std::sqrt(sq_sum / times.size());
            
            return std::make_pair(mean, stddev);
        };
        
        auto [mean_par_v, stddev_par_v] = calculate_stats(times_par_v);
        auto [mean_par_sv, stddev_par_sv] = calculate_stats(times_par_sv);
        
        // Verify correctness by comparing results
        auto verify_results = [total_size](const Kokkos::View<double*>& result1, 
                                          const Kokkos::View<double*>& result2,
                                          const std::string& name1,
                                          const std::string& name2) {
            auto h_result1 = Kokkos::create_mirror_view(result1);
            auto h_result2 = Kokkos::create_mirror_view(result2);
            Kokkos::deep_copy(h_result1, result1);
            Kokkos::deep_copy(h_result2, result2);
            
            double max_diff = 0.0;
            for (int i = 0; i < total_size; i++) {
                max_diff = std::max(max_diff, std::abs(h_result1(i) - h_result2(i)));
            }
            
            std::cout << "Maximum difference between " << name1 << " and " << name2 
                      << ": " << max_diff << std::endl;
        };
        
    
        verify_results(result_par_v, result_par_sv, "sequential", "fully-parallel");
        
        // Print performance results
        std::cout << "\nPerformance Results (average over " << NUM_ITERATIONS << " runs):\n";
        std::cout << "Variance-Only:   " << std::fixed << std::setprecision(6) << mean_par_v * 1000 
                  << " ms (stddev: " << stddev_par_v * 1000 << " ms)\n";
        std::cout << "Full Parallel:   " << std::fixed << std::setprecision(6) << mean_par_sv * 1000 
                  << " ms (stddev: " << stddev_par_sv * 1000 << " ms)\n";
        
        // Calculate speedups
        double speedup_sv_v = mean_par_v / mean_par_sv;
        
        std::cout << "\nSpeedups:\n";
        std::cout << "Full Parallel vs Variance-Only: " << std::fixed << std::setprecision(2) << speedup_sv_v << "x\n";
    }
}

void test_A1_multiplication_performance() {
    using timer = std::chrono::high_resolution_clock;
    
    // Define test dimensions - try different sizes to see scaling behavior
    std::vector<std::pair<int, int>> test_dimensions = {
        {50, 25},    // Small grid
        {100, 50},   // Medium grid
        {200, 100},  // Large grid
        {300, 150}   // Very large grid
    };
    
    // Number of iterations for each test to get reliable timings
    const int NUM_ITERATIONS = 50;
    
    for (const auto& dims : test_dimensions) {
        const int m1 = dims.first;
        const int m2 = dims.second;
        const int total_size = (m1 + 1) * (m2 + 1);
        
        std::cout << "\n=========================================\n";
        std::cout << "Testing A0 multiply with dimensions m1=" << m1 << ", m2=" << m2 << "\n";
        std::cout << "Total grid size: " << total_size << " points\n";
        
        // Create grid
        Grid grid = create_test_grid(m1, m2);
        
        // Initialize A0 matrix
        heston_A1Storage_gpu A1(m1, m2);
        double rho = -0.9;
        double sigma = 0.3;
        double r_d = 0.025;
        double r_f = 0.0;

        A1.build_matrix(grid, rho, sigma, r_d, r_f);
        
        // Create test vectors
        Kokkos::View<double*> x("x", total_size);
        Kokkos::View<double*> result_par_v("result_par_v", total_size);
        Kokkos::View<double*> result_par_sv("result_par_sv", total_size);
        
        // Initialize x with values 1,2,3,...
        Kokkos::parallel_for("init_x", total_size, KOKKOS_LAMBDA(const int idx) {
            x(idx) = static_cast<double>(idx + 1);
        });
        Kokkos::fence();
        
        // Variables to track timing statistics
        std::vector<double> times_par_v(NUM_ITERATIONS);
        std::vector<double> times_par_sv(NUM_ITERATIONS);
        
        // Test 2: Variance-parallel multiply
        std::cout << "Testing variance-parallel multiply (multiply_parallel_v)...\n";
        for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
            Kokkos::deep_copy(result_par_v, 0.0);
            
            auto t_start = timer::now();
            A1.multiply(x, result_par_v);
            auto t_end = timer::now();
            
            times_par_v[iter] = std::chrono::duration<double>(t_end - t_start).count();
        }
        
        // Test 3: Stock and Variance parallel multiply
        std::cout << "Testing fully parallel multiply (multiply_parallel_s_and_v)...\n";
        for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
            Kokkos::deep_copy(result_par_sv, 0.0);
            
            auto t_start = timer::now();
            A1.multiply_parallel_s_and_v(x, result_par_sv);
            auto t_end = timer::now();
            
            times_par_sv[iter] = std::chrono::duration<double>(t_end - t_start).count();
        }
        
        // Calculate statistics
        auto calculate_stats = [](const std::vector<double>& times) {
            double sum = std::accumulate(times.begin(), times.end(), 0.0);
            double mean = sum / times.size();
            
            std::vector<double> diff(times.size());
            std::transform(times.begin(), times.end(), diff.begin(), 
                           [mean](double x) { return x - mean; });
            double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
            double stddev = std::sqrt(sq_sum / times.size());
            
            return std::make_pair(mean, stddev);
        };
        
        auto [mean_par_v, stddev_par_v] = calculate_stats(times_par_v);
        auto [mean_par_sv, stddev_par_sv] = calculate_stats(times_par_sv);
        
        // Verify correctness by comparing results
        auto verify_results = [total_size](const Kokkos::View<double*>& result1, 
                                          const Kokkos::View<double*>& result2,
                                          const std::string& name1,
                                          const std::string& name2) {
            auto h_result1 = Kokkos::create_mirror_view(result1);
            auto h_result2 = Kokkos::create_mirror_view(result2);
            Kokkos::deep_copy(h_result1, result1);
            Kokkos::deep_copy(h_result2, result2);
            
            double max_diff = 0.0;
            for (int i = 0; i < total_size; i++) {
                max_diff = std::max(max_diff, std::abs(h_result1(i) - h_result2(i)));
            }
            
            std::cout << "Maximum difference between " << name1 << " and " << name2 
                      << ": " << max_diff << std::endl;
        };
        
    
        verify_results(result_par_v, result_par_sv, "sequential", "fully-parallel");
        
        // Print performance results
        std::cout << "\nPerformance Results (average over " << NUM_ITERATIONS << " runs):\n";
        std::cout << "Variance-Only:   " << std::fixed << std::setprecision(6) << mean_par_v * 1000 
                  << " ms (stddev: " << stddev_par_v * 1000 << " ms)\n";
        std::cout << "Full Parallel:   " << std::fixed << std::setprecision(6) << mean_par_sv * 1000 
                  << " ms (stddev: " << stddev_par_sv * 1000 << " ms)\n";
        
        // Calculate speedups
        double speedup_sv_v = mean_par_v / mean_par_sv;
        
        std::cout << "\nSpeedups:\n";
        std::cout << "Full Parallel vs Variance-Only: " << std::fixed << std::setprecision(2) << speedup_sv_v << "x\n";
    }
}

void test_A1_implicit_performance() {
    using timer = std::chrono::high_resolution_clock;
    
    // Define test dimensions - try different sizes to see scaling behavior
    std::vector<std::pair<int, int>> test_dimensions = {
        {50, 25},    // Small grid
        {100, 50},   // Medium grid
        {200, 100},  // Large grid
        {300, 150}   // Very large grid
    };
    
    // Number of iterations for each test to get reliable timings
    const int NUM_ITERATIONS = 50;
    
    for (const auto& dims : test_dimensions) {
        const int m1 = dims.first;
        const int m2 = dims.second;
        const int total_size = (m1 + 1) * (m2 + 1);
        
        std::cout << "\n=========================================\n";
        std::cout << "Testing A0 multiply with dimensions m1=" << m1 << ", m2=" << m2 << "\n";
        std::cout << "Total grid size: " << total_size << " points\n";
        
        // Create grid
        Grid grid = create_test_grid(m1, m2);
        
        // Initialize A0 matrix
        heston_A1Storage_gpu A1(m1, m2);
        double rho = -0.9;
        double sigma = 0.3;
        double r_d = 0.025;
        double r_f = 0.0;

        A1.build_matrix(grid, rho, sigma, r_d, r_f);

        double theta = 0.8;
        double delta_t = 1.0/20;

        A1.build_implicit(theta, delta_t);
        
        // Create test vectors
        Kokkos::View<double*> x("x", total_size);
        Kokkos::View<double*> result_par_v("result_par_v", total_size);
        Kokkos::View<double*> result_par_sv("result_par_sv", total_size);
        
        // Initialize x with values 1,2,3,...
        Kokkos::parallel_for("init_x", total_size, KOKKOS_LAMBDA(const int idx) {
            x(idx) = static_cast<double>(idx + 1);
        });
        Kokkos::fence();
        
        // Variables to track timing statistics
        std::vector<double> times_par_v(NUM_ITERATIONS);
        std::vector<double> times_par_sv(NUM_ITERATIONS);
        
        // Test 1: sequential implcicit
        std::cout << "Testing variance-parallel multiply (multiply_parallel_v)...\n";
        for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
            Kokkos::deep_copy(result_par_v, 0.0);
            
            auto t_start = timer::now();
            A1.solve_implicit(result_par_v, x);
            auto t_end = timer::now();
            
            times_par_v[iter] = std::chrono::duration<double>(t_end - t_start).count();
        }
        
        // Test 3: Stock and Variance parallel multiply
        std::cout << "test parallel v implicit \n";
        for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
            Kokkos::deep_copy(result_par_sv, 0.0);
            
            auto t_start = timer::now();
            A1.solve_implicit_parallel_v(result_par_sv, x);
            auto t_end = timer::now();
            
            times_par_sv[iter] = std::chrono::duration<double>(t_end - t_start).count();
        }
        
        // Calculate statistics
        auto calculate_stats = [](const std::vector<double>& times) {
            double sum = std::accumulate(times.begin(), times.end(), 0.0);
            double mean = sum / times.size();
            
            std::vector<double> diff(times.size());
            std::transform(times.begin(), times.end(), diff.begin(), 
                           [mean](double x) { return x - mean; });
            double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
            double stddev = std::sqrt(sq_sum / times.size());
            
            return std::make_pair(mean, stddev);
        };
        
        auto [mean_par_v, stddev_par_v] = calculate_stats(times_par_v);
        auto [mean_par_sv, stddev_par_sv] = calculate_stats(times_par_sv);
        
        // Verify correctness by comparing results
        auto verify_results = [total_size](const Kokkos::View<double*>& result1, 
                                          const Kokkos::View<double*>& result2,
                                          const std::string& name1,
                                          const std::string& name2) {
            auto h_result1 = Kokkos::create_mirror_view(result1);
            auto h_result2 = Kokkos::create_mirror_view(result2);
            Kokkos::deep_copy(h_result1, result1);
            Kokkos::deep_copy(h_result2, result2);
            
            double max_diff = 0.0;
            for (int i = 0; i < total_size; i++) {
                max_diff = std::max(max_diff, std::abs(h_result1(i) - h_result2(i)));
            }
            
            std::cout << "Maximum difference between " << name1 << " and " << name2 
                      << ": " << max_diff << std::endl;
        };
        
    
        verify_results(result_par_v, result_par_sv, "sequential", "fully-parallel");
        
        // Print performance results
        std::cout << "\nPerformance Results (average over " << NUM_ITERATIONS << " runs):\n";
        std::cout << "Variance-Only:   " << std::fixed << std::setprecision(6) << mean_par_v * 1000 
                  << " ms (stddev: " << stddev_par_v * 1000 << " ms)\n";
        std::cout << "Full Parallel:   " << std::fixed << std::setprecision(6) << mean_par_sv * 1000 
                  << " ms (stddev: " << stddev_par_sv * 1000 << " ms)\n";
        
        // Calculate speedups
        double speedup_sv_v = mean_par_v / mean_par_sv;
        
        std::cout << "\nSpeedups:\n";
        std::cout << "Full Parallel vs Variance-Only: " << std::fixed << std::setprecision(2) << speedup_sv_v << "x\n";
    }
}




void test_hes_mat_fac() {
    // Initialize Kokkos
    Kokkos::initialize();
    {
        try {
            std::cout << "Default execution space: " << Kokkos::DefaultExecutionSpace::name() << std::endl;

            //test_heston_A0();
            //test_heston_A1();
            //test_A1_structure();
            
            //test_A0_multiply();
            //test_A1_multiply_and_implicit();


            //test_A0_multiplication_performance();
            //test_A1_multiplication_performance();
            test_A1_implicit_performance();
        }
        catch (std::exception& e) {
            std::cout << "Error: " << e.what() << std::endl;
        }
    } // All test objects destroyed here
    Kokkos::finalize();
}