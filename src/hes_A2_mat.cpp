#include "hes_A2_mat.hpp"
#include <iostream>

#include <iomanip>
//for accumulate
#include <numeric>


/*

A2 class constructor and build function

*/
heston_A2Storage_gpu::heston_A2Storage_gpu(int m1_in, int m2_in) : m1(m1_in), m2(m2_in) {
    main_diag = Kokkos::View<double*>("A2_main_diag", (m2-1)*(m1+1));
    lower_diag = Kokkos::View<double*>("A2_lower_diag", (m2-2)*(m1+1));
    upper_diag = Kokkos::View<double*>("A2_upper_diag", (m2-1)*(m1+1));
    upper2_diag = Kokkos::View<double*>("A2_upper2_diag", m1+1);


    implicit_main_diag = Kokkos::View<double*>("A2_impl_main_diag", (m2+1)*(m1+1));
    implicit_lower_diag = Kokkos::View<double*>("A2_impl_lower_diag", (m2-2)*(m1+1));
    implicit_upper_diag = Kokkos::View<double*>("A2_impl_upper_diag", (m2-1)*(m1+1));
    implicit_upper2_diag = Kokkos::View<double*>("A2_impl_upper2_diag", m1+1);

    // Allocate temporary storage once
    const int spacing = m1 + 1;
    const int num_rows = (m2 - 1) * spacing;
    const int total_size = (m2 + 1) * spacing;

    c_star = Kokkos::View<double*>("A2_c_star", num_rows);
    c2_star = Kokkos::View<double*>("A2_c2_star", spacing);
    d_star = Kokkos::View<double*>("A2_d_star", total_size);

}

void heston_A2Storage_gpu::build_matrix(const Grid& grid, double rho, double sigma, double r_d, 
                                      double kappa, double eta) {
    
    auto h_main = Kokkos::create_mirror_view(main_diag);
    auto h_lower = Kokkos::create_mirror_view(lower_diag);
    auto h_upper = Kokkos::create_mirror_view(upper_diag);
    auto h_upper2 = Kokkos::create_mirror_view(upper2_diag);
    
    
    Kokkos::deep_copy(h_main, 0.0);
    Kokkos::deep_copy(h_lower, 0.0);
    Kokkos::deep_copy(h_upper, 0.0);
    Kokkos::deep_copy(h_upper2, 0.0);

    
    int spacing = m1+1;

    
    // Handle j=0 case first
    for(int i = 0; i < m1+1; i++) {
        double temp = kappa * (eta - grid.Vec_v[0]);
        // Use l_9c = [0,1,2]
        h_main(i) += temp * gamma_v(0, 0, grid.Delta_v)  - 0.5 * r_d;;
        h_upper(i) += temp * gamma_v(0, 1, grid.Delta_v);
        h_upper2(i) += temp * gamma_v(0, 2, grid.Delta_v);
    }
    
    // Handle remaining j values
    for(int j = 1; j < m2-1; j++) {
        for(int i = 0; i < m1+1; i++) {
            double temp = kappa * (eta - grid.Vec_v[j]);
            double temp2 = 0.5 * sigma * sigma * grid.Vec_v[j];

            // Main diagonal index - this is correct as before
            int main_idx = i + j*(m1+1);
            
            // Lower diagonal index - needs to start from 0 and use (j-1) blocks
            int lower_idx = i + (j-1)*(m1+1);
            
            // Upper diagonal index - aligns with main diagonal
            int upper_idx = main_idx;
            
            //else {
                // Standard central difference scheme l_10 = [-1, 0, 1]
                h_lower[lower_idx] += (temp * beta_v(j-1, -1, grid.Delta_v) + 
                                    temp2 * delta_v(j-1, -1, grid.Delta_v));
                h_main[main_idx] += (temp * beta_v(j-1, 0, grid.Delta_v) + 
                                    temp2 * delta_v(j-1, 0, grid.Delta_v));
                h_upper[upper_idx] += (temp * beta_v(j-1, 1, grid.Delta_v) + 
                                    temp2 * delta_v(j-1, 1, grid.Delta_v));
            //}
            
            // Add reaction term to main diagonal
            h_main[main_idx] += -0.5 * r_d;
        }
    }

    // Fill with test values on host
    //Simple structure for debugging
    /*
    std::fill(h_main.data(), h_main.data() + h_main.size(), -2);
    std::fill(h_lower.data(), h_lower.data() + h_lower.size(), 2.0);
    std::fill(h_upper.data(), h_upper.data() + h_upper.size(), 1.0);
    std::fill(h_upper2.data(), h_upper2.data() + h_upper2.size(), 1.0);
    */

    
    Kokkos::deep_copy(main_diag, h_main);
    Kokkos::deep_copy(lower_diag, h_lower);
    Kokkos::deep_copy(upper_diag, h_upper);
    Kokkos::deep_copy(upper2_diag, h_upper2);
    
}

/*
void heston_A2Storage_gpu::build_implicit(const double theta, const double delta_t) {
    Kokkos::parallel_for("build_implicit", 1, KOKKOS_LAMBDA(const int) {
        // Initialize implicit_main_diag with identity
        for(int i = 0; i < (m2+1)*(m1+1); i++) {
            implicit_main_diag(i) = 1.0;
        }

        // Subtract theta*delta_t*A2 from main diagonal where A2 is defined
        for(int i = 0; i < (m2-1)*(m1+1); i++) {
            implicit_main_diag(i) -= theta * delta_t * main_diag(i);
        }

        // Build the off-diagonal terms
        for(int i = 0; i < (m2-2)*(m1+1); i++) {
            implicit_lower_diag(i) = -theta * delta_t * lower_diag(i);
        }

        for(int i = 0; i < (m2-1)*(m1+1); i++) {
            implicit_upper_diag(i) = -theta * delta_t * upper_diag(i);
        }

        for(int i = 0; i < m1+1; i++) {
            implicit_upper2_diag(i) = -theta * delta_t * upper2_diag(i);
        }
    });
    Kokkos::fence();
}
*/

void heston_A2Storage_gpu::build_implicit(const double theta, const double delta_t) {
    // Create host mirrors for all Views
    auto h_main_diag = Kokkos::create_mirror_view(main_diag);
    auto h_lower_diag = Kokkos::create_mirror_view(lower_diag);
    auto h_upper_diag = Kokkos::create_mirror_view(upper_diag);
    auto h_upper2_diag = Kokkos::create_mirror_view(upper2_diag);
    
    auto h_implicit_main_diag = Kokkos::create_mirror_view(implicit_main_diag);
    auto h_implicit_lower_diag = Kokkos::create_mirror_view(implicit_lower_diag);
    auto h_implicit_upper_diag = Kokkos::create_mirror_view(implicit_upper_diag);
    auto h_implicit_upper2_diag = Kokkos::create_mirror_view(implicit_upper2_diag);

    // Copy existing data from device to host
    Kokkos::deep_copy(h_main_diag, main_diag);
    Kokkos::deep_copy(h_lower_diag, lower_diag);
    Kokkos::deep_copy(h_upper_diag, upper_diag);
    Kokkos::deep_copy(h_upper2_diag, upper2_diag);

    // Initialize implicit_main_diag with identity on CPU
    for(int i = 0; i < (m2+1)*(m1+1); i++) {
        h_implicit_main_diag(i) = 1.0;
    }

    // Subtract theta*delta_t*A2 from main diagonal where A2 is defined
    for(int i = 0; i < (m2-1)*(m1+1); i++) {
        h_implicit_main_diag(i) -= theta * delta_t * h_main_diag(i);
    }

    // Build the off-diagonal terms
    for(int i = 0; i < (m2-2)*(m1+1); i++) {
        h_implicit_lower_diag(i) = -theta * delta_t * h_lower_diag(i);
    }

    for(int i = 0; i < (m2-1)*(m1+1); i++) {
        h_implicit_upper_diag(i) = -theta * delta_t * h_upper_diag(i);
    }

    for(int i = 0; i < m1+1; i++) {
        h_implicit_upper2_diag(i) = -theta * delta_t * h_upper2_diag(i);
    }

    // Copy results back to device
    Kokkos::deep_copy(implicit_main_diag, h_implicit_main_diag);
    Kokkos::deep_copy(implicit_lower_diag, h_implicit_lower_diag);
    Kokkos::deep_copy(implicit_upper_diag, h_implicit_upper_diag);
    Kokkos::deep_copy(implicit_upper2_diag, h_implicit_upper2_diag);
}

//A2 class test
void test_heston_A2() {
    // Create scope for Kokkos objects
    int m1 = 2;
    int m2 = 4;
    
    Grid grid = create_test_grid(m1, m2);
    
    
    // Create and build A2 matrix
    heston_A2Storage_gpu A2(m1, m2);
    
    double rho = -0.9;
    double sigma = 0.3;
    double r_d = 0.025;
    double kappa = 1.5;
    double eta = 0.04;
    
    
    A2.build_matrix(grid, rho, sigma, r_d, kappa, eta);
    
    std::cout << "her2";
    // Get Views using getters
    auto main = A2.get_main_diag();
    auto lower = A2.get_lower_diag();
    auto upper = A2.get_upper_diag();
    auto upper2 = A2.get_upper2_diag();

    // Create mirror views
    auto h_main = Kokkos::create_mirror_view(main);
    auto h_lower = Kokkos::create_mirror_view(lower);
    auto h_upper = Kokkos::create_mirror_view(upper);
    auto h_upper2 = Kokkos::create_mirror_view(upper2);
    
    // Copy to host
    Kokkos::deep_copy(h_main, main);
    Kokkos::deep_copy(h_lower, lower);
    Kokkos::deep_copy(h_upper, upper);
    Kokkos::deep_copy(h_upper2, upper2);
    
    // Print matrices
    std::cout << "A2 Matrix Structure:\n";
    std::cout << "--------------------\n";
    
    std::cout << "First block (j=0) upper2 diagonal:\n";
    for(int i = 0; i < m1+1; i++) 
        std::cout << h_upper2[i] << " ";
    std::cout << "\n\n";

    for(int j = 0; j < m2-1; j++) {
        std::cout << "Block j=" << j << ":\n";
        if(j > 0) {
            std::cout << "Lower diagonal: ";
            for(int i = 0; i < m1+1; i++) 
                std::cout << h_lower[i + (j-1)*(m1+1)] << " ";
            std::cout << "\n";
        }
        std::cout << "Main diagonal:  ";
        for(int i = 0; i < m1+1; i++) 
            std::cout << h_main[i + j*(m1+1)] << " ";
        std::cout << "\nUpper diagonal: ";
        for(int i = 0; i < m1+1; i++) 
            std::cout << h_upper[i + j*(m1+1)] << " ";
        std::cout << "\n\n";
    }

    // Print dimensions
    std::cout << "\nDimensions:\n";
    std::cout << "m1: " << A2.get_m1() << ", m2: " << A2.get_m2() << "\n";
    std::cout << "Values shape (main/upper): [" << (m2-1)*(m1+1) << "]\n";
    std::cout << "Values shape (lower): [" << (m2-2)*(m1+1) << "]\n";
    std::cout << "Values shape (upper2): [" << m1+1 << "]\n";
    
}

void test_A2_multiply_and_implicit() {
    using timer = std::chrono::high_resolution_clock;

    // Test dimensions
    int m1 = 50;
    int m2 = 50;
    Grid grid = create_test_grid(m1, m2);

    // Create and build A2 matrix
    heston_A2Storage_gpu A2(m1, m2);
    double rho = -0.9;
    double sigma = 0.3;
    double r_d = 0.025;
    double kappa = 1.5;
    double eta = 0.04;

    
    A2.build_matrix(grid, rho, sigma, r_d, kappa, eta);
    
    const int total_size = (m1 + 1) * (m2 + 1);

    // Create test vectors
    Kokkos::View<double*> x("x", total_size);
    Kokkos::View<double*> b("b", total_size);
    Kokkos::View<double*> result("result", total_size);
    Kokkos::deep_copy(result, 15.0);  // Initialize result to zero

    // Initialize with random values
    auto h_b = Kokkos::create_mirror_view(b);
    auto h_x = Kokkos::create_mirror_view(x);
    for (int i = 0; i < total_size; ++i) {
        h_b(i) = std::rand() / (RAND_MAX + 1.0);
        h_x(i) = std::rand() / (RAND_MAX + 1.0);
    }
    Kokkos::deep_copy(b, h_b);
    Kokkos::deep_copy(x, h_x);

    // Build implicit matrix
    double theta = 0.8;
    double delta_t = 1.0/14;
    A2.build_implicit(theta, delta_t);

    
    // Test multiply
    auto t_start = timer::now();
    A2.multiply(x, result);
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
    A2.solve_implicit(x, b); //x values changed here
    t_end = timer::now();

    std::cout << "Implicit solve time: "
                << std::chrono::duration<double>(t_end - t_start).count()
                << " seconds" << std::endl;

    // Verify solution
    Kokkos::View<double*> verify("verify", total_size);
    A2.multiply(x, verify);

    // Compute residual
    auto h_verify = Kokkos::create_mirror_view(verify);
    Kokkos::deep_copy(h_verify, verify);


    double residual = 0.0;
    Kokkos::deep_copy(h_x, x);  // Make sure we have latest x values
    for(int i = 0; i < total_size; i++) {
        double res = h_x(i) - theta * delta_t * h_verify(i) - h_b(i);
        residual += res * res;
    }
    residual = std::sqrt(residual);

    std::cout << "Residual norm: " << residual << std::endl;
}


/*

Shuffled class tests

*/
// Constructor implementation
heston_A2_shuffled::heston_A2_shuffled(int m1_in, int m2_in) : m1(m1_in), m2(m2_in) {
    // Allocate explicit system diagonals
    main_diags = Kokkos::View<double**>("A2_main_diags", m1+1, m2+1);
    lower_diags = Kokkos::View<double**>("A2_lower_diags", m1+1, m2);
    lower2_diags = Kokkos::View<double**>("A2_lower2_diags", m1+1, m2-1);
    upper_diags = Kokkos::View<double**>("A2_upper_diags", m1+1, m2);
    upper2_diags = Kokkos::View<double**>("A2_upper2_diags", m1+1, m2-1);
    
    // Allocate implicit system diagonals
    implicit_main_diags = Kokkos::View<double**>("A2_impl_main_diags", m1+1, m2+1);
    implicit_lower_diags = Kokkos::View<double**>("A2_impl_lower_diags", m1+1, m2);
    implicit_lower2_diags = Kokkos::View<double**>("A2_impl_lower2_diags", m1+1, m2-1);
    implicit_upper_diags = Kokkos::View<double**>("A2_impl_upper_diags", m1+1, m2);
    implicit_upper2_diags = Kokkos::View<double**>("A2_impl_upper2_diags", m1+1, m2-1);
    
    // Allocate temporary storage for implicit solve
    c_prime = Kokkos::View<double**>("A2_c_prime", m1+1, m2+1);
    c2_prime = Kokkos::View<double**>("A2_c2_prime", m1+1, m2+1);
    d_prime = Kokkos::View<double**>("A2_d_prime", m1+1, m2+1);
}

// Build matrix implementation
void heston_A2_shuffled::build_matrix(const Grid& grid, double rho, double sigma, 
                                    double r_d, double kappa, double eta) {
    auto h_main = Kokkos::create_mirror_view(main_diags);
    auto h_lower = Kokkos::create_mirror_view(lower_diags);
    auto h_lower2 = Kokkos::create_mirror_view(lower2_diags);
    auto h_upper = Kokkos::create_mirror_view(upper_diags);
    auto h_upper2 = Kokkos::create_mirror_view(upper2_diags);
    
    // Initialize to zero
    Kokkos::deep_copy(h_main, 0.0);
    Kokkos::deep_copy(h_lower, 0.0);
    Kokkos::deep_copy(h_lower2, 0.0);
    Kokkos::deep_copy(h_upper, 0.0);
    Kokkos::deep_copy(h_upper2, 0.0);
    
    // Build matrices block by block
    // Build matrices block by block
    for(int i = 0; i < m1 + 1; i++) {
        for(int j = 0; j < m2 - 1; j++) {
            double temp = kappa * (eta - grid.Vec_v[j]);
            double temp2 = 0.5 * sigma * sigma * grid.Vec_v[j];
            
            
            // Add reaction term to main diagonal
            h_main(i, j) += -0.5 * r_d;
            
            if(grid.Vec_v[j] > 1.0) {
                // For upwind scheme
                // Alpha terms
                h_lower2(i, j + 1 -2) += temp * alpha_v(j, -2, grid.Delta_v);  // writes 2 positions back
                h_lower(i, j + 1 - 1) += temp * alpha_v(j, -1, grid.Delta_v);   // writes 1 position back
                h_main(i, j + 1 - 0) += temp * alpha_v(j, 0, grid.Delta_v);       // writes at current position

                // Delta terms
                h_lower(i, j + 1 - 1) += temp2 * delta_v(j-1, -1, grid.Delta_v);  // same as alpha
                h_main(i, j + 1 + 0) += temp2 * delta_v(j-1, 0, grid.Delta_v);      // same as alpha
                h_upper(i, j + 1 ) += temp2 * delta_v(j-1, 1, grid.Delta_v);     // writes at current position
            }
            
            if(j == 0) {
                // v=0 case: uses gamma coefficients
                h_main(i, j) += temp * gamma_v(j, 0, grid.Delta_v);
                h_upper(i, j) += temp * gamma_v(j, 1, grid.Delta_v);
                h_upper2(i, j) += temp * gamma_v(j, 2, grid.Delta_v);
            } 
            
            else{
                // Standard case: uses beta coefficients
                h_lower(i, j-1) += temp * beta_v(j-1, -1, grid.Delta_v) + 
                                            temp2 * delta_v(j-1, -1, grid.Delta_v);
                h_main(i, j) += temp * beta_v(j-1, 0, grid.Delta_v) + 
                            temp2 * delta_v(j-1, 0, grid.Delta_v);
                h_upper(i, j) += temp * beta_v(j-1, 1, grid.Delta_v) + 
                                            temp2 * delta_v(j-1, 1, grid.Delta_v);
            }
        }
    }
    
    // Copy to device
    Kokkos::deep_copy(main_diags, h_main);
    Kokkos::deep_copy(lower_diags, h_lower);
    Kokkos::deep_copy(lower2_diags, h_lower2);
    Kokkos::deep_copy(upper_diags, h_upper);
    Kokkos::deep_copy(upper2_diags, h_upper2);
}

// Build implicit system
void heston_A2_shuffled::build_implicit(const double theta, const double delta_t) {
    auto h_impl_main = Kokkos::create_mirror_view(implicit_main_diags);
    auto h_impl_lower = Kokkos::create_mirror_view(implicit_lower_diags);
    auto h_impl_lower2 = Kokkos::create_mirror_view(implicit_lower2_diags);
    auto h_impl_upper = Kokkos::create_mirror_view(implicit_upper_diags);
    auto h_impl_upper2 = Kokkos::create_mirror_view(implicit_upper2_diags);
    
    auto h_main = Kokkos::create_mirror_view(main_diags);
    auto h_lower = Kokkos::create_mirror_view(lower_diags);
    auto h_lower2 = Kokkos::create_mirror_view(lower2_diags);
    auto h_upper = Kokkos::create_mirror_view(upper_diags);
    auto h_upper2 = Kokkos::create_mirror_view(upper2_diags);
    
    // Copy explicit matrices to host
    Kokkos::deep_copy(h_main, main_diags);
    Kokkos::deep_copy(h_lower, lower_diags);
    Kokkos::deep_copy(h_lower2, lower2_diags);
    Kokkos::deep_copy(h_upper, upper_diags);
    Kokkos::deep_copy(h_upper2, upper2_diags);

    // Initialize to zero
    //do we need to init impl diagonals to zero first?
    //dont think so, since we are setting them to zero in for loops
    //since upper2 is zero after the first entry
    
    // Build implicit matrices
    for(int i = 0; i < m1 + 1; i++) {
        // Main diagonal: I - theta*dt*A
        for(int j = 0; j < m2 + 1; j++) {
            h_impl_main(i, j) = 1.0 - theta * delta_t * h_main(i, j);
        }
        
        // Off diagonals: -theta*dt*A
        for(int j = 0; j < m2; j++) {
            h_impl_lower(i, j) = -theta * delta_t * h_lower(i, j);
            h_impl_upper(i, j) = -theta * delta_t * h_upper(i, j);
        }
        
        for(int j = 0; j < m2-1; j++) {
            h_impl_lower2(i, j) = -theta * delta_t * h_lower2(i, j);
            h_impl_upper2(i, j) = -theta * delta_t * h_upper2(i, j);
        }
    }
    
    // Copy back to device
    Kokkos::deep_copy(implicit_main_diags, h_impl_main);
    Kokkos::deep_copy(implicit_lower_diags, h_impl_lower);
    Kokkos::deep_copy(implicit_lower2_diags, h_impl_lower2);
    Kokkos::deep_copy(implicit_upper_diags, h_impl_upper);
    Kokkos::deep_copy(implicit_upper2_diags, h_impl_upper2);
}


void test_heston_A2_shuffled() {
    // Test dimensions
    int m1 = 2;
    int m2 = 14;
    Grid grid = create_test_grid(m1, m2);
    
    // Create and build A2 matrix
    heston_A2_shuffled A2(m1, m2);
    
    double rho = -0.9;
    double sigma = 0.3;
    double r_d = 0.025;
    double kappa = 1.5;
    double eta = 0.04;
    
    A2.build_matrix(grid, rho, sigma, r_d, kappa, eta);
    
    // Get Views using getters
    auto main = A2.get_main_diags();
    auto lower = A2.get_lower_diags();
    auto lower2 = A2.get_lower2_diags();
    auto upper = A2.get_upper_diags();
    auto upper2 = A2.get_upper2_diags();

    // Create mirror views
    auto h_main = Kokkos::create_mirror_view(main);
    auto h_lower = Kokkos::create_mirror_view(lower);
    auto h_lower2 = Kokkos::create_mirror_view(lower2);
    auto h_upper = Kokkos::create_mirror_view(upper);
    auto h_upper2 = Kokkos::create_mirror_view(upper2);
    
    // Copy to host
    Kokkos::deep_copy(h_main, main);
    Kokkos::deep_copy(h_lower, lower);
    Kokkos::deep_copy(h_lower2, lower2);
    Kokkos::deep_copy(h_upper, upper);
    Kokkos::deep_copy(h_upper2, upper2);
    
    // Print matrices
    std::cout << "A2 Shuffled Matrix Structure:\n";
    std::cout << "-------------------------\n";
    
    for(int i = 0; i < 1; i++){//m1 + 1; i++) {
        std::cout << "\nBlock i=" << i << ":\n";
        // Print all diagonals for this block
        std::cout << "Second lower diagonal: ";
        for(int j = 0; j < m2-1; j++) std::cout << h_lower2(i,j) << " ";
        std::cout << "\nFirst lower diagonal:  ";
        for(int j = 0; j < m2; j++) std::cout << h_lower(i,j) << " ";
        std::cout << "\nMain diagonal:         ";
        for(int j = 0; j < m2+1; j++) std::cout << h_main(i,j) << " ";
        std::cout << "\nFirst upper diagonal:  ";
        for(int j = 0; j < m2; j++) std::cout << h_upper(i,j) << " ";
        std::cout << "\nSecond upper diagonal: ";
        for(int j = 0; j < m2-1; j++) std::cout << h_upper2(i,j) << " ";
        std::cout << "\n";
    }
    
    // Print dimensions
    std::cout << "\nDimensions:\n";
    std::cout << "m1: " << A2.get_m1() << ", m2: " << A2.get_m2() << "\n";
    
    // Test implicit system building and solving
    double theta = 0.8;
    double delta_t = 1.0/40;
    A2.build_implicit(theta, delta_t);

    // Create test vectors
    const int total_size = (m1 + 1) * (m2 + 1);
    Kokkos::View<double*> x("x", total_size);
    Kokkos::View<double*> b("b", total_size);
    Kokkos::View<double*> result("result", total_size);
    
    // Initialize with random values
    auto h_b = Kokkos::create_mirror_view(b);
    auto h_x = Kokkos::create_mirror_view(x);
    for (int i = 0; i < total_size; ++i) {
        h_b(i) = i + 1;//std::rand() / (RAND_MAX + 1.0);
        h_x(i) = i + 1;//std::rand() / (RAND_MAX + 1.0);
    }
    Kokkos::deep_copy(b, h_b);
    Kokkos::deep_copy(x, h_x);

    // Test implicit solve
    using timer = std::chrono::high_resolution_clock;
    for(int i = 0; i<1; i++){
        auto t_start = timer::now();
        A2.solve_implicit(x, b);
        auto t_end = timer::now();

        std::cout << "Implicit solve time: "
                    << std::chrono::duration<double>(t_end - t_start).count()
                    << " seconds" << std::endl;
    }

    // Verify solution by computing residual
    std::cout << std::endl;
    for(int i = 0; i<1; i++){
        auto t_start = timer::now();
        A2.multiply(x, result);
        auto t_end = timer::now();

        std::cout << "Explicit solve time: "
                    << std::chrono::duration<double>(t_end - t_start).count()
                    << " seconds" << std::endl;
    }
    
    // Compute residual
    auto h_result = Kokkos::create_mirror_view(result);
    Kokkos::deep_copy(h_result, result);
    Kokkos::deep_copy(h_x, x);  // Get latest x values
    
    double residual = 0.0;
    for(int i = 0; i < total_size; i++) {
        double res = h_x(i) - theta * delta_t * h_result(i) - h_b(i);
        residual += res * res;
    }
    residual = std::sqrt(residual);
    
    std::cout << "Residual norm: " << residual << std::endl;
    
    std::cout << "First few x values: ";
    for(int i = 0; i < std::min(5, total_size); i++) {
        std::cout << h_x(i) << " ";
    }
    std::cout << std::endl;
}

/*

This is my own penta diagonal solver and it works!!!
I developed it on the cpu for debugging purposes and am using it now in the A2_shuffled scheme. This way 
we dont get oszillation in the variance direction

*/
#include <random>
// Function to calculate the residual of the pentadiagonal system
double calculate_residual(
    const std::vector<double>& lower2,
    const std::vector<double>& lower,
    const std::vector<double>& main,
    const std::vector<double>& upper,
    const std::vector<double>& upper2,
    const std::vector<double>& d,
    const std::vector<double>& x) {
    
    const int N = main.size();
    double max_residual = 0.0;
    
    // Check first row (only has main, upper1, upper2)
    double res = main[0] * x[0] + upper[0] * x[1] + upper2[0] * x[2] - d[0];
    max_residual = std::abs(res);
    
    // Check second row (has lower1, main, upper1, upper2)
    res = lower[0] * x[0] + main[1] * x[1] + upper[1] * x[2];
    if (N > 3) res += upper2[1] * x[3];
    res -= d[1];
    max_residual = std::max(max_residual, std::abs(res));
    
    // Middle rows (have all five diagonals)
    for (int i = 2; i < N-2; i++) {
        res = lower2[i-2] * x[i-2] + lower[i-1] * x[i-1] + main[i] * x[i] +
              upper[i] * x[i+1] + upper2[i] * x[i+2] - d[i];
        max_residual = std::max(max_residual, std::abs(res));
    }
    
    // Second to last row (if exists)
    if (N > 3) {
        res = lower2[N-4] * x[N-4] + lower[N-3] * x[N-3] + main[N-2] * x[N-2] +
              upper[N-2] * x[N-1] - d[N-2];
        max_residual = std::max(max_residual, std::abs(res));
    }
    
    // Last row (if exists)
    if (N > 2) {
        res = lower2[N-3] * x[N-3] + lower[N-2] * x[N-2] + main[N-1] * x[N-1] - d[N-1];
        max_residual = std::max(max_residual, std::abs(res));
    }
    
    return max_residual;
}

void penta_diagonal_solver(
    const std::vector<double>& lower2,  // Second lower diagonal (l2)
    const std::vector<double>& lower,   // First lower diagonal (l1)
    const std::vector<double>& main,    // Main diagonal
    const std::vector<double>& upper,   // First upper diagonal (u1)
    const std::vector<double>& upper2,  // Second upper diagonal (u2)
    const std::vector<double>& d,       // Right-hand side vector
    std::vector<double>& x) {          // Solution vector
    
    const int N = main.size();
    
    // Temporary arrays for modified coefficients
    std::vector<double> c2_star(N, 0.0);  // Modified second upper diagonal
    std::vector<double> c1_star(N, 0.0);  // Modified first upper diagonal
    std::vector<double> d_star(N, 0.0);   // Modified right-hand side
    
    // Initialize first row
    c1_star[0] = upper[0] / main[0];
    c2_star[0] = upper2[0] / main[0];
    d_star[0] = d[0] / main[0];
    
    // Initialize second row
    double m1 = 1.0 / (main[1] - lower[0] * c1_star[0]);
    c1_star[1] = (upper[1] - lower[0] * c2_star[0]) * m1;
    c2_star[1] = upper2[1] * m1;
    d_star[1] = (d[1] - lower[0] * d_star[0]) * m1;
    
    // Forward sweep
    for (int i = 2; i < N; i++) {
        // Compute multiplier for current row
        //double den = main[i] - lower[i-1] * c1_star[i-1] - lower2[i-2] * c1_star[i-2];
        double den = main[i] - (lower[i-1]-lower2[i-2]*c1_star[i-2]) * c1_star[i-1] - lower2[i-2] * c2_star[i-2];
        double m = 1.0 / den;
        
        // Update modified upper diagonals
        //c1_star[i] = (upper[i] - lower[i-1] * c2_star[i-1]) * m;
        c1_star[i] = (upper[i] - (lower[i-1]-lower2[i-2]*c1_star[i-2]) * c2_star[i-1]) * m;
        c2_star[i] = upper2[i] * m;
        
        // Update modified right-hand side
        //d_star[i] = (d[i] - lower[i-1] * d_star[i-1] - lower2[i-2] * d_star[i-2]) * m;
        d_star[i] = (d[i] - (lower[i-1]-lower2[i-2]*c1_star[i-2]) * d_star[i-1] - lower2[i-2] * d_star[i-2]) * m;
    }
    
    // Back substitution
    x[N-1] = d_star[N-1];
    x[N-2] = d_star[N-2] - c1_star[N-2] * x[N-1];
    
    for (int i = N-3; i >= 0; i--) {
        x[i] = d_star[i] - c1_star[i] * x[i+1] - c2_star[i] * x[i+2];
    }
}

//Test fot the cpu Penta-Diagonal solver
void my_test(){
    const int N = 100;
    
    // Create a test system with known solution
    std::vector<double> lower2(N-2, -1.0);  // Second lower diagonal
    std::vector<double> lower(N-1, -1.0);   // First lower diagonal
    std::vector<double> main(N, 4.0);       // Main diagonal
    std::vector<double> upper(N-1, -1.0);   // First upper diagonal
    std::vector<double> upper2(N-2, -1.0);  // Second upper diagonal
    
    // Create right-hand side for a known solution (all ones)
    std::vector<double> x_exact(N, 1.0);    // Exact solution (all ones)
    
    // Create random right-hand side vector
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-10.0, 10.0);  // Random values between -10 and 10
    
    std::vector<double> d(N);
    for (int i = 0; i < N; i++) {
        d[i] = dis(gen);
    }
    
    // Solve the system
    std::vector<double> x(N, 0.0);          // Solution vector
    penta_diagonal_solver(lower2, lower, main, upper, upper2, d, x);
    
    // Calculate and print the residual
    double residual = calculate_residual(lower2, lower, main, upper, upper2, d, x);
    
    // Print solution and error metrics
    
    std::cout << "\nError metrics:" << std::endl;
    std::cout << "Residual: " << residual << std::endl;
}



/*

Tests for the shuffle implementation

*/

//this computes the difference in explcicit and implicit between A2 and A2_shuffled
void compare_A2_implementations() {
    using timer = std::chrono::high_resolution_clock;

    // Test dimensions
    const int m1 = 50;
    const int m2 = 25;
    std::cout << "Testing A2 implementations with dimensions m1=" << m1 << ", m2=" << m2 << "\n\n";

    // Create grid and parameters
    Grid grid = create_test_grid(m1, m2);
    const double rho = -0.9;
    const double sigma = 0.3;
    const double r_d = 0.025;
    const double kappa = 1.5;
    const double eta = 0.04;
    
    // Create both A2 implementations
    heston_A2Storage_gpu A2_original(m1, m2);
    heston_A2_shuffled A2_shuffled(m1, m2);
    
    // Build matrices
    A2_original.build_matrix(grid, rho, sigma, r_d, kappa, eta);
    A2_shuffled.build_matrix(grid, rho, sigma, r_d, kappa, eta);

    // Create test vectors
    const int total_size = (m1 + 1) * (m2 + 1);
    Kokkos::View<double*> x("x", total_size);
    Kokkos::View<double*> x_shuffled("x_shuffled", total_size);
    Kokkos::View<double*> result_orig("result_orig", total_size);
    Kokkos::View<double*> result_shuf("result_shuf", total_size);
    Kokkos::View<double*> result_unshuf("result_unshuf", total_size);

    // Initialize x with random values
    auto h_x = Kokkos::create_mirror_view(x);
    for (int i = 0; i < total_size; i++) {
        h_x(i) = static_cast<double>(rand()) / RAND_MAX;
    }
    Kokkos::deep_copy(x, h_x);

    // Test explicit multiplication
    std::cout << "Testing explicit multiplication:\n";
    
    // Shuffle input
    shuffle_vector(x, x_shuffled, m1, m2);
    
    // Time original implementation
    auto t1 = timer::now();
    A2_original.multiply_parallel_s_and_v(x, result_orig);
    auto t2 = timer::now();
    std::cout << "Original multiply time: " 
              << std::chrono::duration<double>(t2-t1).count() << "s\n";

    // Time shuffled implementation
    t1 = timer::now();
    A2_shuffled.multiply(x_shuffled, result_shuf);
    t2 = timer::now();
    std::cout << "Shuffled multiply time: " 
              << std::chrono::duration<double>(t2-t1).count() << "s\n";

    // Unshuffle result for comparison
    unshuffle_vector(result_shuf, result_unshuf, m1, m2);

    // Compare results
    auto h_result_orig = Kokkos::create_mirror_view(result_orig);
    auto h_result_unshuf = Kokkos::create_mirror_view(result_unshuf);
    Kokkos::deep_copy(h_result_orig, result_orig);
    Kokkos::deep_copy(h_result_unshuf, result_unshuf);

    double max_diff = 0.0;
    for(int i = 0; i < total_size; i++) {
        max_diff = std::max(max_diff, 
                           std::abs(h_result_orig(i) - h_result_unshuf(i)));
    }
    std::cout << "Max difference in multiplication results: " << max_diff << "\n\n";

    // Test implicit solve
    std::cout << "Testing implicit solve:\n";
    
    // Build implicit systems
    double theta = 0.8;
    double delta_t = 1.0/20;
    A2_original.build_implicit(theta, delta_t);
    A2_shuffled.build_implicit(theta, delta_t);

    Kokkos::View<double*> b("b", total_size);
    Kokkos::View<double*> b_shuffled("b_shuffled", total_size);
    
    // Initialize b
    auto h_b = Kokkos::create_mirror_view(b);
    for (int i = 0; i < total_size; i++) {
        h_b(i) = static_cast<double>(rand()) / RAND_MAX;
    }
    Kokkos::deep_copy(b, h_b);

    // Shuffle b
    shuffle_vector(b, b_shuffled, m1, m2);

    // Reset x and x_shuffled to random values
    Kokkos::deep_copy(x, h_x);  // Reuse h_x from before
    shuffle_vector(x, x_shuffled, m1, m2);

    // Time original implementation
    t1 = timer::now();
    A2_original.solve_implicit(x, b);
    t2 = timer::now();
    std::cout << "Original solve time: " 
              << std::chrono::duration<double>(t2-t1).count() << "s\n";

    // Time shuffled implementation
    t1 = timer::now();
    A2_shuffled.solve_implicit(x_shuffled, b_shuffled);
    t2 = timer::now();
    std::cout << "Shuffled solve time: " 
              << std::chrono::duration<double>(t2-t1).count() << "s\n";

    // Unshuffle result for comparison
    Kokkos::View<double*> x_unshuf("x_unshuf", total_size);
    unshuffle_vector(x_shuffled, x_unshuf, m1, m2);

    // Compare results
    auto h_x_orig = Kokkos::create_mirror_view(x);
    auto h_x_unshuf = Kokkos::create_mirror_view(x_unshuf);
    Kokkos::deep_copy(h_x_orig, x);
    Kokkos::deep_copy(h_x_unshuf, x_unshuf);

    max_diff = 0.0;
    for(int i = 0; i < total_size; i++) {
        max_diff = std::max(max_diff, 
                           std::abs(h_x_orig(i) - h_x_unshuf(i)));
    }
    std::cout << "Max difference in solve results: " << max_diff << "\n";
}

//this does basically the same, but for smaller Dim size and prints out the values 
void debug_A2_implementations() {
    // Test with smaller dimensions for easier debugging
    const int m1 = 2;
    const int m2 = 14;
    std::cout << "Testing A2 implementations with dimensions m1=" << m1 << ", m2=" << m2 << "\n\n";

    // Create grid and parameters
    Grid grid = create_test_grid(m1, m2);
    const double rho = -0.9;
    const double sigma = 0.3;
    const double r_d = 0.025;
    const double kappa = 1.5;
    const double eta = 0.04;
    
    // Create both A2 implementations
    heston_A2Storage_gpu A2_original(m1, m2);
    heston_A2_shuffled A2_shuffled(m1, m2);
    
    // Build matrices
    A2_original.build_matrix(grid, rho, sigma, r_d, kappa, eta);
    A2_shuffled.build_matrix(grid, rho, sigma, r_d, kappa, eta);

    // Create test vectors
    const int total_size = (m1 + 1) * (m2 + 1);
    Kokkos::View<double*> x("x", total_size);
    Kokkos::View<double*> x_shuffled("x_shuffled", total_size);
    Kokkos::View<double*> result_orig("result_orig", total_size);
    Kokkos::View<double*> result_shuf("result_shuf", total_size);
    Kokkos::View<double*> result_unshuf("result_unshuf", total_size);

    // Initialize x with sequential values
    auto h_x = Kokkos::create_mirror_view(x);
    for (int i = 0; i < total_size; i++) {
        h_x(i) = i + 1;  // Start from 1 for easier tracking
    }
    Kokkos::deep_copy(x, h_x);

    // Print original vector layout
    std::cout << "Original vector layout (by variance levels):\n";
    /*
    for (int j = 0; j <= m2; j++) {
        std::cout << "v" << j << ": ";
        for (int i = 0; i <= m1; i++) {
            std::cout << h_x(j*(m1+1) + i) << " ";
        }
        std::cout << "\n";
    }
    */

    // Shuffle input
    shuffle_vector(x, x_shuffled, m1, m2);
    
    // Print shuffled vector layout
    auto h_x_shuffled = Kokkos::create_mirror_view(x_shuffled);
    Kokkos::deep_copy(h_x_shuffled, x_shuffled);
    /*
    std::cout << "\nShuffled vector layout (by stock prices):\n";
    for (int i = 0; i <= m1; i++) {
        std::cout << "s" << i << ": ";
        for (int j = 0; j <= m2; j++) {
            std::cout << h_x_shuffled(i*(m2+1) + j) << " ";
        }
        std::cout << "\n";
    }
    */

    // Test multiplication
    std::cout << "\nTesting multiplication:\n";
    A2_original.multiply(x, result_orig);
    A2_shuffled.multiply(x_shuffled, result_shuf);
    
    // Unshuffle result for comparison
    unshuffle_vector(result_shuf, result_unshuf, m1, m2);

    // Compare multiplication results
    auto h_result_orig = Kokkos::create_mirror_view(result_orig);
    auto h_result_unshuf = Kokkos::create_mirror_view(result_unshuf);
    Kokkos::deep_copy(h_result_orig, result_orig);
    Kokkos::deep_copy(h_result_unshuf, result_unshuf);

    std::cout << "Original multiplication results (first block):\n";
    for(int i = 0; i <= m1; i++) {
        std::cout << h_result_orig(i) << " ";
    }
    std::cout << "\n\nShuffled multiplication results (first block):\n";
    for(int i = 0; i < total_size; i++) {
        std::cout << h_result_unshuf(i) << " ";
    }
    std::cout << "\n";

    // Now test implicit solve
    double theta = 0.8;
    double delta_t = 1.0/40;
    A2_original.build_implicit(theta, delta_t);
    A2_shuffled.build_implicit(theta, delta_t);

    // Create RHS vector b with sequential values
    Kokkos::View<double*> b("b", total_size);
    Kokkos::View<double*> b_shuffled("b_shuffled", total_size);
    auto h_b = Kokkos::create_mirror_view(b);
    for (int i = 0; i < total_size; i++) {
        h_b(i) = i + 1;
    }
    Kokkos::deep_copy(b, h_b);
    shuffle_vector(b, b_shuffled, m1, m2);

    // Reset x vectors
    Kokkos::deep_copy(x, h_x);
    shuffle_vector(x, x_shuffled, m1, m2);

    // Solve
    std::cout << "\nTesting implicit solve:\n";
    A2_original.solve_implicit(x, b);
    A2_shuffled.solve_implicit(x_shuffled, b_shuffled);

    // Compare solve results
    Kokkos::View<double*> x_unshuf("x_unshuf", total_size);
    unshuffle_vector(x_shuffled, x_unshuf, m1, m2);
    
    auto h_x_orig = Kokkos::create_mirror_view(x);
    auto h_x_unshuf = Kokkos::create_mirror_view(x_unshuf);
    Kokkos::deep_copy(h_x_orig, x);
    Kokkos::deep_copy(h_x_unshuf, x_unshuf);

    std::cout << "Original solve results (first block):\n";
    for(int i = 0; i <= m1; i++) {
        std::cout << h_x_orig(i) << " ";
    }
    std::cout << "\n\nShuffled solve results (first block):\n";
    for(int i = 0; i < total_size; i++) {
        std::cout << h_x_unshuf(i) << " ";
    }
    std::cout << "\n";

    // Print per-element differences for first few entries
    std::cout << "\nDetailed differences in first few entries:\n";
    std::cout << "Index\tOriginal\tShuffled\tDifference\n";
    for(int i = 0; i < std::min(10, total_size); i++) {
        double diff = h_x_orig(i) - h_x_unshuf(i);
        std::cout << i << "\t" 
                 << h_x_orig(i) << "\t\t"
                 << h_x_unshuf(i) << "\t\t"
                 << diff << "\n";
    }
}

//This prints out the numerical values of the matrices
void compare_A2_matrices() {
    // Use same small dimensions
    const int m1 = 2;
    const int m2 = 14;
    
    Grid grid = create_test_grid(m1, m2);
    const double rho = -0.9;
    const double sigma = 0.3;
    const double r_d = 0.025;
    const double kappa = 1.5;
    const double eta = 0.04;
    
    // Create both implementations
    heston_A2Storage_gpu A2_original(m1, m2);
    heston_A2_shuffled A2_shuffled(m1, m2);
    
    // Build matrices
    A2_original.build_matrix(grid, rho, sigma, r_d, kappa, eta);
    A2_shuffled.build_matrix(grid, rho, sigma, r_d, kappa, eta);
    
    // Get diagonal Views
    auto orig_main = A2_original.get_main_diag();
    auto orig_lower = A2_original.get_lower_diag();
    auto orig_upper = A2_original.get_upper_diag();
    auto orig_upper2 = A2_original.get_upper2_diag();
    
    auto shuf_main = A2_shuffled.get_main_diags();
    auto shuf_lower = A2_shuffled.get_lower_diags();
    auto shuf_lower2 = A2_shuffled.get_lower2_diags();
    auto shuf_upper = A2_shuffled.get_upper_diags();
    auto shuf_upper2 = A2_shuffled.get_upper2_diags();
    
    // Create host mirrors
    auto h_orig_main = Kokkos::create_mirror_view(orig_main);
    auto h_orig_lower = Kokkos::create_mirror_view(orig_lower);
    auto h_orig_upper = Kokkos::create_mirror_view(orig_upper);
    auto h_orig_upper2 = Kokkos::create_mirror_view(orig_upper2);
    
    auto h_shuf_main = Kokkos::create_mirror_view(shuf_main);
    auto h_shuf_lower = Kokkos::create_mirror_view(shuf_lower);
    auto h_shuf_lower2 = Kokkos::create_mirror_view(shuf_lower2);
    auto h_shuf_upper = Kokkos::create_mirror_view(shuf_upper);
    auto h_shuf_upper2 = Kokkos::create_mirror_view(shuf_upper2);
    
    // Copy to host
    Kokkos::deep_copy(h_orig_main, orig_main);
    Kokkos::deep_copy(h_orig_lower, orig_lower);
    Kokkos::deep_copy(h_orig_upper, orig_upper);
    Kokkos::deep_copy(h_orig_upper2, orig_upper2);
    
    Kokkos::deep_copy(h_shuf_main, shuf_main);
    Kokkos::deep_copy(h_shuf_lower, shuf_lower);
    Kokkos::deep_copy(h_shuf_lower2, shuf_lower2);
    Kokkos::deep_copy(h_shuf_upper, shuf_upper);
    Kokkos::deep_copy(h_shuf_upper2, shuf_upper2);
    
    // Print comparison for first few blocks
    //std::cout << std::scientific << std::setprecision(6);
    
    // First compare j=0 block (special case)
    std::cout << "\nj=0 Block Comparison:\n";
    /*
    std::cout << "Original A2:\n";
    std::cout << "main:   ";
    for(int i = 0; i <= m1; i++) {
        std::cout << h_orig_main(i) << " ";
    }
    std::cout << "\nupper:  ";
    for(int i = 0; i < m1; i++) {
        std::cout << h_orig_upper(i) << " ";
    }
    std::cout << "\nupper2: ";
    for(int i = 0; i < m1+1; i++) {
        std::cout << h_orig_upper2(i) << " ";
    }
    */
    
    std::cout << "\n\nShuffled A2 (first stock price block):\n";
    std::cout << "\nlower2:  ";
    for(int j = 0; j < m2-1; j++) {
        std::cout << h_shuf_lower2(0,j) << " ";
    }
    std::cout << "\nlower: ";
    for(int j = 0; j < m2; j++) {
        std::cout << h_shuf_lower(0,j) << " ";
    }
    std::cout << "\nmain:   ";
    for(int j = 0; j <= m2; j++) {
        std::cout << h_shuf_main(0,j) << " ";
    }
    std::cout << "\nupper:  ";
    for(int j = 0; j < m2; j++) {
        std::cout << h_shuf_upper(0,j) << " ";
    }
    std::cout << "\nupper2: ";
    for(int j = 0; j < m2-1; j++) {
        std::cout << h_shuf_upper2(0,j) << " ";
    }
    
    // Now look at a middle block (j=1)
    std::cout << "\n\nj=1 Block Comparison:\n";
    /*
    std::cout << "Original A2:\n";
    const int block_size = m1 + 1;
    std::cout << "lower:  ";
    for(int i = 0; i < block_size; i++) {
        std::cout << h_orig_lower(i) << " ";
    }
    std::cout << "\nmain:   ";
    for(int i = 0; i < block_size; i++) {
        std::cout << h_orig_main(block_size + i) << " ";
    }
    std::cout << "\nupper:  ";
    for(int i = 0; i < block_size - 1; i++) {
        std::cout << h_orig_upper(block_size + i) << " ";
    }
    */
    std::cout << "\n\nShuffled A2 (second stock price block):\n";
    std::cout << "lower2: ";
    for(int j = 0; j < m2-1; j++) {
        std::cout << h_shuf_lower2(1,j) << " ";
    }
    std::cout << "\nlower:  ";
    for(int j = 0; j < m2; j++) {
        std::cout << h_shuf_lower(1,j) << " ";
    }
    std::cout << "\nmain:   ";
    for(int j = 0; j <= m2; j++) {
        std::cout << h_shuf_main(1,j) << " ";
    }
    std::cout << "\nupper:  ";
    for(int j = 0; j < m2; j++) {
        std::cout << h_shuf_upper(1,j) << " ";
    }
    std::cout << "\nupper2: ";
    for(int j = 0; j < m2-1; j++) {
        std::cout << h_shuf_upper2(1,j) << " ";
    }
    
    // Build and compare implicit systems
    double theta = 0.8;
    double delta_t = 1.0/20;
    A2_original.build_implicit(theta, delta_t);
    A2_shuffled.build_implicit(theta, delta_t);
    
    // Get implicit diagonal Views
    auto orig_impl_main = A2_original.get_implicit_main_diag();
    auto orig_impl_lower = A2_original.get_implicit_lower_diag();
    auto orig_impl_upper = A2_original.get_implicit_upper_diag();
    auto orig_impl_upper2 = A2_original.get_implicit_upper2_diag();
    
    auto h_orig_impl_main = Kokkos::create_mirror_view(orig_impl_main);
    auto h_orig_impl_lower = Kokkos::create_mirror_view(orig_impl_lower);
    auto h_orig_impl_upper = Kokkos::create_mirror_view(orig_impl_upper);
    auto h_orig_impl_upper2 = Kokkos::create_mirror_view(orig_impl_upper2);
    
    Kokkos::deep_copy(h_orig_impl_main, orig_impl_main);
    Kokkos::deep_copy(h_orig_impl_lower, orig_impl_lower);
    Kokkos::deep_copy(h_orig_impl_upper, orig_impl_upper);
    Kokkos::deep_copy(h_orig_impl_upper2, orig_impl_upper2);
    
    // Print first block of implicit system
    std::cout << "\n\nImplicit System j=0 Block:\n";
    /*
    std::cout << "Original A2:\n";
    std::cout << "main:   ";
    for(int i = 0; i <= m1; i++) {
        std::cout << h_orig_impl_main(i) << " ";
    }
    std::cout << "\nupper:  ";
    for(int i = 0; i < m1; i++) {
        std::cout << h_orig_impl_upper(i) << " ";
    }
    std::cout << "\nupper2: ";
    for(int i = 0; i < m1+1; i++) {
        std::cout << h_orig_impl_upper2(i) << " ";
    }
    */
    auto shuf_impl_main = A2_shuffled.get_implicit_main_diags();
    auto shuf_impl_lower = A2_shuffled.get_implicit_lower_diags();
    auto shuf_impl_lower2 = A2_shuffled.get_implicit_lower2_diags();
    auto shuf_impl_upper = A2_shuffled.get_implicit_upper_diags();
    auto shuf_impl_upper2 = A2_shuffled.get_implicit_upper2_diags();

    auto h_shuf_impl_main = Kokkos::create_mirror_view(shuf_impl_main);
    auto h_shuf_impl_lower = Kokkos::create_mirror_view(shuf_impl_lower);
    auto h_shuf_impl_lower2 = Kokkos::create_mirror_view(shuf_impl_lower2);
    auto h_shuf_impl_upper = Kokkos::create_mirror_view(shuf_impl_upper);
    auto h_shuf_impl_upper2 = Kokkos::create_mirror_view(shuf_impl_upper2);

    Kokkos::deep_copy(h_shuf_impl_main, shuf_impl_main);
    Kokkos::deep_copy(h_shuf_impl_lower, shuf_impl_lower);
    Kokkos::deep_copy(h_shuf_impl_lower2, shuf_impl_lower2);
    Kokkos::deep_copy(h_shuf_impl_upper, shuf_impl_upper);
    Kokkos::deep_copy(h_shuf_impl_upper2, shuf_impl_upper2);

    std::cout << "\n\nShuffled A2 Implicit (first stock price block):\n";
    std::cout << "lower2: ";
    for(int j = 0; j < m2-1; j++) {
        std::cout << h_shuf_impl_lower2(0,j) << " ";
    }
    std::cout << "\nlower:  ";
    for(int j = 0; j < m2; j++) {
        std::cout << h_shuf_impl_lower(0,j) << " ";
    }
    std::cout << "\nmain:   ";
    for(int j = 0; j <= m2; j++) {
        std::cout << h_shuf_impl_main(0,j) << " ";
    }
    std::cout << "\nupper:  ";
    for(int j = 0; j < m2; j++) {
        std::cout << h_shuf_impl_upper(0,j) << " ";
    }
    std::cout << "\nupper2: ";
    for(int j = 0; j < m2-1; j++) {
        std::cout << h_shuf_impl_upper2(0,j) << " ";
    }
    
    std::cout << "\n\nShuffled A2 Implicit (first stock price block):\n";
    std::cout << "lower2: ";
    for(int j = 0; j < m2-1; j++) {
        std::cout << h_shuf_impl_lower2(1,j) << " ";
    }
    std::cout << "\nlower:  ";
    for(int j = 0; j < m2; j++) {
        std::cout << h_shuf_impl_lower(1,j) << " ";
    }
    std::cout << "\nmain:   ";
    for(int j = 0; j <= m2; j++) {
        std::cout << h_shuf_impl_main(1,j) << " ";
    }
    std::cout << "\nupper:  ";
    for(int j = 0; j < m2; j++) {
        std::cout << h_shuf_impl_upper(1,j) << " ";
    }
    std::cout << "\nupper2: ";
    for(int j = 0; j < m2-1; j++) {
        std::cout << h_shuf_impl_upper2(1,j) << " ";
    }
}

void test_shuffle_functions() {
    using timer = std::chrono::high_resolution_clock;
    // Test dimensions
    const int m1 = 300;  // Small dimensions for easy debugging
    const int m2 = 100;
    const int total_size = (m1 + 1) * (m2 + 1);
    
    // Create test vectors
    Kokkos::View<double*> original("original", total_size);
    Kokkos::View<double*> shuffled("shuffled", total_size);
    Kokkos::View<double*> unshuffled("unshuffled", total_size);
    
    // Initialize original with sequential values for easy checking
    auto h_original = Kokkos::create_mirror_view(original);
    for (int i = 0; i < total_size; i++) {
        h_original(i) = i;
    }
    Kokkos::deep_copy(original, h_original);
    
    // Perform shuffle and unshuffle
    for(int i = 0; i<5; i++){
        auto t1 = timer::now();
        shuffle_vector(original, shuffled, m1, m2);
        unshuffle_vector(shuffled, unshuffled, m1, m2);
        auto t2 = timer::now();
        std::cout << "Shuffle and reshuffle time: " 
                << std::chrono::duration<double>(t2-t1).count() << "s\n";
    }

    
    // Check results
    auto h_shuffled = Kokkos::create_mirror_view(shuffled);
    auto h_unshuffled = Kokkos::create_mirror_view(unshuffled);
    Kokkos::deep_copy(h_shuffled, shuffled);
    Kokkos::deep_copy(h_unshuffled, unshuffled);
    
    // Print original layout (v-first)
    /*
    std::cout << "Original layout (by variance):\n";
    for (int j = 0; j <= m2; j++) {
        std::cout << "v" << j << ": ";
        for (int i = 0; i <= m1; i++) {
            std::cout << h_original(j*(m1+1) + i) << " ";
        }
        std::cout << "\n";
    }
    
    // Print shuffled layout (s-first)
    std::cout << "\nShuffled layout (by stock price):\n";
    for (int i = 0; i <= m1; i++) {
        std::cout << "s" << i << ": ";
        for (int j = 0; j <= m2; j++) {
            std::cout << h_shuffled(i*(m2+1) + j) << " ";
        }
        std::cout << "\n";
    }
    */
    // Print unshuffled and check against original
    //std::cout << "\nUnshuffled layout (should match original):\n";
    double max_diff = 0.0;
    for (int j = 0; j <= m2; j++) {
        //std::cout << "v" << j << ": ";
        for (int i = 0; i <= m1; i++) {
            //std::cout << h_unshuffled(j*(m1+1) + i) << " ";
            max_diff = std::max(max_diff, 
                              std::abs(h_original(j*(m1+1) + i) - 
                                     h_unshuffled(j*(m1+1) + i)));
        }
        //std::cout << "\n";
    }
    
    std::cout << "\nMaximum difference between original and unshuffled: " 
              << max_diff << "\n";
}

void test_A2_implementations() {
    std::cout << "Testing A2 implementations with small dimensions for verification\n";

    // Small dimensions for easy verification
    int m1 = 2;  // 3 stock price levels (0 to 2)
    int m2 = 14;  // 5 variance levels (0 to 4)
    const int total_size = (m1 + 1) * (m2 + 1);
    
    // Create grid using existing helper function
    Grid grid = create_test_grid(m1, m2);
    
    // Parameters (same as Python)
    double rho = -0.9;
    double sigma = 0.3;
    double r_d = 0.025;
    double kappa = 1.5;
    double eta = 0.04;
    
    // Create both A2 implementations
    heston_A2Storage_gpu A2(m1, m2);
    heston_A2_shuffled A2_shuffled(m1, m2);
    
    // Build matrices
    A2.build_matrix(grid, rho, sigma, r_d, kappa, eta);
    A2_shuffled.build_matrix(grid, rho, sigma, r_d, kappa, eta);
    
    // Create test vector [1,2,3,...]
    Kokkos::View<double*> test_vector("test_vector", total_size);
    auto h_test_vector = Kokkos::create_mirror_view(test_vector);
    for(int i = 0; i < total_size; i++) {
        h_test_vector(i) = i + 1;
    }
    Kokkos::deep_copy(test_vector, h_test_vector);
    
    // Print original vector
    std::cout << "\nOriginal test vector:\n";
    for(int i = 0; i < total_size; i++) {
        std::cout << h_test_vector(i) << " ";
    }
    std::cout << "\n";
    
    // Test original A2
    Kokkos::View<double*> result1("result1", total_size);
    A2.multiply(test_vector, result1);
    
    // Get result1 back to host
    auto h_result1 = Kokkos::create_mirror_view(result1);
    Kokkos::deep_copy(h_result1, result1);
    std::cout << "\nResult using original A2:\n";
    for(int i = 0; i < total_size; i++) {
        std::cout << h_result1(i) << " ";
    }
    std::cout << "\n";
    
    // Test shuffled A2
    Kokkos::View<double*> shuffled_vector("shuffled_vector", total_size);
    Kokkos::View<double*> result2_shuffled("result2_shuffled", total_size);
    Kokkos::View<double*> result2("result2", total_size);
    
    // Shuffle test vector
    shuffle_vector(test_vector, shuffled_vector, m1, m2);
    
    // Get shuffled vector to host and print
    auto h_shuffled = Kokkos::create_mirror_view(shuffled_vector);
    Kokkos::deep_copy(h_shuffled, shuffled_vector);
    std::cout << "\nShuffled test vector:\n";
    for(int i = 0; i < total_size; i++) {
        std::cout << h_shuffled(i) << " ";
    }
    std::cout << "\n";
    
    // Multiply with shuffled A2
    A2_shuffled.multiply(shuffled_vector, result2_shuffled);
    
    // Print shuffled result
    auto h_result2_shuffled = Kokkos::create_mirror_view(result2_shuffled);
    Kokkos::deep_copy(h_result2_shuffled, result2_shuffled);
    std::cout << "\nResult using shuffled A2 (before unshuffling):\n";
    for(int i = 0; i < total_size; i++) {
        std::cout << h_result2_shuffled(i) << " ";
    }
    std::cout << "\n";
    
    // Unshuffle result
    unshuffle_vector(result2_shuffled, result2, m1, m2);
    
    // Get final result to host
    auto h_result2 = Kokkos::create_mirror_view(result2);
    Kokkos::deep_copy(h_result2, result2);
    std::cout << "\nResult using shuffled A2 (after unshuffling):\n";
    for(int i = 0; i < total_size; i++) {
        std::cout << h_result2(i) << " ";
    }
    std::cout << "\n";
    
    // Compute max difference
    double max_diff = 0.0;
    for(int i = 0; i < total_size; i++) {
        max_diff = std::max(max_diff, std::abs(h_result1(i) - h_result2(i)));
    }
    std::cout << "\nMax difference: " << max_diff << std::endl;
}



/*

Test performance

*/
void test_A2_multiplication_performance() {
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
        
        // Initialize A2 matrix
        heston_A2Storage_gpu A2(m1, m2);
        double rho = -0.9;
        double sigma = 0.3;
        double r_d = 0.025;
        double kappa = 1.5;
        double eta = 0.04;

        A2.build_matrix(grid, rho, sigma, r_d, kappa, eta);
        
        // Create test vectors
        Kokkos::View<double*> x("x", total_size);
        Kokkos::View<double*> result_seq("result_seq", total_size);
        Kokkos::View<double*> result_par_v("result_par_v", total_size);
        Kokkos::View<double*> result_par_sv("result_par_sv", total_size);
        
        // Initialize x with values 1,2,3,...
        Kokkos::parallel_for("init_x", total_size, KOKKOS_LAMBDA(const int idx) {
            x(idx) = static_cast<double>(idx + 1);
        });
        Kokkos::fence();
        
        // Variables to track timing statistics
        std::vector<double> times_seq(NUM_ITERATIONS);
        std::vector<double> times_par_v(NUM_ITERATIONS);
        std::vector<double> times_par_sv(NUM_ITERATIONS);
        
        // Test 1: Sequential multiply
        std::cout << "\nTesting sequential multiply (multiply_seq)...\n";
        for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
            Kokkos::deep_copy(result_seq, 0.0);
            
            auto t_start = timer::now();
            A2.multiply(x, result_seq);
            auto t_end = timer::now();
            
            times_seq[iter] = std::chrono::duration<double>(t_end - t_start).count();
        }
        
        // Test 2: Variance-parallel multiply
        std::cout << "Testing variance-parallel multiply (multiply_parallel_v)...\n";
        for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
            Kokkos::deep_copy(result_par_v, 0.0);
            
            auto t_start = timer::now();
            A2.multiply_parallel_v(x, result_par_v);
            auto t_end = timer::now();
            
            times_par_v[iter] = std::chrono::duration<double>(t_end - t_start).count();
        }
        
        // Test 3: Stock and Variance parallel multiply
        std::cout << "Testing fully parallel multiply (multiply_parallel_s_and_v)...\n";
        for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
            Kokkos::deep_copy(result_par_sv, 0.0);
            
            auto t_start = timer::now();
            A2.multiply_parallel_s_and_v(x, result_par_sv);
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
        
        auto [mean_seq, stddev_seq] = calculate_stats(times_seq);
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
        
        verify_results(result_seq, result_par_v, "sequential", "variance-parallel");
        verify_results(result_seq, result_par_sv, "sequential", "fully-parallel");
        
        // Print performance results
        std::cout << "\nPerformance Results (average over " << NUM_ITERATIONS << " runs):\n";
        std::cout << "Sequential:      " << std::fixed << std::setprecision(6) << mean_seq * 1000 
                  << " ms (stddev: " << stddev_seq * 1000 << " ms)\n";
        std::cout << "Variance-Only:   " << std::fixed << std::setprecision(6) << mean_par_v * 1000 
                  << " ms (stddev: " << stddev_par_v * 1000 << " ms)\n";
        std::cout << "Full Parallel:   " << std::fixed << std::setprecision(6) << mean_par_sv * 1000 
                  << " ms (stddev: " << stddev_par_sv * 1000 << " ms)\n";
        
        // Calculate speedups
        double speedup_v = mean_seq / mean_par_v;
        double speedup_sv = mean_seq / mean_par_sv;
        double speedup_sv_v = mean_par_v / mean_par_sv;
        
        std::cout << "\nSpeedups:\n";
        std::cout << "Variance-Only vs Sequential:    " << std::fixed << std::setprecision(2) << speedup_v << "x\n";
        std::cout << "Full Parallel vs Sequential:    " << std::fixed << std::setprecision(2) << speedup_sv << "x\n";
        std::cout << "Full Parallel vs Variance-Only: " << std::fixed << std::setprecision(2) << speedup_sv_v << "x\n";
    }
}

void test_A2_implicit_performance() {
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
        std::cout << "Testing A2 implicict with dimensions m1=" << m1 << ", m2=" << m2 << "\n";
        std::cout << "Total grid size: " << total_size << " points\n";
        
        // Create grid
        Grid grid = create_test_grid(m1, m2);
        
        // Initialize A2 matrix
        heston_A2Storage_gpu A2(m1, m2);
        double rho = -0.9;
        double sigma = 0.3;
        double r_d = 0.025;
        double kappa = 1.5;
        double eta = 0.04;

        A2.build_matrix(grid, rho, sigma, r_d, kappa, eta);

        double theta = 0.8;
        double delta_t = 1.0/20;
        A2.build_implicit(theta, delta_t);

        /*
        
        shuffled
        
        */
        heston_A2_shuffled A2_shuff(m1, m2);
        A2_shuff.build_matrix(grid, rho, sigma, r_d, kappa, eta);
        A2_shuff.build_implicit(theta, delta_t);
        
        // Create test vectors
        Kokkos::View<double*> x("x", total_size);
        Kokkos::View<double*> result_sequ("result_seq", total_size);
        Kokkos::View<double*> result_par_v("result_par_v", total_size);
        
        // Initialize x with values 1,2,3,...
        Kokkos::parallel_for("init_x", total_size, KOKKOS_LAMBDA(const int idx) {
            x(idx) = static_cast<double>(idx + 1);
        });
        Kokkos::fence();
        
        // Variables to track timing statistics
        std::vector<double> times_seq(NUM_ITERATIONS);
        std::vector<double> times_par_v(NUM_ITERATIONS);
        
        // Test 1: Sequential multiply
        std::cout << "\nTesting A2 sequential implciit (multiply_seq)...\n";
        for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
            Kokkos::deep_copy(result_sequ, 0.0);
            
            auto t_start = timer::now();
            A2.solve_implicit(result_sequ, x);
            auto t_end = timer::now();
            
            times_seq[iter] = std::chrono::duration<double>(t_end - t_start).count();
        }

        std::cout << "\nTesting A2 shuffle s parallel implciit (multiply_seq)...\n";
        for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
            Kokkos::deep_copy(result_par_v, 0.0);
            
            auto t_start = timer::now();
            A2_shuff.solve_implicit(result_par_v, x);
            auto t_end = timer::now();
            
            times_par_v[iter] = std::chrono::duration<double>(t_end - t_start).count();
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
        
        auto [mean_sequ, stddev_sequ] = calculate_stats(times_seq);
        auto [mean_par_v, stddev_par_v] = calculate_stats(times_par_v);

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

        verify_results(result_par_v, result_sequ, "sequential", "suffle");


        // Print performance results
        std::cout << "\nPerformance Results (average over " << NUM_ITERATIONS << " runs):\n";
        std::cout << "Sequrnail:   " << std::fixed << std::setprecision(6) << mean_sequ * 1000 
                  << " ms (stddev: " << stddev_sequ * 1000 << " ms)\n";
        std::cout << "Suffle:   " << std::fixed << std::setprecision(6) << mean_par_v * 1000 
                  << " ms (stddev: " << stddev_par_v * 1000 << " ms)\n";
        
        // Calculate speedups
        double speedup_sequ_v = mean_sequ / mean_par_v;
        
        std::cout << "\nSpeedups:\n";
        std::cout << "Shuffle vs Sequential: " << std::fixed << std::setprecision(2) << speedup_sequ_v << "x\n";
    }
}






void test_heston_A2_mat(){
    Kokkos::initialize();
    {
        try{
            //test_heston_A2();
            //test_A2_multiply_and_implicit();

            //test_heston_A2_shuffled();

            //test_shuffle_functions();

            //compare_A2_implementations();
            //debug_A2_implementations();
            //compare_A2_matrices();
            //test_A2_implementations();


            //test_A2_multiplication_performance();
            test_A2_implicit_performance();

        }
        catch (std::exception& e) {
            std::cout << "Error: " << e.what() << std::endl;
        }
    } // All test objects destroyed here
    Kokkos::finalize();

    //my_test();
}