#include "hes_A2_mat.hpp"
#include <iostream>


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
        h_main(i) += temp * gamma_v(0, 0, grid.Delta_v);
        h_upper(i) += temp * gamma_v(0, 1, grid.Delta_v);
        h_upper2(i) = temp * gamma_v(0, 2, grid.Delta_v);
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

            /*
            if(grid.Vec_v[j] > 1.0) {
                // Using l_9a = [-2,-1,0] for upwind scheme
                h_lower2.push_back();
                h_lower[] += temp * alpha_v(j, -2, grid.Delta_v);
                h_main[] += temp * alpha_v(j, -1, grid.Delta_v);
                h_upper[] += temp * alpha_v(j, 0, grid.Delta_v);

                // Add regular central differences
                h_lower[lower_idx] += temp2 * delta_v(j-1, -1, grid.Delta_v);
                h_main[main_idx] += temp2 * delta_v(j-1, 0, grid.Delta_v);
                h_upper[upper_idx] += temp2 * delta_v(j-1, 1, grid.Delta_v);
            } 
            */
            
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
    int m1 = 5;
    int m2 = 5;
    
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
            /*
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
            */
            
            if(j == 0) {
                // v=0 case: uses gamma coefficients
                h_main(i, j) += temp * gamma_v(j, 0, grid.Delta_v);
                //std::cout << h_main(i, 0) << std::endl;
                //std::cout << h_main(1,0) << std::endl;
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
    int m2 = 20;
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
    
    for(int i = 0; i < m1 + 1; i++) {
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
    double delta_t = 1.0/14;
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
        h_b(i) = std::rand() / (RAND_MAX + 1.0);
        h_x(i) = std::rand() / (RAND_MAX + 1.0);
    }
    Kokkos::deep_copy(b, h_b);
    Kokkos::deep_copy(x, h_x);

    // Test implicit solve
    using timer = std::chrono::high_resolution_clock;
    for(int i = 0; i<5; i++){
        auto t_start = timer::now();
        A2.solve_implicit(x, b);
        auto t_end = timer::now();

        std::cout << "Implicit solve time: "
                    << std::chrono::duration<double>(t_end - t_start).count()
                    << " seconds" << std::endl;
    }

    // Verify solution by computing residual
    std::cout << std::endl;
    for(int i = 0; i<5; i++){
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
    
}


/*

making sure shuffle and the original A2 matrix perfom the same computations

*/

// Helper functions for shuffling
inline void shuffle_vector(const Kokkos::View<double*>& input, 
                         Kokkos::View<double*>& output,
                         const int m1, 
                         const int m2) {
    // Shuffles from [v0(s0,s1,...), v1(s0,s1,...), ...] 
    // to [s0(v0,v1,...), s1(v0,v1,...), ...]
    Kokkos::parallel_for("shuffle", m1 + 1, KOKKOS_LAMBDA(const int i) {
        for(int j = 0; j <= m2; j++) {
            // From original idx = j*(m1+1) + i 
            // To shuffled idx = i*(m2+1) + j
            output(i*(m2+1) + j) = input(j*(m1+1) + i);
        }
    });
    Kokkos::fence();
}

inline void unshuffle_vector(const Kokkos::View<double*>& input, 
                              Kokkos::View<double*>& output,
                              const int m1, 
                              const int m2) {
    // Shuffles from [s0(v0,v1,...), s1(v0,v1,...), ...] 
    // back to [v0(s0,s1,...), v1(s0,s1,...), ...]
    Kokkos::parallel_for("shuffle_back", m1 + 1, KOKKOS_LAMBDA(const int i) {
        for(int j = 0; j <= m2; j++) {
            // From shuffled idx = i*(m2+1) + j
            // To original idx = j*(m1+1) + i
            output(j*(m1+1) + i) = input(i*(m2+1) + j);
        }
    });
    Kokkos::fence();
}

void compare_A2_implementations() {
    using timer = std::chrono::high_resolution_clock;

    // Test dimensions
    const int m1 = 300;
    const int m2 = 100;
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
    A2_original.multiply(x, result_orig);
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




void test_heston_A2_mat(){
    Kokkos::initialize();
    {
        try{
            //test_heston_A2();
            //test_A2_multiply_and_implicit();
            //test_heston_A2_shuffled();
            compare_A2_implementations();
        }
        catch (std::exception& e) {
            std::cout << "Error: " << e.what() << std::endl;
        }
    } // All test objects destroyed here
    Kokkos::finalize();
}