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
            if(j == 0) {
                // v=0 case: uses gamma coefficients
                h_main(i, j) += temp * gamma_v(j, 0, grid.Delta_v);
                //std::cout << h_main(i, 0) << std::endl;
                //std::cout << h_main(1,0) << std::endl;
                h_upper(i, j) += temp * gamma_v(j, 1, grid.Delta_v);
                h_upper2(i, j) += temp * gamma_v(j, 2, grid.Delta_v);
            } 
            //check indeces again, check against python.
            else if(grid.Vec_v[j] > 1.0) {
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
            else{
                // Standard case: uses beta coefficients
                h_lower(i, j-1) += temp * beta_v(j-1, -1, grid.Delta_v) + 
                                            temp2 * delta_v(j-1, -1, grid.Delta_v);
                h_main(i, j+1) += temp * beta_v(j-1, 0, grid.Delta_v) + 
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


void test_heston_A2_mat(){
    Kokkos::initialize();
    {
        try{
            //test_heston_A2();
            //test_A2_multiply_and_implicit();
            test_heston_A2_shuffled();
        }
        catch (std::exception& e) {
            std::cout << "Error: " << e.what() << std::endl;
        }
    } // All test objects destroyed here
    Kokkos::finalize();
}