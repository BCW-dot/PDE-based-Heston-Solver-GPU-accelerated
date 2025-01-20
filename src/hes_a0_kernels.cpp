#include "hes_a0_kernels.hpp"
#include <iomanip>
#include <iostream>
#include "coeff.hpp"

KOKKOS_FUNCTION
void build_a0_values(
    const Kokkos::View<double**>& values,  // [m2-1][(m1-1)*9]
    const Grid& grid,
    const double rho,
    const double sigma,
    const Kokkos::TeamPolicy<>::member_type& team)
{
    const int m1 = grid.device_Vec_s.extent(0) - 1;
    const int m2 = grid.device_Vec_v.extent(0) - 1;

    // First set all values to zero
    //can be ommited
    /*
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, m2-1), 
        [&](const int j) {
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, (m1-1)*9),
                [&](const int idx) {
                    values(j, idx) = 0.0;
                });
    });
    team.team_barrier();
    */

    // Fill in non-zero values
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, m2-1),
        [&](const int j) {
            for(int i = 0; i < m1-1; i++) {
                const double c = rho * sigma * grid.device_Vec_s[i+1] * grid.device_Vec_v[j+1];
                
                // Loop over k and l in [-1,0,1]
                for(int l = -1; l <= 1; l++) {
                    for(int k = -1; k <= 1; k++) {
                        // Convert k,l to linear index in [0,8]
                        const int val_idx = i * 9 + (l + 1) * 3 + (k + 1);
                        
                        // Compute matrix coefficient using beta coefficients
                        const double beta_s_val = device_beta_s(i, k, grid.device_Delta_s);
                        const double beta_v_val = device_beta_v(j, l, grid.device_Delta_v);
                        
                        values(j, val_idx) = c * beta_s_val * beta_v_val;
                    }
                }
            }
        });
    team.team_barrier();
}

KOKKOS_FUNCTION
void device_multiply_a0(
    const Kokkos::View<const double**>& values,
    const Kokkos::View<double*>& x,
    const Kokkos::View<double*>& result,
    const Kokkos::TeamPolicy<>::member_type& team)
{
    const int m1 = (values.extent(1) / 9) + 1;
    const int m2 = values.extent(0) + 1;
    const int total_size = x.extent(0);

    // First, zero out the entire result vector
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, total_size),
        [&](const int i) {
            result(i) = 0.0;
    });
    team.team_barrier();
    
    //Kokkos::deep_copy(result,0.0);

    // Main computation - only fill in non-zero blocks
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, m2-1),
        [&](const int j) {
            for(int i = 0; i < m1-1; i++) {
                const int row_offset = (j + 1) * (m1 + 1) + (i + 1);
                double sum = 0.0;

                // Sum up contributions from 9 entries
                for(int l = -1; l <= 1; l++) {
                    for(int k = -1; k <= 1; k++) {
                        const int val_idx = i * 9 + (l + 1) * 3 + (k + 1);
                        const int col_idx = (i + 1 + k) + (j + 1 + l) * (m1 + 1);
                        
                        if(col_idx >= 0 && col_idx < total_size) {
                            sum += values(j, val_idx) * x(col_idx);
                        }
                    }
                }
                result(row_offset) = sum;
            }
        });
    
    team.team_barrier();
}

void test_a0_build() {
    using timer = std::chrono::high_resolution_clock;

    // Test dimensions
    const int m1 = 5; // Stock points
    const int m2 = 5;  // Variance points

    std::cout << "Matrix dimensions: m1=" << m1 << ", m2=" << m2 << std::endl;

    // Create grid
    Grid grid = create_test_grid(m1, m2);

    // Create View for A0 values
    Kokkos::View<double**> values("A0_values", m2-1, (m1-1)*9);

    // Parameters
    const double rho = -0.9;
    const double sigma = 0.3;

    // Set up team policy
    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;
    team_policy policy(1, Kokkos::AUTO);

    // Build matrix
    auto t_start = timer::now();
    
    Kokkos::parallel_for("build_a0", policy,
        KOKKOS_LAMBDA(const member_type& team) {
            build_a0_values(values, grid, rho, sigma, team);
    });
    Kokkos::fence();

    auto t_end = timer::now();
    std::cout << "Build matrix time: "
              << std::chrono::duration<double>(t_end - t_start).count()
              << " seconds" << std::endl;
    
    /*
    
    Print matrix values

    */
    // Get host copy of values
    auto h_values = Kokkos::create_mirror_view(values);
    Kokkos::deep_copy(h_values, values);

    // Print matrix structure
    std::cout << "\nA0 Matrix Values:\n";
    std::cout << "Dimensions: [" << m2-1 << "] x [" << (m1-1)*9 << "]" << std::endl;
    
    // Print first few blocks
    const int blocks_to_print = std::min(3, m2-1);  // Print first 3 variance levels
    const int entries_per_row = std::min(5, m1-1);  // Print first 5 stock points per row

    for(int j = 0; j < blocks_to_print; j++) {
        std::cout << "\nVariance level j=" << j << ":\n";
        for(int i = 0; i < entries_per_row; i++) {
            std::cout << "Stock point i=" << i << ": ";
            // Print all 9 values for this (i,j) pair
            for(int val = 0; val < 9; val++) {
                std::cout << std::setw(12) << std::fixed << std::setprecision(6) 
                        << h_values(j, i*9 + val) << " ";
            }
            std::cout << "\n";
        }
        std::cout << "...\n";
    }
    
    // Create test vectors
    const int total_size = (m1 + 1) * (m2 + 1);
    Kokkos::View<double*> x("x", total_size);
    Kokkos::View<double*> result("result", total_size);
    

    // Initialize x with incrementing values
    auto h_x = Kokkos::create_mirror_view(x);
    for(int i = 0; i < total_size; i++) {
        h_x(i) = 2;//(i + 1;
    }
    Kokkos::deep_copy(x, h_x);

    //here we specifically set result to zero, for debugging we shoudl init it to all 1's
    Kokkos::deep_copy(result, 1.0);

    // Test multiply
    t_start = timer::now();
    
    Kokkos::parallel_for("test_multiply", policy,
        KOKKOS_LAMBDA(const member_type& team) {
            device_multiply_a0(values, x, result, team);
    });
    Kokkos::fence();

    t_end = timer::now();
    std::cout << "Multiply time: "
              << std::chrono::duration<double>(t_end - t_start).count()
              << " seconds" << std::endl;

    // Get results back to host
    auto h_result = Kokkos::create_mirror_view(result);
    Kokkos::deep_copy(h_result, result);

    // Print first few results
    /*
    std::cout << "\nFirst 20 multiplication results:" << std::endl;
    for(int i = 0; i < std::min(50, total_size); i++) {
        std::cout << "result[" << i << "] = " << std::fixed 
                 << std::setprecision(6) << h_result(i) << " ";
    }
    */
}

void test_a0_kernel(){
    Kokkos::initialize();
        {
            try{
                test_a0_build();
            }
            catch (std::exception& e) {
                std::cout << "Error: " << e.what() << std::endl;
            }
        } // All test objects destroyed here
    Kokkos::finalize();
}