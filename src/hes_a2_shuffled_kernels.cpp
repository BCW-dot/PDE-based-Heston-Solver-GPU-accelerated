#include "hes_a2_shuffled_kernels.hpp"
#include <iomanip>
#include <iostream>
#include "coeff.hpp"

// Vector shuffling implementations
KOKKOS_FUNCTION
void device_shuffle_vector(
    const Kokkos::View<double*>& input,
    const Kokkos::View<double*>& output,
    const int m1,
    const int m2,
    const Kokkos::TeamPolicy<>::member_type& team) 
{
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, m1 + 1), 
        [&](const int i) {
            for(int j = 0; j <= m2; j++) {
                output(i*(m2+1) + j) = input(j*(m1+1) + i);
            }
        });
    team.team_barrier();
}

KOKKOS_FUNCTION
void device_unshuffle_vector(
    const Kokkos::View<double*>& input,
    const Kokkos::View<double*>& output,
    const int m1,
    const int m2,
    const Kokkos::TeamPolicy<>::member_type& team) 
{
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, m1 + 1), 
        [&](const int i) {
            for(int j = 0; j <= m2; j++) {
                output(j*(m1+1) + i) = input(i*(m2+1) + j);
            }
        });
    team.team_barrier();
}

// Matrix building implementation
KOKKOS_FUNCTION
void build_a2_diagonals_shuffled(
    const Kokkos::View<double**>& main_diag,
    const Kokkos::View<double**>& lower_diag,
    const Kokkos::View<double**>& lower2_diag,
    const Kokkos::View<double**>& upper_diag,
    const Kokkos::View<double**>& upper2_diag,
    const Kokkos::View<double**>& impl_main_diag,
    const Kokkos::View<double**>& impl_lower_diag,
    const Kokkos::View<double**>& impl_lower2_diag,
    const Kokkos::View<double**>& impl_upper_diag,
    const Kokkos::View<double**>& impl_upper2_diag,
    const Grid& grid,
    const double theta,
    const double dt,
    const double r_d,
    const double kappa,
    const double eta,
    const double sigma,
    const Kokkos::TeamPolicy<>::member_type& team)
{
    const int local_m1 = main_diag.extent(0) - 1;
    const int local_m2 = main_diag.extent(1) - 1;

    // Build explicit matrices
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, local_m1 + 1),
        [&](const int i) {
            for(int j = 0; j < local_m2 - 1; j++) {
                const double temp = kappa * (eta - grid.device_Vec_v[j]);
                const double temp2 = 0.5 * sigma * sigma * grid.device_Vec_v[j];
                
                // Add reaction term
                main_diag(i,j) += -0.5 * r_d;

                if(grid.device_Vec_v[j] > 1.0) {
                    // Upwind scheme coefficients
                    lower2_diag(i,j + 1 -2) += temp * device_alpha_v(j, -2, grid.device_Delta_v);
                    lower_diag(i,j + 1 -1) += temp * device_alpha_v(j, -1, grid.device_Delta_v);
                    main_diag(i,j +1 - 0) += temp * device_alpha_v(j, 0, grid.device_Delta_v);

                    lower_diag(i,j+1-1) += temp2 * device_delta_v(j-1, -1, grid.device_Delta_v);
                    main_diag(i,j+1+0) += temp2 * device_delta_v(j-1, 0, grid.device_Delta_v);
                    upper_diag(i,j+1) += temp2 * device_delta_v(j-1, 1, grid.device_Delta_v);
                }
                if(j == 0) {
                    // Special v=0 case
                    main_diag(i,j) += temp * device_gamma_v(j, 0, grid.device_Delta_v);
                    upper_diag(i,j) += temp * device_gamma_v(j, 1, grid.device_Delta_v);
                    upper2_diag(i,j) += temp * device_gamma_v(j, 2, grid.device_Delta_v);
                }
                else {
                    lower_diag(i,j-1) += temp * device_beta_v(j-1, -1, grid.device_Delta_v) +
                                           temp2 * device_delta_v(j-1, -1, grid.device_Delta_v);
                    main_diag(i,j) += temp * device_beta_v(j-1, 0, grid.device_Delta_v) +
                                    temp2 * device_delta_v(j-1, 0, grid.device_Delta_v);
                    upper_diag(i,j) += temp * device_beta_v(j-1, 1, grid.device_Delta_v) +
                                     temp2 * device_delta_v(j-1, 1, grid.device_Delta_v);
                }
            }
        });
    team.team_barrier();

    // Build implicit matrices
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, local_m1 + 1),
        [&](const int i) {
            for(int j = 0; j <= local_m2; j++) {
                impl_main_diag(i,j) = 1.0 - theta * dt * main_diag(i,j);
            }
            for(int j = 0; j < local_m2; j++) {
                impl_lower_diag(i,j) = -theta * dt * lower_diag(i,j);
                impl_upper_diag(i,j) = -theta * dt * upper_diag(i,j);
            }
            for(int j = 0; j < local_m2-1; j++) {
                impl_lower2_diag(i,j) = -theta * dt * lower2_diag(i,j);
                impl_upper2_diag(i,j) = -theta * dt * upper2_diag(i,j);
            }
        });
    team.team_barrier();
}

// Matrix-vector multiplication implementation
KOKKOS_FUNCTION
void device_multiply_shuffled(
    const Kokkos::View<const double**>& main_diag,
    const Kokkos::View<const double**>& lower_diag,
    const Kokkos::View<const double**>& lower2_diag,
    const Kokkos::View<const double**>& upper_diag,
    const Kokkos::View<const double**>& upper2_diag,
    const Kokkos::View<double*>& x,
    const Kokkos::View<double*>& result,
    const Kokkos::TeamPolicy<>::member_type& team)
{
    const int local_m1 = main_diag.extent(0) - 1;
    const int local_m2 = main_diag.extent(1) - 1;

    // Parallelize over stock price blocks
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, local_m1 + 1),
        [&](const int i) {
            const int block_offset = i * (local_m2 + 1);

            // First row of block
            result(block_offset) = main_diag(i, 0) * x(block_offset);
            if(0 < local_m2) {
                result(block_offset) += upper_diag(i, 0) * x(block_offset + 1);
            }
            if(1 < local_m2) {
                result(block_offset) += upper2_diag(i, 0) * x(block_offset + 2);
            }

            // Second row
            if(0 < local_m2) {
                result(block_offset + 1) = lower_diag(i, 0) * x(block_offset) +
                                         main_diag(i, 1) * x(block_offset + 1);
                if(1 < local_m2) {
                    result(block_offset + 1) += upper_diag(i, 1) * x(block_offset + 2);
                }
                if(2 < local_m2) {
                    result(block_offset + 1) += upper2_diag(i, 1) * x(block_offset + 3);
                }
            }

            // Middle rows
            for(int j = 2; j < local_m2 - 1; j++) {
                result(block_offset + j) = lower2_diag(i, j-2) * x(block_offset + j-2) +
                                         lower_diag(i, j-1) * x(block_offset + j-1) +
                                         main_diag(i, j) * x(block_offset + j) +
                                         upper_diag(i, j) * x(block_offset + j+1);
                if(j < local_m2 - 2) {
                    result(block_offset + j) += upper2_diag(i, j) * x(block_offset + j+2);
                }
            }

            // Second-to-last row
            if(local_m2 > 2) {
                const int j = local_m2 - 1;
                result(block_offset + j) = lower2_diag(i, j-2) * x(block_offset + j-2) +
                                         lower_diag(i, j-1) * x(block_offset + j-1) +
                                         main_diag(i, j) * x(block_offset + j);
                if(j < local_m2) {
                    result(block_offset + j) += upper_diag(i, j) * x(block_offset + j+1);
                }
            }

            // Last row
            if(local_m2 > 1) {
                const int j = local_m2;
                result(block_offset + j) = lower2_diag(i, j-2) * x(block_offset + j-2) +
                                         lower_diag(i, j-1) * x(block_offset + j-1) +
                                         main_diag(i, j) * x(block_offset + j);
            }
        });
    team.team_barrier();
}

// Implicit solve implementation
KOKKOS_FUNCTION
void device_solve_implicit_shuffled(
    const Kokkos::View<const double**>& impl_main_diag,
    const Kokkos::View<const double**>& impl_lower_diag,
    const Kokkos::View<const double**>& impl_lower2_diag,
    const Kokkos::View<const double**>& impl_upper_diag,
    const Kokkos::View<const double**>& impl_upper2_diag,
    const Kokkos::View<double*>& x,
    const Kokkos::View<double**>& c_prime,
    const Kokkos::View<double**>& c2_prime,
    const Kokkos::View<double**>& d_prime,
    const Kokkos::View<double*>& b,
    const Kokkos::TeamPolicy<>::member_type& team)
{
    const int local_m1 = impl_main_diag.extent(0) - 1;
    const int local_m2 = impl_main_diag.extent(1) - 1;

    // Parallelize over stock price blocks
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, local_m1 + 1),
        [&](const int i) {
            const int block_offset = i * (local_m2 + 1);
            const int block_size = local_m2 + 1;

            // Forward sweep
            // First row
            c_prime(i, 0) = impl_upper_diag(i, 0) / impl_main_diag(i, 0);
            c2_prime(i, 0) = impl_upper2_diag(i, 0) / impl_main_diag(i, 0);
            d_prime(i, 0) = b(block_offset) / impl_main_diag(i, 0);

            // Second row
            if(block_size > 1) {
                double m1 = 1.0 / (impl_main_diag(i, 1) - impl_lower_diag(i, 0) * c_prime(i, 0));
                c_prime(i, 1) = (impl_upper_diag(i, 1) - impl_lower_diag(i, 0) * c2_prime(i, 0)) * m1;
                c2_prime(i, 1) = impl_upper2_diag(i, 1) * m1;
                d_prime(i, 1) = (b(block_offset + 1) - impl_lower_diag(i, 0) * d_prime(i, 0)) * m1;
            }

            // Main forward sweep
            for(int j = 2; j < block_size; j++) {
                double den = impl_main_diag(i, j) - 
                            (impl_lower_diag(i, j-1) - impl_lower2_diag(i, j-2) * c_prime(i, j-2)) * c_prime(i, j-1) - 
                            impl_lower2_diag(i, j-2) * c2_prime(i, j-2);
                double m = 1.0 / den;

                // Update c coefficients
                c_prime(i, j) = (impl_upper_diag(i, j) - 
                                (impl_lower_diag(i, j-1) - impl_lower2_diag(i, j-2) * c_prime(i, j-2)) * c2_prime(i, j-1)) * m;
                if(j < block_size - 2) {
                    c2_prime(i, j) = impl_upper2_diag(i, j) * m;
                }

                // Update d
                d_prime(i, j) = (b(block_offset + j) - 
                                (impl_lower_diag(i, j-1) - impl_lower2_diag(i, j-2) * c_prime(i, j-2)) * d_prime(i, j-1) - 
                                impl_lower2_diag(i, j-2) * d_prime(i, j-2)) * m;
            }

            // Back substitution
            x(block_offset + block_size - 1) = d_prime(i, block_size - 1);

            if(block_size > 1) {
                x(block_offset + block_size - 2) = d_prime(i, block_size - 2) - 
                                                  c_prime(i, block_size - 2) * x(block_offset + block_size - 1);
            }

            for(int j = block_size - 3; j >= 0; j--) {
                x(block_offset + j) = d_prime(i, j) - 
                                     c_prime(i, j) * x(block_offset + j + 1) - 
                                     c2_prime(i, j) * x(block_offset + j + 2);
            }
        });
    team.team_barrier();
}


/*

Tests

*/

//Residual Test
void test_a2_build() {
    using timer = std::chrono::high_resolution_clock;
    
    // Test dimensions
    const int m1 = 300;  // Stock price points
    const int m2 = 100;   // Variance points
    
    // Create grid
    Grid grid = create_test_grid(m1, m2);
    
    // Create Views for diagonals
    Kokkos::View<double**> main_diag("main_diag", m1+1, m2+1);
    Kokkos::View<double**> lower_diag("lower_diag", m1+1, m2);
    Kokkos::View<double**> lower2_diag("lower2_diag", m1+1, m2-1);
    Kokkos::View<double**> upper_diag("upper_diag", m1+1, m2);
    Kokkos::View<double**> upper2_diag("upper2_diag", m1+1, m2-1);
    
    // Implicit system diagonals
    Kokkos::View<double**> impl_main_diag("impl_main_diag", m1+1, m2+1);
    Kokkos::View<double**> impl_lower_diag("impl_lower_diag", m1+1, m2);
    Kokkos::View<double**> impl_lower2_diag("impl_lower2_diag", m1+1, m2-1);
    Kokkos::View<double**> impl_upper_diag("impl_upper_diag", m1+1, m2);
    Kokkos::View<double**> impl_upper2_diag("impl_upper2_diag", m1+1, m2-1);
    
    // Test parameters
    const double theta = 0.8;
    const double dt = 1.0/14.0;
    const double r_d = 0.025;
    const double kappa = 1.5;
    const double eta = 0.04;
    const double sigma = 0.3;

    // Set up team policy
    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;
    team_policy policy(1, Kokkos::AUTO);

    // Build matrices
    auto t_start = timer::now();
    
    Kokkos::parallel_for("test_build_a2", policy,
        KOKKOS_LAMBDA(const member_type& team) {
            build_a2_diagonals_shuffled(
                main_diag, lower_diag, lower2_diag, upper_diag, upper2_diag,
                impl_main_diag, impl_lower_diag, impl_lower2_diag, impl_upper_diag, impl_upper2_diag,
                grid, theta, dt, r_d, kappa, eta, sigma,
                team);
    });
    Kokkos::fence();

    auto t_end = timer::now();
    std::cout << "Build matrix time: "
              << std::chrono::duration<double>(t_end - t_start).count()
              << " seconds" << std::endl;

    // Create test vectors
    const int total_size = (m1 + 1) * (m2 + 1);
    Kokkos::View<double*> x("x", total_size);
    Kokkos::View<double*> b("b", total_size);
    Kokkos::View<double*> result("result", total_size);

    // Initialize test vectors with random values
    auto h_b = Kokkos::create_mirror_view(b);
    auto h_x = Kokkos::create_mirror_view(x);
    for (int i = 0; i < total_size; ++i) {
        h_b(i) = (double)std::rand() / RAND_MAX;
        h_x(i) = (double)std::rand() / RAND_MAX;
    }
    Kokkos::deep_copy(b, h_b);
    Kokkos::deep_copy(x, h_x);
    Kokkos::deep_copy(result, 0.0);

    //auto x_tmp = x;  // Create non-const copy
    //auto result_tmp = result;

    // Test multiply
    t_start = timer::now();
    
    Kokkos::parallel_for("test_multiply", policy,
        KOKKOS_LAMBDA(const member_type& team) {
            device_multiply_shuffled(
                main_diag, lower_diag, lower2_diag, upper_diag, upper2_diag,
                x, result,
                team);
    });
    Kokkos::fence();

    t_end = timer::now();
    std::cout << "Multiply matrix time: "
              << std::chrono::duration<double>(t_end - t_start).count()
              << " seconds" << std::endl;

    // Test implicit solve
    // Create temporary storage for implicit solve
    Kokkos::View<double**> c_prime("c_prime", m1+1, m2+1);
    Kokkos::View<double**> c2_prime("c2_prime", m1+1, m2+1);
    Kokkos::View<double**> d_prime("d_prime", m1+1, m2+1);

    t_start = timer::now();
    
    Kokkos::parallel_for("test_implicit_solve", policy,
        KOKKOS_LAMBDA(const member_type& team) {
            device_solve_implicit_shuffled(
                impl_main_diag, impl_lower_diag, impl_lower2_diag, 
                impl_upper_diag, impl_upper2_diag,
                x, c_prime, c2_prime, d_prime, b,
                team);
    });
    Kokkos::fence();

    t_end = timer::now();
    std::cout << "Implicit solve time: "
              << std::chrono::duration<double>(t_end - t_start).count()
              << " seconds" << std::endl;

    // Verify solution by computing residual
    Kokkos::deep_copy(result, 0.0);
    
    Kokkos::parallel_for("verify_implicit", policy,
        KOKKOS_LAMBDA(const member_type& team) {
            device_multiply_shuffled(
                main_diag, lower_diag, lower2_diag, upper_diag, upper2_diag,
                x, result,
                team);
    });
    Kokkos::fence();

    // Compute residual norm
    auto h_result = Kokkos::create_mirror_view(result);
    Kokkos::deep_copy(h_result, result);
    Kokkos::deep_copy(h_x, x);

    double residual = 0.0;
    for(int i = 0; i < total_size; i++) {
        double res = h_x(i) - theta * dt * h_result(i) - h_b(i);
        residual += res * res;
    }
    residual = std::sqrt(residual);
    std::cout << "Implicit solve residual norm: " << residual << std::endl;
}

//same numerical outputs as before test
void test_a2_shuffled_structure_function() {
    // Test with smaller dimensions for easier debugging
    const int m1 = 2;
    const int m2 = 14;
    std::cout << "Testing A2 kernels with dimensions m1=" << m1 << ", m2=" << m2 << "\n\n";

    // Create grid and parameters
    Grid grid = create_test_grid(m1, m2);
    const double rho = -0.9;
    const double sigma = 0.3;
    const double r_d = 0.025;
    const double kappa = 1.5;
    const double eta = 0.04;
    const double theta = 0.8;
    const double delta_t = 1.0/20;
    
    // Create matrix diagonals for device version
    Kokkos::View<double**> main_diag("main_diag", m1+1, m2+1);
    Kokkos::View<double**> lower_diag("lower_diag", m1+1, m2);
    Kokkos::View<double**> lower2_diag("lower2_diag", m1+1, m2-1);
    Kokkos::View<double**> upper_diag("upper_diag", m1+1, m2);
    Kokkos::View<double**> upper2_diag("upper2_diag", m1+1, m2-1);
    
    // Implicit system diagonals
    Kokkos::View<double**> impl_main_diag("impl_main_diag", m1+1, m2+1);
    Kokkos::View<double**> impl_lower_diag("impl_lower_diag", m1+1, m2);
    Kokkos::View<double**> impl_lower2_diag("impl_lower2_diag", m1+1, m2-1);
    Kokkos::View<double**> impl_upper_diag("impl_upper_diag", m1+1, m2);
    Kokkos::View<double**> impl_upper2_diag("impl_upper2_diag", m1+1, m2-1);

    // Set up team policy
    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;
    team_policy policy(1, Kokkos::AUTO);

    // Build matrices
    Kokkos::parallel_for("build_a2", policy,
        KOKKOS_LAMBDA(const member_type& team) {
            build_a2_diagonals_shuffled(
                main_diag, lower_diag, lower2_diag, upper_diag, upper2_diag,
                impl_main_diag, impl_lower_diag, impl_lower2_diag, impl_upper_diag, impl_upper2_diag,
                grid, theta, delta_t, r_d, kappa, eta, sigma,
                team);
    });
    Kokkos::fence();

    // Create test vectors
    const int total_size = (m1 + 1) * (m2 + 1);
    Kokkos::View<double*> x("x", total_size);
    Kokkos::View<double*> x_shuffled("x_shuffled", total_size);
    Kokkos::View<double*> result("result", total_size);
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

    //auto x_shuffled_tmp = x_shuffled;
    // Shuffle input
    Kokkos::parallel_for("shuffle", policy,
        KOKKOS_LAMBDA(const member_type& team) {
            device_shuffle_vector(x, x_shuffled, m1, m2, team);
    });
    Kokkos::fence();
    
    // Print shuffled vector layout
    auto h_x_shuffled = Kokkos::create_mirror_view(x_shuffled);
    Kokkos::deep_copy(h_x_shuffled, x_shuffled);
    std::cout << "\nShuffled vector layout (by stock prices):\n";
    /*
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
    Kokkos::parallel_for("multiply", policy,
        KOKKOS_LAMBDA(const member_type& team) {
            device_multiply_shuffled(
                main_diag, lower_diag, lower2_diag, upper_diag, upper2_diag,
                x_shuffled, result,
                team);
    });
    Kokkos::fence();

    //auto result_unshuf_tmp = result_unshuf;
    // Unshuffle result for comparison
    Kokkos::parallel_for("unshuffle", policy,
        KOKKOS_LAMBDA(const member_type& team) {
            device_unshuffle_vector(result, result_unshuf, m1, m2, team);
    });
    Kokkos::fence();

    auto h_result_unshuf = Kokkos::create_mirror_view(result_unshuf);
    Kokkos::deep_copy(h_result_unshuf, result_unshuf);

    std::cout << "Kernel multiplication results (first block):\n";
    for(int i = 0; i <= total_size; i++) {
        std::cout << h_result_unshuf(i) << " ";
    }
    std::cout << "\n";

    // Test implicit solve
    Kokkos::View<double**> c_prime("c_prime", m1+1, m2+1);
    Kokkos::View<double**> c2_prime("c2_prime", m1+1, m2+1);
    Kokkos::View<double**> d_prime("d_prime", m1+1, m2+1);

    // Create RHS vector b with sequential values
    Kokkos::View<double*> b("b", total_size);
    Kokkos::View<double*> b_shuffled("b_shuffled", total_size);
    auto h_b = Kokkos::create_mirror_view(b);
    for (int i = 0; i < total_size; i++) {
        h_b(i) = i + 1;
    }
    Kokkos::deep_copy(b, h_b);

    // Shuffle b
    Kokkos::parallel_for("shuffle_b", policy,
        KOKKOS_LAMBDA(const member_type& team) {
            device_shuffle_vector(b, b_shuffled, m1, m2, team);
    });
    Kokkos::fence();

    // Reset x vectors
    Kokkos::deep_copy(x, h_x);
    Kokkos::parallel_for("shuffle_x", policy,
        KOKKOS_LAMBDA(const member_type& team) {
            device_shuffle_vector(x, x_shuffled, m1, m2, team);
    });
    Kokkos::fence();

    // Solve
    std::cout << "\nTesting implicit solve:\n";
    Kokkos::parallel_for("implicit_solve", policy,
        KOKKOS_LAMBDA(const member_type& team) {
            device_solve_implicit_shuffled(
                impl_main_diag, impl_lower_diag, impl_lower2_diag,
                impl_upper_diag, impl_upper2_diag,
                x_shuffled, c_prime, c2_prime, d_prime, b_shuffled,
                team);
    });
    Kokkos::fence();

    // Compare solve results
    Kokkos::View<double*> x_unshuf("x_unshuf", total_size);
    //auto x_unshuf_tmp = x_unshuf;
    Kokkos::parallel_for("unshuffle_x", policy,
        KOKKOS_LAMBDA(const member_type& team) {
            device_unshuffle_vector(x_shuffled, x_unshuf, m1, m2, team);
    });
    Kokkos::fence();

    auto h_x_unshuf = Kokkos::create_mirror_view(x_unshuf);
    Kokkos::deep_copy(h_x_unshuf, x_unshuf);

    std::cout << "Kernel solve results (first block):\n";
    for(int i = 0; i <= total_size; i++) {
        std::cout << h_x_unshuf(i) << " ";
    }
    std::cout << "\n";

    // Print per-element differences for first few entries
    /*
    std::cout << "\nFirst few entries:\n";
    std::cout << "Index\tValue\n";
    for(int i = 0; i < std::min(10, total_size); i++) {
        std::cout << i << "\t" << h_x_unshuf(i) << "\n";
    }
    */
}

//prints out the diagonal values
void compare_a2_diagonal_kernels() {
    // Use same dimensions
    const int m1 = 2;
    const int m2 = 14;
    
    Grid grid = create_test_grid(m1, m2);
    const double rho = -0.9;
    const double sigma = 0.3;
    const double r_d = 0.025;
    const double kappa = 1.5;
    const double eta = 0.04;
    const double theta = 0.8;
    const double delta_t = 1.0/20;

    // Create matrix diagonals for device version
    Kokkos::View<double**> main_diag("main_diag", m1+1, m2+1);
    Kokkos::View<double**> lower_diag("lower_diag", m1+1, m2);
    Kokkos::View<double**> lower2_diag("lower2_diag", m1+1, m2-1);
    Kokkos::View<double**> upper_diag("upper_diag", m1+1, m2);
    Kokkos::View<double**> upper2_diag("upper2_diag", m1+1, m2-1);
    
    // Implicit system diagonals
    Kokkos::View<double**> impl_main_diag("impl_main_diag", m1+1, m2+1);
    Kokkos::View<double**> impl_lower_diag("impl_lower_diag", m1+1, m2);
    Kokkos::View<double**> impl_lower2_diag("impl_lower2_diag", m1+1, m2-1);
    Kokkos::View<double**> impl_upper_diag("impl_upper_diag", m1+1, m2);
    Kokkos::View<double**> impl_upper2_diag("impl_upper2_diag", m1+1, m2-1);

    // Build matrices using team policy
    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;
    team_policy policy(1, Kokkos::AUTO);

    Kokkos::parallel_for("build_a2", policy,
        KOKKOS_LAMBDA(const member_type& team) {
            build_a2_diagonals_shuffled(
                main_diag, lower_diag, lower2_diag, upper_diag, upper2_diag,
                impl_main_diag, impl_lower_diag, impl_lower2_diag, impl_upper_diag, impl_upper2_diag,
                grid, theta, delta_t, r_d, kappa, eta, sigma,
                team);
    });
    Kokkos::fence();

    // Create host mirrors
    auto h_main = Kokkos::create_mirror_view(main_diag);
    auto h_lower = Kokkos::create_mirror_view(lower_diag);
    auto h_lower2 = Kokkos::create_mirror_view(lower2_diag);
    auto h_upper = Kokkos::create_mirror_view(upper_diag);
    auto h_upper2 = Kokkos::create_mirror_view(upper2_diag);
    
    auto h_impl_main = Kokkos::create_mirror_view(impl_main_diag);
    auto h_impl_lower = Kokkos::create_mirror_view(impl_lower_diag);
    auto h_impl_lower2 = Kokkos::create_mirror_view(impl_lower2_diag);
    auto h_impl_upper = Kokkos::create_mirror_view(impl_upper_diag);
    auto h_impl_upper2 = Kokkos::create_mirror_view(impl_upper2_diag);
    
    // Copy to host
    Kokkos::deep_copy(h_main, main_diag);
    Kokkos::deep_copy(h_lower, lower_diag);
    Kokkos::deep_copy(h_lower2, lower2_diag);
    Kokkos::deep_copy(h_upper, upper_diag);
    Kokkos::deep_copy(h_upper2, upper2_diag);

    Kokkos::deep_copy(h_impl_main, impl_main_diag);
    Kokkos::deep_copy(h_impl_lower, impl_lower_diag);
    Kokkos::deep_copy(h_impl_lower2, impl_lower2_diag);
    Kokkos::deep_copy(h_impl_upper, impl_upper_diag);
    Kokkos::deep_copy(h_impl_upper2, impl_upper2_diag);
    
    // Print comparison for blocks
    std::cout << "j=0 Block Comparison:\n";
    
    std::cout << "\nShuffled A2 (first stock price block):\n";
    std::cout << "\nlower2:  ";
    for(int j = 0; j < m2-1; j++) {
        std::cout << h_lower2(0,j) << " ";
    }
    std::cout << "\nlower: ";
    for(int j = 0; j < m2; j++) {
        std::cout << h_lower(0,j) << " ";
    }
    std::cout << "\nmain:   ";
    for(int j = 0; j <= m2; j++) {
        std::cout << h_main(0,j) << " ";
    }
    std::cout << "\nupper:  ";
    for(int j = 0; j < m2; j++) {
        std::cout << h_upper(0,j) << " ";
    }
    std::cout << "\nupper2: ";
    for(int j = 0; j < m2-1; j++) {
        std::cout << h_upper2(0,j) << " ";
    }
    
    // Middle block (j=1)
    std::cout << "\nj=1 Block Comparison:\n";
    
    std::cout << "\nShuffled A2 (second stock price block):\n";
    std::cout << "lower2: ";
    for(int j = 0; j < m2-1; j++) {
        std::cout << h_lower2(1,j) << " ";
    }
    std::cout << "\nlower:  ";
    for(int j = 0; j < m2; j++) {
        std::cout << h_lower(1,j) << " ";
    }
    std::cout << "\nmain:   ";
    for(int j = 0; j <= m2; j++) {
        std::cout << h_main(1,j) << " ";
    }
    std::cout << "\nupper:  ";
    for(int j = 0; j < m2; j++) {
        std::cout << h_upper(1,j) << " ";
    }
    std::cout << "\nupper2: ";
    for(int j = 0; j < m2-1; j++) {
        std::cout << h_upper2(1,j) << " ";
    }
    
    // Implicit system first block
    std::cout << "\nImplicit System j=0 Block:\n";
    
    std::cout << "\nShuffled A2 Implicit (first stock price block):\n";
    std::cout << "lower2: ";
    for(int j = 0; j < m2-1; j++) {
        std::cout << h_impl_lower2(0,j) << " ";
    }
    std::cout << "\nlower:  ";
    for(int j = 0; j < m2; j++) {
        std::cout << h_impl_lower(0,j) << " ";
    }
    std::cout << "\nmain:   ";
    for(int j = 0; j <= m2; j++) {
        std::cout << h_impl_main(0,j) << " ";
    }
    std::cout << "\nupper:  ";
    for(int j = 0; j < m2; j++) {
        std::cout << h_impl_upper(0,j) << " ";
    }
    std::cout << "\nupper2: ";
    for(int j = 0; j < m2-1; j++) {
        std::cout << h_impl_upper2(0,j) << " ";
    }

    std::cout << "\nImplicit System j=1 Block:\n";
    
    std::cout << "\nShuffled A2 Implicit (first stock price block):\n";
    std::cout << "lower2: ";
    for(int j = 0; j < m2-1; j++) {
        std::cout << h_impl_lower2(1,j) << " ";
    }
    std::cout << "\nlower:  ";
    for(int j = 0; j < m2; j++) {
        std::cout << h_impl_lower(1,j) << " ";
    }
    std::cout << "\nmain:   ";
    for(int j = 0; j <= m2; j++) {
        std::cout << h_impl_main(0,j) << " ";
    }
    std::cout << "\nupper:  ";
    for(int j = 0; j < m2; j++) {
        std::cout << h_impl_upper(1,j) << " ";
    }
    std::cout << "\nupper2: ";
    for(int j = 0; j < m2-1; j++) {
        std::cout << h_impl_upper2(1,j) << " ";
    }
}

void test_a2_shuffled_kernel(){
    Kokkos::initialize();
        {
            try{
                test_a2_build();
                //test_a2_shuffled_structure_function();
                //compare_a2_diagonal_kernels();
            }
            catch (std::exception& e) {
                std::cout << "Error: " << e.what() << std::endl;
            }
        } // All test objects destroyed here
    Kokkos::finalize();
}