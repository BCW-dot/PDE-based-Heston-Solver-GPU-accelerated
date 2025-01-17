#include "hes_a2_shuffled_kernels.hpp"
#include <iomanip>
#include <iostream>
#include "coeff.hpp"

// Vector shuffling implementations
KOKKOS_FUNCTION
void device_shuffle_vector(
    const Kokkos::View<double*>& input,
    Kokkos::View<double*>& output,
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
    Kokkos::View<double*>& output,
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

    // Zero out all arrays first
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, local_m1 + 1),
        [&](const int i) {
            // Main diagonal
            for(int j = 0; j <= local_m2; j++) {
                main_diag(i,j) = 0.0;
                impl_main_diag(i,j) = 1.0;  // Identity matrix baseline
            }
            // Other diagonals
            for(int j = 0; j < local_m2; j++) {
                lower_diag(i,j) = 0.0;
                upper_diag(i,j) = 0.0;
                impl_lower_diag(i,j) = 0.0;
                impl_upper_diag(i,j) = 0.0;
            }
            for(int j = 0; j < local_m2-1; j++) {
                lower2_diag(i,j) = 0.0;
                upper2_diag(i,j) = 0.0;
                impl_lower2_diag(i,j) = 0.0;
                impl_upper2_diag(i,j) = 0.0;
            }
        });
    team.team_barrier();

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
                    if(j >= 2) lower2_diag(i,j-2) += temp * device_alpha_v(j, -2, grid.device_Delta_v);
                    if(j >= 1) lower_diag(i,j-1) += temp * device_alpha_v(j, -1, grid.device_Delta_v);
                    main_diag(i,j) += temp * device_alpha_v(j, 0, grid.device_Delta_v);

                    // Delta terms
                    if(j >= 1) {
                        lower_diag(i,j-1) += temp2 * device_delta_v(j-1, -1, grid.device_Delta_v);
                        main_diag(i,j) += temp2 * device_delta_v(j-1, 0, grid.device_Delta_v);
                        upper_diag(i,j) += temp2 * device_delta_v(j-1, 1, grid.device_Delta_v);
                    }
                }
                else if(j == 0) {
                    // Special v=0 case
                    main_diag(i,0) += temp * device_gamma_v(0, 0, grid.device_Delta_v);
                    upper_diag(i,0) += temp * device_gamma_v(0, 1, grid.device_Delta_v);
                    upper2_diag(i,0) += temp * device_gamma_v(0, 2, grid.device_Delta_v);
                }
                else {
                    // Standard case
                    if(j > 0) {
                        lower_diag(i,j-1) += temp * device_beta_v(j-1, -1, grid.device_Delta_v) +
                                           temp2 * device_delta_v(j-1, -1, grid.device_Delta_v);
                    }
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
    Kokkos::View<double*>& result,
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

void test_a2_build() {
    using timer = std::chrono::high_resolution_clock;
    
    // Test dimensions
    const int m1 = 100;  // Stock price points
    const int m2 = 50;   // Variance points
    
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

void test_a2_shuffled_kernel(){
Kokkos::initialize();
    {
        try{
            test_a2_build();
            //test_a2_structure_function();
        }
        catch (std::exception& e) {
            std::cout << "Error: " << e.what() << std::endl;
        }
    } // All test objects destroyed here
    Kokkos::finalize();
}