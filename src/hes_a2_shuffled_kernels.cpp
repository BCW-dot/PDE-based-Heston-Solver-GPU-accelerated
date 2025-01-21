#include "hes_a2_shuffled_kernels.hpp"
#include <iomanip>
#include <iostream>
#include "coeff.hpp"
//for accumulate
#include <numeric>


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
template <class MDView, class LDView, class L2DView, class UDView, class U2DView,
          class IMDView, class ILDView, class IL2DView, class IUDView, class IU2DView,
          class GridType>
KOKKOS_FUNCTION
void build_a2_diagonals_shuffled(
    const MDView& main_diag,
    const LDView& lower_diag,
    const L2DView& lower2_diag,
    const UDView& upper_diag,
    const U2DView& upper2_diag,
    const IMDView& impl_main_diag,
    const ILDView& impl_lower_diag,
    const IL2DView& impl_lower2_diag,
    const IUDView& impl_upper_diag,
    const IU2DView& impl_upper2_diag,
    const GridType& grid,
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
template<class View2D_const_main, class View2D_const_lower, 
         class View2D_const_lower2, class View2D_const_upper,
         class View2D_const_upper2, class View1D_x, class View1D_result>
KOKKOS_FUNCTION
void device_multiply_shuffled(
    const View2D_const_main& main_diag,
    const View2D_const_lower& lower_diag,
    const View2D_const_lower2& lower2_diag,
    const View2D_const_upper& upper_diag,
    const View2D_const_upper2& upper2_diag,
    const View1D_x& x,
    const View1D_result& result,
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
template<class View2D_const_main, class View2D_const_lower, 
         class View2D_const_lower2, class View2D_const_upper,
         class View2D_const_upper2, class View1D_x, 
         class View2D_c, class View2D_c2, class View2D_d,
         class View1D_b>
KOKKOS_FUNCTION
void device_solve_implicit_shuffled(
    const View2D_const_main& impl_main_diag,
    const View2D_const_lower& impl_lower_diag,
    const View2D_const_lower2& impl_lower2_diag,
    const View2D_const_upper& impl_upper_diag,
    const View2D_const_upper2& impl_upper2_diag,
    const View1D_x& x,
    const View2D_c& c_prime,
    const View2D_c2& c2_prime,
    const View2D_d& d_prime,
    const View1D_b& b,
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
    const int m1 = 2;  // Stock price points
    const int m2 = 14;   // Variance points
    
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
    const double dt = 1.0/40.0;
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
        h_b(i) = i + 1;;//(double)std::rand() / RAND_MAX;
        h_x(i) = i + 1;;//(double)std::rand() / RAND_MAX;
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


    //
    std::cout << "  x[0..4] = ";
    for(int i = 0; i < std::min(20, total_size); i++) {
        std::cout << h_x(i) << " ";
    }
    std::cout << " " << std::endl;
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
    const double delta_t = 1.0/40;
    
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
    for(int i = 0; i < total_size; i++) {
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
    for(int i = 0; i < total_size; i++) {
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

//MOST IMPORTANT TEST
//Kernel is a bit weirdly strucutred but w/e
//this test computes nInstances in parallel of the a2 class free code above
void test_a2_multiple_instances() {
    using timer = std::chrono::high_resolution_clock;

    // Test parameters
    double K = 100.0;
    double S_0 = K;
    double V_0 = 0.04;

    // Test parameters (same as A1 test)
    const int m1 = 50;
    const int m2 = 25;
    std::cout << "A2 Dimension StockxVariance: " << m1+1 << "x" << m2+1 << std::endl;

    // Test parameters
    const double sigma = 0.3;
    const double r_d = 0.025;
    const double kappa = 1.5;
    const double eta = 0.04;

    const double theta = 0.8;
    const double delta_t = 1.0/40;

    // Initialize vectors with grid views and diagonals
    const int nInstances = 1000;          // Same number as A1 test
    std::cout << "Instances: " << nInstances << std::endl;

    // Initialize vectors with grid views and diagonals
    std::vector<GridViews> hostGrids;
    buildMultipleGridViews(hostGrids, nInstances, m1, m2);  // This creates empty views

    // Now we need to actually fill them with proper grid values
    for(int i = 0; i < nInstances; ++i) {
        double K = 90.0 + 10.0 * i;
        
        // Create host mirrors of the views
        auto h_Vec_s = Kokkos::create_mirror_view(hostGrids[i].device_Vec_s);
        auto h_Vec_v = Kokkos::create_mirror_view(hostGrids[i].device_Vec_v);
        auto h_Delta_s = Kokkos::create_mirror_view(hostGrids[i].device_Delta_s);
        auto h_Delta_v = Kokkos::create_mirror_view(hostGrids[i].device_Delta_v);
        
        // Create temporary Grid object to get the values
        //Grid tempGrid(m1, 8*K, S_0, K, K/5, m2, 5.0, V_0, 5.0/500);

        Grid tempGrid = create_test_grid(m1,m2);
        // Copy values to host mirrors
        for(int j = 0; j <= m1; j++) {
            h_Vec_s(j) = tempGrid.Vec_s[j];
        }
        for(int j = 0; j <= m2; j++) {
            h_Vec_v(j) = tempGrid.Vec_v[j];
        }
        for(int j = 0; j < m1; j++) {
            h_Delta_s(j) = tempGrid.Delta_s[j];
        }
        for(int j = 0; j < m2; j++) {
            h_Delta_v(j) = tempGrid.Delta_v[j];
        }
        
        // Copy to device
        Kokkos::deep_copy(hostGrids[i].device_Vec_s, h_Vec_s);
        Kokkos::deep_copy(hostGrids[i].device_Vec_v, h_Vec_v);
        Kokkos::deep_copy(hostGrids[i].device_Delta_s, h_Delta_s);
        Kokkos::deep_copy(hostGrids[i].device_Delta_v, h_Delta_v);
    }

    // Create device view of GridViews array
    Kokkos::View<GridViews*> deviceGrids("deviceGrids", nInstances);
    auto h_deviceGrids = Kokkos::create_mirror_view(deviceGrids);

    // Copy GridViews to device
    for(int i = 0; i < nInstances; ++i) {
        h_deviceGrids(i) = hostGrids[i];
    }
    Kokkos::deep_copy(deviceGrids, h_deviceGrids);

    // 3D arrays for diagonals [nInstances, m1+1, m2+1]
    Kokkos::View<double***> main_diag("main_diag", nInstances, m1+1, m2+1);
    Kokkos::View<double***> lower_diag("lower_diag", nInstances, m1+1, m2);
    Kokkos::View<double***> lower2_diag("lower2_diag", nInstances, m1+1, m2-1);
    Kokkos::View<double***> upper_diag("upper_diag", nInstances, m1+1, m2);
    Kokkos::View<double***> upper2_diag("upper2_diag", nInstances, m1+1, m2-1);

    Kokkos::View<double***> impl_main_diag("impl_main_diag", nInstances, m1+1, m2+1);
    Kokkos::View<double***> impl_lower_diag("impl_lower_diag", nInstances, m1+1, m2);
    Kokkos::View<double***> impl_lower2_diag("impl_lower2_diag", nInstances, m1+1, m2-1);
    Kokkos::View<double***> impl_upper_diag("impl_upper_diag", nInstances, m1+1, m2);
    Kokkos::View<double***> impl_upper2_diag("impl_upper2_diag", nInstances, m1+1, m2-1);

    // Create test vectors
    const int total_size = (m1 + 1) * (m2 + 1);
    Kokkos::View<double**> x("x", nInstances, total_size);
    Kokkos::View<double**> b("b", nInstances, total_size);
    Kokkos::View<double**> result("result", nInstances, total_size);

    // Initialize x and b with random values
    auto h_x = Kokkos::create_mirror_view(x);
    auto h_b = Kokkos::create_mirror_view(b);
    for(int inst = 0; inst < nInstances; ++inst) {
        for(int idx = 0; idx < total_size; ++idx) {
            h_x(inst, idx) = idx + 1;//(double)std::rand() / RAND_MAX;
            h_b(inst, idx) = idx + 1;//(double)std::rand() / RAND_MAX;
        }
    }
    Kokkos::deep_copy(x, h_x);
    Kokkos::deep_copy(b, h_b);

    // Create temporary storage for implicit solve
    Kokkos::View<double***> c_prime("c_prime", nInstances, m1+1, m2+1);
    Kokkos::View<double***> c2_prime("c2_prime", nInstances, m1+1, m2+1);
    Kokkos::View<double***> d_prime("d_prime", nInstances, m1+1, m2+1);

    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;
    team_policy policy(nInstances, Kokkos::AUTO);

    const int NUM_RUNS = 40;
    std::vector<double> timings(NUM_RUNS);
    
    auto total_t_start = timer::now();
    /*
    
    
    */
    //HERE IS THE KERNEL CALL
    for(int run = 0; run < NUM_RUNS; run++) {
        auto t_start = timer::now();

        Kokkos::parallel_for("build_and_solve_all", policy,
            KOKKOS_LAMBDA(const member_type& team) {
                const int instance = team.league_rank();

                // Subview the diagonals for this instance
                auto mainDiag_i = Kokkos::subview(main_diag, instance, Kokkos::ALL, Kokkos::ALL);
                auto lowerDiag_i = Kokkos::subview(lower_diag, instance, Kokkos::ALL, Kokkos::ALL);
                auto lower2Diag_i = Kokkos::subview(lower2_diag, instance, Kokkos::ALL, Kokkos::ALL);
                auto upperDiag_i = Kokkos::subview(upper_diag, instance, Kokkos::ALL, Kokkos::ALL);
                auto upper2Diag_i = Kokkos::subview(upper2_diag, instance, Kokkos::ALL, Kokkos::ALL);

                auto implMain_i = Kokkos::subview(impl_main_diag, instance, Kokkos::ALL, Kokkos::ALL);
                auto implLower_i = Kokkos::subview(impl_lower_diag, instance, Kokkos::ALL, Kokkos::ALL);
                auto implLower2_i = Kokkos::subview(impl_lower2_diag, instance, Kokkos::ALL, Kokkos::ALL);
                auto implUpper_i = Kokkos::subview(impl_upper_diag, instance, Kokkos::ALL, Kokkos::ALL);
                auto implUpper2_i = Kokkos::subview(impl_upper2_diag, instance, Kokkos::ALL, Kokkos::ALL);

                // Subview solution vectors
                auto x_i = Kokkos::subview(x, instance, Kokkos::ALL);
                auto b_i = Kokkos::subview(b, instance, Kokkos::ALL);
                auto result_i = Kokkos::subview(result, instance, Kokkos::ALL);

                // Get temp storage for this instance
                auto c_prime_i = Kokkos::subview(c_prime, instance, Kokkos::ALL, Kokkos::ALL);
                auto c2_prime_i = Kokkos::subview(c2_prime, instance, Kokkos::ALL, Kokkos::ALL);
                auto d_prime_i = Kokkos::subview(d_prime, instance, Kokkos::ALL, Kokkos::ALL);

                // Get grid for this instance
                GridViews grid_i = deviceGrids(instance);

                // Build matrices for this instance
                build_a2_diagonals_shuffled(
                    mainDiag_i, lowerDiag_i, lower2Diag_i, upperDiag_i, upper2Diag_i,
                    implMain_i, implLower_i, implLower2_i, implUpper_i, implUpper2_i,
                    grid_i, theta, delta_t, r_d, kappa, eta, sigma,
                    team
                );

                // Multiply
                device_multiply_shuffled(
                    mainDiag_i, lowerDiag_i, lower2Diag_i, upperDiag_i, upper2Diag_i,
                    x_i, result_i,
                    team
                );

                // Solve implicit system
                device_solve_implicit_shuffled(
                    implMain_i, implLower_i, implLower2_i, implUpper_i, implUpper2_i,
                    x_i, c_prime_i, c2_prime_i, d_prime_i, b_i,
                    team
                );
            });
        Kokkos::fence();

        auto t_end = timer::now();
        timings[run] = std::chrono::duration<double>(t_end - t_start).count();
    }
    auto total_t_end = timer::now();

    std::cout << "Total loop time: "
              << std::chrono::duration<double>(total_t_end - total_t_start).count()
              << " seconds" << std::endl;



    // Compute Kernel statistics
    double avg_time = std::accumulate(timings.begin(), timings.end(), 0.0) / NUM_RUNS;
    double variance = 0.0;
    for(const auto& t : timings) {
        variance += (t - avg_time) * (t - avg_time);
    }
    variance /= NUM_RUNS;
    double std_dev = std::sqrt(variance);

    std::cout << "Average time: " << avg_time << " seconds\n";
    std::cout << "Standard deviation: " << std_dev << " seconds\n";

    // Verify results for each instance
    Kokkos::parallel_for("final_check_multiply", policy,
        KOKKOS_LAMBDA(const member_type& team) {
            const int instance = team.league_rank();

            auto mainDiag_i = Kokkos::subview(main_diag, instance, Kokkos::ALL, Kokkos::ALL);
            auto lowerDiag_i = Kokkos::subview(lower_diag, instance, Kokkos::ALL, Kokkos::ALL);
            auto lower2Diag_i = Kokkos::subview(lower2_diag, instance, Kokkos::ALL, Kokkos::ALL);
            auto upperDiag_i = Kokkos::subview(upper_diag, instance, Kokkos::ALL, Kokkos::ALL);
            auto upper2Diag_i = Kokkos::subview(upper2_diag, instance, Kokkos::ALL, Kokkos::ALL);

            auto x_i = Kokkos::subview(x, instance, Kokkos::ALL);
            auto result_i = Kokkos::subview(result, instance, Kokkos::ALL);

            device_multiply_shuffled(
                mainDiag_i, lowerDiag_i, lower2Diag_i, upperDiag_i, upper2Diag_i,
                x_i, result_i,
                team
            );
        });
    Kokkos::fence();

    // Compute residuals
    auto h_result = Kokkos::create_mirror_view(result);
    Kokkos::deep_copy(h_result, result);
    Kokkos::deep_copy(h_x, x);
    Kokkos::deep_copy(h_b, b);

    for(int inst = 0; inst < min(1,nInstances); ++inst) {
        double residual_sum = 0.0;

        for(int idx = 0; idx < total_size; idx++) {
            double lhs = h_x(inst, idx);
            double rhs = theta * delta_t * h_result(inst, idx) + h_b(inst, idx);
            double diff = lhs - rhs;
            residual_sum += diff * diff;
        }

        double residual_norm = std::sqrt(residual_sum);
        std::cout << "Instance " << inst << " => residual norm = " << residual_norm << std::endl;

        /*
        std::cout << "  x[0..4] = ";
        for(int i = 0; i < std::min(20,total_size); i++){
            std::cout << h_x(inst, i) << " ";
        }
        std::cout << " " << std::endl;
        */

        std::cout << "------------------------------------\n";
    }
}

//this is a debug function for the instances test
void test_a2_shuffled_single_instance_debug() {
    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;

    // Same small dimensions as your debug test
    const int m1 = 2;
    const int m2 = 14;
    std::cout << "Testing A2 single-instance with dimensions m1=" << m1
              << ", m2=" << m2 << " (multi-instance style)\n\n";

    // PDE / PDE-discretization parameters (same as in your test)
    const double sigma = 0.3;
    const double r_d = 0.025;
    const double kappa = 1.5;
    const double eta = 0.04;
    const double theta = 0.8;
    const double delta_t = 1.0/40;

    // Number of "instances" is just 1 here
    const int nInstances = 1;

    // Create a small test grid
    Grid grid = create_test_grid(m1, m2);

    /*
    std::vector<GridViews> hostGrids;
    buildMultipleGridViews(hostGrids, nInstances, m1, m2);  // This creates empty views

    // Now we need to actually fill them with proper grid values
    for(int i = 0; i < nInstances; ++i) {
        //double K = 90.0 + 10.0 * i;
        
        // Create host mirrors of the views
        auto h_Vec_s = Kokkos::create_mirror_view(hostGrids[i].device_Vec_s);
        auto h_Vec_v = Kokkos::create_mirror_view(hostGrids[i].device_Vec_v);
        auto h_Delta_s = Kokkos::create_mirror_view(hostGrids[i].device_Delta_s);
        auto h_Delta_v = Kokkos::create_mirror_view(hostGrids[i].device_Delta_v);
        
        // Create temporary Grid object to get the values
        //Grid tempGrid(m1, 8*K, S_0, K, K/5, m2, 5.0, V_0, 5.0/500);

        Grid tempGrid = create_test_grid(m1,m2);
        // Copy values to host mirrors
        for(int j = 0; j <= m1; j++) {
            h_Vec_s(j) = tempGrid.Vec_s[j];
        }
        for(int j = 0; j <= m2; j++) {
            h_Vec_v(j) = tempGrid.Vec_v[j];
        }
        for(int j = 0; j < m1; j++) {
            h_Delta_s(j) = tempGrid.Delta_s[j];
        }
        for(int j = 0; j < m2; j++) {
            h_Delta_v(j) = tempGrid.Delta_v[j];
        }
        
        // Copy to device
        Kokkos::deep_copy(hostGrids[i].device_Vec_s, h_Vec_s);
        Kokkos::deep_copy(hostGrids[i].device_Vec_v, h_Vec_v);
        Kokkos::deep_copy(hostGrids[i].device_Delta_s, h_Delta_s);
        Kokkos::deep_copy(hostGrids[i].device_Delta_v, h_Delta_v);
    }

    // Create device view of GridViews array
    Kokkos::View<GridViews*> deviceGrids("deviceGrids", nInstances);
    auto h_deviceGrids = Kokkos::create_mirror_view(deviceGrids);
    
    // Copy GridViews to device
    for(int i = 0; i < nInstances; ++i) {
        h_deviceGrids(i) = hostGrids[i];
    }
    Kokkos::deep_copy(deviceGrids, h_deviceGrids);

    auto grid = deviceGrids(0);
    */

    // 3D diagonals: [nInstances, m1+1, m2+1], etc.
    Kokkos::View<double***> main_diag("main_diag", nInstances, m1+1, m2+1);
    Kokkos::View<double***> lower_diag("lower_diag", nInstances, m1+1, m2);
    Kokkos::View<double***> lower2_diag("lower2_diag", nInstances, m1+1, m2-1);
    Kokkos::View<double***> upper_diag("upper_diag", nInstances, m1+1, m2);
    Kokkos::View<double***> upper2_diag("upper2_diag", nInstances, m1+1, m2-1);

    // Implicit system diagonals
    Kokkos::View<double***> impl_main_diag("impl_main_diag", nInstances, m1+1, m2+1);
    Kokkos::View<double***> impl_lower_diag("impl_lower_diag", nInstances, m1+1, m2);
    Kokkos::View<double***> impl_lower2_diag("impl_lower2_diag", nInstances, m1+1, m2-1);
    Kokkos::View<double***> impl_upper_diag("impl_upper_diag", nInstances, m1+1, m2);
    Kokkos::View<double***> impl_upper2_diag("impl_upper2_diag", nInstances, m1+1, m2-1);

    // Single-instance policy
    team_policy policy(nInstances, Kokkos::AUTO);

    // Build the A2 diagonals in a multi-instance kernel (but instance=0)
    Kokkos::parallel_for("build_a2_single_instance", policy,
        KOKKOS_LAMBDA(const member_type& team) {
            int instance = team.league_rank();

            // Subviews for this one instance
            auto mainDiag_i   = Kokkos::subview(main_diag,      instance, Kokkos::ALL, Kokkos::ALL);
            auto lowerDiag_i  = Kokkos::subview(lower_diag,     instance, Kokkos::ALL, Kokkos::ALL);
            auto lower2Diag_i = Kokkos::subview(lower2_diag,    instance, Kokkos::ALL, Kokkos::ALL);
            auto upperDiag_i  = Kokkos::subview(upper_diag,     instance, Kokkos::ALL, Kokkos::ALL);
            auto upper2Diag_i = Kokkos::subview(upper2_diag,    instance, Kokkos::ALL, Kokkos::ALL);

            auto implMain_i   = Kokkos::subview(impl_main_diag, instance, Kokkos::ALL, Kokkos::ALL);
            auto implLower_i  = Kokkos::subview(impl_lower_diag, instance, Kokkos::ALL, Kokkos::ALL);
            auto implLower2_i = Kokkos::subview(impl_lower2_diag,instance, Kokkos::ALL, Kokkos::ALL);
            auto implUpper_i  = Kokkos::subview(impl_upper_diag, instance, Kokkos::ALL, Kokkos::ALL);
            auto implUpper2_i = Kokkos::subview(impl_upper2_diag,instance, Kokkos::ALL, Kokkos::ALL);

            // Build the diagonals
            build_a2_diagonals_shuffled(
                mainDiag_i, lowerDiag_i, lower2Diag_i,
                upperDiag_i, upper2Diag_i,
                implMain_i, implLower_i, implLower2_i,
                implUpper_i, implUpper2_i,
                grid, theta, delta_t, r_d, kappa, eta, sigma,
                team
            );
        }
    );
    Kokkos::fence();

    // Create x and result vectors (single instance in row 0).
    const int total_size = (m1 + 1) * (m2 + 1);
    std::cout<<"total size" << total_size << std::endl;
    Kokkos::View<double**> x("x", nInstances, total_size);
    Kokkos::View<double**> result("result", nInstances, total_size);

    // Initialize x with sequential values
    auto h_x = Kokkos::create_mirror_view(x);
    for(int i = 0; i < total_size; ++i) {
        h_x(0, i) = i + 1.0;  // for easier tracking
    }
    Kokkos::deep_copy(x, h_x);

    // Print original vector layout
    std::cout << "Original vector layout (by variance levels):\n";
    // (Optional) If you want the full grid, uncomment:
    for(int j = 0; j <= m2; j++) {
        std::cout << "  v" << j << ": ";
        for(int i = 0; i <= m1; i++) {
            int idx = j*(m1+1) + i;
            std::cout << h_x(0, idx) << " ";
        }
        std::cout << "\n";
    }
    

    // Shuffle x in a parallel_for
    Kokkos::View<double**> x_shuffled("x_shuffled", nInstances, total_size);
    Kokkos::parallel_for("shuffle_x", policy,
        KOKKOS_LAMBDA(const member_type& team) {
            int instance = team.league_rank();
            device_shuffle_vector(
                Kokkos::subview(x, instance, Kokkos::ALL),
                Kokkos::subview(x_shuffled, instance, Kokkos::ALL),
                m1, m2, team
            );
        }
    );
    Kokkos::fence();

    std::cout << "\nShuffled vector layout (by stock prices):\n";
    // (Optional) If you want the full grid, fetch x_shuffled back and print:
    
    auto h_x_shuf = Kokkos::create_mirror_view(x_shuffled);
    Kokkos::deep_copy(h_x_shuf, x_shuffled);
    for(int i = 0; i <= m1; i++) {
        std::cout << "  s" << i << ": ";
        for(int j = 0; j <= m2; j++) {
            int idx = i*(m2+1) + j;
            std::cout << h_x_shuf(0, idx) << " ";
        }
        std::cout << "\n";
    }
    

    // Test multiplication
    std::cout << "\nTesting multiplication:\n";
    Kokkos::parallel_for("multiply_shuffled", policy,
        KOKKOS_LAMBDA(const member_type& team) {
            int instance = team.league_rank();
            device_multiply_shuffled(
                Kokkos::subview(main_diag,   instance, Kokkos::ALL, Kokkos::ALL),
                Kokkos::subview(lower_diag,  instance, Kokkos::ALL, Kokkos::ALL),
                Kokkos::subview(lower2_diag, instance, Kokkos::ALL, Kokkos::ALL),
                Kokkos::subview(upper_diag,  instance, Kokkos::ALL, Kokkos::ALL),
                Kokkos::subview(upper2_diag, instance, Kokkos::ALL, Kokkos::ALL),
                Kokkos::subview(x_shuffled,  instance, Kokkos::ALL),
                Kokkos::subview(result,       instance, Kokkos::ALL),
                team
            );
        }
    );
    Kokkos::fence();

    // Unshuffle the result to compare in original indexing
    Kokkos::View<double*> single_result_unshuf("single_result_unshuf", total_size);
    {
        team_policy singlePolicy(1, Kokkos::AUTO);
        Kokkos::parallel_for("unshuffle_result", singlePolicy,
            KOKKOS_LAMBDA(const member_type& team) {
                device_unshuffle_vector(
                    Kokkos::subview(result, 0, Kokkos::ALL),
                    single_result_unshuf,
                    m1, m2, team
                );
            }
        );
        Kokkos::fence();
    }

    // Print "Kernel multiplication results (first block)"
    auto h_result_unshuf = Kokkos::create_mirror_view(single_result_unshuf);
    Kokkos::deep_copy(h_result_unshuf, single_result_unshuf);
    std::cout << "Kernel multiplication results (first block):\n";
    for(int i = 0; i < total_size; i++) {
        // printing up to 42 entries just to match your debug code
        std::cout << h_result_unshuf(i) << " ";
    }
    std::cout << "\n";

    // Test implicit solve: create a b vector, shuffle it, then solve
    Kokkos::View<double**> b("b", nInstances, total_size);
    auto h_b = Kokkos::create_mirror_view(b);
    for(int i = 0; i < total_size; ++i) {
        h_b(0, i) = i + 1.0; // same pattern
    }
    Kokkos::deep_copy(b, h_b);

    // Shuffle b
    Kokkos::View<double**> b_shuffled("b_shuffled", nInstances, total_size);
    Kokkos::parallel_for("shuffle_b", policy,
        KOKKOS_LAMBDA(const member_type& team) {
            int instance = team.league_rank();
            device_shuffle_vector(
                Kokkos::subview(b, instance, Kokkos::ALL),
                Kokkos::subview(b_shuffled, instance, Kokkos::ALL),
                m1, m2, team
            );
        }
    );
    Kokkos::fence();

    // Re-shuffle x again so we start from the same known initial guess
    Kokkos::deep_copy(x, h_x);
    Kokkos::parallel_for("shuffle_x_again", policy,
        KOKKOS_LAMBDA(const member_type& team) {
            device_shuffle_vector(
                Kokkos::subview(x, team.league_rank(), Kokkos::ALL),
                Kokkos::subview(x_shuffled, team.league_rank(), Kokkos::ALL),
                m1, m2, team
            );
        }
    );
    Kokkos::fence();

    // Temporary storage for the solver
    Kokkos::View<double***> c_prime("c_prime",   nInstances, m1+1, m2+1);
    Kokkos::View<double***> c2_prime("c2_prime", nInstances, m1+1, m2+1);
    Kokkos::View<double***> d_prime("d_prime",   nInstances, m1+1, m2+1);

    // Solve
    std::cout << "\nTesting implicit solve:\n";
    Kokkos::parallel_for("implicit_solve_shuffled", policy,
        KOKKOS_LAMBDA(const member_type& team) {
            int instance = team.league_rank();
            device_solve_implicit_shuffled(
                Kokkos::subview(impl_main_diag,   instance, Kokkos::ALL, Kokkos::ALL),
                Kokkos::subview(impl_lower_diag,  instance, Kokkos::ALL, Kokkos::ALL),
                Kokkos::subview(impl_lower2_diag, instance, Kokkos::ALL, Kokkos::ALL),
                Kokkos::subview(impl_upper_diag,  instance, Kokkos::ALL, Kokkos::ALL),
                Kokkos::subview(impl_upper2_diag, instance, Kokkos::ALL, Kokkos::ALL),
                Kokkos::subview(x_shuffled,       instance, Kokkos::ALL),
                Kokkos::subview(c_prime,          instance, Kokkos::ALL, Kokkos::ALL),
                Kokkos::subview(c2_prime,         instance, Kokkos::ALL, Kokkos::ALL),
                Kokkos::subview(d_prime,          instance, Kokkos::ALL, Kokkos::ALL),
                Kokkos::subview(b_shuffled,       instance, Kokkos::ALL),
                team
            );
        }
    );
    Kokkos::fence();

    // Unshuffle the final x
    Kokkos::View<double*> x_unshuffled("x_unshuffled", total_size);
    {
        team_policy singlePolicy(1, Kokkos::AUTO);
        Kokkos::parallel_for("unshuffle_x_final", singlePolicy,
            KOKKOS_LAMBDA(const member_type& team) {
                device_unshuffle_vector(
                    Kokkos::subview(x_shuffled, 0, Kokkos::ALL),
                    x_unshuffled, m1, m2, team
                );
            }
        );
        Kokkos::fence();
    }

    // Print final solve results
    auto h_x_unshuf = Kokkos::create_mirror_view(x_unshuffled);
    Kokkos::deep_copy(h_x_unshuf, x_unshuffled);

    std::cout << "Kernel solve results (first block):\n";
    for(int i = 0; i < total_size; i++) {
        std::cout << h_x_unshuf(i) << " ";
    }
    std::cout << "\n\nTests have completed successfully.\n";
}

//compares to test_heston_A2_shuffled()
//I thought the instnaces are not working properly, but they are
void test_heston_A2_shuffled_class_free() {
    using timer = std::chrono::high_resolution_clock;
    
    // Test dimensions (same as original test)
    const int m1 = 300;
    const int m2 = 100;
    
    // Parameters (same as original test)
    const double sigma = 0.3;
    const double r_d = 0.025;
    const double kappa = 1.5;
    const double eta = 0.04;

    const double theta = 0.8;
    const double delta_t = 1.0/40;

    // Create grid
    Grid grid = create_test_grid(m1, m2);

    // Initialize GridViews structure
    std::vector<GridViews> hostGrids;
    buildMultipleGridViews(hostGrids, 1, m1, m2);  // Just one instance

    // Fill grid values
    auto h_Vec_s = Kokkos::create_mirror_view(hostGrids[0].device_Vec_s);
    auto h_Vec_v = Kokkos::create_mirror_view(hostGrids[0].device_Vec_v);
    auto h_Delta_s = Kokkos::create_mirror_view(hostGrids[0].device_Delta_s);
    auto h_Delta_v = Kokkos::create_mirror_view(hostGrids[0].device_Delta_v);

    // Copy values from test grid
    for(int j = 0; j <= m1; j++) {
        h_Vec_s(j) = grid.Vec_s[j];
    }
    for(int j = 0; j <= m2; j++) {
        h_Vec_v(j) = grid.Vec_v[j];
    }
    for(int j = 0; j < m1; j++) {
        h_Delta_s(j) = grid.Delta_s[j];
    }
    for(int j = 0; j < m2; j++) {
        h_Delta_v(j) = grid.Delta_v[j];
    }

    // Copy to device
    Kokkos::deep_copy(hostGrids[0].device_Vec_s, h_Vec_s);
    Kokkos::deep_copy(hostGrids[0].device_Vec_v, h_Vec_v);
    Kokkos::deep_copy(hostGrids[0].device_Delta_s, h_Delta_s);
    Kokkos::deep_copy(hostGrids[0].device_Delta_v, h_Delta_v);

    // Create device view of GridViews
    Kokkos::View<GridViews*> deviceGrids("deviceGrids", 1);
    auto h_deviceGrids = Kokkos::create_mirror_view(deviceGrids);
    h_deviceGrids(0) = hostGrids[0];
    Kokkos::deep_copy(deviceGrids, h_deviceGrids);

    // Create matrix diagonals
    Kokkos::View<double**> main_diag("main_diag", m1+1, m2+1);
    Kokkos::View<double**> lower_diag("lower_diag", m1+1, m2);
    Kokkos::View<double**> lower2_diag("lower2_diag", m1+1, m2-1);
    Kokkos::View<double**> upper_diag("upper_diag", m1+1, m2);
    Kokkos::View<double**> upper2_diag("upper2_diag", m1+1, m2-1);

    Kokkos::View<double**> impl_main_diag("impl_main_diag", m1+1, m2+1);
    Kokkos::View<double**> impl_lower_diag("impl_lower_diag", m1+1, m2);
    Kokkos::View<double**> impl_lower2_diag("impl_lower2_diag", m1+1, m2-1);
    Kokkos::View<double**> impl_upper_diag("impl_upper_diag", m1+1, m2);
    Kokkos::View<double**> impl_upper2_diag("impl_upper2_diag", m1+1, m2-1);

    // Create test vectors
    const int total_size = (m1 + 1) * (m2 + 1);
    Kokkos::View<double*> x("x", total_size);
    Kokkos::View<double*> b("b", total_size);
    Kokkos::View<double*> result("result", total_size);

    // Initialize with same values as original test
    auto h_b = Kokkos::create_mirror_view(b);
    auto h_x = Kokkos::create_mirror_view(x);
    for (int i = 0; i < total_size; ++i) {
        h_b(i) = i + 1;
        h_x(i) = i + 1;
    }
    Kokkos::deep_copy(b, h_b);
    Kokkos::deep_copy(x, h_x);

    // Create temporary storage for implicit solve
    Kokkos::View<double**> c_prime("c_prime", m1+1, m2+1);
    Kokkos::View<double**> c2_prime("c2_prime", m1+1, m2+1);
    Kokkos::View<double**> d_prime("d_prime", m1+1, m2+1);

    // Set up team policy
    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;
    team_policy policy(1, Kokkos::AUTO);

    // Build matrices and solve
    auto t_start = timer::now();
    
    Kokkos::parallel_for("build_and_solve", policy,
        KOKKOS_LAMBDA(const member_type& team) {
            // Build matrices
            build_a2_diagonals_shuffled(
                main_diag, lower_diag, lower2_diag, upper_diag, upper2_diag,
                impl_main_diag, impl_lower_diag, impl_lower2_diag, impl_upper_diag, impl_upper2_diag,
                deviceGrids(0), theta, delta_t, r_d, kappa, eta, sigma,
                team
            );

            // Solve implicit system
            device_solve_implicit_shuffled(
                impl_main_diag, impl_lower_diag, impl_lower2_diag, impl_upper_diag, impl_upper2_diag,
                x, c_prime, c2_prime, d_prime, b,
                team
            );
    });
    Kokkos::fence();

    auto t_end = timer::now();
    std::cout << "Implicit solve time: "
              << std::chrono::duration<double>(t_end - t_start).count()
              << " seconds" << std::endl;

    // Test multiply
    t_start = timer::now();
    
    Kokkos::parallel_for("multiply", policy,
        KOKKOS_LAMBDA(const member_type& team) {
            device_multiply_shuffled(
                main_diag, lower_diag, lower2_diag, upper_diag, upper2_diag,
                x, result,
                team
            );
    });
    Kokkos::fence();

    t_end = timer::now();
    std::cout << "Multiply time: "
              << std::chrono::duration<double>(t_end - t_start).count()
              << " seconds" << std::endl;

    // Compute residual
    auto h_result = Kokkos::create_mirror_view(result);
    Kokkos::deep_copy(h_result, result);
    Kokkos::deep_copy(h_x, x);

    double residual = 0.0;
    for(int i = 0; i < total_size; i++) {
        double res = h_x(i) - theta * delta_t * h_result(i) - h_b(i);
        residual += res * res;
    }
    residual = std::sqrt(residual);

    std::cout << "Residual norm: " << residual << std::endl;

    // Print first few results for comparison
    std::cout << "First few x values: ";
    for(int i = 0; i < std::min(5, total_size); i++) {
        std::cout << h_x(i) << " ";
    }
    std::cout << std::endl;
}

void test_a2_shuffled_kernel(){
    Kokkos::initialize();
        {
            try{
                //test_a2_build();
                //test_a2_shuffled_structure_function();
                //compare_a2_diagonal_kernels();

                test_a2_multiple_instances();
                //test_a2_shuffled_single_instance_debug();

                //test_heston_A2_shuffled_class_free();
            }
            catch (std::exception& e) {
                std::cout << "Error: " << e.what() << std::endl;
            }
        } // All test objects destroyed here
    Kokkos::finalize();
}
