#ifndef HES_A2_SHUFFLED_KERNELS_HPP
#define HES_A2_SHUFFLED_KERNELS_HPP

#include <Kokkos_Core.hpp>
#include "grid.hpp"
#include "grid_pod.hpp"
#include "coeff.hpp"


template<class InputView, class OutputView>
KOKKOS_FUNCTION
void device_shuffle_vector(
    const InputView& input,
    OutputView& output,
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

template<class InputView, class OutputView>
KOKKOS_FUNCTION
void device_unshuffle_vector(
    const InputView& input,
    OutputView& output,
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


/*

Device A2 class

*/
// A2 Shuffled Device Class
template<class DeviceType>
struct Device_A2_shuffled_heston {
    typedef DeviceType execution_space;
    typedef typename DeviceType::memory_space memory_space;
    
    // Matrix diagonals
    Kokkos::View<double**, DeviceType> main_diags;
    Kokkos::View<double**, DeviceType> lower_diags;
    Kokkos::View<double**, DeviceType> lower2_diags;
    Kokkos::View<double**, DeviceType> upper_diags;
    Kokkos::View<double**, DeviceType> upper2_diags;

    // Implicit system diagonals
    Kokkos::View<double**, DeviceType> impl_main_diags;
    Kokkos::View<double**, DeviceType> impl_lower_diags;
    Kokkos::View<double**, DeviceType> impl_lower2_diags;
    Kokkos::View<double**, DeviceType> impl_upper_diags;
    Kokkos::View<double**, DeviceType> impl_upper2_diags;

    // Temporary storage
    Kokkos::View<double**, DeviceType> c_prime;
    Kokkos::View<double**, DeviceType> c2_prime;
    Kokkos::View<double**, DeviceType> d_prime;

    int m1, m2;

    KOKKOS_FUNCTION Device_A2_shuffled_heston() = default;
    Device_A2_shuffled_heston(int m1_in, int m2_in): m1(m1_in), m2(m2_in) {
        // Allocate explicit system diagonals
        main_diags = Kokkos::View<double**>("A2_main_diags", m1+1, m2+1);
        lower_diags = Kokkos::View<double**>("A2_lower_diags", m1+1, m2);
        lower2_diags = Kokkos::View<double**>("A2_lower2_diags", m1+1, m2-1);
        upper_diags = Kokkos::View<double**>("A2_upper_diags", m1+1, m2);
        upper2_diags = Kokkos::View<double**>("A2_upper2_diags", m1+1, m2-1);
        
        // Allocate implicit system diagonals
        impl_main_diags = Kokkos::View<double**>("A2_impl_main_diags", m1+1, m2+1);
        impl_lower_diags = Kokkos::View<double**>("A2_impl_lower_diags", m1+1, m2);
        impl_lower2_diags = Kokkos::View<double**>("A2_impl_lower2_diags", m1+1, m2-1);
        impl_upper_diags = Kokkos::View<double**>("A2_impl_upper_diags", m1+1, m2);
        impl_upper2_diags = Kokkos::View<double**>("A2_impl_upper2_diags", m1+1, m2-1);
        
        // Allocate temporary storage for implicit solve
        c_prime = Kokkos::View<double**>("A2_c_prime", m1+1, m2+1);
        c2_prime = Kokkos::View<double**>("A2_c2_prime", m1+1, m2+1);
        d_prime = Kokkos::View<double**>("A2_d_prime", m1+1, m2+1);
    }

    template<class GridType>
    KOKKOS_FUNCTION
    inline void build_matrix(const GridType& grid,
                     const double r_d, const double kappa, const double eta, 
                     const double sigma, const double theta, const double dt,
                     const Kokkos::TeamPolicy<>::member_type& team){
    // Build explicit matrices
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, m1 + 1),
        [&](const int i) {
            for(int j = 0; j < m2 + 1; j++) {
                main_diags(i, j) = 0.0;
                if(j < m2) {
                lower_diags(i, j) = 0.0;
                upper_diags(i, j) = 0.0;
                }
                if(j < m2 - 1) {
                lower2_diags(i, j) = 0.0;
                upper2_diags(i, j) = 0.0;
                }
            }
            
            for(int j = 0; j < m2 - 1; j++) {
                const double temp = kappa * (eta - grid.device_Vec_v[j]);
                const double temp2 = 0.5 * sigma * sigma * grid.device_Vec_v[j];
                
                // Add reaction term
                main_diags(i,j) += -0.5 * r_d;

                if(grid.device_Vec_v[j] > 1.0) {
                    // Upwind scheme coefficients
                    lower2_diags(i,j + 1 -2) += temp * device_alpha_v(j, -2, grid.device_Delta_v);
                    lower_diags(i,j + 1 -1) += temp * device_alpha_v(j, -1, grid.device_Delta_v);
                    main_diags(i,j +1 - 0) += temp * device_alpha_v(j, 0, grid.device_Delta_v);

                    lower_diags(i,j+1-1) += temp2 * device_delta_v(j-1, -1, grid.device_Delta_v);
                    main_diags(i,j+1+0) += temp2 * device_delta_v(j-1, 0, grid.device_Delta_v);
                    upper_diags(i,j+1) += temp2 * device_delta_v(j-1, 1, grid.device_Delta_v);
                }
                if(j == 0) {
                    // Special v=0 case
                    main_diags(i,j) += temp * device_gamma_v(j, 0, grid.device_Delta_v);
                    upper_diags(i,j) += temp * device_gamma_v(j, 1, grid.device_Delta_v);
                    upper2_diags(i,j) += temp * device_gamma_v(j, 2, grid.device_Delta_v);
                }
                else {
                    lower_diags(i,j-1) += temp * device_beta_v(j-1, -1, grid.device_Delta_v) +
                                           temp2 * device_delta_v(j-1, -1, grid.device_Delta_v);
                    main_diags(i,j) += temp * device_beta_v(j-1, 0, grid.device_Delta_v) +
                                    temp2 * device_delta_v(j-1, 0, grid.device_Delta_v);
                    upper_diags(i,j) += temp * device_beta_v(j-1, 1, grid.device_Delta_v) +
                                     temp2 * device_delta_v(j-1, 1, grid.device_Delta_v);
                }
            }
        });
    team.team_barrier();
    
    // Build implicit matrices
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, m1 + 1),
        [&](const int i) {
            for(int j = 0; j <= m2; j++) {
                impl_main_diags(i,j) = 1.0 - theta * dt * main_diags(i,j);
            }
            for(int j = 0; j < m2; j++) {
                impl_lower_diags(i,j) = -theta * dt * lower_diags(i,j);
                impl_upper_diags(i,j) = -theta * dt * upper_diags(i,j);
            }
            for(int j = 0; j < m2-1; j++) {
                impl_lower2_diags(i,j) = -theta * dt * lower2_diags(i,j);
                impl_upper2_diags(i,j) = -theta * dt * upper2_diags(i,j);
            }
        });
    team.team_barrier();
    
    //I tested to build both expl and impl diagonals inside one loop, hwoever it was actually slower. 
    //I guess becasue of warp divergence maybe, since the A2 matrix has tougher syntax to handle
    }

    template<class XView, class ResultView>
    KOKKOS_FUNCTION
    void multiply_parallel_s(const XView& x, const ResultView& result,
                         const Kokkos::TeamPolicy<>::member_type& team){
    // Parallelize over stock price blocks
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, m1 + 1),
        [&](const int i) {
            const int block_offset = i * (m2 + 1);

            // First row of block
            result(block_offset) = main_diags(i, 0) * x(block_offset);
            if(0 < m2) {
                result(block_offset) += upper_diags(i, 0) * x(block_offset + 1);
            }
            if(1 < m2) {
                result(block_offset) += upper2_diags(i, 0) * x(block_offset + 2);
            }

            // Second row
            if(0 < m2) {
                result(block_offset + 1) = lower_diags(i, 0) * x(block_offset) +
                                         main_diags(i, 1) * x(block_offset + 1);
                if(1 < m2) {
                    result(block_offset + 1) += upper_diags(i, 1) * x(block_offset + 2);
                }
                if(2 < m2) {
                    result(block_offset + 1) += upper2_diags(i, 1) * x(block_offset + 3);
                }
            }

            // Middle rows
            for(int j = 2; j < m2 - 1; j++) {
                result(block_offset + j) = lower2_diags(i, j-2) * x(block_offset + j-2) +
                                         lower_diags(i, j-1) * x(block_offset + j-1) +
                                         main_diags(i, j) * x(block_offset + j) +
                                         upper_diags(i, j) * x(block_offset + j+1);
                if(j < m2 - 2) {
                    result(block_offset + j) += upper2_diags(i, j) * x(block_offset + j+2);
                }
            }

            // Second-to-last row
            if(m2 > 2) {
                const int j = m2 - 1;
                result(block_offset + j) = lower2_diags(i, j-2) * x(block_offset + j-2) +
                                         lower_diags(i, j-1) * x(block_offset + j-1) +
                                         main_diags(i, j) * x(block_offset + j);
                if(j < m2) {
                    result(block_offset + j) += upper_diags(i, j) * x(block_offset + j+1);
                }
            }

            // Last row
            if(m2 > 1) {
                const int j = m2;
                result(block_offset + j) = lower2_diags(i, j-2) * x(block_offset + j-2) +
                                         lower_diags(i, j-1) * x(block_offset + j-1) +
                                         main_diags(i, j) * x(block_offset + j);
            }
        });
    team.team_barrier();
    }

    template<class XView, class BView>
    KOKKOS_FUNCTION
    void solve_implicit_parallel_s(XView& x, const BView& b,
                               const Kokkos::TeamPolicy<>::member_type& team){
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, m1 + 1),
        [&](const int i) {
            const int block_offset = i * (m2 + 1);
            const int block_size = m2 + 1;

            // Forward sweep
            // First row
            c_prime(i, 0) = impl_upper_diags(i, 0) / impl_main_diags(i, 0);
            c2_prime(i, 0) = impl_upper2_diags(i, 0) / impl_main_diags(i, 0);
            d_prime(i, 0) = b(block_offset) / impl_main_diags(i, 0);

            // Second row
            if(block_size > 1) {
                double m1 = 1.0 / (impl_main_diags(i, 1) - impl_lower_diags(i, 0) * c_prime(i, 0));
                c_prime(i, 1) = (impl_upper_diags(i, 1) - impl_lower_diags(i, 0) * c2_prime(i, 0)) * m1;
                c2_prime(i, 1) = impl_upper2_diags(i, 1) * m1;
                d_prime(i, 1) = (b(block_offset + 1) - impl_lower_diags(i, 0) * d_prime(i, 0)) * m1;
            }

            // Main forward sweep
            for(int j = 2; j < block_size; j++) {
                double den = impl_main_diags(i, j) - 
                            (impl_lower_diags(i, j-1) - impl_lower2_diags(i, j-2) * c_prime(i, j-2)) * c_prime(i, j-1) - 
                            impl_lower2_diags(i, j-2) * c2_prime(i, j-2);
                double m = 1.0 / den;

                // Update c coefficients
                c_prime(i, j) = (impl_upper_diags(i, j) - 
                                (impl_lower_diags(i, j-1) - impl_lower2_diags(i, j-2) * c_prime(i, j-2)) * c2_prime(i, j-1)) * m;
                if(j < block_size - 2) {
                    c2_prime(i, j) = impl_upper2_diags(i, j) * m;
                }

                // Update d
                d_prime(i, j) = (b(block_offset + j) - 
                                (impl_lower_diags(i, j-1) - impl_lower2_diags(i, j-2) * c_prime(i, j-2)) * d_prime(i, j-1) - 
                                impl_lower2_diags(i, j-2) * d_prime(i, j-2)) * m;
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
};

// Test function
void test_a2_shuffled_kernel();

#endif