// In hes_a1_kernels.hpp
#ifndef HES_A1_KERNELS_HPP
#define HES_A1_KERNELS_HPP

#include <Kokkos_Core.hpp>
#include "grid.hpp"
#include "coeff.hpp"


/*

A1 class

*/
template<class DeviceType>
struct Device_A1_heston {
    typedef DeviceType execution_space;
    typedef typename DeviceType::memory_space memory_space;
    
    // Matrix diagonals 
    Kokkos::View<double**, DeviceType> main_diags;
    Kokkos::View<double**, DeviceType> lower_diags;
    Kokkos::View<double**, DeviceType> upper_diags;

    Kokkos::View<double**, DeviceType> impl_main_diags;
    Kokkos::View<double**, DeviceType> impl_lower_diags;
    Kokkos::View<double**, DeviceType> impl_upper_diags;

    // Temporary storage for implicit solve
    Kokkos::View<double**, DeviceType> temp_para;

    // Dimensions and parameters
    int m1, m2;
    
    KOKKOS_FUNCTION Device_A1_heston() = default;

    Device_A1_heston(int m1_in, int m2_in) : m1(m1_in), m2(m2_in) {
        main_diags = Kokkos::View<double**, DeviceType>("A1_main_diags", m2+1, m1+1);
        lower_diags = Kokkos::View<double**, DeviceType>("A1_lower_diags", m2+1, m1);
        upper_diags = Kokkos::View<double**, DeviceType>("A1_upper_diags", m2+1, m1);

        impl_main_diags = Kokkos::View<double**, DeviceType>("A1_impl_main_diags", m2+1, m1+1);
        impl_lower_diags = Kokkos::View<double**, DeviceType>("A1_impl_lower_diags", m2+1, m1);
        impl_upper_diags = Kokkos::View<double**, DeviceType>("A1_impl_upper_diags", m2+1, m1);

        temp_para = Kokkos::View<double**>("temp_para", m2+1, m1+1);
    }

    template<class GridType>
    KOKKOS_FUNCTION
    void build_matrix(const GridType& grid, 
                     const double r_d, const double r_f,
                     const double theta, const double dt,
                     const Kokkos::TeamPolicy<>::member_type& team) {
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, m2+1),
            [&](const int j) {
            
            // First point (i=0) is boundary
            main_diags(j,0) = 0.0;
            impl_main_diags(j,0) = 1.0;
            if(j < m2) {
                upper_diags(j,0) = 0.0;
                impl_upper_diags(j,0) = 0.0;
            }

            // Interior points
            for(int i = 1; i < m1; i++) {
                // Compute PDE coefficients
                const double s = grid.device_Vec_s[i];
                const double v = grid.device_Vec_v[j];
                
                // a = 0.5*s^2*v (diffusion)
                // b = (r_d - r_f)*s (drift)
                const double a = 0.5 * s * s * v;
                const double b = (r_d - r_f) * s;

                // Build explicit diagonals using central differences
                // PDE coefficients

                // Build tridiagonal system for this level
                // Lower diagonal
                lower_diags(j,i-1) = a * device_delta_s(i-1, -1, grid.device_Delta_s) + 
                                    b * device_beta_s(i-1, -1, grid.device_Delta_s);
                
                // Main diagonal
                main_diags(j,i) = a * device_delta_s(i-1, 0, grid.device_Delta_s) + 
                                    b * device_beta_s(i-1, 0, grid.device_Delta_s) - 0.5 * r_d;
                
                // Upper diagonal
                upper_diags(j,i) = a * device_delta_s(i-1, 1, grid.device_Delta_s) + 
                                    b * device_beta_s(i-1, 1, grid.device_Delta_s);

                // Build implicit diagonals: (I - theta*dt*A)
                impl_lower_diags(j,i-1) = -theta * dt * lower_diags(j,i-1);
                impl_main_diags(j,i) = 1.0 - theta * dt * main_diags(j,i);
                impl_upper_diags(j,i) = -theta * dt * upper_diags(j,i);
            }

            // Last point (i=m1)
            main_diags(j,m1) = -0.5 * r_d;
            impl_main_diags(j,m1) = 1.0 - theta * dt * main_diags(j,m1);
            lower_diags(j,m1-1) = 0.0;
            impl_lower_diags(j,m1-1) = 0.0;

            });
        team.team_barrier();
    }

    template<class XView, class ResultView>
    KOKKOS_FUNCTION
    void multiply_parallel_v(const XView& x, const ResultView& result,
                           const Kokkos::TeamPolicy<>::member_type& team) {
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, m2+1),
            [&](const int j) {
                const int offset = j * (m1 + 1);
                // First point (i=0): only has main and upper diagonal
                double sum = main_diags(j, 0) * x(offset);
                sum += upper_diags(j, 0) * x(offset + 1);
                result(offset) = sum;

                // Middle points: have all three diagonals
                for (int i = 1; i < m1; i++) {
                    double sum = lower_diags(j, i-1) * x(offset + i-1) +
                                main_diags(j, i) * x(offset + i) +
                                upper_diags(j, i) * x(offset + i+1);
                    result(offset + i) = sum;
                }

                // Last point (i=m1): only has main and lower diagonal
                sum = lower_diags(j, m1-1) * x(offset + m1-1) +
                    main_diags(j, m1) * x(offset + m1);
                result(offset + m1) = sum;
            });
        team.team_barrier();
    }

    template<class XView, class BView>
    KOKKOS_FUNCTION
    void solve_implicit_parallel_v(XView& x, const BView& b,
                                 const Kokkos::TeamPolicy<>::member_type& team) {
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, m2+1),
            [&](const int j) {
                const int offset = j * (m1 + 1);

                temp_para(j,0) = impl_main_diags(j,0);
                x(offset) = b(offset);

                for(int i = 1; i <= m1; i++) {
                    const double m = impl_lower_diags(j,i-1) / temp_para(j,i-1);
                    temp_para(j,i) = impl_main_diags(j,i) - m * impl_upper_diags(j,i-1);
                    x(offset + i) = b(offset + i) - m * x(offset + i-1);
                }

                x(offset + m1) /= temp_para(j,m1);
                for(int i = m1-1; i >= 0; i--) {
                    x(offset + i) = (x(offset + i) - impl_upper_diags(j,i) * x(offset + i+1)) 
                                   / temp_para(j,i);
                }
            });
        team.team_barrier();
    }
};








void test_a1_kernel();

#endif