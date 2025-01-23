#ifndef HES_A0_KERNELS_HPP
#define HES_A0_KERNELS_HPP

#include <Kokkos_Core.hpp>
#include "grid.hpp"
#include "coeff.hpp"


/*

A0 device class

*/
// A0 Device Class
template<class DeviceType>
struct Device_A0_heston {
    typedef DeviceType execution_space;
    typedef typename DeviceType::memory_space memory_space;
    
    Kokkos::View<double**, DeviceType> values; // [m2-1][(m1-1)*9]
    int m1, m2;

    KOKKOS_FUNCTION Device_A0_heston() = default;
    Device_A0_heston(int m1_in, int m2_in) : m1(m1_in), m2(m2_in) {
        values = Kokkos::View<double**>("A0_values", m2 - 1, (m1 - 1) * 9);
    }

    template<class GridType>
    KOKKOS_FUNCTION
    void build_matrix(const GridType& grid,
                     const double rho, const double sigma,
                     const Kokkos::TeamPolicy<>::member_type& team){
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

    template<class XView, class ResultView>
    KOKKOS_FUNCTION
    void multiply_parallel_s_and_v(const XView& x, const ResultView& result,
                                 const Kokkos::TeamPolicy<>::member_type& team){
    
    //Zero out result vector, since A0 matrix has a "complex" zero strucutre appearing in the values we can not account for those
    //inside the function itself. We have to zero out first
    const int total_size = (m1+1)*(m2+1);
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
};









// Test function
void test_a0_kernel();

#endif // HES_A0_KERNELS_HPP