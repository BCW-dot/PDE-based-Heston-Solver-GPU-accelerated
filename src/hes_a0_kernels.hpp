#ifndef HES_A0_KERNELS_HPP
#define HES_A0_KERNELS_HPP

#include <Kokkos_Core.hpp>
#include "grid.hpp"

// Function to build A0 matrix diagonals
KOKKOS_FUNCTION
void build_a0_values(
    const Kokkos::View<double**>& values,  // [m2-1][(m1-1)*9]
    const Grid& grid,
    const double rho,
    const double sigma,
    const Kokkos::TeamPolicy<>::member_type& team);

// Matrix-vector multiplication
KOKKOS_FUNCTION
void device_multiply_a0(
    const Kokkos::View<const double**>& values,
    const Kokkos::View<double*>& x,
    const Kokkos::View<double*>& result,
    const Kokkos::TeamPolicy<>::member_type& team);

// Test function
void test_a0_kernel();

#endif // HES_A0_KERNELS_HPP