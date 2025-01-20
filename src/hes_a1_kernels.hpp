// In hes_a1_kernels.hpp
#ifndef HES_A1_KERNELS_HPP
#define HES_A1_KERNELS_HPP

#include <Kokkos_Core.hpp>
#include "grid.hpp"
#include "grid_pod.hpp"

KOKKOS_FUNCTION
void build_a1_diagonals(
    const Kokkos::View<double**>& main_diag,
    const Kokkos::View<double**>& lower_diag,
    const Kokkos::View<double**>& upper_diag,
    const Kokkos::View<double**>& impl_main_diag,
    const Kokkos::View<double**>& impl_lower_diag,
    const Kokkos::View<double**>& impl_upper_diag,
    const Grid& grid,
    const double theta,
    const double dt,
    const double r_d,
    const double r_f);

void test_a1_kernel();

#endif