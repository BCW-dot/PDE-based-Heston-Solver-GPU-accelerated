#ifndef HES_A0_KERNELS_HPP
#define HES_A0_KERNELS_HPP

#include <Kokkos_Core.hpp>
#include "grid.hpp"

// Build the A0 matrix values on device with templates
template <class ValuesView, class GridType>
KOKKOS_FUNCTION
void build_a0_values(
    const ValuesView& values,  // [m2-1][(m1-1)*9]
    const GridType& grid,      // Now accepts either Grid or GridViews
    const double rho,
    const double sigma,
    const Kokkos::TeamPolicy<>::member_type& team);

// Templated multiply operations
template <class ValuesView, class XView, class ResultView>
KOKKOS_FUNCTION
void device_multiply_a0(
    const ValuesView& values,
    const XView& x,
    const ResultView& result,
    const Kokkos::TeamPolicy<>::member_type& team);

// Test function
void test_a0_kernel();

#endif // HES_A0_KERNELS_HPP