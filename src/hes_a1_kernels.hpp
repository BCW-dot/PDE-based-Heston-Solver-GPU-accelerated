// In hes_a1_kernels.hpp
#ifndef HES_A1_KERNELS_HPP
#define HES_A1_KERNELS_HPP

#include <Kokkos_Core.hpp>
#include "grid.hpp"
#include "grid_pod.hpp"

template <class MDView, class LDView, class UDView,
          class IMDView, class ILDView, class IUDView,
          class GridType>  // New template parameter for Grid type
KOKKOS_FUNCTION
void build_a1_diagonals(
    const MDView& main_diag,
    const LDView& lower_diag,
    const UDView& upper_diag,
    const IMDView& impl_main_diag,
    const ILDView& impl_lower_diag,
    const IUDView& impl_upper_diag,
    const GridType& grid,    // Now accepts either Grid or GridViews
    const double theta,
    const double dt,
    const double r_d,
    const double r_f,
    const Kokkos::TeamPolicy<>::member_type& team);

void test_a1_kernel();

#endif