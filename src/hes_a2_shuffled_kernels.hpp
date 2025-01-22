#ifndef HES_A2_SHUFFLED_KERNELS_HPP
#define HES_A2_SHUFFLED_KERNELS_HPP

#include <Kokkos_Core.hpp>
#include "grid.hpp"
#include "grid_pod.hpp"

// Build the A2 matrix diagonals on device
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
    const Kokkos::TeamPolicy<>::member_type& team);

// Multiply template
template<class View2D_const_main, class View2D_const_lower, 
         class View2D_const_lower2, class View2D_const_upper,
         class View2D_const_upper2, class View1D_x, class View1D_result>
KOKKOS_FUNCTION
void a2_device_multiply_shuffled(
    const View2D_const_main& main_diag,
    const View2D_const_lower& lower_diag,
    const View2D_const_lower2& lower2_diag,
    const View2D_const_upper& upper_diag,
    const View2D_const_upper2& upper2_diag,
    const View1D_x& x,
    const View1D_result& result,
    const Kokkos::TeamPolicy<>::member_type& team);

// Device implicit solve
template<class View2D_const_main, class View2D_const_lower, 
         class View2D_const_lower2, class View2D_const_upper,
         class View2D_const_upper2, class View1D_x, 
         class View2D_c, class View2D_c2, class View2D_d,
         class View1D_b>
KOKKOS_FUNCTION
void a2_device_solve_implicit_shuffled(
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
    const Kokkos::TeamPolicy<>::member_type& team);

// Helper functions for vector shuffling
KOKKOS_FUNCTION
void device_shuffle_vector(
    const Kokkos::View<double*>& input,
    Kokkos::View<double*>& output,
    const int m1,
    const int m2,
    const Kokkos::TeamPolicy<>::member_type& team);

KOKKOS_FUNCTION
void device_unshuffle_vector(
    const Kokkos::View<double*>& input,
    Kokkos::View<double*>& output,
    const int m1,
    const int m2,
    const Kokkos::TeamPolicy<>::member_type& team);

// Test function
void test_a2_shuffled_kernel();

#endif