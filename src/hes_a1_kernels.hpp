// In hes_a1_kernels.hpp
#ifndef HES_A1_KERNELS_HPP
#define HES_A1_KERNELS_HPP

#include <Kokkos_Core.hpp>
#include "grid.hpp"

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

//name is missleading, we are just parallising in v
template<
    class View2D_const_main,  // e.g. Kokkos::View<const double**, LayoutStride, ...>
    class View2D_const_lower,
    class View2D_const_upper,
    class View1D_x,           // e.g. Kokkos::View<double*, ...>
    class View1D_result>
KOKKOS_FUNCTION
void a1_device_multiply_parallel_v(
    const View2D_const_main&  main_diag,
    const View2D_const_lower& lower_diag,
    const View2D_const_upper& upper_diag,
    const View1D_x&           x,
    const View1D_result&      result,
    const Kokkos::TeamPolicy<>::member_type& team);

template<
    class View2D_const_main,
    class View2D_const_lower,
    class View2D_const_upper,
    class View1D_x,     // x is 1D
    class View2D_temp,  // temp is rank-2
    class View1D_b>
KOKKOS_FUNCTION
void a1_device_solve_implicit_parallel_v(
    const View2D_const_main&  impl_main,
    const View2D_const_lower& impl_lower,
    const View2D_const_upper& impl_upper,
    const View1D_x&           x,
    const View2D_temp&        temp,
    const View1D_b&           b,
    const Kokkos::TeamPolicy<>::member_type& team);

void test_a1_kernel();

#endif