#ifndef HES_A2_SHUFFLED_KERNELS_HPP
#define HES_A2_SHUFFLED_KERNELS_HPP

#include <Kokkos_Core.hpp>
#include "grid.hpp"

// Build the A2 matrix diagonals on device
KOKKOS_FUNCTION
void build_a2_diagonals_shuffled(
    const Kokkos::View<double**>& main_diag,
    const Kokkos::View<double**>& lower_diag,
    const Kokkos::View<double**>& lower2_diag,
    const Kokkos::View<double**>& upper_diag,
    const Kokkos::View<double**>& upper2_diag,
    const Kokkos::View<double**>& impl_main_diag,
    const Kokkos::View<double**>& impl_lower_diag,
    const Kokkos::View<double**>& impl_lower2_diag,
    const Kokkos::View<double**>& impl_upper_diag,
    const Kokkos::View<double**>& impl_upper2_diag,
    const Grid& grid,
    const double theta,
    const double dt,
    const double r_d,
    const double kappa,
    const double eta,
    const double sigma,
    const Kokkos::TeamPolicy<>::member_type& team);

// Device multiply operation
KOKKOS_FUNCTION
void device_multiply_shuffled(
    const Kokkos::View<const double**>& main_diag,
    const Kokkos::View<const double**>& lower_diag,
    const Kokkos::View<const double**>& lower2_diag,
    const Kokkos::View<const double**>& upper_diag,
    const Kokkos::View<const double**>& upper2_diag,
    const Kokkos::View<double*>& x,
    Kokkos::View<double*>& result,
    const Kokkos::TeamPolicy<>::member_type& team);

// Device implicit solve
KOKKOS_FUNCTION
void device_solve_implicit_shuffled(
    const Kokkos::View<const double**>& impl_main_diag,
    const Kokkos::View<const double**>& impl_lower_diag,
    const Kokkos::View<const double**>& impl_lower2_diag,
    const Kokkos::View<const double**>& impl_upper_diag,
    const Kokkos::View<const double**>& impl_upper2_diag,
    const Kokkos::View<double*>& x,
    const Kokkos::View<double**>& c_prime,
    const Kokkos::View<double**>& c2_prime,
    const Kokkos::View<double**>& d_prime,
    const Kokkos::View<double*>& b,
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