#pragma once

#include <Kokkos_Core.hpp>
#include "grid.hpp"
#include "coeff.hpp"

#include <iomanip>

void initialize_A0_values(
    Kokkos::View<double**>& values,
    const Grid& grid,
    const double rho,
    const double sigma
);

void initialize_A1_matrices(
    Kokkos::View<double**>& main_diags,
    Kokkos::View<double**>& lower_diags,
    Kokkos::View<double**>& upper_diags,
    const Grid& grid,
    const double rho,
    const double sigma,
    const double r_d,
    const double r_f
);

void initialize_A2_matrices(
    Kokkos::View<double*>& main_diag,
    Kokkos::View<double*>& lower_diag,
    Kokkos::View<double*>& upper_diag,
    Kokkos::View<double*>& upper2_diag,
    const Grid& grid,
    const double rho,
    const double sigma,
    const double r_d,
    const double kappa,
    const double eta
);

//impliccit
void build_A1_implicit(
    Kokkos::View<double**>& impl_main,
    Kokkos::View<double**>& impl_lower,
    Kokkos::View<double**>& impl_upper,
    const Kokkos::View<double**>& main,
    const Kokkos::View<double**>& lower,
    const Kokkos::View<double**>& upper,
    const double theta,
    const double delta_t,
    const int m1,
    const int m2
);

void build_A2_implicit(
    Kokkos::View<double*>& impl_main,
    Kokkos::View<double*>& impl_lower,
    Kokkos::View<double*>& impl_upper,
    Kokkos::View<double*>& impl_upper2,
    const Kokkos::View<double*>& main,
    const Kokkos::View<double*>& lower,
    const Kokkos::View<double*>& upper,
    const Kokkos::View<double*>& upper2,
    const double theta,
    const double delta_t,
    const int m1,
    const int m2
);

void build_boundary_conditions(
    Kokkos::View<double*> b0,
    Kokkos::View<double*> b1,
    Kokkos::View<double*> b2,
    Kokkos::View<double*> b,
    const int m1,
    const int m2,
    const int m,
    const double r_d,
    const double r_f,
    const Grid& grid,
    const int N,
    const double delta_t
);

double DO_scheme_optimized(
    const int m1, const int m2,        // Grid dimensions
    const int N,                       // Number of time steps
    const double delta_t,              // Time step size
    const double theta,                // Weight parameter
    const double r_f,                  // Foreign interest rate
    const Grid& grid,                  // Grid information
    const double rho,                  // Correlation coefficient
    const double sigma,                // Volatility of variance
    const double r_d,                  // Domestic interest rate 
    const double kappa,                // Mean reversion rate
    const double eta,                  // Long-term variance
    const double S_0,                  // Initial stock price
    const double V_0                   // Initial variance
);

void test_DO_scheme_optimized();



