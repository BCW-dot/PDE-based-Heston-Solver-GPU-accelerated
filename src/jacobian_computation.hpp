#pragma once

#include <vector>
#include <Kokkos_Core.hpp>
#include "device_solver.hpp"

//Generates Black Scholes Call Option prices
void generate_market_data(
    const double S_0,          // Spot price
    const double T,            // Time to maturity
    const double r_d,          // Risk-free rate
    const std::vector<double>& strikes,  // Array of strikes
    Kokkos::View<double*>::HostMirror& h_market_prices  // Output market prices on host
);

//Solves an implicit 5x5 matrix on the gpu
void solve_5x5_device(
    const Kokkos::View<double**> &A_device,  // shape (5,5)
    const Kokkos::View<double*>  &b_device,  // shape (5)
    const Kokkos::View<double*>  &x_device   // shape (5)
);

//Perfoms the LVBM steps
void compute_parameter_update_on_device(
    const Kokkos::View<double**>& J,        // [num_data x 5]
    const Kokkos::View<double*>&  residual, // [num_data]
    const double                  lambda,
    Kokkos::View<double*>&        delta     // [5]
);

//Computes the Jacobian in parallel in rows, but sequentially in columns
void compute_jacobian(
    // Market/model parameters
    const double S_0, const double V_0, const double T,
    const double r_d, const double r_f,
    const double rho, const double sigma, const double kappa, const double eta,
    // Numerical parameters
    const int m1, const int m2, const int total_size, const int N, const double theta, const double delta_t,
    // Pre-computed data structures
    const int num_strikes,
    const Kokkos::View<Device_A0_heston<Kokkos::DefaultExecutionSpace>*>& A0_solvers,
    const Kokkos::View<Device_A1_heston<Kokkos::DefaultExecutionSpace>*>& A1_solvers,
    const Kokkos::View<Device_A2_shuffled_heston<Kokkos::DefaultExecutionSpace>*>& A2_solvers,
    const Kokkos::View<Device_BoundaryConditions<Kokkos::DefaultExecutionSpace>*>& bounds_d,
    const Kokkos::View<GridViews*>& deviceGrids,
    const Kokkos::View<double**>& U_0,
    DO_Workspace<Kokkos::DefaultExecutionSpace>& workspace,
    // Output matrix
    Kokkos::View<double**>& J,
    Kokkos::View<double*>& base_prices,
    // Optional: perturbation size
    const double eps = 1e-6
);

//computes the option prices in parallel for different strikes
void compute_base_prices(
    // Market/model parameters
    const double S_0, const double V_0, const double T,
    const double r_d, const double r_f,
    const double rho, const double sigma, const double kappa, const double eta,
    // Numerical parameters
    const int m1, const int m2, const int total_size, const int N, const double theta, const double delta_t,
    // Pre-computed data structures
    const int num_strikes,
    const Kokkos::View<Device_A0_heston<Kokkos::DefaultExecutionSpace>*>& A0_solvers,
    const Kokkos::View<Device_A1_heston<Kokkos::DefaultExecutionSpace>*>& A1_solvers,
    const Kokkos::View<Device_A2_shuffled_heston<Kokkos::DefaultExecutionSpace>*>& A2_solvers,
    const Kokkos::View<Device_BoundaryConditions<Kokkos::DefaultExecutionSpace>*>& bounds_d,
    const Kokkos::View<GridViews*>& deviceGrids,
    DO_Workspace<Kokkos::DefaultExecutionSpace>& workspace,
    Kokkos::View<double*>& base_prices
);


void compute_jacobian_american(
    // Market/model parameters
    const double S_0, const double V_0, const double T,
    const double r_d, const double r_f,
    const double rho, const double sigma, const double kappa, const double eta,
    // Numerical parameters
    const int m1, const int m2, const int total_size, const int N, const double theta, const double delta_t,
    // Pre-computed data structures
    const int num_strikes,
    const Kokkos::View<Device_A0_heston<Kokkos::DefaultExecutionSpace>*>& A0_solvers,
    const Kokkos::View<Device_A1_heston<Kokkos::DefaultExecutionSpace>*>& A1_solvers,
    const Kokkos::View<Device_A2_shuffled_heston<Kokkos::DefaultExecutionSpace>*>& A2_solvers,
    const Kokkos::View<Device_BoundaryConditions<Kokkos::DefaultExecutionSpace>*>& bounds_d,
    const Kokkos::View<GridViews*>& deviceGrids,
    const Kokkos::View<double**>& U_0,
    DO_Workspace<Kokkos::DefaultExecutionSpace>& workspace,
    // Output matrix
    Kokkos::View<double**>& J,
    Kokkos::View<double*>& base_prices,
    // Optional: perturbation size
    const double eps
);

void compute_base_prices_american(
    // Market/model parameters
    // Market/model parameters
    const double S_0, const double V_0, const double T,
    const double r_d, const double r_f,
    const double rho, const double sigma, const double kappa, const double eta,
    // Numerical parameters
    const int m1, const int m2, const int total_size, const int N, const double theta, const double delta_t,
    // Pre-computed data structures
    const int num_strikes,
    const Kokkos::View<Device_A0_heston<Kokkos::DefaultExecutionSpace>*>& A0_solvers,
    const Kokkos::View<Device_A1_heston<Kokkos::DefaultExecutionSpace>*>& A1_solvers,
    const Kokkos::View<Device_A2_shuffled_heston<Kokkos::DefaultExecutionSpace>*>& A2_solvers,
    const Kokkos::View<Device_BoundaryConditions<Kokkos::DefaultExecutionSpace>*>& bounds_d,
    const Kokkos::View<GridViews*>& deviceGrids,
    const Kokkos::View<double**>& U_0,
    DO_Workspace<Kokkos::DefaultExecutionSpace>& workspace,
    Kokkos::View<double*>& base_prices
);

void compute_jacobian_dividends(
    // Market/model parameters
    const double S_0, const double V_0, const double T,
    const double r_d, const double r_f,
    const double rho, const double sigma, const double kappa, const double eta,
    // Numerical parameters
    const int m1, const int m2, const int total_size, const int N, const double theta, const double delta_t,
    // Pre-computed data structures
    const int num_strikes,
    const Kokkos::View<Device_A0_heston<Kokkos::DefaultExecutionSpace>*>& A0_solvers,
    const Kokkos::View<Device_A1_heston<Kokkos::DefaultExecutionSpace>*>& A1_solvers,
    const Kokkos::View<Device_A2_shuffled_heston<Kokkos::DefaultExecutionSpace>*>& A2_solvers,
    const Kokkos::View<Device_BoundaryConditions<Kokkos::DefaultExecutionSpace>*>& bounds_d,
    const Kokkos::View<GridViews*>& deviceGrids,
    const Kokkos::View<double**>& U_0,
    DO_Workspace<Kokkos::DefaultExecutionSpace>& workspace,
    //dividend specifics
    const int num_dividends,                        // Number of dividends
    const Kokkos::View<double*>& dividend_dates,    // Device view of dates
    const Kokkos::View<double*>& dividend_amounts,  // Device view of amounts
    const Kokkos::View<double*>& dividend_percentages,  // Device view of percentages
    // Output matrix
    Kokkos::View<double**>& J,
    Kokkos::View<double*>& base_prices,
    // Optional: perturbation size
    const double eps
);

void compute_base_prices_dividends(
    // Market/model parameters
    const double S_0, const double V_0, const double T,
    const double r_d, const double r_f,
    const double rho, const double sigma, const double kappa, const double eta,
    // Numerical parameters
    const int m1, const int m2, const int total_size, const int N, const double theta, const double delta_t,
    // Pre-computed data structures
    const int num_strikes,
    const Kokkos::View<Device_A0_heston<Kokkos::DefaultExecutionSpace>*>& A0_solvers,
    const Kokkos::View<Device_A1_heston<Kokkos::DefaultExecutionSpace>*>& A1_solvers,
    const Kokkos::View<Device_A2_shuffled_heston<Kokkos::DefaultExecutionSpace>*>& A2_solvers,
    const Kokkos::View<Device_BoundaryConditions<Kokkos::DefaultExecutionSpace>*>& bounds_d,
    const Kokkos::View<GridViews*>& deviceGrids,
    const Kokkos::View<double**>& U_0,
    DO_Workspace<Kokkos::DefaultExecutionSpace>& workspace,
    //dividend specifics
    const int num_dividends,                        // Number of dividends
    const Kokkos::View<double*>& dividend_dates,    // Device view of dates
    const Kokkos::View<double*>& dividend_amounts,  // Device view of amounts
    const Kokkos::View<double*>& dividend_percentages,  // Device view of percentages
    Kokkos::View<double*>& base_prices
);


void compute_jacobian_american_dividends(
    // Market/model parameters
    const double S_0, const double V_0, const double T,
    const double r_d, const double r_f,
    const double rho, const double sigma, const double kappa, const double eta,
    // Numerical parameters
    const int m1, const int m2, const int total_size, const int N, const double theta, const double delta_t,
    // Pre-computed data structures
    const int num_strikes,
    const Kokkos::View<Device_A0_heston<Kokkos::DefaultExecutionSpace>*>& A0_solvers,
    const Kokkos::View<Device_A1_heston<Kokkos::DefaultExecutionSpace>*>& A1_solvers,
    const Kokkos::View<Device_A2_shuffled_heston<Kokkos::DefaultExecutionSpace>*>& A2_solvers,
    const Kokkos::View<Device_BoundaryConditions<Kokkos::DefaultExecutionSpace>*>& bounds_d,
    const Kokkos::View<GridViews*>& deviceGrids,
    const Kokkos::View<double**>& U_0,
    DO_Workspace<Kokkos::DefaultExecutionSpace>& workspace,
    //dividend specifics
    const int num_dividends,                        // Number of dividends
    const Kokkos::View<double*>& dividend_dates,    // Device view of dates
    const Kokkos::View<double*>& dividend_amounts,  // Device view of amounts
    const Kokkos::View<double*>& dividend_percentages,  // Device view of percentages
    // Output matrix
    Kokkos::View<double**>& J,
    Kokkos::View<double*>& base_prices,
    // Optional: perturbation size
    const double eps
);

void compute_base_prices_american_dividends(
    // Market/model parameters
    const double S_0, const double V_0, const double T,
    const double r_d, const double r_f,
    const double rho, const double sigma, const double kappa, const double eta,
    // Numerical parameters
    const int m1, const int m2, const int total_size, const int N, const double theta, const double delta_t,
    // Pre-computed data structures
    const int num_strikes,
    const Kokkos::View<Device_A0_heston<Kokkos::DefaultExecutionSpace>*>& A0_solvers,
    const Kokkos::View<Device_A1_heston<Kokkos::DefaultExecutionSpace>*>& A1_solvers,
    const Kokkos::View<Device_A2_shuffled_heston<Kokkos::DefaultExecutionSpace>*>& A2_solvers,
    const Kokkos::View<Device_BoundaryConditions<Kokkos::DefaultExecutionSpace>*>& bounds_d,
    const Kokkos::View<GridViews*>& deviceGrids,
    const Kokkos::View<double**>& U_0,
    DO_Workspace<Kokkos::DefaultExecutionSpace>& workspace,
    //dividend specifics
    const int num_dividends,                        // Number of dividends
    const Kokkos::View<double*>& dividend_dates,    // Device view of dates
    const Kokkos::View<double*>& dividend_amounts,  // Device view of amounts
    const Kokkos::View<double*>& dividend_percentages,  // Device view of percentages
    Kokkos::View<double*>& base_prices
);



void test_jacobian_computation();
