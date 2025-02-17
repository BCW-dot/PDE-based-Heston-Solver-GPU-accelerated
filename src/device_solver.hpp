#pragma once
#include <Kokkos_Core.hpp>

#include "hes_a0_kernels.hpp"
#include "hes_a1_kernels.hpp"
#include "hes_a2_shuffled_kernels.hpp"
#include "hes_boundary_kernels.hpp"

#include "DO_solver_workspace.hpp"

/*
template<class Device>
struct DO_Workspace {
    // Main solution arrays
    Kokkos::View<double**, Device> U;
    Kokkos::View<double**, Device> Y_0;
    Kokkos::View<double**, Device> Y_1;
    
    // Results arrays
    Kokkos::View<double**, Device> A0_result;
    Kokkos::View<double**, Device> A1_result;
    Kokkos::View<double**, Device> A2_result_unshuf;
    
    // Shuffled arrays
    Kokkos::View<double**, Device> U_shuffled;
    Kokkos::View<double**, Device> Y_1_shuffled;
    Kokkos::View<double**, Device> A2_result_shuffled;
    Kokkos::View<double**, Device> U_next_shuffled;

    DO_Workspace(int nInstances, int total_size) {
        U = Kokkos::View<double**, Device>("U", nInstances, total_size);
        Y_0 = Kokkos::View<double**, Device>("Y_0", nInstances, total_size);
        Y_1 = Kokkos::View<double**, Device>("Y_1", nInstances, total_size);
        A0_result = Kokkos::View<double**, Device>("A0_result", nInstances, total_size);
        A1_result = Kokkos::View<double**, Device>("A1_result", nInstances, total_size);
        A2_result_unshuf = Kokkos::View<double**, Device>("A2_result_unshuf", nInstances, total_size);
        U_shuffled = Kokkos::View<double**, Device>("U_shuffled", nInstances, total_size);
        Y_1_shuffled = Kokkos::View<double**, Device>("Y_1_shuffled", nInstances, total_size);
        A2_result_shuffled = Kokkos::View<double**, Device>("A2_result_shuffled", nInstances, total_size);
        U_next_shuffled = Kokkos::View<double**, Device>("U_next_shuffled", nInstances, total_size);
    }
};
*/


/*

Fixed parameters, parallising over strikes

*/

template<class Device>
void parallel_DO_solve(
    // Grid dimensions
    const int nInstances,
    const int S_0,
    const int V_0,
    const int m1,
    const int m2,
    // Time discretization
    const int N,
    const double T,
    const double delta_t,
    const double theta,
    // Heston parameters
    const double r_d,
    const double r_f,
    const double rho,
    const double sigma,
    const double kappa,
    const double eta,
    // Problem components
    const Kokkos::View<Device_A0_heston<Device>*>& A0_solvers,
    const Kokkos::View<Device_A1_heston<Device>*>& A1_solvers,
    const Kokkos::View<Device_A2_shuffled_heston<Device>*>& A2_solvers,
    const Kokkos::View<Device_BoundaryConditions<Device>*>& bounds_d,
    const Kokkos::View<GridViews*>& deviceGrids,
    DO_Workspace<Device>& workspace,
    Kokkos::View<double*>& base_prices) {  // Now just pass the workspace

    const int total_size = (m1+1)*(m2+1);

    using team_policy = Kokkos::TeamPolicy<>;
    team_policy policy(nInstances, Kokkos::AUTO);

    Kokkos::parallel_for("DO_scheme", policy,
        KOKKOS_LAMBDA(const team_policy::member_type& team) {
            const int instance = team.league_rank();
            
            // Get subviews from workspace
            auto U_i = Kokkos::subview(workspace.U, instance, Kokkos::ALL);
            auto Y_0_i = Kokkos::subview(workspace.Y_0, instance, Kokkos::ALL);
            auto Y_1_i = Kokkos::subview(workspace.Y_1, instance, Kokkos::ALL);
            auto A0_result_i = Kokkos::subview(workspace.A0_result, instance, Kokkos::ALL);
            auto A1_result_i = Kokkos::subview(workspace.A1_result, instance, Kokkos::ALL);
            auto A2_result_unshuf_i = Kokkos::subview(workspace.A2_result_unshuf, instance, Kokkos::ALL);
            
            auto U_shuffled_i = Kokkos::subview(workspace.U_shuffled, instance, Kokkos::ALL);
            auto Y_1_shuffled_i = Kokkos::subview(workspace.Y_1_shuffled, instance, Kokkos::ALL);
            auto A2_result_shuffled_i = Kokkos::subview(workspace.A2_result_shuffled, instance, Kokkos::ALL);
            auto U_next_shuffled_i = Kokkos::subview(workspace.U_next_shuffled, instance, Kokkos::ALL);

            //Inits Grid views
            GridViews grid_i = deviceGrids(instance);

            //grid_i.rebuild_variance_views(V_0, 5.0, 5.0/500, team);


            // Initialize boundaries AFTER the Grids are initilized 
            bounds_d(instance).initialize(grid_i, team);
            auto bounds = bounds_d(instance);
            
            // Build matrices
            A0_solvers(instance).build_matrix(grid_i, rho, sigma, team);
            A1_solvers(instance).build_matrix(grid_i, r_d, r_f, theta, delta_t, team);
            A2_solvers(instance).build_matrix(grid_i, r_d, kappa, eta, sigma, theta, delta_t, team);

            for(int n = 1; n <= N; n++) {
                // Step 1: Y0 computation
                A0_solvers(instance).multiply_parallel_s_and_v(U_i, A0_result_i, team);
                A1_solvers(instance).multiply_parallel_v(U_i, A1_result_i, team);
                
                device_shuffle_vector(U_i, U_shuffled_i, m1, m2, team);
                A2_solvers(instance).multiply_parallel_s(U_shuffled_i, A2_result_shuffled_i, team);
                device_unshuffle_vector(A2_result_shuffled_i, A2_result_unshuf_i, m1, m2, team);

                // Y0 computation with boundary terms
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, total_size), 
                    [&](const int i) {
                        double exp_factor = std::exp(r_f * delta_t * (n-1));
                        Y_0_i(i) = U_i(i) + delta_t * (A0_result_i(i) + A1_result_i(i) + 
                                  A2_result_unshuf_i(i) + bounds.b_(i) * exp_factor);
                });

                // Step 2: A1 implicit solve
                A1_solvers(instance).multiply_parallel_v(U_i, A1_result_i, team);
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, total_size),
                    [&](const int i) {
                        double exp_factor_n = std::exp(r_f * delta_t * n);
                        double exp_factor_nm1 = std::exp(r_f * delta_t * (n-1));
                        Y_0_i(i) = Y_0_i(i) + theta * delta_t * (bounds.b1_(i) * exp_factor_n - 
                                  (A1_result_i(i) + bounds.b1_(i) * exp_factor_nm1));
                });
                A1_solvers(instance).solve_implicit_parallel_v(Y_1_i, Y_0_i, team);

                // Step 3: A2 shuffled implicit solve
                device_shuffle_vector(U_i, U_shuffled_i, m1, m2, team);
                A2_solvers(instance).multiply_parallel_s(U_shuffled_i, A2_result_shuffled_i, team);
                device_unshuffle_vector(A2_result_shuffled_i, A2_result_unshuf_i, m1, m2, team);

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, total_size),
                    [&](const int i) {
                        double exp_factor_n = std::exp(r_f * delta_t * n);
                        double exp_factor_nm1 = std::exp(r_f * delta_t * (n-1));
                        Y_1_i(i) = Y_1_i(i) + theta * delta_t * (bounds.b2_(i) * exp_factor_n - 
                                  (A2_result_unshuf_i(i) + bounds.b2_(i) * exp_factor_nm1));
                });

                device_shuffle_vector(Y_1_i, Y_1_shuffled_i, m1, m2, team);
                A2_solvers(instance).solve_implicit_parallel_s(U_next_shuffled_i, Y_1_shuffled_i, team);
                device_unshuffle_vector(U_next_shuffled_i, U_i, m1, m2, team);
            }
            // Find s index
            int index_s = -1;
            int index_v = -1;
            for(int i = 0; i <= m1; i++) {
                if(Kokkos::abs(grid_i.device_Vec_s(i) - S_0) < 1e-10) {
                    index_s = i;  // Store s index
                    break;
                }
            }

            for(int i = 0; i <= m2; i++) {
                if(Kokkos::abs(grid_i.device_Vec_v(i) - V_0) < 1e-10) {
                    index_v = i;  // Store s index
                    break;
                }
            }

            // Now you can use these indices directly
            const double base_price = U_i(index_s + index_v*(m1+1));
            base_prices(instance) = base_price;
    });
    Kokkos::fence();
}



/*

This is a device callable time stepping of the heston pde

*/
template<class Device, class ViewType>  // Add ViewType template parameter
KOKKOS_FUNCTION 
void device_DO_timestepping(
    // Grid dimensions
    const int m1,
    const int m2,
    // Time discretization
    const int N,
    const double delta_t,
    const double theta,
    const double r_f,
    // Problem components
    Device_A0_heston<Device>& A0,
    Device_A1_heston<Device>& A1,
    Device_A2_shuffled_heston<Device>& A2,
    const Device_BoundaryConditions<Device>& bounds,
    // Workspace views for this instance - now using ViewType
    ViewType& U_i,
    ViewType& Y_0_i,
    ViewType& Y_1_i,
    ViewType& A0_result_i,
    ViewType& A1_result_i,
    ViewType& A2_result_unshuf_i,
    ViewType& U_shuffled_i,
    ViewType& Y_1_shuffled_i,
    ViewType& A2_result_shuffled_i,
    ViewType& U_next_shuffled_i,
    // Team handle
    const typename Kokkos::TeamPolicy<>::member_type& team
) {
    const int total_size = (m1+1)*(m2+1);

    for(int n = 1; n <= N; n++) {
        // Step 1: Y0 computation
        A0.multiply_parallel_s_and_v(U_i, A0_result_i, team);
        A1.multiply_parallel_v(U_i, A1_result_i, team);
        
        device_shuffle_vector(U_i, U_shuffled_i, m1, m2, team);
        A2.multiply_parallel_s(U_shuffled_i, A2_result_shuffled_i, team);
        device_unshuffle_vector(A2_result_shuffled_i, A2_result_unshuf_i, m1, m2, team);

        // Y0 computation with boundary terms
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, total_size), 
            [&](const int i) {
                double exp_factor = std::exp(r_f * delta_t * (n-1));
                Y_0_i(i) = U_i(i) + delta_t * (A0_result_i(i) + A1_result_i(i) + 
                          A2_result_unshuf_i(i) + bounds.b_(i) * exp_factor);
            });

        // Step 2: A1 implicit solve
        A1.multiply_parallel_v(U_i, A1_result_i, team);
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, total_size),
            [&](const int i) {
                double exp_factor_n = std::exp(r_f * delta_t * n);
                double exp_factor_nm1 = std::exp(r_f * delta_t * (n-1));
                Y_0_i(i) = Y_0_i(i) + theta * delta_t * (bounds.b1_(i) * exp_factor_n - 
                          (A1_result_i(i) + bounds.b1_(i) * exp_factor_nm1));
            });
        A1.solve_implicit_parallel_v(Y_1_i, Y_0_i, team);

        // Step 3: A2 shuffled implicit solve
        device_shuffle_vector(U_i, U_shuffled_i, m1, m2, team);
        A2.multiply_parallel_s(U_shuffled_i, A2_result_shuffled_i, team);
        device_unshuffle_vector(A2_result_shuffled_i, A2_result_unshuf_i, m1, m2, team);

        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, total_size),
            [&](const int i) {
                double exp_factor_n = std::exp(r_f * delta_t * n);
                double exp_factor_nm1 = std::exp(r_f * delta_t * (n-1));
                Y_1_i(i) = Y_1_i(i) + theta * delta_t * (bounds.b2_(i) * exp_factor_n - 
                          (A2_result_unshuf_i(i) + bounds.b2_(i) * exp_factor_nm1));
            });

        device_shuffle_vector(Y_1_i, Y_1_shuffled_i, m1, m2, team);
        A2.solve_implicit_parallel_s(U_next_shuffled_i, Y_1_shuffled_i, team);
        device_unshuffle_vector(U_next_shuffled_i, U_i, m1, m2, team);
    }
}


/*

Device callable american solver

*/
template<class Device, class ViewType>  // Add ViewType template parameter
KOKKOS_FUNCTION 
void device_DO_timestepping_american(
    // Grid dimensions
    const int m1,
    const int m2,
    // Time discretization
    const int N,
    const double delta_t,
    const double theta,
    const double r_f,
    // Problem components
    Device_A0_heston<Device>& A0,
    Device_A1_heston<Device>& A1,
    Device_A2_shuffled_heston<Device>& A2,
    const Device_BoundaryConditions<Device>& bounds,
    // Workspace views for this instance - now using ViewType
    ViewType& U_i,
    ViewType& Y_0_i,
    ViewType& Y_1_i,
    ViewType& A0_result_i,
    ViewType& A1_result_i,
    ViewType& A2_result_unshuf_i,
    ViewType& U_shuffled_i,
    ViewType& Y_1_shuffled_i,
    ViewType& A2_result_shuffled_i,
    ViewType& U_next_shuffled_i,
    //american specifics
    ViewType& lambda_bar_i,
    const ViewType& U_0_i,
    // Team handle
    const typename Kokkos::TeamPolicy<>::member_type& team
) {
    const int total_size = (m1+1)*(m2+1);

     //Need to reset the lambda function to zero 
     Kokkos::parallel_for(Kokkos::TeamThreadRange(team, total_size),
        [&](const int i) {
        lambda_bar_i(i) = 0;
    });

    for(int n = 1; n <= N; n++) {
        // Step 1: Y0 computation
        A0.multiply_parallel_s_and_v(U_i, A0_result_i, team);
        A1.multiply_parallel_v(U_i, A1_result_i, team);
        
        device_shuffle_vector(U_i, U_shuffled_i, m1, m2, team);
        A2.multiply_parallel_s(U_shuffled_i, A2_result_shuffled_i, team);
        device_unshuffle_vector(A2_result_shuffled_i, A2_result_unshuf_i, m1, m2, team);

        // Y0 computation with boundary terms
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, total_size), 
            [&](const int i) {
                double exp_factor = std::exp(r_f * delta_t * (n-1));
                Y_0_i(i) = U_i(i) + delta_t * (A0_result_i(i) + A1_result_i(i) + 
                          A2_result_unshuf_i(i) + bounds.b_(i) * exp_factor + 
                          lambda_bar_i(i));  // Add lambda contribution
            });

        // Step 2: A1 implicit solve
        A1.multiply_parallel_v(U_i, A1_result_i, team);
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, total_size),
            [&](const int i) {
                double exp_factor_n = std::exp(r_f * delta_t * n);
                double exp_factor_nm1 = std::exp(r_f * delta_t * (n-1));
                Y_0_i(i) = Y_0_i(i) + theta * delta_t * (bounds.b1_(i) * exp_factor_n - 
                          (A1_result_i(i) + bounds.b1_(i) * exp_factor_nm1));
            });
        A1.solve_implicit_parallel_v(Y_1_i, Y_0_i, team);

        // Step 3: A2 shuffled implicit solve
        device_shuffle_vector(U_i, U_shuffled_i, m1, m2, team);
        A2.multiply_parallel_s(U_shuffled_i, A2_result_shuffled_i, team);
        device_unshuffle_vector(A2_result_shuffled_i, A2_result_unshuf_i, m1, m2, team);

        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, total_size),
            [&](const int i) {
                double exp_factor_n = std::exp(r_f * delta_t * n);
                double exp_factor_nm1 = std::exp(r_f * delta_t * (n-1));
                Y_1_i(i) = Y_1_i(i) + theta * delta_t * (bounds.b2_(i) * exp_factor_n - 
                          (A2_result_unshuf_i(i) + bounds.b2_(i) * exp_factor_nm1));
            });

        device_shuffle_vector(Y_1_i, Y_1_shuffled_i, m1, m2, team);
        A2.solve_implicit_parallel_s(U_next_shuffled_i, Y_1_shuffled_i, team);
        device_unshuffle_vector(U_next_shuffled_i, U_i, m1, m2, team);

        // American option early exercise check and lambda update
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, total_size),
            [&](const int i) {
            // Apply early exercise condition
            const double U_bar = U_i(i);
            U_i(i) = Kokkos::max(U_bar - delta_t * lambda_bar_i(i), U_0_i(i));

            // Update lambda multiplier
            lambda_bar_i(i) = Kokkos::max(0.0, lambda_bar_i(i) + (U_0_i(i) - U_bar) / delta_t);

            // Set lambda to zero at S_max for all variance levels
            // Every (m1+1)th entry starting at index m1 is an S_max entry
            if(i % (m1 + 1) == m1) {  // This hits exactly S_max entries at all variance levels
                lambda_bar_i(i) = 0.0;
            }
        });
    }
}


/*

Device callable solver with dividends

*/
template<class Device, class ViewType>  // Add ViewType template parameter
KOKKOS_FUNCTION 
void device_DO_timestepping_dividend(
    // Grid dimensions
    const int m1,
    const int m2,
    // Time discretization
    const int N,
    const double delta_t,
    const double theta,
    const double r_f,
    // Problem components
    Device_A0_heston<Device>& A0,
    Device_A1_heston<Device>& A1,
    Device_A2_shuffled_heston<Device>& A2,
    const Device_BoundaryConditions<Device>& bounds,
    // Workspace views for this instance - now using ViewType
    ViewType& U_i,
    ViewType& Y_0_i,
    ViewType& Y_1_i,
    ViewType& A0_result_i,
    ViewType& A1_result_i,
    ViewType& A2_result_unshuf_i,
    ViewType& U_shuffled_i,
    ViewType& Y_1_shuffled_i,
    ViewType& A2_result_shuffled_i,
    ViewType& U_next_shuffled_i,
    //dividend specifics
    const int num_dividends,                        // Number of dividends
    const Kokkos::View<double*>& dividend_dates,    // Device view of dates
    const Kokkos::View<double*>& dividend_amounts,  // Device view of amounts
    const Kokkos::View<double*>& dividend_percentages,  // Device view of percentages
    const ViewType& device_Vec_s_i,                         // Stock price grid
    ViewType& U_temp_i,  // Temporary storage passed in from workspace
    // Team handle
    const typename Kokkos::TeamPolicy<>::member_type& team
) { 
    const int total_size = (m1+1)*(m2+1);
int current_div_idx = 0;

for(int n = 1; n <= N; n++) {
    double t = n * delta_t;

    // One-thread logic to check for dividend
    int dividend_int = 0;
    if (team.team_rank() == 0) {
        // Use exactly the same condition as sequential version
        dividend_int = (current_div_idx < num_dividends && 
                       t <= dividend_dates(current_div_idx) && 
                       dividend_dates(current_div_idx) < (n+1) * delta_t) ? 1 : 0;
    }
    // Broadcast the decision to all threads
    team.team_broadcast(dividend_int, 0);
    team.team_barrier();

    if(dividend_int) {
        // First, parallel copy of current solution
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, total_size),
        [&](const int i) {
            U_temp_i(i) = U_i(i);
        });
        team.team_barrier();

        // Get dividend parameters
        const double div_amount = dividend_amounts(current_div_idx);
        const double div_percentage = dividend_percentages(current_div_idx);

        // Process dividend adjustment in parallel over variance levels
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, m2+1),
            [&](const int j) {
                const int offset = j * (m1 + 1);
                
                // For each stock price level - keep sequential loop as in original
                for(int i = 0; i <= m1; i++) {
                    double old_s = device_Vec_s_i(i);
                    double new_s = old_s * (1.0 - div_percentage) - div_amount;

                    if(new_s > 0) {
                        // Keep linear search as in sequential version
                        int idx = 0;
                        for(int k = 0; k <= m1; k++) {
                            if(device_Vec_s_i(k) > new_s) {
                                idx = k;
                                break;
                            }
                        }

                        if(idx > 0 && idx < m1 + 1) {
                            // Interpolate
                            double s_low = device_Vec_s_i(idx-1);
                            double s_high = device_Vec_s_i(idx);
                            double weight = (new_s - s_low) / (s_high - s_low);
                            
                            double val_low = U_temp_i(offset + idx-1);
                            double val_high = U_temp_i(offset + idx);
                            U_i(offset + i) = (1.0 - weight) * val_low + weight * val_high;
                        }
                        else if(idx == 0) {
                            U_i(offset + i) = U_temp_i(offset);
                        }
                        else {
                            U_i(offset + i) = U_temp_i(offset + m1);
                        }
                    }
                    else {
                        U_i(offset + i) = 0.0;
                    }
                }
            });
        team.team_barrier();

        // Increment dividend index (single thread)
        if(team.team_rank() == 0) {
            current_div_idx++;
        }
        team.team_barrier();
    }
    
    //parallel handling of dividents. Almost works, some small mistake
    /*
    const int total_size = (m1+1)*(m2+1);

    int current_div_idx = 0;

    for(int n = 1; n <= N; n++) {
        double t = n * delta_t;

        //one thread determines if it is a divident date. 
        //Then broadcast the bool to all other threads
        // One-thread logic
        int dividend_int = 0;
        if (team.team_rank() == 0) {
        // Evaluate the condition on one thread
        dividend_int = (current_div_idx < num_dividends &&
                        t <= dividend_dates(current_div_idx) &&
                        dividend_dates(current_div_idx) < (n+1)*delta_t)
                        ? 1 : 0;
        }
        // Broadcast the int
        // Broadcast in place; modifies dividend_int in-place
        team.team_broadcast(dividend_int, 0);
        team.team_barrier();

        // Now, all threads have the correct value in dividend_int
        bool process_dividend = (dividend_int != 0);

        if(process_dividend) {

            // All threads process dividend
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, total_size),
            [&](const int i) {
                U_temp_i(i) = U_i(i);
            });
            team.team_barrier();

            double div_date       = dividend_dates(current_div_idx);
            double div_amount     = dividend_amounts(current_div_idx);
            double div_percentage = dividend_percentages(current_div_idx);

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, m2+1),
                [&](const int j) {
                    const int offset = j * (m1 + 1);
                    
                    // For each stock price level
                    for(int i = 0; i <= m1; i++) {
                        double old_s = device_Vec_s_i(i);
                        double new_s = old_s * (1.0 - div_percentage) - div_amount;
                        
                        if(new_s > 0) {
                            // Find interpolation points
                            int idx = 0;
                            for(int k = 0; k <= m1; k++) {
                                if(device_Vec_s_i(k) > new_s) {
                                    idx = k;
                                    break;
                                }
                            }

                            if(idx > 0 && idx < m1 + 1) {
                                // Interpolate
                                double s_low = device_Vec_s_i(idx-1);
                                double s_high = device_Vec_s_i(idx);
                                double weight = (new_s - s_low) / (s_high - s_low);
                                
                                double val_low = U_temp_i(offset + idx-1);
                                double val_high = U_temp_i(offset + idx);
                                U_i(offset + i) = (1.0 - weight) * val_low + weight * val_high;
                            }
                            else if(idx == 0) {
                                // Left extrapolation
                                U_i(offset + i) = U_temp_i(offset);
                            }
                            else {
                                // Right extrapolation
                                U_i(offset + i) = U_temp_i(offset + m1);
                            }
                        }
                        else {
                            U_i(offset + i) = 0.0;
                        }
                    }
                });
        }
        team.team_barrier();

        // Update index separately
        if(team.team_rank() == 0 && 
           current_div_idx < num_dividends && 
           t > dividend_dates(current_div_idx)) {
            if(team.league_rank() == 0) {
                printf("Incrementing div_idx from %d to %d at t = %.4f\n", 
                       current_div_idx, current_div_idx + 1, t);
            }
            current_div_idx++;
        }
        team.team_barrier();
        */
        

    //This works, but is sequential in divident handling
    /*
    const int total_size = (m1+1)*(m2+1);

    int current_div_idx = 0;

    for(int n = 1; n <= N; n++) {
        double t = n * delta_t;
        
        // Check for dividends - only one thread per team checks
        if (team.team_rank() == 0) {
            // Process only if there are still dividends left
            int div_idx = current_div_idx;
            if (div_idx < num_dividends) {
                // Check if we're in the dividend window for the current dividend
                if (t <= dividend_dates(div_idx) &&
                    dividend_dates(div_idx) < (n+1) * delta_t)
                {
                    // Get current dividend values
                    double div_date       = dividend_dates(div_idx);
                    double div_amount     = dividend_amounts(div_idx);
                    double div_percentage = dividend_percentages(div_idx);
        
                    // 1) Copy current solution U_i -> U_temp_i in a *sequential* double loop
                    for (int j = 0; j <= m2; j++) {
                        int offset = j * (m1 + 1);
                        for (int i = 0; i <= m1; i++) {
                            U_temp_i(offset + i) = U_i(offset + i);
                        }
                    }
        
                    // 2) Process dividend adjustment in another *sequential* double loop
                    for (int j = 0; j <= m2; j++) {
                        int offset = j * (m1 + 1);
                        for (int i = 0; i <= m1; i++) {
                            double old_s = device_Vec_s_i(i);
                            double new_s = old_s * (1.0 - div_percentage) - div_amount;
        
                            if (new_s > 0.0) {
                                // Find interpolation points
                                int idx = 0;
                                for (int k = 0; k <= m1; k++) {
                                    if (device_Vec_s_i(k) > new_s) {
                                        idx = k;
                                        break;
                                    }
                                }
        
                                if (idx > 0 && idx < m1 + 1) {
                                    // Interpolate
                                    double s_low  = device_Vec_s_i(idx - 1);
                                    double s_high = device_Vec_s_i(idx);
                                    double weight = (new_s - s_low) / (s_high - s_low);
        
                                    double val_low  = U_temp_i(offset + idx - 1);
                                    double val_high = U_temp_i(offset + idx);
        
                                    U_i(offset + i) = (1.0 - weight) * val_low
                                                    + weight * val_high;
                                }
                                else if (idx == 0) {
                                    U_i(offset + i) = U_temp_i(offset + 0);
                                }
                                else {
                                    U_i(offset + i) = U_temp_i(offset + m1);
                                }
                            } 
                            else {
                                U_i(offset + i) = 0.0;
                            }
                        }
                    }
                    // 3) Move to the next dividend
                    current_div_idx++;
                }
            }
        }
        // 4) Synchronize the entire team so that everyone sees updated U_i
        team.team_barrier();
        */
        
        
        
        
        
        // Step 1: Y0 computation
        A0.multiply_parallel_s_and_v(U_i, A0_result_i, team);
        A1.multiply_parallel_v(U_i, A1_result_i, team);
        
        device_shuffle_vector(U_i, U_shuffled_i, m1, m2, team);
        A2.multiply_parallel_s(U_shuffled_i, A2_result_shuffled_i, team);
        device_unshuffle_vector(A2_result_shuffled_i, A2_result_unshuf_i, m1, m2, team);

        // Y0 computation with boundary terms
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, total_size), 
            [&](const int i) {
                double exp_factor = std::exp(r_f * delta_t * (n-1));
                Y_0_i(i) = U_i(i) + delta_t * (A0_result_i(i) + A1_result_i(i) + 
                          A2_result_unshuf_i(i) + bounds.b_(i) * exp_factor);
            });

        // Step 2: A1 implicit solve
        A1.multiply_parallel_v(U_i, A1_result_i, team);
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, total_size),
            [&](const int i) {
                double exp_factor_n = std::exp(r_f * delta_t * n);
                double exp_factor_nm1 = std::exp(r_f * delta_t * (n-1));
                Y_0_i(i) = Y_0_i(i) + theta * delta_t * (bounds.b1_(i) * exp_factor_n - 
                          (A1_result_i(i) + bounds.b1_(i) * exp_factor_nm1));
            });
        A1.solve_implicit_parallel_v(Y_1_i, Y_0_i, team);

        // Step 3: A2 shuffled implicit solve
        device_shuffle_vector(U_i, U_shuffled_i, m1, m2, team);
        A2.multiply_parallel_s(U_shuffled_i, A2_result_shuffled_i, team);
        device_unshuffle_vector(A2_result_shuffled_i, A2_result_unshuf_i, m1, m2, team);

        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, total_size),
            [&](const int i) {
                double exp_factor_n = std::exp(r_f * delta_t * n);
                double exp_factor_nm1 = std::exp(r_f * delta_t * (n-1));
                Y_1_i(i) = Y_1_i(i) + theta * delta_t * (bounds.b2_(i) * exp_factor_n - 
                          (A2_result_unshuf_i(i) + bounds.b2_(i) * exp_factor_nm1));
            });

        device_shuffle_vector(Y_1_i, Y_1_shuffled_i, m1, m2, team);
        A2.solve_implicit_parallel_s(U_next_shuffled_i, Y_1_shuffled_i, team);
        device_unshuffle_vector(U_next_shuffled_i, U_i, m1, m2, team);
    }
}



/*

Device callable american with dividends

*/
template<class Device, class ViewType>  // Add ViewType template parameter
KOKKOS_FUNCTION 
void device_DO_timestepping_american_dividend(
    // Grid dimensions
    const int m1,
    const int m2,
    // Time discretization
    const int N,
    const double delta_t,
    const double theta,
    const double r_f,
    // Problem components
    Device_A0_heston<Device>& A0,
    Device_A1_heston<Device>& A1,
    Device_A2_shuffled_heston<Device>& A2,
    const Device_BoundaryConditions<Device>& bounds,
    // Workspace views for this instance - now using ViewType
    ViewType& U_i,
    ViewType& Y_0_i,
    ViewType& Y_1_i,
    ViewType& A0_result_i,
    ViewType& A1_result_i,
    ViewType& A2_result_unshuf_i,
    ViewType& U_shuffled_i,
    ViewType& Y_1_shuffled_i,
    ViewType& A2_result_shuffled_i,
    ViewType& U_next_shuffled_i,
    //american specifics
    ViewType& lambda_bar_i,
    const ViewType& U_0_i,
    //dividend specifics
    const int num_dividends,                        // Number of dividends
    const Kokkos::View<double*>& dividend_dates,    // Device view of dates
    const Kokkos::View<double*>& dividend_amounts,  // Device view of amounts
    const Kokkos::View<double*>& dividend_percentages,  // Device view of percentages
    const ViewType& device_Vec_s_i,                         // Stock price grid
    ViewType& U_temp_i,  // Temporary storage passed in from workspace
    // Team handle
    const typename Kokkos::TeamPolicy<>::member_type& team
) { 

    const int total_size = (m1+1)*(m2+1);

    //Need to reset the lambda function to zero 
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, total_size),
    [&](const int i) {
        lambda_bar_i(i) = 0;
    });

    int current_div_idx = 0;

    for(int n = 1; n <= N; n++) {
        double t = n * delta_t;
        
        // Check for dividends - only one thread per team checks
        if (team.team_rank() == 0) {
            // Process only if there are still dividends left
            int div_idx = current_div_idx;
            if (div_idx < num_dividends) {
                // Check if we're in the dividend window for the current dividend
                if (t <= dividend_dates(div_idx) &&
                    dividend_dates(div_idx) < (n+1) * delta_t)
                {
                    // Get current dividend values
                    double div_date       = dividend_dates(div_idx);
                    double div_amount     = dividend_amounts(div_idx);
                    double div_percentage = dividend_percentages(div_idx);
        
                    // 1) Copy current solution U_i -> U_temp_i in a *sequential* double loop
                    for (int j = 0; j <= m2; j++) {
                        int offset = j * (m1 + 1);
                        for (int i = 0; i <= m1; i++) {
                            U_temp_i(offset + i) = U_i(offset + i);
                        }
                    }
        
                    // 2) Process dividend adjustment in another *sequential* double loop
                    for (int j = 0; j <= m2; j++) {
                        int offset = j * (m1 + 1);
                        for (int i = 0; i <= m1; i++) {
                            double old_s = device_Vec_s_i(i);
                            double new_s = old_s * (1.0 - div_percentage) - div_amount;
        
                            if (new_s > 0.0) {
                                // Find interpolation points
                                int idx = 0;
                                for (int k = 0; k <= m1; k++) {
                                    if (device_Vec_s_i(k) > new_s) {
                                        idx = k;
                                        break;
                                    }
                                }
        
                                if (idx > 0 && idx < m1 + 1) {
                                    // Interpolate
                                    double s_low  = device_Vec_s_i(idx - 1);
                                    double s_high = device_Vec_s_i(idx);
                                    double weight = (new_s - s_low) / (s_high - s_low);
        
                                    double val_low  = U_temp_i(offset + idx - 1);
                                    double val_high = U_temp_i(offset + idx);
        
                                    U_i(offset + i) = (1.0 - weight) * val_low
                                                    + weight * val_high;
                                }
                                else if (idx == 0) {
                                    U_i(offset + i) = U_temp_i(offset + 0);
                                }
                                else {
                                    U_i(offset + i) = U_temp_i(offset + m1);
                                }
                            } 
                            else {
                                U_i(offset + i) = 0.0;
                            }
                        }
                    }
                    // 3) Move to the next dividend
                    current_div_idx++;
                }
            }
        }
        // 4) Synchronize the entire team so that everyone sees updated U_i
        team.team_barrier();
        
        // Step 1: Y0 computation
        A0.multiply_parallel_s_and_v(U_i, A0_result_i, team);
        A1.multiply_parallel_v(U_i, A1_result_i, team);
        
        device_shuffle_vector(U_i, U_shuffled_i, m1, m2, team);
        A2.multiply_parallel_s(U_shuffled_i, A2_result_shuffled_i, team);
        device_unshuffle_vector(A2_result_shuffled_i, A2_result_unshuf_i, m1, m2, team);

        // Y0 computation with boundary terms
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, total_size), 
            [&](const int i) {
                double exp_factor = std::exp(r_f * delta_t * (n-1));
                Y_0_i(i) = U_i(i) + delta_t * (A0_result_i(i) + A1_result_i(i) + 
                          A2_result_unshuf_i(i) + bounds.b_(i) * exp_factor + 
                          lambda_bar_i(i));  // Add lambda contribution
            });

        // Step 2: A1 implicit solve
        A1.multiply_parallel_v(U_i, A1_result_i, team);
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, total_size),
            [&](const int i) {
                double exp_factor_n = std::exp(r_f * delta_t * n);
                double exp_factor_nm1 = std::exp(r_f * delta_t * (n-1));
                Y_0_i(i) = Y_0_i(i) + theta * delta_t * (bounds.b1_(i) * exp_factor_n - 
                          (A1_result_i(i) + bounds.b1_(i) * exp_factor_nm1));
            });
        A1.solve_implicit_parallel_v(Y_1_i, Y_0_i, team);

        // Step 3: A2 shuffled implicit solve
        device_shuffle_vector(U_i, U_shuffled_i, m1, m2, team);
        A2.multiply_parallel_s(U_shuffled_i, A2_result_shuffled_i, team);
        device_unshuffle_vector(A2_result_shuffled_i, A2_result_unshuf_i, m1, m2, team);

        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, total_size),
            [&](const int i) {
                double exp_factor_n = std::exp(r_f * delta_t * n);
                double exp_factor_nm1 = std::exp(r_f * delta_t * (n-1));
                Y_1_i(i) = Y_1_i(i) + theta * delta_t * (bounds.b2_(i) * exp_factor_n - 
                          (A2_result_unshuf_i(i) + bounds.b2_(i) * exp_factor_nm1));
            });

        device_shuffle_vector(Y_1_i, Y_1_shuffled_i, m1, m2, team);
        A2.solve_implicit_parallel_s(U_next_shuffled_i, Y_1_shuffled_i, team);
        device_unshuffle_vector(U_next_shuffled_i, U_i, m1, m2, team);

        // American option early exercise check and lambda update
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, total_size),
            [&](const int i) {
            // Apply early exercise condition
            const double U_bar = U_i(i);
            U_i(i) = Kokkos::max(U_bar - delta_t * lambda_bar_i(i), U_0_i(i));

            // Update lambda multiplier
            lambda_bar_i(i) = Kokkos::max(0.0, lambda_bar_i(i) + (U_0_i(i) - U_bar) / delta_t);

            // Set lambda to zero at S_max for all variance levels
            // Every (m1+1)th entry starting at index m1 is an S_max entry
            if(i % (m1 + 1) == m1) {  // This hits exactly S_max entries at all variance levels
                lambda_bar_i(i) = 0.0;
            }
        });
    }
}


void test_device_class();