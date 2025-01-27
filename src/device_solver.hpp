#pragma once
#include <Kokkos_Core.hpp>
#include "hes_a0_kernels.hpp"
#include "hes_a1_kernels.hpp"
#include "hes_a2_shuffled_kernels.hpp"
#include "hes_boundary_kernels.hpp"

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


/*

Fixed parameters, parallising over strikes

*/

template<class Device>
void parallel_DO_solve(
    // Grid dimensions
    const int nInstances,
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
    DO_Workspace<Device>& workspace) {  // Now just pass the workspace

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
    });
    Kokkos::fence();
}


/*

Variying parameters, used to compute Jacobian matrix

*/
struct HestonParams {
    double r_d;    // Domestic rate
    double r_f;    // Foreign / dividend rate
    double rho;    // Correlation
    double sigma;  // Vol of vol
    double kappa; 
    double eta;
    
    HestonParams() = default;
    HestonParams(double rd, double rf, double rh, double s, double k, double e)
     : r_d(rd), r_f(rf), rho(rh), sigma(s), kappa(k), eta(e) {}
};

template<class Device>
void parallel_DO_solve_params(
    const int nInstances,
    const int m1,
    const int m2,
    const int N,
    const double T,
    const double delta_t,
    const double theta,
    // Arrays of parameters, one per instance
    Kokkos::View<HestonParams*, Device> paramsView,
    // Problem components, one set per instance
    const Kokkos::View<Device_A0_heston<Device>*>& A0_solvers,
    const Kokkos::View<Device_A1_heston<Device>*>& A1_solvers,
    const Kokkos::View<Device_A2_shuffled_heston<Device>*>& A2_solvers,
    const Kokkos::View<Device_BoundaryConditions<Device>*>& bounds_d,
    const Kokkos::View<GridViews*>& deviceGrids,
    DO_Workspace<Device>& workspace)
{
    const int total_size = (m1+1)*(m2+1);

    using team_policy = Kokkos::TeamPolicy<>;
    team_policy policy(nInstances, Kokkos::AUTO);

    Kokkos::parallel_for("DO_scheme_params", policy,
      KOKKOS_LAMBDA(const team_policy::member_type& team) {
        const int instance = team.league_rank();
        
        //--- 1) Grab the parameter set for THIS instance
        double r_d   = paramsView(instance).r_d;
        double r_f   = paramsView(instance).r_f;
        double rho   = paramsView(instance).rho;
        double sigma = paramsView(instance).sigma;
        double kappa = paramsView(instance).kappa;
        double eta   = paramsView(instance).eta;
        
        //--- 2) Subviews from workspace for this instance
        auto U_i               = Kokkos::subview(workspace.U, instance, Kokkos::ALL);
        auto Y_0_i             = Kokkos::subview(workspace.Y_0, instance, Kokkos::ALL);
        auto Y_1_i             = Kokkos::subview(workspace.Y_1, instance, Kokkos::ALL);
        auto A0_result_i       = Kokkos::subview(workspace.A0_result, instance, Kokkos::ALL);
        auto A1_result_i       = Kokkos::subview(workspace.A1_result, instance, Kokkos::ALL);
        auto A2_result_unshuf_i= Kokkos::subview(workspace.A2_result_unshuf, instance, Kokkos::ALL);
        
        auto U_shuffled_i      = Kokkos::subview(workspace.U_shuffled, instance, Kokkos::ALL);
        auto Y_1_shuffled_i    = Kokkos::subview(workspace.Y_1_shuffled, instance, Kokkos::ALL);
        auto A2_result_shuffled_i = Kokkos::subview(workspace.A2_result_shuffled, instance, Kokkos::ALL);
        auto U_next_shuffled_i = Kokkos::subview(workspace.U_next_shuffled, instance, Kokkos::ALL);

        //--- 3) Grid & boundary
        GridViews grid_i = deviceGrids(instance);
        bounds_d(instance).initialize(grid_i, team);
        auto bounds = bounds_d(instance);

        //--- 4) Build the PDE matrices for this instance
        A0_solvers(instance).build_matrix(grid_i, rho, sigma, team);
        A1_solvers(instance).build_matrix(grid_i, r_d, r_f, theta, delta_t, team);
        A2_solvers(instance).build_matrix(grid_i, r_d, kappa, eta, sigma, theta, delta_t, team);

        //--- 5) Time stepping
        for(int n = 1; n <= N; n++) {
            // Step 1: Y0
            A0_solvers(instance).multiply_parallel_s_and_v(U_i, A0_result_i, team);
            A1_solvers(instance).multiply_parallel_v(U_i, A1_result_i, team);
            
            device_shuffle_vector(U_i, U_shuffled_i, m1, m2, team);
            A2_solvers(instance).multiply_parallel_s(U_shuffled_i, A2_result_shuffled_i, team);
            device_unshuffle_vector(A2_result_shuffled_i, A2_result_unshuf_i, m1, m2, team);

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, total_size), 
                [&](const int idx) {
                    double exp_factor = std::exp(r_f * delta_t * (n-1));
                    Y_0_i(idx) = U_i(idx)
                                 + delta_t*(A0_result_i(idx)
                                            + A1_result_i(idx)
                                            + A2_result_unshuf_i(idx)
                                            + bounds.b_(idx)*exp_factor);
            });

            // Step 2: A1 implicit
            A1_solvers(instance).multiply_parallel_v(U_i, A1_result_i, team);
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, total_size),
                [&](const int idx) {
                    double exp_factor_n   = std::exp(r_f * delta_t * n);
                    double exp_factor_nm1 = std::exp(r_f * delta_t * (n-1));
                    Y_0_i(idx) += theta*delta_t*( bounds.b1_(idx)*exp_factor_n
                                       - ( A1_result_i(idx) + bounds.b1_(idx)*exp_factor_nm1 ));
            });
            A1_solvers(instance).solve_implicit_parallel_v(Y_1_i, Y_0_i, team);

            // Step 3: A2 implicit (shuffled)
            device_shuffle_vector(U_i, U_shuffled_i, m1, m2, team);
            A2_solvers(instance).multiply_parallel_s(U_shuffled_i, A2_result_shuffled_i, team);
            device_unshuffle_vector(A2_result_shuffled_i, A2_result_unshuf_i, m1, m2, team);

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, total_size),
                [&](const int idx) {
                    double exp_factor_n   = std::exp(r_f * delta_t * n);
                    double exp_factor_nm1 = std::exp(r_f * delta_t * (n-1));
                    Y_1_i(idx) += theta*delta_t*( bounds.b2_(idx)*exp_factor_n
                                     - ( A2_result_unshuf_i(idx) + bounds.b2_(idx)*exp_factor_nm1 ));
            });

            device_shuffle_vector(Y_1_i, Y_1_shuffled_i, m1, m2, team);
            A2_solvers(instance).solve_implicit_parallel_s(U_next_shuffled_i, Y_1_shuffled_i, team);
            device_unshuffle_vector(U_next_shuffled_i, U_i, m1, m2, team);
        }
    });
    
    Kokkos::fence();
}


void test_device_class();