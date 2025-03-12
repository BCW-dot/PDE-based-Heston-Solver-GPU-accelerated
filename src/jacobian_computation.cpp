#include "jacobian_computation.hpp"

#include <iostream>
#include <numeric>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <string>

#include <fstream>  // For std::ofstream
#include <iostream> // For std::cerr, std::cout

#include "bs.hpp" //for market prices generating

#include <KokkosBlas2_gemv.hpp> // for gemv
#include <KokkosBlas3_gemm.hpp> // for gemm


void solve_5x5_device(
    const Kokkos::View<double**> &A_device,  // shape (5,5)
    const Kokkos::View<double*>  &b_device,  // shape (5)
    const Kokkos::View<double*>  &x_device   // shape (5)
){
  // We run one kernel with a single iteration so that everything is done on GPU.
  Kokkos::parallel_for("solve_5x5", Kokkos::RangePolicy<>(0, 1),
    KOKKOS_LAMBDA(const int /*dummy*/)
    {
      constexpr int N = 5;

      // 1) Copy A_device, b_device into local arrays in GPU registers/shared memory.
      double A[25]; // row-major: A[i*N + j]
      double b[5];
      for (int i = 0; i < N; i++) {
        b[i] = b_device(i);
        for (int j = 0; j < N; j++) {
          A[i*N + j] = A_device(i,j);
        }
      }

      // 2) Perform partial pivot Gaussian elimination in-place.
      //    We'll pivot on the largest absolute value in the column for each step.
      for (int k = 0; k < N; k++) {

        // 2a) Find pivot row = row p in [k..N-1] with max |A[p,k]|.
        double maxA = Kokkos::abs(A[k*N + k]);
        int pivotRow = k;
        for(int p = k+1; p < N; p++){
          double val = Kokkos::abs(A[p*N + k]);
          if(val > maxA){
            maxA = val;
            pivotRow = p;
          }
        }
        // If pivotRow != k, swap the two rows in A and the entries in b.
        if(pivotRow != k){
          for(int col = 0; col < N; col++){
            double tmp = A[k*N + col];
            A[k*N + col] = A[pivotRow*N + col];
            A[pivotRow*N + col] = tmp;
          }
          double tmpb = b[k];
          b[k] = b[pivotRow];
          b[pivotRow] = tmpb;
        }

        // 2b) Divide pivot row by pivot
        double pivot = A[k*N + k];
        // (Assume matrix is non-singular, pivot != 0)
        for(int col = k+1; col < N; col++){
          A[k*N + col] /= pivot;
        }
        b[k] /= pivot;
        A[k*N + k] = 1.0;

        // 2c) Eliminate below pivot
        for(int i = k+1; i < N; i++){
          double factor = A[i*N + k];
          for(int col = k+1; col < N; col++){
            A[i*N + col] -= factor * A[k*N + col];
          }
          b[i] -= factor * b[k];
          A[i*N + k] = 0.0;
        }
      }

      // 3) Back-substitution
      for(int k = N-1; k >= 0; k--){
        double val = b[k];
        for(int col = k+1; col < N; col++){
          val -= A[k*N + col] * b[col];
        }
        b[k] = val;
      }

      // 4) Now b[] holds the solution. Copy to x_device.
      for(int i = 0; i < N; i++){
        x_device(i) = b[i];
      }
    }
  ); // parallel_for

  Kokkos::fence();
}

// This is the same as compute_parameter_update, but no KokkosBatched:
void compute_parameter_update_on_device(
    const Kokkos::View<double**>& J,        // [num_data x 5]
    const Kokkos::View<double*>&  residual, // [num_data]
    const double                  lambda,
    Kokkos::View<double*>&        delta     // [5]
){
    constexpr int N = 5;

    // 1. Build J^T J => [5 x 5]
    Kokkos::View<double**> JTJ("JTJ", N, N);
    KokkosBlas::gemm("T", "N", 1.0, J, J, 0.0, JTJ);

    // Print J^T J
    /*
    std::cout << std::fixed << std::setprecision(6);
    auto h_JTJ = Kokkos::create_mirror_view(JTJ);
    Kokkos::deep_copy(h_JTJ, JTJ);
    std::cout << "\nJ^T J matrix before lambda modification:\n";
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            std::cout << std::scientific << std::setw(15) << h_JTJ(i,j) << " ";
        }
        std::cout << "\n";
    }
    */
    
    

    // 2. Add lambda on diagonal
    Kokkos::parallel_for("add_lambda_diag", N, KOKKOS_LAMBDA(const int i){
        JTJ(i,i) *= (1.0 + lambda);
    });

    // Print modified matrix
    /*
    Kokkos::deep_copy(h_JTJ, JTJ);
    std::cout << "\nJ^T J matrix after lambda modification (lambda = " << lambda << "):\n";
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            std::cout << std::scientific << std::setw(15) << h_JTJ(i,j) << " ";
        }
        std::cout << "\n";
    }
    */

    // 3. Build J^T r => [5]
    Kokkos::View<double*> JTr("JTr", N);
    KokkosBlas::gemv("T", 1.0, J, residual, 0.0, JTr);

    // Print J^T r
    /*
    auto h_JTr = Kokkos::create_mirror_view(JTr);
    Kokkos::deep_copy(h_JTr, JTr);
    std::cout << "\nJ^T r vector:\n";
    for(int i = 0; i < N; i++) {
        std::cout << std::scientific << std::setw(15) << h_JTr(i) << "\n";
    }
    */

    // 4. Solve (JTJ) * delta = (JTr) on device with our manual 5x5 routine
    solve_5x5_device(JTJ, JTr, delta);

    // Print solution (delta)
    /*
    auto h_delta = Kokkos::create_mirror_view(delta);
    Kokkos::deep_copy(h_delta, delta);
    std::cout << "\nSolution delta:\n";
    for(int i = 0; i < N; i++) {
        std::cout << std::scientific << std::setw(15) << h_delta(i) << "\n";
    }
    */

    // Optional: verify solution by computing residual JTJ * delta - JTr
    /*
    Kokkos::View<double*> verify("verify", N);
    KokkosBlas::gemv("N", 1.0, JTJ, delta, 0.0, verify);
    auto h_verify = Kokkos::create_mirror_view(verify);
    Kokkos::deep_copy(h_verify, verify);

    std::cout << "\nVerification - residual norm of (JTJ * delta - JTr):\n";
    double res_norm = 0.0;
    for(int i = 0; i < N; i++) {
        double res = h_verify(i) - h_JTr(i);
        res_norm += res * res;
    }
    res_norm = std::sqrt(res_norm);
    std::cout << "||JTJ * delta - JTr|| = " << std::scientific << res_norm << "\n";
    */
}


/*

The next two methods are used for calibrating european options

*/
// Function to compute Jacobian matrix of a european options
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
    const double eps
) {
    using Device = Kokkos::DefaultExecutionSpace;
    // Create team policy
    using team_policy = Kokkos::TeamPolicy<>;
    team_policy policy(num_strikes, Kokkos::AUTO);

    // Main Jacobian computation kernel 
    Kokkos::parallel_for("Jacobian_computation", policy,
        KOKKOS_LAMBDA(const team_policy::member_type& team) {
            const int instance = team.league_rank();
            
            // Setup workspace views
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

            auto U_0_i = Kokkos::subview(U_0, instance, Kokkos::ALL);  // Get initial condition

            GridViews grid_i = deviceGrids(instance);

            grid_i.rebuild_variance_views(V_0, 5.0, 5.0/500, team);
            
            bounds_d(instance).initialize(grid_i, team);
            auto bounds = bounds_d(instance);
            
            A0_solvers(instance).build_matrix(grid_i, rho, sigma, team);
            A1_solvers(instance).build_matrix(grid_i, r_d, r_f, theta, delta_t, team);
            A2_solvers(instance).build_matrix(grid_i, r_d, kappa, eta, sigma, theta, delta_t, team);

            device_DO_timestepping<Device, decltype(U_i)>(
                m1, m2, N, delta_t, theta, r_f,
                A0_solvers(instance), A1_solvers(instance), A2_solvers(instance),
                bounds,
                U_i, Y_0_i, Y_1_i,
                A0_result_i, A1_result_i, A2_result_unshuf_i,
                U_shuffled_i, Y_1_shuffled_i, A2_result_shuffled_i, U_next_shuffled_i,
                team
            );
            
            //The s direction can be precomputed
            // Get indices for price extraction
            // Find s index
            int index_s = -1;
            //int index_v = -1;
            for(int i = 0; i <= m1; i++) {
                if(Kokkos::abs(grid_i.device_Vec_s(i) - S_0) < 1e-10) {
                    index_s = i;  // Store s index
                    break;
                }
            }
            //finding v index
            const int index_v = grid_i.find_v0_index(V_0);

            // Now we can use these indices directly
            const double base_price = U_i(index_s + index_v*(m1+1));
            base_prices(instance) = base_price;

            // Loop over parameters for finite differences, this excludes v0 in the loop and treats it seperately
            //same result of course but surprisingly not a whole lot faster
            for(int param = 0; param < 4; param++) {
                // Handle other parameters as before
                double kappa_p = kappa;
                double eta_p = eta;
                double sigma_p = sigma;
                double rho_p = rho;

                switch(param) {
                    case 0: kappa_p += eps; break;
                    case 1: eta_p += eps; break;
                    case 2: sigma_p += eps; break;
                    case 3: rho_p += eps; break;
                }

                // Reset initial condition
                for(int idx = 0; idx < total_size; idx++) {
                    U_i(idx) = U_0_i(idx);
                }

                // Rebuild matrices with perturbed parameter
                A0_solvers(instance).build_matrix(grid_i, rho_p, sigma_p, team);
                A1_solvers(instance).build_matrix(grid_i, r_d, r_f, theta, delta_t, team);
                A2_solvers(instance).build_matrix(grid_i, r_d, kappa_p, eta_p, sigma_p, theta, delta_t, team);

                // Compute perturbed solution
                device_DO_timestepping<Device, decltype(U_i)>(
                    m1, m2, N, delta_t, theta, r_f,
                    A0_solvers(instance), A1_solvers(instance), A2_solvers(instance),
                    bounds, U_i, Y_0_i, Y_1_i,
                    A0_result_i, A1_result_i, A2_result_unshuf_i,
                    U_shuffled_i, Y_1_shuffled_i, A2_result_shuffled_i, U_next_shuffled_i,
                    team
                );

                // Store results
                double pert_price = U_i(index_s + index_v*(m1+1));
                //pert_prices(instance, param) = pert_price;
                J(instance, param) = (pert_price - base_price) / eps;
            }
            // Special handling for V0 (param == 4)
            const int param = 4;
            for(int idx = 0; idx < total_size; idx++) {
                U_i(idx) = U_0_i(idx);
            }

            // Rebuild variance views with perturbed V0
            //This will change the V direction views for everyone!
            grid_i.rebuild_variance_views(V_0 + eps, 5.0, 5.0/500, team);
            //new v index for option price
            const int index_v_pertubed = grid_i.find_v0_index(V_0 + eps);
            
            // Rebuild matrices with updated grid
            A0_solvers(instance).build_matrix(grid_i, rho, sigma, team);
            A1_solvers(instance).build_matrix(grid_i, r_d, r_f, theta, delta_t, team);
            A2_solvers(instance).build_matrix(grid_i, r_d, kappa, eta, sigma, theta, delta_t, team);

            // Recompute solution
            // Compute perturbed solution
            device_DO_timestepping<Device, decltype(U_i)>(
                m1, m2, N, delta_t, theta, r_f,
                A0_solvers(instance), A1_solvers(instance), A2_solvers(instance),
                bounds, U_i, Y_0_i, Y_1_i,
                A0_result_i, A1_result_i, A2_result_unshuf_i,
                U_shuffled_i, Y_1_shuffled_i, A2_result_shuffled_i, U_next_shuffled_i,
                team
            );

            // Get price and compute gradient
            double pert_price = U_i(index_s + index_v_pertubed*(m1+1));
            J(instance, param) = (pert_price - base_price) / eps;
        });
        Kokkos::fence();
}


// Function to compute the base price in parallel over strikes, called from host. For european options
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
) {
    using Device = Kokkos::DefaultExecutionSpace;
    // Create team policy
    using team_policy = Kokkos::TeamPolicy<>;
    team_policy policy(num_strikes, Kokkos::AUTO);

    // Main Jacobian computation kernel 
    Kokkos::parallel_for("Base_Price_computation", policy,
        KOKKOS_LAMBDA(const team_policy::member_type& team) {
            const int instance = team.league_rank();
            
            // Setup workspace views
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

            GridViews grid_i = deviceGrids(instance);

            //need to rebuild the Variance direction, since we are chagning V0 throughout the computation
            grid_i.rebuild_variance_views(V_0, 5.0, 5.0/500, team);
            
            bounds_d(instance).initialize(grid_i, team);
            auto bounds = bounds_d(instance);
            
            A0_solvers(instance).build_matrix(grid_i, rho, sigma, team);
            A1_solvers(instance).build_matrix(grid_i, r_d, r_f, theta, delta_t, team);
            A2_solvers(instance).build_matrix(grid_i, r_d, kappa, eta, sigma, theta, delta_t, team);

            device_DO_timestepping<Device, decltype(U_i)>(
                m1, m2, N, delta_t, theta, r_f,
                A0_solvers(instance), A1_solvers(instance), A2_solvers(instance),
                bounds,
                U_i, Y_0_i, Y_1_i,
                A0_result_i, A1_result_i, A2_result_unshuf_i,
                U_shuffled_i, Y_1_shuffled_i, A2_result_shuffled_i, U_next_shuffled_i,
                team
            );
            
            //The s direction can be precomputed
            // Get indices for price extraction
            // Find s index
            int index_s = -1;
            //int index_v = -1;
            for(int i = 0; i <= m1; i++) {
                if(Kokkos::abs(grid_i.device_Vec_s(i) - S_0) < 1e-10) {
                    index_s = i;  // Store s index
                    break;
                }
            }
            const int index_v = grid_i.find_v0_index(V_0);

            // Now you can use these indices directly
            const double base_price = U_i(index_s + index_v*(m1+1));
            base_prices(instance) = base_price;
        });
        Kokkos::fence();
}


/*

The next two functions are used for american options

*/
// Function to compute Jacobian matrix for american options
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
) {
    using Device = Kokkos::DefaultExecutionSpace;
    // Create team policy
    using team_policy = Kokkos::TeamPolicy<>;
    team_policy policy(num_strikes, Kokkos::AUTO);

    // Main Jacobian computation kernel 
    Kokkos::parallel_for("Jacobian_computation", policy,
        KOKKOS_LAMBDA(const team_policy::member_type& team) {
            const int instance = team.league_rank();
            
            // Setup workspace views
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

            auto U_0_i = Kokkos::subview(U_0, instance, Kokkos::ALL);  // Get initial condition
            //american specific
            auto lambda_bar_i = Kokkos::subview(workspace.lambda_bar, instance, Kokkos::ALL);

            GridViews grid_i = deviceGrids(instance);

            grid_i.rebuild_variance_views(V_0, 5.0, 5.0/500, team);
            
            bounds_d(instance).initialize(grid_i, team);
            auto bounds = bounds_d(instance);
            
            A0_solvers(instance).build_matrix(grid_i, rho, sigma, team);
            A1_solvers(instance).build_matrix(grid_i, r_d, r_f, theta, delta_t, team);
            A2_solvers(instance).build_matrix(grid_i, r_d, kappa, eta, sigma, theta, delta_t, team);

            
            // Call device timestepping
            device_DO_timestepping_american<Device, decltype(U_i)>(
                m1, m2, N, delta_t, theta, r_f,
                A0_solvers(instance), A1_solvers(instance), A2_solvers(instance),
                bounds,
                U_i, Y_0_i, Y_1_i,
                A0_result_i, A1_result_i, A2_result_unshuf_i,
                U_shuffled_i, Y_1_shuffled_i, A2_result_shuffled_i, U_next_shuffled_i,
                lambda_bar_i, U_0_i,  
                team
            );
            
            //The s direction can be precomputed
            // Get indices for price extraction
            // Find s index
            int index_s = -1;
            //int index_v = -1;
            for(int i = 0; i <= m1; i++) {
                if(Kokkos::abs(grid_i.device_Vec_s(i) - S_0) < 1e-10) {
                    index_s = i;  // Store s index
                    break;
                }
            }
            const int index_v = grid_i.find_v0_index(V_0);

            // Now you can use these indices directly
            const double base_price = U_i(index_s + index_v*(m1+1));
            base_prices(instance) = base_price;

            // Loop over parameters for finite differences, this excludes v0 in the loop and treats it seperately
            //same result of course but surprisingly not a whole lot faster
            for(int param = 0; param < 4; param++) {
                // Handle other parameters as before
                double kappa_p = kappa;
                double eta_p = eta;
                double sigma_p = sigma;
                double rho_p = rho;

                switch(param) {
                    case 0: kappa_p += eps; break;
                    case 1: eta_p += eps; break;
                    case 2: sigma_p += eps; break;
                    case 3: rho_p += eps; break;
                }

                // Reset initial condition
                for(int idx = 0; idx < total_size; idx++) {
                    U_i(idx) = U_0_i(idx);
                }

                // Rebuild matrices with perturbed parameter
                A0_solvers(instance).build_matrix(grid_i, rho_p, sigma_p, team);
                A1_solvers(instance).build_matrix(grid_i, r_d, r_f, theta, delta_t, team);
                A2_solvers(instance).build_matrix(grid_i, r_d, kappa_p, eta_p, sigma_p, theta, delta_t, team);

                // Compute perturbed solution
                device_DO_timestepping_american<Device, decltype(U_i)>(
                    m1, m2, N, delta_t, theta, r_f,
                    A0_solvers(instance), A1_solvers(instance), A2_solvers(instance),
                    bounds,
                    U_i, Y_0_i, Y_1_i,
                    A0_result_i, A1_result_i, A2_result_unshuf_i,
                    U_shuffled_i, Y_1_shuffled_i, A2_result_shuffled_i, U_next_shuffled_i,
                    lambda_bar_i, U_0_i,  
                    team
                );

                // Store results
                double pert_price = U_i(index_s + index_v*(m1+1));
                //pert_prices(instance, param) = pert_price;
                J(instance, param) = (pert_price - base_price) / eps;
            }
            // Special handling for V0 (param == 4)
            const int param = 4;
            for(int idx = 0; idx < total_size; idx++) {
                U_i(idx) = U_0_i(idx);
            }

            // Rebuild variance views with perturbed V0
            //This will change the V direction views for everyone!
            grid_i.rebuild_variance_views(V_0 + eps, 5.0, 5.0/500, team);
            //new v index for option price
            const int index_v_pertubed = grid_i.find_v0_index(V_0 + eps);
            
            // Rebuild matrices with updated grid
            A0_solvers(instance).build_matrix(grid_i, rho, sigma, team);
            A1_solvers(instance).build_matrix(grid_i, r_d, r_f, theta, delta_t, team);
            A2_solvers(instance).build_matrix(grid_i, r_d, kappa, eta, sigma, theta, delta_t, team);

            // Recompute solution
            // Compute perturbed solution
            device_DO_timestepping_american<Device, decltype(U_i)>(
                m1, m2, N, delta_t, theta, r_f,
                A0_solvers(instance), A1_solvers(instance), A2_solvers(instance),
                bounds,
                U_i, Y_0_i, Y_1_i,
                A0_result_i, A1_result_i, A2_result_unshuf_i,
                U_shuffled_i, Y_1_shuffled_i, A2_result_shuffled_i, U_next_shuffled_i,
                lambda_bar_i, U_0_i,  
                team
            );

            // Get price and compute gradient
            double pert_price = U_i(index_s + index_v_pertubed*(m1+1));
            J(instance, param) = (pert_price - base_price) / eps;
        });
        Kokkos::fence();
}
    

//Computes the base prices for american options in parallel
void compute_base_prices_american(
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
) {
    using Device = Kokkos::DefaultExecutionSpace;
    // Create team policy
    using team_policy = Kokkos::TeamPolicy<>;
    team_policy policy(num_strikes, Kokkos::AUTO);

    // Main Jacobian computation kernel 
    Kokkos::parallel_for("Jacobian_computation", policy,
        KOKKOS_LAMBDA(const team_policy::member_type& team) {
            const int instance = team.league_rank();
            
            // Setup workspace views
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

            auto U_0_i = Kokkos::subview(U_0, instance, Kokkos::ALL);  // Get initial condition
            //american specific
            auto lambda_bar_i = Kokkos::subview(workspace.lambda_bar, instance, Kokkos::ALL);

            GridViews grid_i = deviceGrids(instance);

            grid_i.rebuild_variance_views(V_0, 5.0, 5.0/500, team);
            
            bounds_d(instance).initialize(grid_i, team);
            auto bounds = bounds_d(instance);
            
            A0_solvers(instance).build_matrix(grid_i, rho, sigma, team);
            A1_solvers(instance).build_matrix(grid_i, r_d, r_f, theta, delta_t, team);
            A2_solvers(instance).build_matrix(grid_i, r_d, kappa, eta, sigma, theta, delta_t, team);

            
            // Call device timestepping
            device_DO_timestepping_american<Device, decltype(U_i)>(
                m1, m2, N, delta_t, theta, r_f,
                A0_solvers(instance), A1_solvers(instance), A2_solvers(instance),
                bounds,
                U_i, Y_0_i, Y_1_i,
                A0_result_i, A1_result_i, A2_result_unshuf_i,
                U_shuffled_i, Y_1_shuffled_i, A2_result_shuffled_i, U_next_shuffled_i,
                lambda_bar_i, U_0_i,  
                team
            );
            
            //The s direction can be precomputed
            // Get indices for price extraction
            // Find s index
            int index_s = -1;
            //int index_v = -1;
            for(int i = 0; i <= m1; i++) {
                if(Kokkos::abs(grid_i.device_Vec_s(i) - S_0) < 1e-10) {
                    index_s = i;  // Store s index
                    break;
                }
            }
            const int index_v = grid_i.find_v0_index(V_0);

            // Now you can use these indices directly
            const double base_price = U_i(index_s + index_v*(m1+1));
            base_prices(instance) = base_price;
        });
        Kokkos::fence();
}
  


/*

The next two functions are used for european options with an underlying paying dividends

*/
// Function to compute Jacobian matrix for american options
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
) {
    using Device = Kokkos::DefaultExecutionSpace;
    // Create team policy
    using team_policy = Kokkos::TeamPolicy<>;
    team_policy policy(num_strikes, Kokkos::AUTO);

    // Main Jacobian computation kernel 
    Kokkos::parallel_for("Jacobian_computation", policy,
        KOKKOS_LAMBDA(const team_policy::member_type& team) {
            const int instance = team.league_rank();
            
            // Setup workspace views
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

            auto U_0_i = Kokkos::subview(U_0, instance, Kokkos::ALL);  // Get initial condition
            
            GridViews grid_i = deviceGrids(instance);

            //Dividend specifics
            auto U_temp_i = Kokkos::subview(workspace.U_temp, instance, Kokkos::ALL);
            auto device_Vec_s_i = grid_i.device_Vec_s;

            grid_i.rebuild_variance_views(V_0, 5.0, 5.0/500, team);
            
            bounds_d(instance).initialize(grid_i, team);
            auto bounds = bounds_d(instance);
            
            A0_solvers(instance).build_matrix(grid_i, rho, sigma, team);
            A1_solvers(instance).build_matrix(grid_i, r_d, r_f, theta, delta_t, team);
            A2_solvers(instance).build_matrix(grid_i, r_d, kappa, eta, sigma, theta, delta_t, team);

            
            // Call device timestepping
            device_DO_timestepping_dividend<Device, decltype(U_i)>(
                m1, m2, N, delta_t, theta, r_f,
                A0_solvers(instance), A1_solvers(instance), A2_solvers(instance),
                bounds,
                U_i, Y_0_i, Y_1_i,
                A0_result_i, A1_result_i, A2_result_unshuf_i,
                U_shuffled_i, Y_1_shuffled_i, A2_result_shuffled_i, U_next_shuffled_i,
                //Dividend - Use device views
                num_dividends,
                dividend_dates,     // Use device view instead of vector
                dividend_amounts,   // Use device view instead of vector
                dividend_percentages, // Use device view instead of vector
                device_Vec_s_i,        
                U_temp_i,  
                team
            );
            
            //The s direction can be precomputed
            // Get indices for price extraction
            // Find s index
            int index_s = -1;
            //int index_v = -1;
            for(int i = 0; i <= m1; i++) {
                if(Kokkos::abs(grid_i.device_Vec_s(i) - S_0) < 1e-10) {
                    index_s = i;  // Store s index
                    break;
                }
            }
            const int index_v = grid_i.find_v0_index(V_0);

            // Now you can use these indices directly
            const double base_price = U_i(index_s + index_v*(m1+1));
            base_prices(instance) = base_price;

            // Loop over parameters for finite differences, this excludes v0 in the loop and treats it seperately
            //same result of course but surprisingly not a whole lot faster
            for(int param = 0; param < 4; param++) {
                // Handle other parameters as before
                double kappa_p = kappa;
                double eta_p = eta;
                double sigma_p = sigma;
                double rho_p = rho;

                switch(param) {
                    case 0: kappa_p += eps; break;
                    case 1: eta_p += eps; break;
                    case 2: sigma_p += eps; break;
                    case 3: rho_p += eps; break;
                }

                // Reset initial condition
                for(int idx = 0; idx < total_size; idx++) {
                    U_i(idx) = U_0_i(idx);
                }

                // Rebuild matrices with perturbed parameter
                A0_solvers(instance).build_matrix(grid_i, rho_p, sigma_p, team);
                A1_solvers(instance).build_matrix(grid_i, r_d, r_f, theta, delta_t, team);
                A2_solvers(instance).build_matrix(grid_i, r_d, kappa_p, eta_p, sigma_p, theta, delta_t, team);

                // Compute perturbed solution
                device_DO_timestepping_dividend<Device, decltype(U_i)>(
                    m1, m2, N, delta_t, theta, r_f,
                    A0_solvers(instance), A1_solvers(instance), A2_solvers(instance),
                    bounds,
                    U_i, Y_0_i, Y_1_i,
                    A0_result_i, A1_result_i, A2_result_unshuf_i,
                    U_shuffled_i, Y_1_shuffled_i, A2_result_shuffled_i, U_next_shuffled_i,
                    //Dividend - Use device views
                    num_dividends,
                    dividend_dates,     // Use device view instead of vector
                    dividend_amounts,   // Use device view instead of vector
                    dividend_percentages, // Use device view instead of vector
                    device_Vec_s_i,        
                    U_temp_i,  
                    team
                );

                // Store results
                double pert_price = U_i(index_s + index_v*(m1+1));
                //pert_prices(instance, param) = pert_price;
                J(instance, param) = (pert_price - base_price) / eps;
            }
            // Special handling for V0 (param == 4)
            const int param = 4;
            for(int idx = 0; idx < total_size; idx++) {
                U_i(idx) = U_0_i(idx);
            }

            // Rebuild variance views with perturbed V0
            //This will change the V direction views for everyone!
            grid_i.rebuild_variance_views(V_0 + eps, 5.0, 5.0/500, team);
            //new v index for option price
            const int index_v_pertubed = grid_i.find_v0_index(V_0 + eps);
            
            // Rebuild matrices with updated grid
            A0_solvers(instance).build_matrix(grid_i, rho, sigma, team);
            A1_solvers(instance).build_matrix(grid_i, r_d, r_f, theta, delta_t, team);
            A2_solvers(instance).build_matrix(grid_i, r_d, kappa, eta, sigma, theta, delta_t, team);

            // Recompute solution
            // Compute perturbed solution
            device_DO_timestepping_dividend<Device, decltype(U_i)>(
                m1, m2, N, delta_t, theta, r_f,
                A0_solvers(instance), A1_solvers(instance), A2_solvers(instance),
                bounds,
                U_i, Y_0_i, Y_1_i,
                A0_result_i, A1_result_i, A2_result_unshuf_i,
                U_shuffled_i, Y_1_shuffled_i, A2_result_shuffled_i, U_next_shuffled_i,
                //Dividend - Use device views
                num_dividends,
                dividend_dates,     // Use device view instead of vector
                dividend_amounts,   // Use device view instead of vector
                dividend_percentages, // Use device view instead of vector
                device_Vec_s_i,        
                U_temp_i,  
                team
            );

            // Get price and compute gradient
            double pert_price = U_i(index_s + index_v_pertubed*(m1+1));
            J(instance, param) = (pert_price - base_price) / eps;
        });
        Kokkos::fence();
}


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
) {
    using Device = Kokkos::DefaultExecutionSpace;
    // Create team policy
    using team_policy = Kokkos::TeamPolicy<>;
    team_policy policy(num_strikes, Kokkos::AUTO);

    // Main Jacobian computation kernel 
    Kokkos::parallel_for("Jacobian_computation", policy,
        KOKKOS_LAMBDA(const team_policy::member_type& team) {
            const int instance = team.league_rank();
            
            // Setup workspace views
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

            auto U_0_i = Kokkos::subview(U_0, instance, Kokkos::ALL);  // Get initial condition
            
            GridViews grid_i = deviceGrids(instance);

            //Dividend specifics
            auto U_temp_i = Kokkos::subview(workspace.U_temp, instance, Kokkos::ALL);
            auto device_Vec_s_i = grid_i.device_Vec_s;

            grid_i.rebuild_variance_views(V_0, 5.0, 5.0/500, team);
            
            bounds_d(instance).initialize(grid_i, team);
            auto bounds = bounds_d(instance);
            
            A0_solvers(instance).build_matrix(grid_i, rho, sigma, team);
            A1_solvers(instance).build_matrix(grid_i, r_d, r_f, theta, delta_t, team);
            A2_solvers(instance).build_matrix(grid_i, r_d, kappa, eta, sigma, theta, delta_t, team);

            
            // Call device timestepping
            device_DO_timestepping_dividend<Device, decltype(U_i)>(
                m1, m2, N, delta_t, theta, r_f,
                A0_solvers(instance), A1_solvers(instance), A2_solvers(instance),
                bounds,
                U_i, Y_0_i, Y_1_i,
                A0_result_i, A1_result_i, A2_result_unshuf_i,
                U_shuffled_i, Y_1_shuffled_i, A2_result_shuffled_i, U_next_shuffled_i,
                //Dividend - Use device views
                num_dividends,
                dividend_dates,     // Use device view instead of vector
                dividend_amounts,   // Use device view instead of vector
                dividend_percentages, // Use device view instead of vector
                device_Vec_s_i,        
                U_temp_i,  
                team
            );
            
            //The s direction can be precomputed
            // Get indices for price extraction
            // Find s index
            int index_s = -1;
            //int index_v = -1;
            for(int i = 0; i <= m1; i++) {
                if(Kokkos::abs(grid_i.device_Vec_s(i) - S_0) < 1e-10) {
                    index_s = i;  // Store s index
                    break;
                }
            }
            const int index_v = grid_i.find_v0_index(V_0);

            // Now you can use these indices directly
            const double base_price = U_i(index_s + index_v*(m1+1));
            base_prices(instance) = base_price;
        });
        Kokkos::fence();
}


/*

The next two functions are used for american options with an underlying paying dividends

*/
// Function to compute Jacobian matrix for american options
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
) {
    using Device = Kokkos::DefaultExecutionSpace;
    // Create team policy
    using team_policy = Kokkos::TeamPolicy<>;
    team_policy policy(num_strikes, Kokkos::AUTO);

    // Main Jacobian computation kernel 
    Kokkos::parallel_for("Jacobian_computation", policy,
        KOKKOS_LAMBDA(const team_policy::member_type& team) {
            const int instance = team.league_rank();
            
            // Setup workspace views
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

            auto U_0_i = Kokkos::subview(U_0, instance, Kokkos::ALL);  // Get initial condition
            //american specific
            auto lambda_bar_i = Kokkos::subview(workspace.lambda_bar, instance, Kokkos::ALL);

            GridViews grid_i = deviceGrids(instance);

            //Dividend specifics
            auto U_temp_i = Kokkos::subview(workspace.U_temp, instance, Kokkos::ALL);
            auto device_Vec_s_i = grid_i.device_Vec_s;

            grid_i.rebuild_variance_views(V_0, 5.0, 5.0/500, team);
            
            bounds_d(instance).initialize(grid_i, team);
            auto bounds = bounds_d(instance);
            
            A0_solvers(instance).build_matrix(grid_i, rho, sigma, team);
            A1_solvers(instance).build_matrix(grid_i, r_d, r_f, theta, delta_t, team);
            A2_solvers(instance).build_matrix(grid_i, r_d, kappa, eta, sigma, theta, delta_t, team);

            
            // Call device timestepping
            device_DO_timestepping_american_dividend<Device, decltype(U_i)>(
                m1, m2, N, delta_t, theta, r_f,
                A0_solvers(instance), A1_solvers(instance), A2_solvers(instance),
                bounds,
                U_i, Y_0_i, Y_1_i,
                A0_result_i, A1_result_i, A2_result_unshuf_i,
                U_shuffled_i, Y_1_shuffled_i, A2_result_shuffled_i, U_next_shuffled_i,
                lambda_bar_i, U_0_i,
                //Dividend - Use device views
                num_dividends,
                dividend_dates,     // Use device view instead of vector
                dividend_amounts,   // Use device view instead of vector
                dividend_percentages, // Use device view instead of vector
                device_Vec_s_i,        
                U_temp_i,  
                team
            );
            
            //The s direction can be precomputed
            // Get indices for price extraction
            // Find s index
            int index_s = -1;
            //int index_v = -1;
            for(int i = 0; i <= m1; i++) {
                if(Kokkos::abs(grid_i.device_Vec_s(i) - S_0) < 1e-10) {
                    index_s = i;  // Store s index
                    break;
                }
            }
            const int index_v = grid_i.find_v0_index(V_0);

            // Now you can use these indices directly
            const double base_price = U_i(index_s + index_v*(m1+1));
            base_prices(instance) = base_price;

            // Loop over parameters for finite differences, this excludes v0 in the loop and treats it seperately
            //same result of course but surprisingly not a whole lot faster
            for(int param = 0; param < 4; param++) {
                // Handle other parameters as before
                double kappa_p = kappa;
                double eta_p = eta;
                double sigma_p = sigma;
                double rho_p = rho;

                switch(param) {
                    case 0: kappa_p += eps; break;
                    case 1: eta_p += eps; break;
                    case 2: sigma_p += eps; break;
                    case 3: rho_p += eps; break;
                }

                // Reset initial condition
                for(int idx = 0; idx < total_size; idx++) {
                    U_i(idx) = U_0_i(idx);
                }

                // Rebuild matrices with perturbed parameter
                A0_solvers(instance).build_matrix(grid_i, rho_p, sigma_p, team);
                A1_solvers(instance).build_matrix(grid_i, r_d, r_f, theta, delta_t, team);
                A2_solvers(instance).build_matrix(grid_i, r_d, kappa_p, eta_p, sigma_p, theta, delta_t, team);

                // Compute perturbed solution
                device_DO_timestepping_american_dividend<Device, decltype(U_i)>(
                    m1, m2, N, delta_t, theta, r_f,
                    A0_solvers(instance), A1_solvers(instance), A2_solvers(instance),
                    bounds,
                    U_i, Y_0_i, Y_1_i,
                    A0_result_i, A1_result_i, A2_result_unshuf_i,
                    U_shuffled_i, Y_1_shuffled_i, A2_result_shuffled_i, U_next_shuffled_i,
                    lambda_bar_i, U_0_i,
                    //Dividend - Use device views
                    num_dividends,
                    dividend_dates,     // Use device view instead of vector
                    dividend_amounts,   // Use device view instead of vector
                    dividend_percentages, // Use device view instead of vector
                    device_Vec_s_i,        
                    U_temp_i,  
                    team
                );

                // Store results
                double pert_price = U_i(index_s + index_v*(m1+1));
                //pert_prices(instance, param) = pert_price;
                J(instance, param) = (pert_price - base_price) / eps;
            }
            // Special handling for V0 (param == 4)
            const int param = 4;
            for(int idx = 0; idx < total_size; idx++) {
                U_i(idx) = U_0_i(idx);
            }

            // Rebuild variance views with perturbed V0
            //This will change the V direction views for everyone!
            grid_i.rebuild_variance_views(V_0 + eps, 5.0, 5.0/500, team);
            //new v index for option price
            const int index_v_pertubed = grid_i.find_v0_index(V_0 + eps);
            
            // Rebuild matrices with updated grid
            A0_solvers(instance).build_matrix(grid_i, rho, sigma, team);
            A1_solvers(instance).build_matrix(grid_i, r_d, r_f, theta, delta_t, team);
            A2_solvers(instance).build_matrix(grid_i, r_d, kappa, eta, sigma, theta, delta_t, team);

            // Recompute solution
            // Compute perturbed solution
            device_DO_timestepping_american_dividend<Device, decltype(U_i)>(
                m1, m2, N, delta_t, theta, r_f,
                A0_solvers(instance), A1_solvers(instance), A2_solvers(instance),
                bounds,
                U_i, Y_0_i, Y_1_i,
                A0_result_i, A1_result_i, A2_result_unshuf_i,
                U_shuffled_i, Y_1_shuffled_i, A2_result_shuffled_i, U_next_shuffled_i,
                lambda_bar_i, U_0_i,
                //Dividend - Use device views
                num_dividends,
                dividend_dates,     // Use device view instead of vector
                dividend_amounts,   // Use device view instead of vector
                dividend_percentages, // Use device view instead of vector
                device_Vec_s_i,        
                U_temp_i,  
                team
            );

            // Get price and compute gradient
            double pert_price = U_i(index_s + index_v_pertubed*(m1+1));
            J(instance, param) = (pert_price - base_price) / eps;
        });
        Kokkos::fence();
}


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
) {
    using Device = Kokkos::DefaultExecutionSpace;
    // Create team policy
    using team_policy = Kokkos::TeamPolicy<>;
    team_policy policy(num_strikes, Kokkos::AUTO);

    // Main Jacobian computation kernel 
    Kokkos::parallel_for("Jacobian_computation", policy,
        KOKKOS_LAMBDA(const team_policy::member_type& team) {
            const int instance = team.league_rank();
            
            // Setup workspace views
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

            auto U_0_i = Kokkos::subview(U_0, instance, Kokkos::ALL);  // Get initial condition
            //american specific
            auto lambda_bar_i = Kokkos::subview(workspace.lambda_bar, instance, Kokkos::ALL);

            GridViews grid_i = deviceGrids(instance);

            //Dividend specifics
            auto U_temp_i = Kokkos::subview(workspace.U_temp, instance, Kokkos::ALL);
            auto device_Vec_s_i = grid_i.device_Vec_s;

            grid_i.rebuild_variance_views(V_0, 5.0, 5.0/500, team);
            
            bounds_d(instance).initialize(grid_i, team);
            auto bounds = bounds_d(instance);
            
            A0_solvers(instance).build_matrix(grid_i, rho, sigma, team);
            A1_solvers(instance).build_matrix(grid_i, r_d, r_f, theta, delta_t, team);
            A2_solvers(instance).build_matrix(grid_i, r_d, kappa, eta, sigma, theta, delta_t, team);

            
            // Call device timestepping
            device_DO_timestepping_american_dividend<Device, decltype(U_i)>(
                m1, m2, N, delta_t, theta, r_f,
                A0_solvers(instance), A1_solvers(instance), A2_solvers(instance),
                bounds,
                U_i, Y_0_i, Y_1_i,
                A0_result_i, A1_result_i, A2_result_unshuf_i,
                U_shuffled_i, Y_1_shuffled_i, A2_result_shuffled_i, U_next_shuffled_i,
                lambda_bar_i, U_0_i,
                //Dividend - Use device views
                num_dividends,
                dividend_dates,     // Use device view instead of vector
                dividend_amounts,   // Use device view instead of vector
                dividend_percentages, // Use device view instead of vector
                device_Vec_s_i,        
                U_temp_i,  
                team
            );
            
            //The s direction can be precomputed
            // Get indices for price extraction
            // Find s index
            int index_s = -1;
            //int index_v = -1;
            for(int i = 0; i <= m1; i++) {
                if(Kokkos::abs(grid_i.device_Vec_s(i) - S_0) < 1e-10) {
                    index_s = i;  // Store s index
                    break;
                }
            }
            const int index_v = grid_i.find_v0_index(V_0);

            // Now you can use these indices directly
            const double base_price = U_i(index_s + index_v*(m1+1));
            base_prices(instance) = base_price;
        });
        Kokkos::fence();
}


//This does not work and is under testing
/*

Purely device sideable calibration code

*/
/*
template<class Device>
KOKKOS_FUNCTION
void device_compute_jacobian(
    // Market/model parameters
    const double S_0, const double V_0, const double T,
    const double r_d, const double r_f,
    const double rho, const double sigma, const double kappa, const double eta,
    // Numerical parameters
    const int m1, const int m2, const int total_size, const int N, const double theta, const double delta_t,
    // Pre-computed data structures
    const int num_strikes,
    const Kokkos::View<Device_A0_heston<Device>*>& A0_solvers,
    const Kokkos::View<Device_A1_heston<Device>*>& A1_solvers,
    const Kokkos::View<Device_A2_shuffled_heston<Device>*>& A2_solvers,
    const Kokkos::View<Device_BoundaryConditions<Device>*>& bounds_d,
    const Kokkos::View<GridViews*>& deviceGrids,
    const Kokkos::View<double**>& U_0,
    const DO_Workspace<Device>& workspace,
    // Output matrix
    const Kokkos::View<double**>& J,
    const Kokkos::View<double*>& base_prices,
    // Optional: perturbation size
    const double eps,
    // Team handle for parallelism
    const typename Kokkos::TeamPolicy<>::member_type& team
) {
    //Here i would have specified the team based parallsim to be accross strikes, this i dont have anymore

    // Main Jacobian computation kernel 
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, num_strikes),
        [&](const int strike_idx) {
            
            // Setup workspace views
            auto U_i = Kokkos::subview(workspace.U, strike_idx, Kokkos::ALL);
            auto Y_0_i = Kokkos::subview(workspace.Y_0, strike_idx, Kokkos::ALL);
            auto Y_1_i = Kokkos::subview(workspace.Y_1, strike_idx, Kokkos::ALL);
            auto A0_result_i = Kokkos::subview(workspace.A0_result, strike_idx, Kokkos::ALL);
            auto A1_result_i = Kokkos::subview(workspace.A1_result, strike_idx, Kokkos::ALL);
            auto A2_result_unshuf_i = Kokkos::subview(workspace.A2_result_unshuf, strike_idx, Kokkos::ALL);
            
            auto U_shuffled_i = Kokkos::subview(workspace.U_shuffled, strike_idx, Kokkos::ALL);
            auto Y_1_shuffled_i = Kokkos::subview(workspace.Y_1_shuffled, strike_idx, Kokkos::ALL);
            auto A2_result_shuffled_i = Kokkos::subview(workspace.A2_result_shuffled, strike_idx, Kokkos::ALL);
            auto U_next_shuffled_i = Kokkos::subview(workspace.U_next_shuffled, strike_idx, Kokkos::ALL);

            auto U_0_i = Kokkos::subview(U_0, strike_idx, Kokkos::ALL);  // Get initial condition

            GridViews grid_i = deviceGrids(strike_idx);

            grid_i.rebuild_variance_views(V_0, 5.0, 5.0/500, team);
            
            bounds_d(strike_idx).initialize(grid_i, team);
            auto bounds = bounds_d(strike_idx);
            
            A0_solvers(strike_idx).build_matrix(grid_i, rho, sigma, team);
            A1_solvers(strike_idx).build_matrix(grid_i, r_d, r_f, theta, delta_t, team);
            A2_solvers(strike_idx).build_matrix(grid_i, r_d, kappa, eta, sigma, theta, delta_t, team);

            device_DO_timestepping<Device, decltype(U_i)>(
                m1, m2, N, delta_t, theta, r_f,
                A0_solvers(strike_idx), A1_solvers(strike_idx), A2_solvers(strike_idx),
                bounds,
                U_i, Y_0_i, Y_1_i,
                A0_result_i, A1_result_i, A2_result_unshuf_i,
                U_shuffled_i, Y_1_shuffled_i, A2_result_shuffled_i, U_next_shuffled_i,
                team
            );
            
            //The s direction can be precomputed
            // Get indices for price extraction
            // Find s index
            int index_s = -1;
            //int index_v = -1;
            for(int i = 0; i <= m1; i++) {
                if(Kokkos::abs(grid_i.device_Vec_s(i) - S_0) < 1e-10) {
                    index_s = i;  // Store s index
                    break;
                }
            }
            const int index_v = grid_i.find_v0_index(V_0);

            // Now you can use these indices directly
            const double base_price = U_i(index_s + index_v*(m1+1));
            base_prices(strike_idx) = base_price;

            // Loop over parameters for finite differences, this excludes v0 in the loop and treats it seperately
            //same result of course but surprisingly not a whole lot faster
            for(int param = 0; param < 4; param++) {
                // Handle other parameters as before
                double kappa_p = kappa;
                double eta_p = eta;
                double sigma_p = sigma;
                double rho_p = rho;

                switch(param) {
                    case 0: kappa_p += eps; break;
                    case 1: eta_p += eps; break;
                    case 2: sigma_p += eps; break;
                    case 3: rho_p += eps; break;
                }

                // Reset initial condition
                for(int idx = 0; idx < total_size; idx++) {
                    U_i(idx) = U_0_i(idx);
                }

                // Rebuild matrices with perturbed parameter
                A0_solvers(strike_idx).build_matrix(grid_i, rho_p, sigma_p, team);
                A1_solvers(strike_idx).build_matrix(grid_i, r_d, r_f, theta, delta_t, team);
                A2_solvers(strike_idx).build_matrix(grid_i, r_d, kappa_p, eta_p, sigma_p, theta, delta_t, team);

                // Compute perturbed solution
                device_DO_timestepping<Device, decltype(U_i)>(
                    m1, m2, N, delta_t, theta, r_f,
                    A0_solvers(strike_idx), A1_solvers(strike_idx), A2_solvers(strike_idx),
                    bounds, U_i, Y_0_i, Y_1_i,
                    A0_result_i, A1_result_i, A2_result_unshuf_i,
                    U_shuffled_i, Y_1_shuffled_i, A2_result_shuffled_i, U_next_shuffled_i,
                    team
                );

                // Store results
                double pert_price = U_i(index_s + index_v*(m1+1));
                //pert_prices(instance, param) = pert_price;
                J(strike_idx, param) = (pert_price - base_price) / eps;
            }
            // Special handling for V0 (param == 4)
            const int param = 4;
            for(int idx = 0; idx < total_size; idx++) {
                U_i(idx) = U_0_i(idx);
            }

            // Rebuild variance views with perturbed V0
            //This will change the V direction views for everyone!
            grid_i.rebuild_variance_views(V_0 + eps, 5.0, 5.0/500, team);
            //new v index for option price
            const int index_v_pertubed = grid_i.find_v0_index(V_0 + eps);
            
            // Rebuild matrices with updated grid
            A0_solvers(strike_idx).build_matrix(grid_i, rho, sigma, team);
            A1_solvers(strike_idx).build_matrix(grid_i, r_d, r_f, theta, delta_t, team);
            A2_solvers(strike_idx).build_matrix(grid_i, r_d, kappa, eta, sigma, theta, delta_t, team);

            // Recompute solution
            // Compute perturbed solution
            device_DO_timestepping<Device, decltype(U_i)>(
                m1, m2, N, delta_t, theta, r_f,
                A0_solvers(strike_idx), A1_solvers(strike_idx), A2_solvers(strike_idx),
                bounds, U_i, Y_0_i, Y_1_i,
                A0_result_i, A1_result_i, A2_result_unshuf_i,
                U_shuffled_i, Y_1_shuffled_i, A2_result_shuffled_i, U_next_shuffled_i,
                team
            );

            // Get price and compute gradient
            double pert_price = U_i(index_s + index_v_pertubed*(m1+1));
            J(strike_idx, param) = (pert_price - base_price) / eps;
        });
        Kokkos::fence();
}
*/


/*

Test the basic jacobian computation

*/
void test_jacobian_method(){
    using Device = Kokkos::DefaultExecutionSpace;
    using timer = std::chrono::high_resolution_clock;

    // Market parameters
    const double S_0 = 100.0;
    const double V_0 = 0.04;
    const double T = 1.0;

    const double r_d = 0.025;
    const double r_f = 0.0;

    // Current parameter set
    const double rho = -0.9;
    const double sigma = 0.3;
    const double kappa = 1.5;
    const double eta = 0.04;
    
    // Numerical parameters
    const int m1 = 25;
    const int m2 = 20;

    const int N = 20;
    const double theta = 0.8;
    const double delta_t = T/N;

    const double eps = 1e-6;  // Perturbation size

    // Setup strikes and market data
    const int num_strikes = 5;
    std::vector<double> strikes(num_strikes);
    std::cout << "Strikes: ";
    for(int i = 0; i < num_strikes; ++i) {
        strikes[i] = 90.0 + i;  // Strikes
        std::cout << strikes[i] << ", ";
    }
    std::cout << "" << std::endl;

    std::cout << "Computing Jacobian for " << num_strikes << " strikes\n";
    std::cout << "Total PDE solves: " << num_strikes * 6 << std::endl;
    std::cout << "Base parameters: kappa=" << kappa << ", eta=" << eta 
              << ", sigma=" << sigma << ", rho=" << rho << ", V_0=" << V_0 << "\n";

    Kokkos::View<double*> market_prices("market_prices", num_strikes);
    auto h_market_prices = Kokkos::create_mirror_view(market_prices);

    // Compute market prices on host using Black-Scholes
    // Generate synthetic market prices
    BlackScholes::generate_market_data(S_0, T, r_d, strikes, h_market_prices);
    Kokkos::deep_copy(market_prices, h_market_prices);


    // Create solver arrays for each strike
    Kokkos::View<Device_A0_heston<Device>*> A0_solvers("A0_solvers", num_strikes);
    Kokkos::View<Device_A1_heston<Device>*> A1_solvers("A1_solvers", num_strikes);
    Kokkos::View<Device_A2_shuffled_heston<Device>*> A2_solvers("A2_solvers", num_strikes);
    
    // Initialize solvers
    auto h_A0 = Kokkos::create_mirror_view(A0_solvers);
    auto h_A1 = Kokkos::create_mirror_view(A1_solvers);
    auto h_A2 = Kokkos::create_mirror_view(A2_solvers);
    
    for(int i = 0; i < num_strikes; i++) {
        h_A0(i) = Device_A0_heston<Device>(m1, m2);
        h_A1(i) = Device_A1_heston<Device>(m1, m2);
        h_A2(i) = Device_A2_shuffled_heston<Device>(m1, m2);
    }
    Kokkos::deep_copy(A0_solvers, h_A0);
    Kokkos::deep_copy(A1_solvers, h_A1);
    Kokkos::deep_copy(A2_solvers, h_A2);

    // Create boundary conditions array
    Kokkos::View<Device_BoundaryConditions<Device>*> bounds_d("bounds_d", num_strikes);
    auto h_bounds = Kokkos::create_mirror_view(bounds_d);
    for(int i = 0; i < num_strikes; ++i) {
        h_bounds(i) = Device_BoundaryConditions<Device>(m1, m2, r_d, r_f, N, delta_t);
    }
    Kokkos::deep_copy(bounds_d, h_bounds);

    // Initialize grid views
    std::vector<GridViews> hostGrids;
    buildMultipleGridViews(hostGrids, num_strikes, m1, m2);
    for(int i = 0; i < num_strikes; ++i) {
        double K = strikes[i];
        auto h_Vec_s = Kokkos::create_mirror_view(hostGrids[i].device_Vec_s);
        auto h_Vec_v = Kokkos::create_mirror_view(hostGrids[i].device_Vec_v);
        auto h_Delta_s = Kokkos::create_mirror_view(hostGrids[i].device_Delta_s);
        auto h_Delta_v = Kokkos::create_mirror_view(hostGrids[i].device_Delta_v);

        //if we change something here we need to change it also inside the gradient method!
        //v0 grid views are rebuild there curretnly with 5.0, V_0+eps, 5.0/500
        Grid tempGrid(m1, 8*K, S_0, K, K/5, m2, 5.0, V_0, 5.0/500);
        
        for(int j = 0; j <= m1; j++) h_Vec_s(j) = tempGrid.Vec_s[j];
        for(int j = 0; j <= m2; j++) h_Vec_v(j) = tempGrid.Vec_v[j];
        for(int j = 0; j < m1; j++) h_Delta_s(j) = tempGrid.Delta_s[j];
        for(int j = 0; j < m2; j++) h_Delta_v(j) = tempGrid.Delta_v[j];

        Kokkos::deep_copy(hostGrids[i].device_Vec_s, h_Vec_s);
        Kokkos::deep_copy(hostGrids[i].device_Vec_v, h_Vec_v);
        Kokkos::deep_copy(hostGrids[i].device_Delta_s, h_Delta_s);
        Kokkos::deep_copy(hostGrids[i].device_Delta_v, h_Delta_v);
    }

    Kokkos::View<GridViews*> deviceGrids("deviceGrids", num_strikes);
    auto h_deviceGrids = Kokkos::create_mirror_view(deviceGrids);
    for(int i = 0; i < num_strikes; ++i) h_deviceGrids(i) = hostGrids[i];
    Kokkos::deep_copy(deviceGrids, h_deviceGrids);

    const int total_size = (m1+1)*(m2+1);
    
    // Create workspace
    DO_Workspace<Device> workspace(num_strikes, total_size);

    // Initialize initial conditions U_0
    Kokkos::View<double**> U_0("U_0", num_strikes, total_size);
    auto h_U_0 = Kokkos::create_mirror_view(U_0);

    // Fill initial conditions on host
    for(int inst = 0; inst < num_strikes; ++inst) {
        auto grid = hostGrids[inst];
        auto h_Vec_s = Kokkos::create_mirror_view(grid.device_Vec_s);
        Kokkos::deep_copy(h_Vec_s, grid.device_Vec_s);
        double K = strikes[inst];
        
        for(int j = 0; j <= m2; j++) {
            for(int i = 0; i <= m1; i++) {
                h_U_0(inst, i + j*(m1+1)) = std::max(h_Vec_s(i) - K, 0.0);
            }
        }
    }
    Kokkos::deep_copy(U_0, h_U_0);
    Kokkos::deep_copy(workspace.U, U_0);  // Copy initial condition to workspace

    // Storage for Jacobian matrix
    Kokkos::View<double**> J("Jacobian", num_strikes, 5);
    
    // Storage for base and perturbed prices
    Kokkos::View<double*> base_prices("base_prices", num_strikes);
    Kokkos::View<double**> pert_prices("pert_prices", num_strikes, 5); 

    // First compute base prices
    auto t_start = timer::now();

    compute_jacobian(
        // Market/model parameters  
        S_0, V_0, T,
        r_d, r_f,
        rho, sigma, kappa, eta,
        // Numerical parameters
        m1, m2, total_size, N, theta, delta_t,
        // Pre-computed structures
        num_strikes,
        A0_solvers, A1_solvers, A2_solvers,
        bounds_d, deviceGrids,
        U_0, workspace,
        // Output matrix
        J, base_prices,
        // Perturbation size (optional)
        eps
    );

    auto t_end = timer::now();
    std::cout << "Jacobian computation time: "
              << std::chrono::duration<double>(t_end - t_start).count()
              << " seconds" << std::endl;

    // Copy base_prices to host for printing
    auto h_base_prices = Kokkos::create_mirror_view(base_prices);
    Kokkos::deep_copy(h_base_prices, base_prices);
    
    // Print results
    std::cout << "\nCalibration Results:\n";
    std::cout << "==================\n";
    
    // Print some price comparisons
    std::cout << "\nPrice Comparisons (first 5 strikes):\n";
    std::cout << "Strike    Market    Model     Residual\n";
    std::cout << "----------------------------------------\n";
    for(int i = 0; i < std::min(5, num_strikes); i++) {
        std::cout << std::setw(8) << std::setprecision(3) << strikes[i] 
                  << std::setw(10 + 12) << std::setprecision(16) << h_market_prices(i)
                  << std::setw(10 + 12) << std::setprecision(16) << h_base_prices(i)
                  << "\n";
    }
    
    // Print Jacobian matrix
    
    auto h_J = Kokkos::create_mirror_view(J);
    Kokkos::deep_copy(h_J, J);
    
    // Column headers
    int precision = 12;
    int column_width = precision + 10; // Ensure enough space for large numbers
    int strike_width = 10;

    std::cout << "\nJacobian matrix:\n";
    std::cout << std::fixed << std::setprecision(precision);

    // Print header
    std::cout << std::setw(strike_width) << "Strike" 
              << std::setw(column_width) << "κ"
              << std::setw(column_width) << "η"
              << std::setw(column_width) << "σ"
              << std::setw(column_width) << "ρ"
              << std::setw(column_width) << "v0"
              << "\n";

    // Separator line (adjusted width)
    std::cout << std::string(strike_width + 5 * column_width, '-') << "\n";

    // Data rows
    for(int i = 0; i < num_strikes; i++) {
        std::cout << std::setw(strike_width) << strikes[i];
        for(int j = 0; j < 5; j++) {
            std::cout << std::setw(column_width) << h_J(i,j);
        }
        std::cout << "\n";
    }
    
}


void test_jacobian_computation(){
    Kokkos::initialize();
    {
      test_jacobian_method();
    }
    Kokkos::finalize();
   
  }