#include "heston_calibration.hpp"

//for the callable kernel DO method 
#include "device_solver.hpp"

#include <iostream>
#include <numeric>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>
#include <KokkosBlas2_gemv.hpp> // for gemv
#include <KokkosBlas3_gemm.hpp> // for gemm

void solve_5x5_device(
    const Kokkos::View<double**> &A_device,  // shape (5,5)
    const Kokkos::View<double*>  &b_device,  // shape (5)
    const Kokkos::View<double*>  &x_device   // shape (5)
)
{
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
)
{
  constexpr int N = 5;

  // 1. Build J^T J => [5 x 5]
  Kokkos::View<double**> JTJ("JTJ", N, N);
  KokkosBlas::gemm("T", "N", 
                   1.0, J, J,
                   0.0, JTJ);

  // 2. Add lambda on diagonal
  Kokkos::parallel_for("add_lambda_diag", N, KOKKOS_LAMBDA(const int i){
    JTJ(i,i) *= (1.0 + lambda);
  });

  // 3. Build J^T r => [5]
  Kokkos::View<double*> JTr("JTr", N);
  KokkosBlas::gemv("T", 1.0, J, residual, 0.0, JTr);

  // 4. Solve (JTJ) * delta = (JTr) on device with our manual 5x5 routine
  solve_5x5_device(JTJ, JTr, delta);
}



// Function to compute Jacobian matrix
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
    //precomputed Price extraction as well as Interpolation
    const Kokkos::View<int**>& price_indices,
    const Kokkos::View<double**>& v0_interp_data,
    // Output matrix
    Kokkos::View<double**>& J,
    // Optional: perturbation size
    const double eps = 1e-6
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
            
            // Get indices for price extraction
            const int index_s = price_indices(instance, 0);
            const int index_v = price_indices(instance, 1);

            // Now you can use these indices directly
            const double base_price = U_i(index_s + index_v*(m1+1));

            //FD approx of the parameters
            for(int param = 0; param < 5; param++) {
                // Special handling for V0 (param == 4)
               if(param == 4) {  // V0 perturbation
                    const int lower_idx = v0_interp_data(instance, 0);
                    const int upper_idx = v0_interp_data(instance, 1);
                    const double weight = v0_interp_data(instance, 2);
                    
                    // Get prices at S_0 for both variance levels
                    const double price_lower = U_i(index_s + lower_idx*(m1+1));
                    const double price_upper = U_i(index_s + upper_idx*(m1+1));
                    
                    // Interpolate price at perturbed V0
                    const double pert_price = price_lower + weight * (price_upper - price_lower);
                    J(instance, param) = (pert_price - base_price) / eps;
                }
                else {
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
                    J(instance, param) = (pert_price - base_price) / eps;
                }
            }

            // Loop over parameters for finite differences, this excludes v0 in the loop and treats it seperately
            //same result of course but surprisingly not a whole lot faster
            /*
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
                pert_prices(instance, param) = pert_price;
                J(instance, param) = (pert_price - base_price) / eps;
            }
            // Special handling for V0 (param == 4)
            const int param = 4;
            const int lower_idx = v0_interp_data(instance, 0);
            const int upper_idx = v0_interp_data(instance, 1);
            const double weight = v0_interp_data(instance, 2);
            
            // Get prices at S_0 for both variance levels
            const double price_lower = U_i(index_s + lower_idx*(m1+1));
            const double price_upper = U_i(index_s + upper_idx*(m1+1));
            
            // Interpolate price at perturbed V0
            const double pert_price = price_lower + weight * (price_upper - price_lower);
            pert_prices(instance, param) = pert_price;
            J(instance, param) = (pert_price - base_price) / eps;
            */
        });
        Kokkos::fence();
}


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
    const int m1 = 50;
    const int m2 = 25;

    const int N = 20;
    const double theta = 0.8;
    const double delta_t = T/N;

    const double eps = 1e-6;  // Perturbation size, should be Order of the error we are making in the Option computation

    // Setup strikes and market data
    const int num_strikes = 60;
    std::vector<double> strikes(num_strikes);
    for(int i = 0; i < num_strikes; ++i) {
        strikes[i] = 90.0 + i;  // Strikes 
    }

    std::cout << "Computing Jacobian for " << num_strikes << " strikes\n";
    std::cout << "Total PDE solves: " << num_strikes * 6 << std::endl;
    std::cout << "Base parameters: kappa=" << kappa << ", eta=" << eta 
              << ", sigma=" << sigma << ", rho=" << rho << ", V_0=" << V_0 << "\n";

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

    
    // Get indices for price extraction
    // Before your main computation, create a view for storing indices
    Kokkos::View<int**> price_indices("price_indices", num_strikes, 2);  // [num_strikes][2] for s and v indices
    auto h_price_indices = Kokkos::create_mirror_view(price_indices);

    // Compute indices on host for price extraction
    for(int strike_idx = 0; strike_idx < num_strikes; ++strike_idx) {
        auto grid = hostGrids[strike_idx];
        auto h_Vec_s = Kokkos::create_mirror_view(grid.device_Vec_s);
        auto h_Vec_v = Kokkos::create_mirror_view(grid.device_Vec_v);
        
        Kokkos::deep_copy(h_Vec_s, grid.device_Vec_s);
        Kokkos::deep_copy(h_Vec_v, grid.device_Vec_v);
        
        // Find s index
        for(int i = 0; i <= m1; i++) {
            if(std::abs(h_Vec_s(i) - S_0) < 1e-10) {
                h_price_indices(strike_idx, 0) = i;  // Store s index
                break;
            }
        }
        
        // Find v index
        for(int i = 0; i <= m2; i++) {
            if(std::abs(h_Vec_v(i) - V_0) < 1e-10) {
                h_price_indices(strike_idx, 1) = i;  // Store v index
                break;
            }
        }
    }
    
    // Copy indices to device
    Kokkos::deep_copy(price_indices, h_price_indices);  

    // Create a view for V0 interpolation data [num_strikes][4]:
    // [0]: lower_idx
    // [1]: upper_idx
    // [2]: interpolation weight
    // [3]: (v_upper - v_lower) for validation
    Kokkos::View<double**> v0_interp_data("v0_interp_data", num_strikes, 4);
    auto h_v0_interp_data = Kokkos::create_mirror_view(v0_interp_data);

    // Compute interpolation data on host
    const double v0_pert = V_0 + eps;
    for(int strike_idx = 0; strike_idx < num_strikes; ++strike_idx) {
        auto grid = hostGrids[strike_idx];
        auto h_Vec_v = Kokkos::create_mirror_view(grid.device_Vec_v);
        Kokkos::deep_copy(h_Vec_v, grid.device_Vec_v);
        
        // Find bracketing points
        for(int i = 0; i < m2; i++) {
            if(h_Vec_v(i) <= v0_pert && v0_pert <= h_Vec_v(i+1)) {
                // Store indices
                h_v0_interp_data(strike_idx, 0) = i;          // lower_idx
                h_v0_interp_data(strike_idx, 1) = i + 1;      // upper_idx
                
                // Compute and store interpolation weight
                double v_lower = h_Vec_v(i);
                double v_upper = h_Vec_v(i+1);
                h_v0_interp_data(strike_idx, 2) = (v0_pert - v_lower) / (v_upper - v_lower);  // weight
                h_v0_interp_data(strike_idx, 3) = v_upper - v_lower;  // store difference for validation
                break;
            }
        }
    }

    // Copy to device
    Kokkos::deep_copy(v0_interp_data, h_v0_interp_data);

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
        // Price extraction and interpolation data
        price_indices, v0_interp_data,
        // Output matrix
        J,
        // Perturbation size (optional)
        eps
    );

    auto t_end = timer::now();
    std::cout << "Jacobian computation time: "
              << std::chrono::duration<double>(t_end - t_start).count()
              << " seconds" << std::endl;

    Kokkos::View<double*> residuals("residuals", num_strikes);
    Kokkos::View<double*> delta("delta", 5);

    Kokkos::deep_copy(residuals, 1.0);

    // 4) Now call your on-device parameter update with a chosen lambda
    double lambda = 0.1;
    compute_parameter_update_on_device(J, residuals, lambda, delta);

    // 5) delta (size 5) now contains the solution (the LM update).
    //    If you need to access or print it on the host, do:
    auto delta_host = Kokkos::create_mirror_view(delta);
    Kokkos::deep_copy(delta_host, delta);
    std::cout << "Delta: ";
    for(int i = 0; i < 5; i++){
        std::cout << delta_host(i) << " ";
    }
    std::cout << std::endl;

    auto t_end_second = timer::now();
    std::cout << "Updating parameters: "
              << std::chrono::duration<double>(t_end_second - t_start).count()
              << " seconds" << std::endl;
    
    // Print Jacobian matrix
    /*
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
    */
}


void test_heston_calibration(){
  Kokkos::initialize();
  {
    test_jacobian_method();
  }
  Kokkos::finalize();
 
}