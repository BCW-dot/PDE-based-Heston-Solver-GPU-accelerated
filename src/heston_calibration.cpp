#include "heston_calibration.hpp"

//for the callable kernel DO method 
#include "device_solver.hpp"
//for jacobian
#include "jacobian_computation.hpp"
//for implied vols
#include "bs.hpp"

#include <iostream>
#include <numeric>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <string>
#include <vector>

#include <fstream>  // For std::ofstream
#include <iostream> // For std::cerr, std::cout

#include <Kokkos_Core.hpp>


//This calibrates to european call options
void test_calibration_european(){
    std::cout<< "Testing european" << std::endl;
    using Device = Kokkos::DefaultExecutionSpace;
    using timer = std::chrono::high_resolution_clock;

    // Market parameters
    const double S_0 = 100.0;
    const double T = 1.0;

    const double r_d = 0.025;
    const double r_f = 0.0;

    // Current parameter set
    const double V_0 = 0.04;
    
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

    const double eps = 1e-6;  // Perturbation size for finite difference approximation in Jacobian matrix

    // Setup strikes and market data
    const int num_strikes = 60;
    std::vector<double> strikes(num_strikes);
    std::cout << "Strikes: ";
    for(int i = 0; i < num_strikes; ++i) {
        strikes[i] = S_0 * 0.7 + i * 1;//S_0 * (0.5 + i * 0.01); //S_0 - num_strikes + i;  // Strikes
        std::cout << strikes[i] << ", ";
    }
    std::cout << "" << std::endl;

    const int max_iter = 15;
    const double tol = 0.1;//0.001 * num_strikes * (S_0/100.0)*(S_0/100.0); //0.01;


    std::cout << "Computing Jacobian for " << num_strikes << " strikes\n";
    std::cout << "Total PDE solves per iteration: " << num_strikes * (1 + 5 + 1) << std::endl; //base_price + param_pertubation + new error computation
    std::cout << "Base parameters: kappa=" << kappa << ", eta=" << eta 
              << ", sigma=" << sigma << ", rho=" << rho << ", V_0=" << V_0 << "\n";
    std::cout << "Tolerance: " << tol << std::endl;

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

    // Create views for tracking errors
    Kokkos::View<double*> current_residuals("current_residuals", num_strikes);
    Kokkos::View<double*> new_residuals("new_residuals", num_strikes);
    Kokkos::View<double*> delta("delta", 5);

    // Current parameters that will be updated
    // will be implicitely copied ifrom host to device. This is faster than keeping it on device
    double current_kappa = kappa;
    double current_eta = eta;
    double current_sigma = sigma;
    double current_rho = rho;
    double current_v0 = V_0;

    double lambda = 0.01; // Initial LM parameter
    bool converged = false;

    double final_error = 100.0; // for plot information
    int iteration_count = 0;

    // Define bounds for updating
    static constexpr double rho_min = -1.0, rho_max = 1.0;

    /*
    static constexpr double kappa_min = 1e-6, kappa_max = 20.0;
    static constexpr double eta_min = 1e-6, eta_max = 1.0;
    static constexpr double sigma_min = 1e-6, sigma_max = 5.0;
    static constexpr double v0_min = 1e-6, v0_max = 1.0;
    */
    
    
    // Main iteration loop
    auto iter_start = timer::now();
    for(int iter = 0; iter < max_iter && !converged; iter++) {
        //std::cout << "\nIteration " << iter + 1 << " of " << max_iter << std::endl;

        //std::cout << "Current parameters: κ=" << current_kappa 
                //<< ", η=" << current_eta 
                //<< ", σ=" << current_sigma 
                //<< ", ρ=" << current_rho 
                //<< ", v₀=" << current_v0 << std::endl;

        //need to reset the Inital condition
        Kokkos::deep_copy(workspace.U, U_0); 
        // Compute Jacobian and base prices with current parameters
        compute_jacobian(
            S_0, current_v0, T,
            r_d, r_f,
            current_rho, current_sigma, current_kappa, current_eta,
            m1, m2, total_size, N, theta, delta_t,
            num_strikes,
            A0_solvers, A1_solvers, A2_solvers,
            bounds_d, deviceGrids,
            U_0, workspace,
            J, base_prices,
            eps
        );

        
        //printing Jacobian
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

        
        //base prices are already computed in compute_jacobian and stored in base_price
        // Compute current residuals
        Kokkos::parallel_for("compute_residuals", num_strikes, 
            KOKKOS_LAMBDA(const int i) {
                current_residuals(i) = market_prices(i) - base_prices(i);
        });
        Kokkos::fence();

        // Compute parameter update
        compute_parameter_update_on_device(J, current_residuals, lambda, delta);

        // Get delta on host
        // maybe it would be faster to store the new_paras on device so we dont have a data transfer here
        auto h_delta = Kokkos::create_mirror_view(delta);
        Kokkos::deep_copy(h_delta, delta);

        // Try new parameters
        double new_kappa = std::max(1e-3, current_kappa + h_delta(0));
        double new_eta = std::max(1e-2, current_eta + h_delta(1));
        double new_sigma = std::max(1e-2, current_sigma + h_delta(2));
        double new_rho = std::min(rho_max, std::max(rho_min, current_rho + h_delta(3)));
        double new_v0 = std::max(1e-2, current_v0 + h_delta(4));
        
        
        /*
        double new_kappa = std::min(kappa_max, std::max(kappa_min, current_kappa + h_delta(0)));
        double new_eta = std::min(eta_max, std::max(eta_min, current_eta + h_delta(1)));
        double new_sigma = std::min(sigma_max, std::max(sigma_min, current_sigma + h_delta(2)));
        double new_rho = std::min(rho_max, std::max(rho_min, current_rho + h_delta(3)));
        double new_v0 = std::min(v0_max, std::max(v0_min, current_v0 + h_delta(4)));
        */

        //print the new params
        /*
        std::cout << std::fixed << std::setprecision(10);
        std::vector<double> new_params = {new_kappa, new_eta, new_sigma, new_rho, new_v0};
        std::cout << "The new parameters: ";
        for(int i =0; i< new_params.size(); i++){
            std::cout << new_params[i] << ", ";
        }
        std::cout << "\n";
        */
        

        // Compute delta norm (matching np.linalg.norm(delta))
        double delta_norm = 0.0;
        for(int i = 0; i < 5; ++i) {
            delta_norm += h_delta(i) * h_delta(i);
        }
        delta_norm = std::sqrt(delta_norm);

        //computing current error
        double current_error = 0;
        auto h_current_residuals = Kokkos::create_mirror_view(current_residuals);
        Kokkos::deep_copy(h_current_residuals, current_residuals);

        for(int i = 0; i < num_strikes; i++) {
            current_error += h_current_residuals(i) * h_current_residuals(i);
        }
        //std::cout << "Current error: " << current_error << std::endl;

        // Check convergence
        if(delta_norm < tol ||   
        (current_error < tol)) {
            converged = true;
            std::cout << "Converged!" << std::endl;
            std::cout << "Error Toleranze: " << tol << std::endl;
            std::cout << "Converged Error: " << current_error << std::endl;
            std::cout << "Converged Delta Norm: " << delta_norm << std::endl;

            // Update parameters to final values (matching params = new_params)
            current_kappa = new_kappa;
            current_eta = new_eta;
            current_sigma = new_sigma;
            current_rho = new_rho;
            current_v0 = new_v0;

            final_error = current_error;
            iteration_count = iter + 1;
            break; // Exit the loop
        }

        // Compute new prices with updated parameters
        Kokkos::deep_copy(workspace.U, U_0); // reset init condition
        //Rebuilding the variance direction for new_v0 is done inside the kernel
        compute_base_prices(S_0, new_v0, T,
                r_d, r_f,
                new_rho, new_sigma, new_kappa, new_eta,
                m1, m2, total_size, N, theta, delta_t,
                num_strikes,
                A0_solvers, A1_solvers, A2_solvers,
                bounds_d, deviceGrids,
                workspace,
                base_prices
            );

        //print the new base prices
        /*
        std::cout << std::fixed << std::setprecision(12);
        auto h_base_prices   = Kokkos::create_mirror_view(base_prices);
        Kokkos::deep_copy(h_base_prices,   base_prices);
        std::cout << "The new base prices: ";
        for(int i =0; i< num_strikes; i++){
            std::cout << h_base_prices(i) << ", ";
        }
        std::cout << "\n";
        */

        // Compute new residuals
        Kokkos::parallel_for("compute_new_residuals", num_strikes, 
            KOKKOS_LAMBDA(const int i) {
                new_residuals(i) = market_prices(i) - base_prices(i);
        });
        Kokkos::fence();

        // Compute new error norms
        double new_error = 0.0;
        auto h_new_residuals = Kokkos::create_mirror_view(new_residuals);
        Kokkos::deep_copy(h_new_residuals, new_residuals);

        for(int i = 0; i < num_strikes; i++) {
            new_error += h_new_residuals(i) * h_new_residuals(i);
        }

        //std::cout << "Current error: " << std::sqrt(current_error) << std::endl;
        //std::cout << "New error: " << new_error << std::endl;
        

        // Update parameters based on error improvement
        if(new_error < current_error) {
            current_kappa = new_kappa;
            current_eta = new_eta;
            current_sigma = new_sigma;
            current_rho = new_rho;
            current_v0 = new_v0;
            lambda = std::max(lambda / 10.0, 1e-7);  // Decrease lambda but not too small
        } 
        else {
            lambda = std::min(lambda * 10.0, 1e7);  // Increase lambda but not too large
        }

        //auto iter_end = timer::now();
        //std::cout << "Iteration time: "
                //<< std::chrono::duration<double>(iter_end - iter_start).count()
                //<< " seconds" << std::endl;
        //if delta norm and error do not get lower than "tol" and max iter is hit
        final_error = std::min(new_error,current_error);
        iteration_count = iter + 1;
    }

    // Print final results
    std::cout << "\nFinal calibrated parameters:" << std::endl;
    std::cout << "κ = " << current_kappa << std::endl;
    std::cout << "η = " << current_eta << std::endl;
    std::cout << "σ = " << current_sigma << std::endl;
    std::cout << "ρ = " << current_rho << std::endl;
    std::cout << "v₀ = " << current_v0 << std::endl;
    std::cout << "final error = " << final_error << std::endl;
    std::cout << "total iterations = " << iteration_count << std::endl;
    int total_pde_solves = num_strikes * (1 + 5 + 1) * iteration_count - num_strikes;
    std::cout << "Total PDE solves: " << total_pde_solves << std::endl;

    auto t_end_second = timer::now();
    std::cout << "Total time after Updating parameters: "
              << std::chrono::duration<double>(t_end_second - t_start).count()
              << " seconds" << std::endl;

    //for ploting
    // Copy device -> host for market and fitted prices
    auto h_base_prices   = Kokkos::create_mirror_view(base_prices);
    Kokkos::deep_copy(h_base_prices,   base_prices);
    Kokkos::deep_copy(h_market_prices, market_prices);

    //computing implied vols
    double epsilon = 0.01;
    // Before the export loop, calculate IVs and their differences
    std::vector<double> iv_differences(num_strikes);
    for(int i = 0; i < num_strikes; i++) {
        double K = strikes[i];
        double market_price = h_market_prices(i);
        double heston_price = h_base_prices(i);
        
        // Compute implied vol from market price 
        double market_impl_vol = BlackScholes::reverse_BS(1, S_0, K, r_d, T, 0.5, market_price, epsilon);
        
        // Compute implied vol from calibrated Heston price
        double heston_impl_vol = BlackScholes::reverse_BS(1, S_0, K, r_d, T, 0.5, heston_price, epsilon);
        
        iv_differences[i] = std::abs(market_impl_vol - heston_impl_vol);
        // Compare - should both be close to 0.2
        //std::cout << "Strike " << K << ": Market IV = " << market_impl_vol 
                  //<< ", Heston IV = " << heston_impl_vol << std::endl;
    }

    // ================================================================
    // ============ Export final results to CSV for Python ============
    // ================================================================

    // Choose a filename
    std::string csv_filename = "fitted_heston_vs_market.csv";
    std::ofstream out(csv_filename);
    if (!out.is_open()) {
        std::cerr << "Error: Cannot open file for writing: " << csv_filename << std::endl;
        return;
    }

    // Write a title/comment line:
    //   - number of options
    //   - total runtime
    //   - final error
    // Write metadata with calibrated parameters
    double total_time = std::chrono::duration<double>(t_end_second - t_start).count();
    out << "# " << num_strikes 
        << " options, Time=" << total_time 
        << " s, FinalError=" << final_error
        << ", iterationCount=" << iteration_count
        << ", TotalPdeSolves=" << total_pde_solves
        << ", init_kappa=" << kappa
        << ", init_eta="  << eta
        << ", init_sigma="<< sigma
        << ", init_rho="  << rho
        << ", init_v0="   << V_0
        << ", kappa="     << current_kappa
        << ", eta="       << current_eta
        << ", sigma="     << current_sigma
        << ", rho="       << current_rho
        << ", v0="        << current_v0
        << "\n";

    // Write CSV header
    out << "Strike,MarketPrice,FittedPrice,IVDifference\n";

    // Write each line: Strike, MarketPrice, FittedHestonPrice
    for (int i = 0; i < num_strikes; ++i) {
        out << strikes[i] 
            << "," << h_market_prices(i) 
            << "," << h_base_prices(i) 
            << "," << iv_differences[i]
            << "\n";
    }

    out.close();
    std::cout << "Exported final results to " << csv_filename << std::endl;
}

//This calibrates to american call options
void test_calibration_american(){
    std::cout<< "Testing american" << std::endl;
    using Device = Kokkos::DefaultExecutionSpace;
    using timer = std::chrono::high_resolution_clock;

    // Market parameters
    const double S_0 = 100.0;
    const double T = 1.0;

    const double r_d = 0.025;
    const double r_f = 0.0;

    // Current parameter set
    const double V_0 = 0.04;
    
    
    const double rho = -0.9;
    const double sigma = 0.3;
    const double kappa = 1.5;
    const double eta = 0.04;
    
   
   /*
    const double rho = 0.03;
    const double sigma = 0.02;
    const double kappa = 3.0;
    const double eta = 0.01;
    */
    
    
    // Numerical parameters
    const int m1 = 50;
    const int m2 = 25;

    const int N = 20;
    const double theta = 0.8;
    const double delta_t = T/N;

    const double eps = 1e-6;  // Perturbation size

    // Setup strikes and market data
    const int num_strikes = 10;
    std::vector<double> strikes(num_strikes);
    std::cout << "Strikes: ";
    for(int i = 0; i < num_strikes; ++i) {
        strikes[i] = S_0 * 0.55 + i * 1.0;//S_0 * (0.5 + i * 0.01); //S_0 - num_strikes + i;  // Strikes
        std::cout << strikes[i] << ", ";
    }
    std::cout << "" << std::endl;

    const int max_iter = 15;
    const double tol = 0.1;//0.001 * num_strikes * (S_0/100.0)*(S_0/100.0); //0.01;



    std::cout << "Computing Jacobian for " << num_strikes << " strikes\n";
    std::cout << "Total PDE solves: " << num_strikes * (6 + 1) << std::endl; //base_price + param_pertubation + new error computation
    std::cout << "Base parameters: kappa=" << kappa << ", eta=" << eta 
              << ", sigma=" << sigma << ", rho=" << rho << ", V_0=" << V_0 << "\n";
    std::cout << "Tolerance: " << tol << std::endl;

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

    // Create views for tracking errors
    Kokkos::View<double*> current_residuals("current_residuals", num_strikes);
    Kokkos::View<double*> new_residuals("new_residuals", num_strikes);
    Kokkos::View<double*> delta("delta", 5);

    // Current parameters that will be updated
    // will be implicitely copied ifrom host to device. This is faster than keeping it on device
    double current_kappa = kappa;
    double current_eta = eta;
    double current_sigma = sigma;
    double current_rho = rho;
    double current_v0 = V_0;

    double lambda = 0.01; // Initial LM parameter
    bool converged = false;

    double final_error = 100.0; // for plot information
    int iteration_count = 0;

    // Define bounds for updating
    static constexpr double rho_min = -1.0, rho_max = 1.0;

    /*
    static constexpr double kappa_min = 1e-6, kappa_max = 20.0;
    static constexpr double eta_min = 1e-6, eta_max = 1.0;
    static constexpr double sigma_min = 1e-6, sigma_max = 5.0;
    static constexpr double v0_min = 1e-6, v0_max = 1.0;
    */
    
    
    // Main iteration loop
    auto iter_start = timer::now();
    for(int iter = 0; iter < max_iter && !converged; iter++) {
        std::cout << "\nIteration " << iter + 1 << " of " << max_iter << std::endl;

        //std::cout << "Current parameters: κ=" << current_kappa 
                //<< ", η=" << current_eta 
                //<< ", σ=" << current_sigma 
                //<< ", ρ=" << current_rho 
                //<< ", v₀=" << current_v0 << std::endl;

        //need to reset the Inital condition
        Kokkos::deep_copy(workspace.U, U_0); 
        // Compute Jacobian and base prices with current parameters

        
        compute_jacobian_american(
            S_0, current_v0, T,
            r_d, r_f,
            current_rho, current_sigma, current_kappa, current_eta,
            m1, m2, total_size, N, theta, delta_t,
            num_strikes,
            A0_solvers, A1_solvers, A2_solvers,
            bounds_d, deviceGrids,
            U_0, workspace,
            J, base_prices,
            eps
        );
        

        /*
        compute_jacobian(
            S_0, current_v0, T,
            r_d, r_f,
            current_rho, current_sigma, current_kappa, current_eta,
            m1, m2, total_size, N, theta, delta_t,
            num_strikes,
            A0_solvers, A1_solvers, A2_solvers,
            bounds_d, deviceGrids,
            U_0, workspace,
            J, base_prices,
            eps
        );
        */
        
        //printing Jacobian
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

        
        //base prices are already computed in compute_jacobian and stored in base_price
        // Compute current residuals
        Kokkos::parallel_for("compute_residuals", num_strikes, 
            KOKKOS_LAMBDA(const int i) {
                current_residuals(i) = market_prices(i) - base_prices(i);
        });
        Kokkos::fence();

        // Compute parameter update
        compute_parameter_update_on_device(J, current_residuals, lambda, delta);

        // Get delta on host
        // maybe it would be faster to store the new_paras on device so we dont have a data transfer here
        auto h_delta = Kokkos::create_mirror_view(delta);
        Kokkos::deep_copy(h_delta, delta);

        // Try new parameters
        double new_kappa = std::max(1e-3, current_kappa + h_delta(0));
        double new_eta = std::max(1e-2, current_eta + h_delta(1));
        double new_sigma = std::max(1e-2, current_sigma + h_delta(2));
        double new_rho = std::min(rho_max, std::max(rho_min, current_rho + h_delta(3)));
        double new_v0 = std::max(1e-2, current_v0 + h_delta(4));
        
        
        /*
        double new_kappa = std::min(kappa_max, std::max(kappa_min, current_kappa + h_delta(0)));
        double new_eta = std::min(eta_max, std::max(eta_min, current_eta + h_delta(1)));
        double new_sigma = std::min(sigma_max, std::max(sigma_min, current_sigma + h_delta(2)));
        double new_rho = std::min(rho_max, std::max(rho_min, current_rho + h_delta(3)));
        double new_v0 = std::min(v0_max, std::max(v0_min, current_v0 + h_delta(4)));
        */

        //print the new params
        /*
        std::cout << std::fixed << std::setprecision(10);
        std::vector<double> new_params = {new_kappa, new_eta, new_sigma, new_rho, new_v0};
        std::cout << "The new parameters: ";
        for(int i =0; i< new_params.size(); i++){
            std::cout << new_params[i] << ", ";
        }
        std::cout << "\n";
        */
        

        // Compute delta norm (matching np.linalg.norm(delta))
        double delta_norm = 0.0;
        for(int i = 0; i < 5; ++i) {
            delta_norm += h_delta(i) * h_delta(i);
        }
        delta_norm = std::sqrt(delta_norm);

        //computing current error
        double current_error = 0;
        auto h_current_residuals = Kokkos::create_mirror_view(current_residuals);
        Kokkos::deep_copy(h_current_residuals, current_residuals);

        for(int i = 0; i < num_strikes; i++) {
            current_error += h_current_residuals(i) * h_current_residuals(i);
        }
        //std::cout << "Current error: " << current_error << std::endl;

        // Check convergence
        if(delta_norm < tol ||   
        (current_error < tol)) {
            converged = true;
            std::cout << "Converged!" << std::endl;
            std::cout << "Error Toleranze: " << tol << std::endl;
            std::cout << "Converged Error: " << current_error << std::endl;
            std::cout << "Converged Delta Norm: " << delta_norm << std::endl;

            // Update parameters to final values (matching params = new_params)
            current_kappa = new_kappa;
            current_eta = new_eta;
            current_sigma = new_sigma;
            current_rho = new_rho;
            current_v0 = new_v0;

            final_error = current_error;
            iteration_count = iter + 1;
            break; // Exit the loop
        }

        // Compute new prices with updated parameters
        Kokkos::deep_copy(workspace.U, U_0); // reset init condition
        //Rebuilding the variance direction for new_v0 is done inside the kernel

        
        compute_base_prices_american(S_0, new_v0, T,
                r_d, r_f,
                new_rho, new_sigma, new_kappa, new_eta,
                m1, m2, total_size, N, theta, delta_t,
                num_strikes,
                A0_solvers, A1_solvers, A2_solvers,
                bounds_d, deviceGrids, U_0,
                workspace,
                base_prices
            );
        

        //print the new base prices
        /*
        std::cout << std::fixed << std::setprecision(12);
        auto h_base_prices   = Kokkos::create_mirror_view(base_prices);
        Kokkos::deep_copy(h_base_prices,   base_prices);
        std::cout << "The new base prices: ";
        for(int i =0; i< num_strikes; i++){
            std::cout << h_base_prices(i) << ", ";
        }
        std::cout << "\n";
        */

        // Compute new residuals
        Kokkos::parallel_for("compute_new_residuals", num_strikes, 
            KOKKOS_LAMBDA(const int i) {
                new_residuals(i) = market_prices(i) - base_prices(i);
        });
        Kokkos::fence();

        // Compute new error norms
        double new_error = 0.0;
        auto h_new_residuals = Kokkos::create_mirror_view(new_residuals);
        Kokkos::deep_copy(h_new_residuals, new_residuals);

        for(int i = 0; i < num_strikes; i++) {
            new_error += h_new_residuals(i) * h_new_residuals(i);
        }

        //std::cout << "Current error: " << std::sqrt(current_error) << std::endl;
        //std::cout << "New error: " << new_error << std::endl;
        

        // Update parameters based on error improvement
        if(new_error < current_error) {
            current_kappa = new_kappa;
            current_eta = new_eta;
            current_sigma = new_sigma;
            current_rho = new_rho;
            current_v0 = new_v0;
            lambda = std::max(lambda / 10.0, 1e-7);  // Decrease lambda but not too small
        } 
        else {
            lambda = std::min(lambda * 10.0, 1e7);  // Increase lambda but not too large
        }

        //auto iter_end = timer::now();
        //std::cout << "Iteration time: "
                //<< std::chrono::duration<double>(iter_end - iter_start).count()
                //<< " seconds" << std::endl;
        //if delta norm and error do not get lower than "tol" and max iter is hit
        final_error = std::min(new_error,current_error);
        iteration_count = iter + 1;
    }

    // Print final results
    std::cout << "\nFinal calibrated parameters:" << std::endl;
    std::cout << "κ = " << current_kappa << std::endl;
    std::cout << "η = " << current_eta << std::endl;
    std::cout << "σ = " << current_sigma << std::endl;
    std::cout << "ρ = " << current_rho << std::endl;
    std::cout << "v₀ = " << current_v0 << std::endl;
    std::cout << "final error = " << final_error << std::endl;
    std::cout << "total iterations = " << iteration_count << std::endl;

    auto t_end_second = timer::now();
    std::cout << "Total time after Updating parameters: "
              << std::chrono::duration<double>(t_end_second - t_start).count()
              << " seconds" << std::endl;


    //for ploting
    // Copy device -> host for market and fitted prices
    auto h_base_prices   = Kokkos::create_mirror_view(base_prices);
    Kokkos::deep_copy(h_base_prices,   base_prices);
    Kokkos::deep_copy(h_market_prices, market_prices);

    //computing implied vols
    double epsilon = 0.01;
    // Before the export loop, calculate IVs and their differences
    std::vector<double> iv_differences(num_strikes);
    for(int i = 0; i < num_strikes; i++) {
        double K = strikes[i];
        double market_price = h_market_prices(i);
        double heston_price = h_base_prices(i);
        
        // Compute implied vol from market price 
        double market_impl_vol = BlackScholes::reverse_BS(1, S_0, K, r_d, T, 0.5, market_price, epsilon);
        
        // Compute implied vol from calibrated Heston price
        double heston_impl_vol = BlackScholes::reverse_BS(1, S_0, K, r_d, T, 0.5, heston_price, epsilon);
        
        iv_differences[i] = std::abs(market_impl_vol - heston_impl_vol);
        // Compare - should both be close to 0.2
        //std::cout << "Strike " << K << ": Market IV = " << market_impl_vol 
                  //<< ", Heston IV = " << heston_impl_vol << std::endl;
    }

    // ================================================================
    // ============ Export final results to CSV for Python ============
    // ================================================================

    // Choose a filename
    std::string csv_filename = "fitted_heston_vs_market_american.csv";
    std::ofstream out(csv_filename);
    if (!out.is_open()) {
        std::cerr << "Error: Cannot open file for writing: " << csv_filename << std::endl;
        return;
    }

    // Write a title/comment line:
    //   - number of options
    //   - total runtime
    //   - final error
    // Write metadata with calibrated parameters
    double total_time = std::chrono::duration<double>(t_end_second - t_start).count();
    out << "# " << num_strikes 
        << " options, Time=" << total_time 
        << " s, FinalError=" << final_error
        << ", iterationCount=" << iteration_count
        << ", init_kappa=" << kappa
        << ", init_eta="  << eta
        << ", init_sigma="<< sigma
        << ", init_rho="  << rho
        << ", init_v0="   << V_0
        << ", kappa="     << current_kappa
        << ", eta="       << current_eta
        << ", sigma="     << current_sigma
        << ", rho="       << current_rho
        << ", v0="        << current_v0
        << "\n";

    // Write CSV header
    out << "Strike,MarketPrice,FittedPrice,IVDifference\n";

    // Write each line: Strike, MarketPrice, FittedHestonPrice
    for (int i = 0; i < num_strikes; ++i) {
        out << strikes[i] 
            << "," << h_market_prices(i) 
            << "," << h_base_prices(i) 
            << "," << iv_differences[i]
            << "\n";
    }

    out.close();
    std::cout << "Exported final results to " << csv_filename << std::endl;
}

//This calibrates to european calls with an underlying paying dividends
void test_calibration_dividends(){
    std::cout<< "Testing european with dividends" << std::endl;
    using Device = Kokkos::DefaultExecutionSpace;
    using timer = std::chrono::high_resolution_clock;

    // Market parameters
    const double S_0 = 100.0;
    const double T = 1.0;

    const double r_d = 0.025;
    const double r_f = 0.0;

    // Current parameter set
    const double V_0 = 0.04;
    
    const double rho = -0.9;
    const double sigma = 0.3;
    const double kappa = 1.5;
    const double eta = 0.04;
    
   
   /*
    const double rho = 0.03;
    const double sigma = 0.02;
    const double kappa = 3.0;
    const double eta = 0.01;
    */
    
    
    // Numerical parameters
    const int m1 = 50;
    const int m2 = 25;

    const int N = 20;
    const double theta = 0.8;
    const double delta_t = T/N;

    const double eps = 1e-6;  // Perturbation size

    // Setup strikes and market data
    const int num_strikes = 10;
    std::vector<double> strikes(num_strikes);
    std::cout << "Strikes: ";
    for(int i = 0; i < num_strikes; ++i) {
        strikes[i] = S_0 * 0.85 + i * 1.1;//S_0 * (0.5 + i * 0.01); //S_0 - num_strikes + i;  // Strikes
        std::cout << strikes[i] << ", ";
    }
    std::cout << "" << std::endl;

    const int max_iter = 20;
    const double tol = 0.1;//0.001 * num_strikes * (S_0/100.0)*(S_0/100.0); //0.01;

    //Handling dividend host device transfer
    //{0.0, 0.0, 0.0, 0.0}
    std::vector<double> dividend_dates = {0.2, 0.4, 0.6, 0.8};
    std::vector<double> dividend_amounts = {0.2, 0.2, 0.2, 0.2};
    std::vector<double> dividend_percentages = {0.001, 0.001, 0.001, 0.001};//{0.02, 0.02, 0.02, 0.02};

    // On host side, create views for dividend data
    Kokkos::View<double*> d_dividend_dates("dividend_dates", dividend_dates.size());
    Kokkos::View<double*> d_dividend_amounts("dividend_amounts", dividend_amounts.size());
    Kokkos::View<double*> d_dividend_percentages("dividend_percentages", dividend_percentages.size());

    // Copy dividend data to device
    auto h_dividend_dates = Kokkos::create_mirror_view(d_dividend_dates);
    auto h_dividend_amounts = Kokkos::create_mirror_view(d_dividend_amounts);
    auto h_dividend_percentages = Kokkos::create_mirror_view(d_dividend_percentages);

    for(size_t i = 0; i < dividend_dates.size(); i++) {
        h_dividend_dates(i) = dividend_dates[i];
        h_dividend_amounts(i) = dividend_amounts[i];
        h_dividend_percentages(i) = dividend_percentages[i];
    }

    Kokkos::deep_copy(d_dividend_dates, h_dividend_dates);
    Kokkos::deep_copy(d_dividend_amounts, h_dividend_amounts);
    Kokkos::deep_copy(d_dividend_percentages, h_dividend_percentages);

    const int num_dividends = dividend_dates.size();


    std::cout << "Computing Jacobian for " << num_strikes << " strikes\n";
    std::cout << "Total PDE solves: " << num_strikes * (6 + 1) << std::endl; //base_price + param_pertubation + new error computation
    std::cout << "Base parameters: kappa=" << kappa << ", eta=" << eta 
              << ", sigma=" << sigma << ", rho=" << rho << ", V_0=" << V_0 << "\n";
    std::cout << "Tolerance: " << tol << std::endl;

    Kokkos::View<double*> market_prices("market_prices", num_strikes);
    auto h_market_prices = Kokkos::create_mirror_view(market_prices);

    // Compute market prices on host using Black-Scholes
    // Generate synthetic market prices

    BlackScholes::generate_market_data_with_dividends(S_0, T, r_d, strikes, dividend_dates, dividend_amounts, dividend_percentages, h_market_prices);
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

    // Create views for tracking errors
    Kokkos::View<double*> current_residuals("current_residuals", num_strikes);
    Kokkos::View<double*> new_residuals("new_residuals", num_strikes);
    Kokkos::View<double*> delta("delta", 5);

    // Current parameters that will be updated
    // will be implicitely copied ifrom host to device. This is faster than keeping it on device
    double current_kappa = kappa;
    double current_eta = eta;
    double current_sigma = sigma;
    double current_rho = rho;
    double current_v0 = V_0;

    double lambda = 0.01; // Initial LM parameter
    bool converged = false;

    double final_error = 100.0; // for plot information
    int iteration_count = 0;

    // Define bounds for updating
    static constexpr double rho_min = -1.0, rho_max = 1.0;

    /*
    static constexpr double kappa_min = 1e-6, kappa_max = 20.0;
    static constexpr double eta_min = 1e-6, eta_max = 1.0;
    static constexpr double sigma_min = 1e-6, sigma_max = 5.0;
    static constexpr double v0_min = 1e-6, v0_max = 1.0;
    */
    
    
    // Main iteration loop
    auto iter_start = timer::now();
    for(int iter = 0; iter < max_iter && !converged; iter++) {
        std::cout << "\nIteration " << iter + 1 << " of " << max_iter << std::endl;

        //std::cout << "Current parameters: κ=" << current_kappa 
                //<< ", η=" << current_eta 
                //<< ", σ=" << current_sigma 
                //<< ", ρ=" << current_rho 
                //<< ", v₀=" << current_v0 << std::endl;

        //need to reset the Inital condition
        Kokkos::deep_copy(workspace.U, U_0); 
        // Compute Jacobian and base prices with current parameters

        compute_jacobian_dividends(
            S_0, current_v0, T,
            r_d, r_f,
            current_rho, current_sigma, current_kappa, current_eta,
            m1, m2, total_size, N, theta, delta_t,
            num_strikes,
            A0_solvers, A1_solvers, A2_solvers,
            bounds_d, deviceGrids,
            U_0, workspace,
            num_dividends,
            d_dividend_dates,     // Use device view instead of vector
            d_dividend_amounts,   // Use device view instead of vector
            d_dividend_percentages, // Use device view instead of vector 
            J, base_prices,
            eps
        );

        
        //printing Jacobian
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

        
        //base prices are already computed in compute_jacobian and stored in base_price
        // Compute current residuals
        Kokkos::parallel_for("compute_residuals", num_strikes, 
            KOKKOS_LAMBDA(const int i) {
                current_residuals(i) = market_prices(i) - base_prices(i);
        });
        Kokkos::fence();

        // Compute parameter update
        compute_parameter_update_on_device(J, current_residuals, lambda, delta);

        // Get delta on host
        // maybe it would be faster to store the new_paras on device so we dont have a data transfer here
        auto h_delta = Kokkos::create_mirror_view(delta);
        Kokkos::deep_copy(h_delta, delta);

        // Try new parameters
        double new_kappa = std::max(1e-3, current_kappa + h_delta(0));
        double new_eta = std::max(1e-2, current_eta + h_delta(1));
        double new_sigma = std::max(1e-2, current_sigma + h_delta(2));
        double new_rho = std::min(rho_max, std::max(rho_min, current_rho + h_delta(3)));
        double new_v0 = std::max(1e-2, current_v0 + h_delta(4));
        
        
        /*
        double new_kappa = std::min(kappa_max, std::max(kappa_min, current_kappa + h_delta(0)));
        double new_eta = std::min(eta_max, std::max(eta_min, current_eta + h_delta(1)));
        double new_sigma = std::min(sigma_max, std::max(sigma_min, current_sigma + h_delta(2)));
        double new_rho = std::min(rho_max, std::max(rho_min, current_rho + h_delta(3)));
        double new_v0 = std::min(v0_max, std::max(v0_min, current_v0 + h_delta(4)));
        */

        //print the new params
        /*
        std::cout << std::fixed << std::setprecision(10);
        std::vector<double> new_params = {new_kappa, new_eta, new_sigma, new_rho, new_v0};
        std::cout << "The new parameters: ";
        for(int i =0; i< new_params.size(); i++){
            std::cout << new_params[i] << ", ";
        }
        std::cout << "\n";
        */
        

        // Compute delta norm (matching np.linalg.norm(delta))
        double delta_norm = 0.0;
        for(int i = 0; i < 5; ++i) {
            delta_norm += h_delta(i) * h_delta(i);
        }
        delta_norm = std::sqrt(delta_norm);

        //computing current error
        double current_error = 0;
        auto h_current_residuals = Kokkos::create_mirror_view(current_residuals);
        Kokkos::deep_copy(h_current_residuals, current_residuals);

        for(int i = 0; i < num_strikes; i++) {
            current_error += h_current_residuals(i) * h_current_residuals(i);
        }
        //std::cout << "Current error: " << current_error << std::endl;

        // Check convergence
        if(delta_norm < tol ||   
        (current_error < tol)) {
            converged = true;
            std::cout << "Converged!" << std::endl;
            std::cout << "Error Toleranze: " << tol << std::endl;
            std::cout << "Converged Error: " << current_error << std::endl;
            std::cout << "Converged Delta Norm: " << delta_norm << std::endl;

            // Update parameters to final values (matching params = new_params)
            current_kappa = new_kappa;
            current_eta = new_eta;
            current_sigma = new_sigma;
            current_rho = new_rho;
            current_v0 = new_v0;

            final_error = current_error;
            iteration_count = iter + 1;
            break; // Exit the loop
        }

        // Compute new prices with updated parameters
        Kokkos::deep_copy(workspace.U, U_0); // reset init condition
        //Rebuilding the variance direction for new_v0 is done inside the kernel

        compute_base_prices_dividends(S_0, new_v0, T,
                r_d, r_f,
                new_rho, new_sigma, new_kappa, new_eta,
                m1, m2, total_size, N, theta, delta_t,
                num_strikes,
                A0_solvers, A1_solvers, A2_solvers,
                bounds_d, deviceGrids, U_0,
                workspace,
                num_dividends,
                d_dividend_dates,     // Use device view instead of vector
                d_dividend_amounts,   // Use device view instead of vector
                d_dividend_percentages, // Use device view instead of vector
                base_prices
            );
            

        //print the new base prices
        /*
        std::cout << std::fixed << std::setprecision(12);
        auto h_base_prices   = Kokkos::create_mirror_view(base_prices);
        Kokkos::deep_copy(h_base_prices,   base_prices);
        std::cout << "The new base prices: ";
        for(int i =0; i< num_strikes; i++){
            std::cout << h_base_prices(i) << ", ";
        }
        std::cout << "\n";
        */

        // Compute new residuals
        Kokkos::parallel_for("compute_new_residuals", num_strikes, 
            KOKKOS_LAMBDA(const int i) {
                new_residuals(i) = market_prices(i) - base_prices(i);
        });
        Kokkos::fence();

        // Compute new error norms
        double new_error = 0.0;
        auto h_new_residuals = Kokkos::create_mirror_view(new_residuals);
        Kokkos::deep_copy(h_new_residuals, new_residuals);

        for(int i = 0; i < num_strikes; i++) {
            new_error += h_new_residuals(i) * h_new_residuals(i);
        }

        //std::cout << "Current error: " << std::sqrt(current_error) << std::endl;
        //std::cout << "New error: " << new_error << std::endl;
        

        // Update parameters based on error improvement
        if(new_error < current_error) {
            current_kappa = new_kappa;
            current_eta = new_eta;
            current_sigma = new_sigma;
            current_rho = new_rho;
            current_v0 = new_v0;
            lambda = std::max(lambda / 10.0, 1e-7);  // Decrease lambda but not too small
        } 
        else {
            lambda = std::min(lambda * 10.0, 1e7);  // Increase lambda but not too large
        }

        //auto iter_end = timer::now();
        //std::cout << "Iteration time: "
                //<< std::chrono::duration<double>(iter_end - iter_start).count()
                //<< " seconds" << std::endl;
        //if delta norm and error do not get lower than "tol" and max iter is hit
        final_error = std::min(new_error,current_error);
        iteration_count = iter + 1;
    }

    // Print final results
    std::cout << "\nFinal calibrated parameters:" << std::endl;
    std::cout << "κ = " << current_kappa << std::endl;
    std::cout << "η = " << current_eta << std::endl;
    std::cout << "σ = " << current_sigma << std::endl;
    std::cout << "ρ = " << current_rho << std::endl;
    std::cout << "v₀ = " << current_v0 << std::endl;
    std::cout << "final error = " << final_error << std::endl;
    std::cout << "total iterations = " << iteration_count << std::endl;

    auto t_end_second = timer::now();
    std::cout << "Total time after Updating parameters: "
              << std::chrono::duration<double>(t_end_second - t_start).count()
              << " seconds" << std::endl;

    //for ploting
    // Copy device -> host for market and fitted prices
    auto h_base_prices   = Kokkos::create_mirror_view(base_prices);
    Kokkos::deep_copy(h_base_prices,   base_prices);
    Kokkos::deep_copy(h_market_prices, market_prices);

    //adjust the starting spot for dividnets
    double S_adjusted = S_0;
    for(size_t i = 0; i < dividend_dates.size(); ++i) {
        if(dividend_dates[i] < T) {  // Only consider dividends before maturity
            //std::cout<< "div applied at " << dividend_dates[i] << std::endl;
            // Fixed amount dividend
            S_adjusted -= dividend_amounts[i] * std::exp(-r_d * dividend_dates[i]);
            //std::cout<< "stock after cash " << S_adjusted << std::endl;
            // Percentage dividend
            S_adjusted -= (S_0 * dividend_percentages[i]) * std::exp(-r_d * dividend_dates[i]);
            //std::cout<< "stock after percentage " << S_adjusted << std::endl;
        }
    }


    //computing implied vols
    double epsilon = 0.01;
    // Before the export loop, calculate IVs and their differences
    std::vector<double> iv_differences(num_strikes);
    for(int i = 0; i < num_strikes; i++) {
        double K = strikes[i];
        double market_price = h_market_prices(i);
        double heston_price = h_base_prices(i);
        
        // Compute implied vol from market price 
        //std::cout << "Market prices implied vol computation" << std::endl;
        double market_impl_vol = BlackScholes::reverse_BS(1, S_adjusted, K, r_d, T, 0.5, market_price, epsilon);
        
        // Compute implied vol from calibrated Heston price
        //std::cout << "Fitted prices implied vol computation" << std::endl;
        double heston_impl_vol = BlackScholes::reverse_BS(1, S_adjusted, K, r_d, T, 0.5, heston_price, epsilon);
        
        iv_differences[i] = std::abs(market_impl_vol - heston_impl_vol);
        // Compare - should both be close to 0.2
        //std::cout << "Strike " << K << ": Market IV = " << market_impl_vol 
                  //<< ", Heston IV = " << heston_impl_vol << std::endl;
    }

    // ================================================================
    // ============ Export final results to CSV for Python ============
    // ================================================================

    // Choose a filename
    std::string csv_filename = "fitted_heston_vs_market_dividends.csv";
    std::ofstream out(csv_filename);
    if (!out.is_open()) {
        std::cerr << "Error: Cannot open file for writing: " << csv_filename << std::endl;
        return;
    }

    // Write a title/comment line:
    //   - number of options
    //   - total runtime
    //   - final error
    // Write metadata with calibrated parameters
    double total_time = std::chrono::duration<double>(t_end_second - t_start).count();
    out << "# " << num_strikes 
        << " options, Time=" << total_time 
        << " s, FinalError=" << final_error
        << ", iterationCount=" << iteration_count
        << ", init_kappa=" << kappa
        << ", init_eta="  << eta
        << ", init_sigma="<< sigma
        << ", init_rho="  << rho
        << ", init_v0="   << V_0
        << ", kappa="     << current_kappa
        << ", eta="       << current_eta
        << ", sigma="     << current_sigma
        << ", rho="       << current_rho
        << ", v0="        << current_v0
        << "\n";

    // Write CSV header
    out << "Strike,MarketPrice,FittedPrice,IVDifference\n";

    // Write each line: Strike, MarketPrice, FittedHestonPrice
    for (int i = 0; i < num_strikes; ++i) {
        out << strikes[i] 
            << "," << h_market_prices(i) 
            << "," << h_base_prices(i) 
            << "," << iv_differences[i]
            << "\n";
    }

    out.close();
    std::cout << "Exported final results to " << csv_filename << std::endl;
}

//This calibrates to american calls with an underlying paying dividends
void test_calibration_american_dividends(){
    std::cout<< "Testing american with dividends" << std::endl;
    using Device = Kokkos::DefaultExecutionSpace;
    using timer = std::chrono::high_resolution_clock;

    // Market parameters
    const double S_0 = 100.0;
    const double T = 1.0;

    const double r_d = 0.025;
    const double r_f = 0.0;

    // Current parameter set
    const double V_0 = 0.04;
    
    
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

    const double eps = 1e-6;  // Perturbation size

    // Setup strikes and market data
    const int num_strikes = 30;
    std::vector<double> strikes(num_strikes);
    std::cout << "Strikes: ";
    for(int i = 0; i < num_strikes; ++i) {
        strikes[i] = S_0 * 0.9 + i * 1.0;//S_0 * (0.5 + i * 0.01); //S_0 - num_strikes + i;  // Strikes
        std::cout << strikes[i] << ", ";
    }
    std::cout << "" << std::endl;

    const int max_iter = 20;
    const double tol = 0.1;//0.001 * num_strikes * (S_0/100.0)*(S_0/100.0); //0.01;

    //Handling dividend host device transfer
    //{0.0, 0.0, 0.0, 0.0}
    std::vector<double> dividend_dates = {0.2, 0.4, 0.6, 0.8};
    std::vector<double> dividend_amounts = {0.3, 0.3, 0.3, 0.3};
    std::vector<double> dividend_percentages = {0.02, 0.02, 0.02, 0.02};

    // On host side, create views for dividend data
    Kokkos::View<double*> d_dividend_dates("dividend_dates", dividend_dates.size());
    Kokkos::View<double*> d_dividend_amounts("dividend_amounts", dividend_amounts.size());
    Kokkos::View<double*> d_dividend_percentages("dividend_percentages", dividend_percentages.size());

    // Copy dividend data to device
    auto h_dividend_dates = Kokkos::create_mirror_view(d_dividend_dates);
    auto h_dividend_amounts = Kokkos::create_mirror_view(d_dividend_amounts);
    auto h_dividend_percentages = Kokkos::create_mirror_view(d_dividend_percentages);

    for(size_t i = 0; i < dividend_dates.size(); i++) {
        h_dividend_dates(i) = dividend_dates[i];
        h_dividend_amounts(i) = dividend_amounts[i];
        h_dividend_percentages(i) = dividend_percentages[i];
    }

    Kokkos::deep_copy(d_dividend_dates, h_dividend_dates);
    Kokkos::deep_copy(d_dividend_amounts, h_dividend_amounts);
    Kokkos::deep_copy(d_dividend_percentages, h_dividend_percentages);

    const int num_dividends = dividend_dates.size();


    std::cout << "Computing Jacobian for " << num_strikes << " strikes\n";
    std::cout << "Total PDE solves: " << num_strikes * (6 + 1) << std::endl; //base_price + param_pertubation + new error computation
    std::cout << "Base parameters: kappa=" << kappa << ", eta=" << eta 
              << ", sigma=" << sigma << ", rho=" << rho << ", V_0=" << V_0 << "\n";
    std::cout << "Tolerance: " << tol << std::endl;

    Kokkos::View<double*> market_prices("market_prices", num_strikes);
    auto h_market_prices = Kokkos::create_mirror_view(market_prices);

    //We can not use Black scholes market prices to calibrate too. The model
    //would produce too "simple" rpices where we can not calibrate to. 
    


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

    // Create views for tracking errors
    Kokkos::View<double*> current_residuals("current_residuals", num_strikes);
    Kokkos::View<double*> new_residuals("new_residuals", num_strikes);
    Kokkos::View<double*> delta("delta", 5);

    // Current parameters that will be updated
    // will be implicitely copied ifrom host to device. This is faster than keeping it on device
    double current_kappa = kappa;
    double current_eta = eta;
    double current_sigma = sigma;
    double current_rho = rho;
    double current_v0 = V_0;

    double lambda = 0.01; // Initial LM parameter
    bool converged = false;

    double final_error = 100.0; // for plot information
    int iteration_count = 0;

    // Define bounds for updating
    static constexpr double rho_min = -1.0, rho_max = 1.0;

    /*
    static constexpr double kappa_min = 1e-6, kappa_max = 20.0;
    static constexpr double eta_min = 1e-6, eta_max = 1.0;
    static constexpr double sigma_min = 1e-6, sigma_max = 5.0;
    static constexpr double v0_min = 1e-6, v0_max = 1.0;
    */

    //generate synthetic american divident option prices:
    double market_kappa = 3.0;
    double market_eta = 0.1;
    double market_sigma = 0.05;
    double market_rho = 0.2;
    double market_v0 = 0.06;

    compute_base_prices_american_dividends(S_0, market_v0, T,
        r_d, r_f,
        market_rho, market_sigma, market_kappa, market_eta,
        m1, m2, total_size, N, theta, delta_t,
        num_strikes,
        A0_solvers, A1_solvers, A2_solvers,
        bounds_d, deviceGrids, U_0,
        workspace,
        num_dividends,
        d_dividend_dates,     // Use device view instead of vector
        d_dividend_amounts,   // Use device view instead of vector
        d_dividend_percentages, // Use device view instead of vector
        base_prices
    );
    // Both Views are on device, so we can directly copy
    Kokkos::deep_copy(market_prices, base_prices);  
    
    
    // Main iteration loop
    auto iter_start = timer::now();
    for(int iter = 0; iter < max_iter && !converged; iter++) {
        std::cout << "\nIteration " << iter + 1 << " of " << max_iter << std::endl;

        //std::cout << "Current parameters: κ=" << current_kappa 
                //<< ", η=" << current_eta 
                //<< ", σ=" << current_sigma 
                //<< ", ρ=" << current_rho 
                //<< ", v₀=" << current_v0 << std::endl;

        //need to reset the Inital condition
        Kokkos::deep_copy(workspace.U, U_0); 
        // Compute Jacobian and base prices with current parameters

        compute_jacobian_american_dividends(
            S_0, current_v0, T,
            r_d, r_f,
            current_rho, current_sigma, current_kappa, current_eta,
            m1, m2, total_size, N, theta, delta_t,
            num_strikes,
            A0_solvers, A1_solvers, A2_solvers,
            bounds_d, deviceGrids,
            U_0, workspace,
            num_dividends,
            d_dividend_dates,     // Use device view instead of vector
            d_dividend_amounts,   // Use device view instead of vector
            d_dividend_percentages, // Use device view instead of vector 
            J, base_prices,
            eps
        );

        
        //printing Jacobian
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

        
        //base prices are already computed in compute_jacobian and stored in base_price
        // Compute current residuals
        Kokkos::parallel_for("compute_residuals", num_strikes, 
            KOKKOS_LAMBDA(const int i) {
                current_residuals(i) = market_prices(i) - base_prices(i);
        });
        Kokkos::fence();

        // Compute parameter update
        compute_parameter_update_on_device(J, current_residuals, lambda, delta);

        // Get delta on host
        // maybe it would be faster to store the new_paras on device so we dont have a data transfer here
        auto h_delta = Kokkos::create_mirror_view(delta);
        Kokkos::deep_copy(h_delta, delta);

        // Try new parameters
        double new_kappa = std::max(1e-3, current_kappa + h_delta(0));
        double new_eta = std::max(1e-2, current_eta + h_delta(1));
        double new_sigma = std::max(1e-2, current_sigma + h_delta(2));
        double new_rho = std::min(rho_max, std::max(rho_min, current_rho + h_delta(3)));
        double new_v0 = std::max(1e-2, current_v0 + h_delta(4));
        
        
        /*
        double new_kappa = std::min(kappa_max, std::max(kappa_min, current_kappa + h_delta(0)));
        double new_eta = std::min(eta_max, std::max(eta_min, current_eta + h_delta(1)));
        double new_sigma = std::min(sigma_max, std::max(sigma_min, current_sigma + h_delta(2)));
        double new_rho = std::min(rho_max, std::max(rho_min, current_rho + h_delta(3)));
        double new_v0 = std::min(v0_max, std::max(v0_min, current_v0 + h_delta(4)));
        */

        //print the new params
        /*
        std::cout << std::fixed << std::setprecision(10);
        std::vector<double> new_params = {new_kappa, new_eta, new_sigma, new_rho, new_v0};
        std::cout << "The new parameters: ";
        for(int i =0; i< new_params.size(); i++){
            std::cout << new_params[i] << ", ";
        }
        std::cout << "\n";
        */
        

        // Compute delta norm (matching np.linalg.norm(delta))
        double delta_norm = 0.0;
        for(int i = 0; i < 5; ++i) {
            delta_norm += h_delta(i) * h_delta(i);
        }
        delta_norm = std::sqrt(delta_norm);

        //computing current error
        double current_error = 0;
        auto h_current_residuals = Kokkos::create_mirror_view(current_residuals);
        Kokkos::deep_copy(h_current_residuals, current_residuals);

        for(int i = 0; i < num_strikes; i++) {
            current_error += h_current_residuals(i) * h_current_residuals(i);
        }
        //std::cout << "Current error: " << current_error << std::endl;

        // Check convergence
        if(delta_norm < tol ||   
        (current_error < tol)) {
            converged = true;
            std::cout << "Converged!" << std::endl;
            std::cout << "Error Toleranze: " << tol << std::endl;
            std::cout << "Converged Error: " << current_error << std::endl;
            std::cout << "Converged Delta Norm: " << delta_norm << std::endl;

            // Update parameters to final values (matching params = new_params)
            current_kappa = new_kappa;
            current_eta = new_eta;
            current_sigma = new_sigma;
            current_rho = new_rho;
            current_v0 = new_v0;

            final_error = current_error;
            iteration_count = iter + 1;
            break; // Exit the loop
        }

        // Compute new prices with updated parameters
        Kokkos::deep_copy(workspace.U, U_0); // reset init condition
        //Rebuilding the variance direction for new_v0 is done inside the kernel

        compute_base_prices_american_dividends(S_0, new_v0, T,
                r_d, r_f,
                new_rho, new_sigma, new_kappa, new_eta,
                m1, m2, total_size, N, theta, delta_t,
                num_strikes,
                A0_solvers, A1_solvers, A2_solvers,
                bounds_d, deviceGrids, U_0,
                workspace,
                num_dividends,
                d_dividend_dates,     // Use device view instead of vector
                d_dividend_amounts,   // Use device view instead of vector
                d_dividend_percentages, // Use device view instead of vector
                base_prices
            );
            

        //print the new base prices
        /*
        std::cout << std::fixed << std::setprecision(12);
        auto h_base_prices   = Kokkos::create_mirror_view(base_prices);
        Kokkos::deep_copy(h_base_prices,   base_prices);
        std::cout << "The new base prices: ";
        for(int i =0; i< num_strikes; i++){
            std::cout << h_base_prices(i) << ", ";
        }
        std::cout << "\n";
        */

        // Compute new residuals
        Kokkos::parallel_for("compute_new_residuals", num_strikes, 
            KOKKOS_LAMBDA(const int i) {
                new_residuals(i) = market_prices(i) - base_prices(i);
        });
        Kokkos::fence();

        // Compute new error norms
        double new_error = 0.0;
        auto h_new_residuals = Kokkos::create_mirror_view(new_residuals);
        Kokkos::deep_copy(h_new_residuals, new_residuals);

        for(int i = 0; i < num_strikes; i++) {
            new_error += h_new_residuals(i) * h_new_residuals(i);
        }

        //std::cout << "Current error: " << std::sqrt(current_error) << std::endl;
        //std::cout << "New error: " << new_error << std::endl;
        

        // Update parameters based on error improvement
        if(new_error < current_error) {
            current_kappa = new_kappa;
            current_eta = new_eta;
            current_sigma = new_sigma;
            current_rho = new_rho;
            current_v0 = new_v0;
            lambda = std::max(lambda / 10.0, 1e-7);  // Decrease lambda but not too small
        } 
        else {
            lambda = std::min(lambda * 10.0, 1e7);  // Increase lambda but not too large
        }

        //auto iter_end = timer::now();
        //std::cout << "Iteration time: "
                //<< std::chrono::duration<double>(iter_end - iter_start).count()
                //<< " seconds" << std::endl;
        //if delta norm and error do not get lower than "tol" and max iter is hit
        final_error = std::min(new_error,current_error);
        iteration_count = iter + 1;
    }

    // Print final results
    std::cout << "\nFinal calibrated parameters:" << std::endl;
    std::cout << "κ = " << current_kappa << std::endl;
    std::cout << "η = " << current_eta << std::endl;
    std::cout << "σ = " << current_sigma << std::endl;
    std::cout << "ρ = " << current_rho << std::endl;
    std::cout << "v₀ = " << current_v0 << std::endl;
    std::cout << "final error = " << final_error << std::endl;
    std::cout << "total iterations = " << iteration_count << std::endl;

    auto t_end_second = timer::now();
    std::cout << "Total time after Updating parameters: "
              << std::chrono::duration<double>(t_end_second - t_start).count()
              << " seconds" << std::endl;

    //for ploting
    // Copy device -> host for market and fitted prices
    auto h_base_prices   = Kokkos::create_mirror_view(base_prices);
    Kokkos::deep_copy(h_base_prices,   base_prices);
    Kokkos::deep_copy(h_market_prices, market_prices);

    //adjust the starting spot for dividnets
    double S_adjusted = S_0;
    for(size_t i = 0; i < dividend_dates.size(); ++i) {
        if(dividend_dates[i] < T) {  // Only consider dividends before maturity
            //std::cout<< "div applied at " << dividend_dates[i] << std::endl;
            // Fixed amount dividend
            S_adjusted -= dividend_amounts[i] * std::exp(-r_d * dividend_dates[i]);
            //std::cout<< "stock after cash " << S_adjusted << std::endl;
            // Percentage dividend
            S_adjusted -= (S_0 * dividend_percentages[i]) * std::exp(-r_d * dividend_dates[i]);
            //std::cout<< "stock after percentage " << S_adjusted << std::endl;
        }
    }

    //computing implied vols
    double epsilon = 0.01;
    // Before the export loop, calculate IVs and their differences
    std::vector<double> iv_differences(num_strikes);
    for(int i = 0; i < num_strikes; i++) {
        double K = strikes[i];
        double market_price = h_market_prices(i);
        double heston_price = h_base_prices(i);
        
        // Compute implied vol from market price 
        double market_impl_vol = BlackScholes::reverse_BS(1, S_adjusted, K, r_d, T, 0.5, market_price, epsilon);
        
        // Compute implied vol from calibrated Heston price
        double heston_impl_vol = BlackScholes::reverse_BS(1, S_adjusted, K, r_d, T, 0.5, heston_price, epsilon);
        
        iv_differences[i] = std::abs(market_impl_vol - heston_impl_vol);
        // Compare - should both be close to 0.2
        //std::cout << "Strike " << K << ": Market IV = " << market_impl_vol 
                  //<< ", Heston IV = " << heston_impl_vol << std::endl;
    }

    // ================================================================
    // ============ Export final results to CSV for Python ============
    // ================================================================

    // Choose a filename
    std::string csv_filename = "fitted_heston_vs_market_american_dividends.csv";
    std::ofstream out(csv_filename);
    if (!out.is_open()) {
        std::cerr << "Error: Cannot open file for writing: " << csv_filename << std::endl;
        return;
    }

    // Write a title/comment line:
    //   - number of options
    //   - total runtime
    //   - final error
    // Write metadata with calibrated parameters
    double total_time = std::chrono::duration<double>(t_end_second - t_start).count();
    out << "# " << num_strikes 
        << " options, Time=" << total_time 
        << " s, FinalError=" << final_error
        << ", iterationCount=" << iteration_count
        << ", init_kappa=" << kappa
        << ", init_eta="  << eta
        << ", init_sigma="<< sigma
        << ", init_rho="  << rho
        << ", init_v0="   << V_0
        << ", kappa="     << current_kappa
        << ", eta="       << current_eta
        << ", sigma="     << current_sigma
        << ", rho="       << current_rho
        << ", v0="        << current_v0
        << "\n";

    // Write CSV header
    out << "Strike,MarketPrice,FittedPrice,IVDifference\n";

    // Write each line: Strike, MarketPrice, FittedHestonPrice
    for (int i = 0; i < num_strikes; ++i) {
        out << strikes[i] 
            << "," << h_market_prices(i) 
            << "," << h_base_prices(i) 
            << "," << iv_differences[i]
            << "\n";
    }

    out.close();
    std::cout << "Exported final results to " << csv_filename << std::endl;
}




/*

This calibrates a model to a number of maturites at a number of strikes
Making the Jacobian of dim [num_maturity*num_strikes x 5] form the original [num_strikes x 5]


Does not increase speefd when compared to the single maturity code 
*/
// Market data organization
struct CalibrationPoint {
    double strike;
    double maturity;
    int time_steps;      // N for this maturity
    double delta_t;      // Time step for this maturity
    int global_index;    // Flat index in the global arrays
};

// Function to compute Jacobian matrix of a european options at multiply maturities
void compute_jacobian_multi_maturity(
    // Market/model parameters
    const double S_0, const double V_0,
    const double r_d, const double r_f,
    const double rho, const double sigma, const double kappa, const double eta,
    // Numerical parameters
    const int m1, const int m2, const int total_size,
    const double theta,
    // Calibration points
    const Kokkos::View<CalibrationPoint*>& d_calibration_points,  // Note the & here
    // Pre-computed data structures
    const int total_calibration_size,
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
    const Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>& policy,
    // Optional: perturbation size
    const double eps = 1e-6
) {
    // Create team policy for the total number of calibration points
    //using Device = Kokkos::DefaultExecutionSpace;
    //using team_policy = Kokkos::TeamPolicy<>;
    //team_policy policy(total_calibration_size, Kokkos::AUTO);


    Kokkos::parallel_for("Jacobian_computation", policy,
        KOKKOS_LAMBDA(const typename Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::member_type& team) {
            const int instance = team.league_rank();
            
            // Get calibration point data
            const CalibrationPoint& point = d_calibration_points(instance);
            const double maturity = point.maturity;
            const int N = point.time_steps;
            const double delta_t = point.delta_t;
    
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
    
            // Use instance-specific time steps and delta_t from calibration point
            device_DO_timestepping<Kokkos::DefaultExecutionSpace, decltype(U_i)>(
                m1, m2, N, delta_t, theta, r_f,
                A0_solvers(instance), A1_solvers(instance), A2_solvers(instance),
                bounds,
                U_i, Y_0_i, Y_1_i,
                A0_result_i, A1_result_i, A2_result_unshuf_i,
                U_shuffled_i, Y_1_shuffled_i, A2_result_shuffled_i, U_next_shuffled_i,
                team
            );
            
            // Find price at spot price
            int index_s = -1;
            for(int i = 0; i <= m1; i++) {
                if(Kokkos::abs(grid_i.device_Vec_s(i) - S_0) < 1e-10) {
                    index_s = i;
                    break;
                }
            }
            const int index_v = grid_i.find_v0_index(V_0);
    
            const double base_price = U_i(index_s + index_v*(m1+1));
            base_prices(instance) = base_price;
    
            // Loop over parameters for finite differences
            for(int param = 0; param < 4; param++) {
                // Handle parameters other than V0
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
    
                // Compute perturbed solution with maturity-specific time steps
                device_DO_timestepping<Kokkos::DefaultExecutionSpace, decltype(U_i)>(
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
    
            // Special handling for V0 (param == 4)
            const int param = 4;
            for(int idx = 0; idx < total_size; idx++) {
                U_i(idx) = U_0_i(idx);
            }
    
            // Rebuild variance views with perturbed V0
            grid_i.rebuild_variance_views(V_0 + eps, 5.0, 5.0/500, team);
            const int index_v_pertubed = grid_i.find_v0_index(V_0 + eps);
            
            // Rebuild matrices with updated grid
            A0_solvers(instance).build_matrix(grid_i, rho, sigma, team);
            A1_solvers(instance).build_matrix(grid_i, r_d, r_f, theta, delta_t, team);
            A2_solvers(instance).build_matrix(grid_i, r_d, kappa, eta, sigma, theta, delta_t, team);
    
            // Compute perturbed solution
            device_DO_timestepping<Kokkos::DefaultExecutionSpace, decltype(U_i)>(
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


void compute_base_prices_multi_maturity(
    // Market/model parameters
    const double S_0, const double V_0,
    const double r_d, const double r_f,
    const double rho, const double sigma, const double kappa, const double eta,
    // Numerical parameters
    const int m1, const int m2, const int total_size,
    const double theta,
    // Calibration points
    const Kokkos::View<CalibrationPoint*>& d_calibration_points,
    // Pre-computed data structures
    const int total_calibration_size,
    const Kokkos::View<Device_A0_heston<Kokkos::DefaultExecutionSpace>*>& A0_solvers,
    const Kokkos::View<Device_A1_heston<Kokkos::DefaultExecutionSpace>*>& A1_solvers,
    const Kokkos::View<Device_A2_shuffled_heston<Kokkos::DefaultExecutionSpace>*>& A2_solvers,
    const Kokkos::View<Device_BoundaryConditions<Kokkos::DefaultExecutionSpace>*>& bounds_d,
    const Kokkos::View<GridViews*>& deviceGrids,
    DO_Workspace<Kokkos::DefaultExecutionSpace>& workspace,
    Kokkos::View<double*>& base_prices,
    const Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>& policy
) {
    //using Device = Kokkos::DefaultExecutionSpace;
    // Create team policy
    //using team_policy = Kokkos::TeamPolicy<>;
    //team_policy policy(total_calibration_size, Kokkos::AUTO);

    // Main computation kernel 
    Kokkos::parallel_for("Base_Price_computation", policy,
        KOKKOS_LAMBDA(const typename Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::member_type& team) {
            const int instance = team.league_rank();
            
            // Get calibration point data
            const CalibrationPoint& point = d_calibration_points(instance);
            const int N = point.time_steps;
            const double delta_t = point.delta_t;
            
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

            grid_i.rebuild_variance_views(V_0, 5.0, 5.0/500, team);
            
            bounds_d(instance).initialize(grid_i, team);
            auto bounds = bounds_d(instance);
            
            A0_solvers(instance).build_matrix(grid_i, rho, sigma, team);
            A1_solvers(instance).build_matrix(grid_i, r_d, r_f, theta, delta_t, team);
            A2_solvers(instance).build_matrix(grid_i, r_d, kappa, eta, sigma, theta, delta_t, team);

            // Use instance-specific time steps and delta_t
            device_DO_timestepping<Kokkos::DefaultExecutionSpace, decltype(U_i)>(
                m1, m2, N, delta_t, theta, r_f,
                A0_solvers(instance), A1_solvers(instance), A2_solvers(instance),
                bounds,
                U_i, Y_0_i, Y_1_i,
                A0_result_i, A1_result_i, A2_result_unshuf_i,
                U_shuffled_i, Y_1_shuffled_i, A2_result_shuffled_i, U_next_shuffled_i,
                team
            );
            
            // Find spot price index
            int index_s = -1;
            for(int i = 0; i <= m1; i++) {
                if(Kokkos::abs(grid_i.device_Vec_s(i) - S_0) < 1e-10) {
                    index_s = i;
                    break;
                }
            }
            const int index_v = grid_i.find_v0_index(V_0);

            // Store base price
            base_prices(instance) = U_i(index_s + index_v*(m1+1));
        });
    Kokkos::fence();
}


//This calibrates to european call options
void test_calibration_european_multi_maturity(){
    std::cout<< "Testing european multi maturity" << std::endl;
    using Device = Kokkos::DefaultExecutionSpace;
    using timer = std::chrono::high_resolution_clock;


    // Market parameters
    const double S_0 = 100.0;
    
    const double r_d = 0.025;
    const double r_f = 0.0;

    // Current parameter set
    const double V_0 = 0.04;
    
    const double rho = -0.9;
    const double sigma = 0.3;
    const double kappa = 1.5;
    const double eta = 0.04;
    
    
    // Numerical parameters
    const int m1 = 50;
    const int m2 = 25;

    //use the same theta for all computations
    const double theta = 0.8;

    const double eps = 1e-6;  // Perturbation size for finite difference approximation in Jacobian matrix

    // Setup maturity, strikes and market data
    //each maturity has the same amount of strikes and the same strike values
    const int num_maturities = 3;
    const int num_strikes = 60;

    const int total_calibration_size = num_maturities*num_strikes;

    std::vector<double> maturities(num_maturities);
    std::cout << "Maturities: ";
    for(int i = 0; i < num_maturities; ++i) {
        maturities[i] = 1 + i * 0.5;//0.8 + i * 0.1;

        // Weekly maturities for first 12 weeks, then bi-weekly up to 3 months
        /*
        if (i < 12) {
            maturities[i] = (i + 1) * (1.0/52.0); // Weekly (in years)
        } else {
            maturities[i] = (12 + (i-12)*2) * (1.0/52.0); // Bi-weekly
        }
        */

        // Monthly maturities from 3 to 12 months
        //maturities_medium[i] = 0.25 + i * (1.0/12.0); // Start at 3 months, monthly increments


        // Quarterly for first 2 years, then semi-annually up to 5 years
        /*
        if (i < 8) {
            maturities[i] = 1.0 + i * 0.25; // Quarterly (in years) starting at 1 year
        } else {
            maturities[i] = 3.0 + (i-8) * 0.5; // Semi-annually after 3 years
        }
        */
        

        std::cout << maturities[i] << ", ";
    }
    std::cout << "" << std::endl;

    std::vector<double> strikes(num_strikes);
    std::cout << "Strikes: ";
    double strike_width = 1.0; // Percentage of spot
    for(int i = 0; i < num_strikes; ++i) {
        strikes[i] = S_0 * 0.7 + i * 1;

        // Center around S_0 with finer spacing near the money
        //double percent_away = strike_width * (i - num_strikes/2);
        //strikes[i] = S_0 * (1.0 + percent_away/100.0);
        //std::cout << strikes[i] << ", ";
    }
    std::cout << "" << std::endl;

    // Build calibration points
    std::vector<CalibrationPoint> calibration_points;
    calibration_points.reserve(total_calibration_size);

    for(int m = 0; m < num_maturities; ++m) {
        double T_m = maturities[m];
        int N_m = std::max(20, static_cast<int>(T_m * 20));  // Base steps on maturity
        double dt_m = T_m / N_m;
        std::cout << "maturity " << T_m << " time points " << N_m << " dt " << dt_m << std::endl;
        
        for(int s = 0; s < num_strikes; ++s) {
            int idx = m * num_strikes + s;
            calibration_points.push_back({
                strikes[s],      // strike
                T_m,            // maturity
                N_m,            // time steps
                dt_m,           // delta_t
                idx             // global index
            });
            
        }
    }

    //copy to device
    Kokkos::View<CalibrationPoint*> d_calibration_points("d_calibration_points", calibration_points.size());
    auto h_calibration_points = Kokkos::create_mirror_view(d_calibration_points);
    for(size_t i = 0; i < calibration_points.size(); i++) {
        h_calibration_points(i) = calibration_points[i];
    }
    Kokkos::deep_copy(d_calibration_points, h_calibration_points);

    //CONVERGENCE CHEKC CHANGED!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //look isnide claibration delta check
    const int max_iter = 15;
    const double tol = 0.1;//0.001 * num_strikes * (S_0/100.0)*(S_0/100.0); //0.01;


    std::cout << "Computing Jacobian for " << num_maturities << " maturities and " << num_strikes << " strikes\n";
    std::cout << "Total PDE solves per iteration: " << total_calibration_size * (1 + 5 + 1) << std::endl; //base_price + param_pertubation + new error computation
    std::cout << "Base parameters: kappa=" << kappa << ", eta=" << eta 
              << ", sigma=" << sigma << ", rho=" << rho << ", V_0=" << V_0 << "\n";
    std::cout << "Tolerance: " << tol << std::endl;

    // Market data - store as a flat array, indexed by global_index
    Kokkos::View<double*> market_prices("market_prices", total_calibration_size);
    auto h_market_prices = Kokkos::create_mirror_view(market_prices);

    // Fill market data - generate synthetic data for each maturity/strike pair
    for(const auto& point : calibration_points) {
        // Generate synthetic price for this maturity/strike
        double syn_price = BlackScholes::call_price(1, S_0, point.strike, r_d, 0.2, point.maturity);
        h_market_prices(point.global_index) = syn_price;
    }
    Kokkos::deep_copy(market_prices, h_market_prices);

    // Create solver arrays for each calibration point
    Kokkos::View<Device_A0_heston<Device>*> A0_solvers("A0_solvers", total_calibration_size);
    Kokkos::View<Device_A1_heston<Device>*> A1_solvers("A1_solvers", total_calibration_size);
    Kokkos::View<Device_A2_shuffled_heston<Device>*> A2_solvers("A2_solvers", total_calibration_size);

    // Initialize solvers
    auto h_A0 = Kokkos::create_mirror_view(A0_solvers);
    auto h_A1 = Kokkos::create_mirror_view(A1_solvers);
    auto h_A2 = Kokkos::create_mirror_view(A2_solvers);

    for(int i = 0; i < total_calibration_size; i++) {
        h_A0(i) = Device_A0_heston<Device>(m1, m2);
        h_A1(i) = Device_A1_heston<Device>(m1, m2);
        h_A2(i) = Device_A2_shuffled_heston<Device>(m1, m2);
    }
    Kokkos::deep_copy(A0_solvers, h_A0);
    Kokkos::deep_copy(A1_solvers, h_A1);
    Kokkos::deep_copy(A2_solvers, h_A2);

    // Create boundary conditions array - with maturity-specific parameters
    Kokkos::View<Device_BoundaryConditions<Device>*> bounds_d("bounds_d", total_calibration_size);
    auto h_bounds = Kokkos::create_mirror_view(bounds_d);

    for(const auto& point : calibration_points) {
        int idx = point.global_index;
        h_bounds(idx) = Device_BoundaryConditions<Device>(
            m1, m2, r_d, r_f, point.time_steps, point.delta_t);
    }
    Kokkos::deep_copy(bounds_d, h_bounds);

    // Initialize grid views
    std::vector<GridViews> hostGrids;
    buildMultipleGridViews(hostGrids, total_calibration_size, m1, m2);

    for(const auto& point : calibration_points) {
        int idx = point.global_index;
        double K = point.strike;
        
        auto h_Vec_s = Kokkos::create_mirror_view(hostGrids[idx].device_Vec_s);
        auto h_Vec_v = Kokkos::create_mirror_view(hostGrids[idx].device_Vec_v);
        auto h_Delta_s = Kokkos::create_mirror_view(hostGrids[idx].device_Delta_s);
        auto h_Delta_v = Kokkos::create_mirror_view(hostGrids[idx].device_Delta_v);

        // Create grid for this strike
        Grid tempGrid(m1, 8*K, S_0, K, K/5, m2, 5.0, V_0, 5.0/500);
        
        // Copy grid data
        for(int j = 0; j <= m1; j++) h_Vec_s(j) = tempGrid.Vec_s[j];
        for(int j = 0; j <= m2; j++) h_Vec_v(j) = tempGrid.Vec_v[j];
        for(int j = 0; j < m1; j++) h_Delta_s(j) = tempGrid.Delta_s[j];
        for(int j = 0; j < m2; j++) h_Delta_v(j) = tempGrid.Delta_v[j];

        Kokkos::deep_copy(hostGrids[idx].device_Vec_s, h_Vec_s);
        Kokkos::deep_copy(hostGrids[idx].device_Vec_v, h_Vec_v);
        Kokkos::deep_copy(hostGrids[idx].device_Delta_s, h_Delta_s);
        Kokkos::deep_copy(hostGrids[idx].device_Delta_v, h_Delta_v);
    }

    Kokkos::View<GridViews*> deviceGrids("deviceGrids", total_calibration_size);
    auto h_deviceGrids = Kokkos::create_mirror_view(deviceGrids);
    for(int i = 0; i < total_calibration_size; ++i) {
        h_deviceGrids(i) = hostGrids[i];
    }
    Kokkos::deep_copy(deviceGrids, h_deviceGrids);

    const int total_size = (m1+1)*(m2+1);
    
    // Create workspace
    DO_Workspace<Device> workspace(total_calibration_size, total_size);

    // Initialize initial conditions U_0
    Kokkos::View<double**> U_0("U_0", total_calibration_size, total_size);
    auto h_U_0 = Kokkos::create_mirror_view(U_0);

    // Fill initial conditions on host
    for(const auto& point : calibration_points) {
        int idx = point.global_index;
        double K = point.strike;
        
        auto grid = hostGrids[idx];
        auto h_Vec_s = Kokkos::create_mirror_view(grid.device_Vec_s);
        Kokkos::deep_copy(h_Vec_s, grid.device_Vec_s);
        
        for(int j = 0; j <= m2; j++) {
            for(int i = 0; i <= m1; i++) {
                h_U_0(idx, i + j*(m1+1)) = std::max(h_Vec_s(i) - K, 0.0);
            }
        }
    }
    Kokkos::deep_copy(U_0, h_U_0);
    Kokkos::deep_copy(workspace.U, U_0);  // Copy initial condition to workspace

    // Storage for Jacobian matrix
    Kokkos::View<double**> J("Jacobian", total_calibration_size, 5);
    
    // Storage for base and perturbed prices
    Kokkos::View<double*> base_prices("base_prices", total_calibration_size);
    Kokkos::View<double**> pert_prices("pert_prices", total_calibration_size, 5);  

    // First compute base prices
    auto t_start = timer::now();

    // Create views for tracking errors
    Kokkos::View<double*> current_residuals("current_residuals", total_calibration_size);
    Kokkos::View<double*> new_residuals("new_residuals", total_calibration_size);
    Kokkos::View<double*> delta("delta", 5);

    // Current parameters that will be updated
    // will be implicitely copied ifrom host to device. This is faster than keeping it on device
    double current_kappa = kappa;
    double current_eta = eta;
    double current_sigma = sigma;
    double current_rho = rho;
    double current_v0 = V_0;

    double lambda = 0.01; // Initial LM parameter
    bool converged = false;

    double final_error = 100.0; // for plot information
    int iteration_count = 0;

    // Define bounds for updating
    static constexpr double rho_min = -1.0, rho_max = 1.0;

    using team_policy = Kokkos::TeamPolicy<Device>;
    team_policy my_policy(total_calibration_size, Kokkos::AUTO);    

    // Main iteration loop
    auto iter_start = timer::now();
    for(int iter = 0; iter < max_iter && !converged; iter++) {
        //std::cout << "\nIteration " << iter + 1 << " of " << max_iter << std::endl;

        //std::cout << "Current parameters: κ=" << current_kappa 
                //<< ", η=" << current_eta 
                //<< ", σ=" << current_sigma 
                //<< ", ρ=" << current_rho 
                //<< ", v₀=" << current_v0 << std::endl;

        //need to reset the Inital condition
        Kokkos::deep_copy(workspace.U, U_0); 
        // Compute Jacobian and base prices with current parameters
        
        compute_jacobian_multi_maturity(
            S_0, current_v0, 
            r_d, r_f,
            current_rho, current_sigma, current_kappa, current_eta,
            m1, m2, total_size, theta,
            d_calibration_points,  // Make sure this parameter is passed
            total_calibration_size,
            A0_solvers, A1_solvers, A2_solvers,
            bounds_d, deviceGrids,
            U_0, workspace,
            J, base_prices,
            my_policy,
            eps
        );
        
        
        //base prices are already computed in compute_jacobian and stored in base_price
        // Compute current residuals (for all calibration points)
        Kokkos::parallel_for("compute_residuals", total_calibration_size, 
            KOKKOS_LAMBDA(const int i) {
                current_residuals(i) = market_prices(i) - base_prices(i);
        });
        Kokkos::fence();

        // Compute parameter update
        compute_parameter_update_on_device(J, current_residuals, lambda, delta);

        // Get delta on host
        auto h_delta = Kokkos::create_mirror_view(delta);
        Kokkos::deep_copy(h_delta, delta);

        // Try new parameters
        double new_kappa = std::max(1e-3, current_kappa + h_delta(0));
        double new_eta = std::max(1e-2, current_eta + h_delta(1));
        double new_sigma = std::max(1e-2, current_sigma + h_delta(2));
        double new_rho = std::min(rho_max, std::max(rho_min, current_rho + h_delta(3)));
        double new_v0 = std::max(1e-2, current_v0 + h_delta(4));

        // Compute delta norm
        double delta_norm = 0.0;
        for(int i = 0; i < 5; ++i) {
            delta_norm += h_delta(i) * h_delta(i);
        }
        delta_norm = std::sqrt(delta_norm);

        // Computing current error across all calibration points
        double current_error = 0;
        auto h_current_residuals = Kokkos::create_mirror_view(current_residuals);
        Kokkos::deep_copy(h_current_residuals, current_residuals);

        for(int i = 0; i < total_calibration_size; i++) {
            current_error += h_current_residuals(i) * h_current_residuals(i);
        }

        // Check convergence
        //maybe do two errir checks one for delta normn and the other for the toleranze
        if(delta_norm < 0.3 * tol || (current_error < tol)) {
            converged = true;
            std::cout << "Converged!" << std::endl;
            std::cout << "Error Tolerance: " << tol << std::endl;
            std::cout << "Converged Error: " << current_error << std::endl;
            std::cout << "Converged Delta Norm: " << delta_norm << std::endl;

            // Update parameters to final values
            current_kappa = new_kappa;
            current_eta = new_eta;
            current_sigma = new_sigma;
            current_rho = new_rho;
            current_v0 = new_v0;

            final_error = current_error;
            iteration_count = iter + 1;
            break; // Exit the loop
        }

        // Compute new prices with updated parameters
        Kokkos::deep_copy(workspace.U, U_0); // reset initial condition

        // Call the multi-maturity base price computation
        compute_base_prices_multi_maturity(
            S_0, new_v0,
            r_d, r_f,
            new_rho, new_sigma, new_kappa, new_eta,
            m1, m2, total_size, theta,
            d_calibration_points,
            total_calibration_size,
            A0_solvers, A1_solvers, A2_solvers,
            bounds_d, deviceGrids,
            workspace,
            base_prices,
            my_policy
        );

        // Compute new residuals
        Kokkos::parallel_for("compute_new_residuals", total_calibration_size, 
            KOKKOS_LAMBDA(const int i) {
                new_residuals(i) = market_prices(i) - base_prices(i);
        });
        Kokkos::fence();

        // Compute new error norms across all calibration points
        double new_error = 0.0;
        auto h_new_residuals = Kokkos::create_mirror_view(new_residuals);
        Kokkos::deep_copy(h_new_residuals, new_residuals);

        for(int i = 0; i < total_calibration_size; i++) {
            new_error += h_new_residuals(i) * h_new_residuals(i);
        }

        // Update parameters based on error improvement
        if(new_error < current_error) {
            current_kappa = new_kappa;
            current_eta = new_eta;
            current_sigma = new_sigma;
            current_rho = new_rho;
            current_v0 = new_v0;
            lambda = std::max(lambda / 10.0, 1e-7);  // Decrease lambda but not too small
        } 
        else {
            lambda = std::min(lambda * 10.0, 1e7);  // Increase lambda but not too large
        }

        final_error = std::min(new_error, current_error);
        iteration_count = iter + 1;
    }

    
    // Print final results
    std::cout << "\nFinal calibrated parameters:" << std::endl;
    std::cout << "κ = " << current_kappa << std::endl;
    std::cout << "η = " << current_eta << std::endl;
    std::cout << "σ = " << current_sigma << std::endl;
    std::cout << "ρ = " << current_rho << std::endl;
    std::cout << "v₀ = " << current_v0 << std::endl;
    std::cout << "final error = " << final_error << std::endl;
    std::cout << "total iterations = " << iteration_count << std::endl;
    int total_pde_solves = total_calibration_size * (1 + 5 + 1) * iteration_count - total_calibration_size;
    std::cout << "Total PDE solves: " << total_pde_solves << std::endl;
    

    auto t_end_second = timer::now();
    std::cout << "Total time after Updating parameters: "
              << std::chrono::duration<double>(t_end_second - t_start).count()
              << " seconds" << std::endl;


    // Compute implied volatilities and export to CSV
    std::cout << "Exporting calibration results to CSV..." << std::endl;

    // Choose a filename for the combined results
    std::string combined_csv_filename = "fitted_heston_vs_market_multi_maturity.csv";
    std::ofstream out(combined_csv_filename);
    if (!out.is_open()) {
        std::cerr << "Error: Cannot open file for writing: " << combined_csv_filename << std::endl;
        return;
    }

    // Copy results to host
    auto h_base_prices = Kokkos::create_mirror_view(base_prices);
    Kokkos::deep_copy(h_base_prices, base_prices);
    Kokkos::deep_copy(h_market_prices, market_prices);

    // Write metadata with calibrated parameters
    double total_time = std::chrono::duration<double>(t_end_second - t_start).count();
    out << "# Calibration with " << num_maturities << " maturities, " 
        << num_strikes << " strikes per maturity, "
        << "Time=" << total_time << " s, "
        << "FinalError=" << final_error << ", "
        << "IterationCount=" << iteration_count << ", "
        << "TotalPdeSolves=" << total_pde_solves << ", "
        << "init_kappa=" << kappa << ", "
        << "init_eta=" << eta << ", "
        << "init_sigma=" << sigma << ", "
        << "init_rho=" << rho << ", "
        << "init_v0=" << V_0 << ", "
        << "kappa=" << current_kappa << ", "
        << "eta=" << current_eta << ", "
        << "sigma=" << current_sigma << ", "
        << "rho=" << current_rho << ", "
        << "v0=" << current_v0 << "\n";

    // Write CSV header
    out << "Maturity,Strike,MarketPrice,FittedPrice,MarketIV,FittedIV,IVDifference\n";

    // Compute implied volatilities and write data
    double epsilon = 0.01; // Tolerance for implied vol calculation

    for (int m = 0; m < num_maturities; ++m) {
        double T_m = maturities[m];
        for (int s = 0; s < num_strikes; ++s) {
            int idx = m * num_strikes + s;
            double K = strikes[s];
            double market_price = h_market_prices(idx);
            double heston_price = h_base_prices(idx);
            
            // Compute implied vol from market price 
            double market_impl_vol = BlackScholes::reverse_BS(1, S_0, K, r_d, T_m, 0.5, market_price, epsilon);
            
            // Compute implied vol from calibrated Heston price
            double heston_impl_vol = BlackScholes::reverse_BS(1, S_0, K, r_d, T_m, 0.5, heston_price, epsilon);
            
            // Compute difference in implied vols
            double iv_difference = std::abs(market_impl_vol - heston_impl_vol);
            
            // Write row
            out << T_m << "," 
                << K << "," 
                << market_price << "," 
                << heston_price << "," 
                << market_impl_vol << "," 
                << heston_impl_vol << "," 
                << iv_difference << "\n";
        }
    }

    out.close();
    std::cout << "Exported final results to " << combined_csv_filename << std::endl;

}





/*

Completely Deivce callable calibration to get rid of host device transfers

*/
//This calibrates to european call options
//this is a work in progress and isnt even compiling at the moment
/*
void test_DEVICE_calibration_european(){
    std::cout<< "Testing european" << std::endl;
    using Device = Kokkos::DefaultExecutionSpace;
    using timer = std::chrono::high_resolution_clock;

    // Market parameters
    const double S_0 = 100.0;
    const double T = 1.0;

    const double r_d = 0.025;
    const double r_f = 0.0;

    // Current parameter set
    const double V_0 = 0.04;
    
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

    const double eps = 1e-6;  // Perturbation size for finite difference approximation in Jacobian matrix

    // Setup strikes and market data
    const int num_strikes = 30;
    std::vector<double> strikes(num_strikes);
    std::cout << "Strikes: ";
    for(int i = 0; i < num_strikes; ++i) {
        strikes[i] = S_0 * 0.8 + i * 0.1;//S_0 * (0.5 + i * 0.01); //S_0 - num_strikes + i;  // Strikes
        std::cout << strikes[i] << ", ";
    }
    std::cout << "" << std::endl;

    const int max_iter = 15;
    const double tol = 0.1;//0.001 * num_strikes * (S_0/100.0)*(S_0/100.0); //0.01;


    std::cout << "Computing Jacobian for " << num_strikes << " strikes\n";
    std::cout << "Total PDE solves per iteration: " << num_strikes * (1 + 5 + 1) << std::endl; //base_price + param_pertubation + new error computation
    std::cout << "Base parameters: kappa=" << kappa << ", eta=" << eta 
              << ", sigma=" << sigma << ", rho=" << rho << ", V_0=" << V_0 << "\n";
    std::cout << "Tolerance: " << tol << std::endl;

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

    // Create views for tracking errors
    Kokkos::View<double*> current_residuals("current_residuals", num_strikes);
    Kokkos::View<double*> new_residuals("new_residuals", num_strikes);
    Kokkos::View<double*> delta("delta", 5);

    // Current parameters that will be updated
    // will be implicitely copied ifrom host to device. This is faster than keeping it on device
    double current_kappa = kappa;
    double current_eta = eta;
    double current_sigma = sigma;
    double current_rho = rho;
    double current_v0 = V_0;

    double lambda = 0.01; // Initial LM parameter
    bool converged = false;

    double final_error = 100.0; // for plot information
    int iteration_count = 0;

    // Define bounds for updating
    static constexpr double rho_min = -1.0, rho_max = 1.0;
    
    // Main iteration loop
    const int nInstances = 1; // Only one team will execute this
    using team_policy = Kokkos::TeamPolicy<>;
    team_policy policy(nInstances, Kokkos::AUTO);

    auto iter_start = timer::now();
    // Main kernel launch with modified internals
    Kokkos::parallel_for("Device_calibration_Eu_call", policy,
        KOKKOS_LAMBDA(const team_policy::member_type& team) {
            const int instance = team.league_rank(); // Will always be 0 since nInstances=1

            
            for(int iter = 0; iter < max_iter && !converged; iter++) {
                // Reset initial condition for all strikes
                // Reset initial condition using nested parallelism
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, num_strikes),
                    [&](const int strike_idx) {
                    auto U_i = Kokkos::subview(workspace.U, strike_idx, Kokkos::ALL);
                    auto U_0_i = Kokkos::subview(U_0, strike_idx, Kokkos::ALL);
                    
                    // Inner parallelism over the elements of each solution vector
                    Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, total_size),
                        [&](const int i) {
                            U_i(i) = U_0_i(i);
                        });
                });
            
                team.team_barrier(); // Ensure all threads complete before proceeding

                
                device_compute_jacobian<Device>(
                    S_0, current_v0, T,
                    r_d, r_f,
                    current_rho, current_sigma, current_kappa, current_eta,
                    m1, m2, total_size, N, theta, delta_t,
                    num_strikes,
                    A0_solvers, A1_solvers, A2_solvers,
                    bounds_d, deviceGrids,
                    U_0, workspace,
                    J, base_prices,
                    eps,
                    team
                );
                
                
                // Continue with optimization steps...
            }
        });
}
*/




void test_heston_calibration(){
  Kokkos::initialize();
  {
    test_calibration_european();
    //test_calibration_american();
    //test_calibration_dividends();
    //test_calibration_american_dividends();

    /*
    
    does not work yet adn will liekly not be implemented

    */
    //test_DEVICE_calibration_european();

    /*
    
    Calibrates multiple maturities of strikes
    
    */
    //test_calibration_european_multi_maturity();
  }
  Kokkos::finalize();
}