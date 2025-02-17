#include "heston_calibration.hpp"

//for the callable kernel DO method 
#include "device_solver.hpp"
//for jacobian
#include "jacobian_computation.hpp"

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
    const int num_strikes = 20;
    std::vector<double> strikes(num_strikes);
    std::cout << "Strikes: ";
    for(int i = 0; i < num_strikes; ++i) {
        strikes[i] = S_0 * 0.95 + i * 1.0;//S_0 * (0.5 + i * 0.01); //S_0 - num_strikes + i;  // Strikes
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
    generate_market_data(S_0, T, r_d, strikes, h_market_prices);
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
    for(int iter = 0; iter < max_iter && !converged; iter++) {
        auto iter_start = timer::now();
        std::cout << "\nIteration " << iter + 1 << " of " << max_iter << std::endl;

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

    auto t_end_second = timer::now();
    std::cout << "Total time after Updating parameters: "
              << std::chrono::duration<double>(t_end_second - t_start).count()
              << " seconds" << std::endl;

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
    out << "Strike,MarketPrice,FittedPrice\n";

    // Copy device -> host for market and fitted prices
    auto h_base_prices   = Kokkos::create_mirror_view(base_prices);
    Kokkos::deep_copy(h_base_prices,   base_prices);
    Kokkos::deep_copy(h_market_prices, market_prices);

    // Write each line: Strike, MarketPrice, FittedHestonPrice
    for (int i = 0; i < num_strikes; ++i) {
        out << strikes[i] 
            << "," << h_market_prices(i) 
            << "," << h_base_prices(i) 
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
    const int num_strikes = 60;
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
    generate_market_data(S_0, T, r_d, strikes, h_market_prices);
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
    for(int iter = 0; iter < max_iter && !converged; iter++) {
        auto iter_start = timer::now();
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
    out << "Strike,MarketPrice,FittedPrice\n";

    // Copy device -> host for market and fitted prices
    auto h_base_prices   = Kokkos::create_mirror_view(base_prices);
    Kokkos::deep_copy(h_base_prices,   base_prices);
    Kokkos::deep_copy(h_market_prices, market_prices);

    // Write each line: Strike, MarketPrice, FittedHestonPrice
    for (int i = 0; i < num_strikes; ++i) {
        out << strikes[i] 
            << "," << h_market_prices(i) 
            << "," << h_base_prices(i) 
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
    const int num_strikes = 50;
    std::vector<double> strikes(num_strikes);
    std::cout << "Strikes: ";
    for(int i = 0; i < num_strikes; ++i) {
        strikes[i] = S_0 * 0.50 + i * 1.0;//S_0 * (0.5 + i * 0.01); //S_0 - num_strikes + i;  // Strikes
        std::cout << strikes[i] << ", ";
    }
    std::cout << "" << std::endl;

    const int max_iter = 15;
    const double tol = 0.1;//0.001 * num_strikes * (S_0/100.0)*(S_0/100.0); //0.01;

    //Handling dividend host device transfer
    //{0.0, 0.0, 0.0, 0.0}
    std::vector<double> dividend_dates = {0.2, 0.4, 0.6, 0.8};
    std::vector<double> dividend_amounts = {0.5, 0.3, 0.2, 0.1};
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

    // Compute market prices on host using Black-Scholes
    // Generate synthetic market prices
    generate_market_data(S_0, T, r_d, strikes, h_market_prices);
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
    for(int iter = 0; iter < max_iter && !converged; iter++) {
        auto iter_start = timer::now();
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
    out << "Strike,MarketPrice,FittedPrice\n";

    // Copy device -> host for market and fitted prices
    auto h_base_prices   = Kokkos::create_mirror_view(base_prices);
    Kokkos::deep_copy(h_base_prices,   base_prices);
    Kokkos::deep_copy(h_market_prices, market_prices);

    // Write each line: Strike, MarketPrice, FittedHestonPrice
    for (int i = 0; i < num_strikes; ++i) {
        out << strikes[i] 
            << "," << h_market_prices(i) 
            << "," << h_base_prices(i) 
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
    const int num_strikes = 20;
    std::vector<double> strikes(num_strikes);
    std::cout << "Strikes: ";
    for(int i = 0; i < num_strikes; ++i) {
        strikes[i] = S_0 * 0.95 + i * 1.0;//S_0 * (0.5 + i * 0.01); //S_0 - num_strikes + i;  // Strikes
        std::cout << strikes[i] << ", ";
    }
    std::cout << "" << std::endl;

    const int max_iter = 15;
    const double tol = 0.1;//0.001 * num_strikes * (S_0/100.0)*(S_0/100.0); //0.01;

    //Handling dividend host device transfer
    //{0.0, 0.0, 0.0, 0.0}
    std::vector<double> dividend_dates = {0.2, 0.4, 0.6, 0.8};
    std::vector<double> dividend_amounts = {0.5, 0.3, 0.2, 0.1};
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

    // Compute market prices on host using Black-Scholes
    // Generate synthetic market prices
    generate_market_data(S_0, T, r_d, strikes, h_market_prices);
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
    for(int iter = 0; iter < max_iter && !converged; iter++) {
        auto iter_start = timer::now();
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
    out << "Strike,MarketPrice,FittedPrice\n";

    // Copy device -> host for market and fitted prices
    auto h_base_prices   = Kokkos::create_mirror_view(base_prices);
    Kokkos::deep_copy(h_base_prices,   base_prices);
    Kokkos::deep_copy(h_market_prices, market_prices);

    // Write each line: Strike, MarketPrice, FittedHestonPrice
    for (int i = 0; i < num_strikes; ++i) {
        out << strikes[i] 
            << "," << h_market_prices(i) 
            << "," << h_base_prices(i) 
            << "\n";
    }

    out.close();
    std::cout << "Exported final results to " << csv_filename << std::endl;
}


void test_heston_calibration(){
  Kokkos::initialize();
  {
    //test_calibration_european();
    //test_calibration_american();
    test_calibration_dividends();
    //test_calibration_american_dividends();
  }
  Kokkos::finalize();
 
}