#include <Kokkos_Core.hpp>
#include "device_solver.hpp"
#include "perfomance_test.hpp"

#include <fstream>


#include <iostream>
#include <numeric>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <string>
#include <vector>

std::string base_dir = "/home/benedikt/ADIKokkos/benchmark_results/";

//Benchmark for european call option
void benchmark_DO_solver_performance_european(const std::string& output_filename = "kokkos_performance_call.csv") {
    using timer = std::chrono::high_resolution_clock;
    using Device = Kokkos::DefaultExecutionSpace;
    
    // Open CSV file for output
    std::cout << "Writing benchmark results to: " << output_filename << std::endl;
    std::ofstream outfile(output_filename);
    if (!outfile.is_open()) {
        std::cerr << "Error: Cannot open file for writing: " << output_filename << std::endl;
        return;
    }
    
    // Write header row
    outfile << "m1,m2,instances,total_size,total_runtime,runtime_per_instance,throughput\n";
    
    
    // Fixed parameters
    const double S_0 = 100.0;
    const double V_0 = 0.04;
    const double T = 1.0;
    const double r_d = 0.025;
    const double r_f = 0.0;
    const double rho = -0.9;
    const double sigma = 0.3;
    const double kappa = 1.5;
    const double eta = 0.04;
    const int N = 20;
    const double theta = 0.8;
    
    // Vary grid sizes and instance counts
    std::vector<int> grid_sizes_m2 = {25};//, 50, 100, 150};
    std::vector<int> instance_counts = {1, 10, 20, 50, 100, 200, 300, 500};
    
    // Number of runs for each configuration to average
    const int NUM_RUNS = 10;
    
    for (int m2 : grid_sizes_m2) {
        int m1 = 2*m2;
        const int total_size = (m1+1) * (m2+1);
        
        for (int nInstances : instance_counts) {
            // Skip very large configurations that might be too slow or memory-intensive
            if (static_cast<size_t>(total_size) * nInstances > 100000000){
                std::cout << "skipped " << nInstances << " with total size " << m1 << ", " << m2 << std::endl;
                continue;
            }
            
            std::cout << "Testing m1=" << m1 << ", m2=" << m2 
                        << ", instances=" << nInstances << std::endl;
            
            std::vector<double> timings;
            
            // Generate strike prices
            std::vector<double> strikes(nInstances, 0.0);
            for (int i = 0; i < nInstances; ++i) {
                strikes[i] = 85.0;
            }
            
            
            // Reset Kokkos for clean state in each run
            // Create solver arrays
            Kokkos::View<Device_A0_heston<Device>*> A0_solvers("A0_solvers", nInstances);
            Kokkos::View<Device_A1_heston<Device>*> A1_solvers("A1_solvers", nInstances);
            Kokkos::View<Device_A2_shuffled_heston<Device>*> A2_solvers("A2_solvers", nInstances);
            
            // Initialize solvers
            auto h_A0 = Kokkos::create_mirror_view(A0_solvers);
            auto h_A1 = Kokkos::create_mirror_view(A1_solvers);
            auto h_A2 = Kokkos::create_mirror_view(A2_solvers);
            
            for (int i = 0; i < nInstances; i++) {
                h_A0(i) = Device_A0_heston<Device>(m1, m2);
                h_A1(i) = Device_A1_heston<Device>(m1, m2);
                h_A2(i) = Device_A2_shuffled_heston<Device>(m1, m2);
            }
            Kokkos::deep_copy(A0_solvers, h_A0);
            Kokkos::deep_copy(A1_solvers, h_A1);
            Kokkos::deep_copy(A2_solvers, h_A2);
            
            // Time step size
            const double delta_t = T / N;
            
            // Create boundary conditions array
            Kokkos::View<Device_BoundaryConditions<Device>*> bounds_d("bounds_d", nInstances);
            auto h_bounds = Kokkos::create_mirror_view(bounds_d);
            for (int i = 0; i < nInstances; ++i) {
                h_bounds(i) = Device_BoundaryConditions<Device>(m1, m2, r_d, r_f, N, delta_t);
            }
            Kokkos::deep_copy(bounds_d, h_bounds);
            
            // Initialize grid views
            std::vector<GridViews> hostGrids;
            buildMultipleGridViews(hostGrids, nInstances, m1, m2);
            for (int i = 0; i < nInstances; ++i) {
                double K = strikes[i];
                auto h_Vec_s = Kokkos::create_mirror_view(hostGrids[i].device_Vec_s);
                auto h_Vec_v = Kokkos::create_mirror_view(hostGrids[i].device_Vec_v);
                auto h_Delta_s = Kokkos::create_mirror_view(hostGrids[i].device_Delta_s);
                auto h_Delta_v = Kokkos::create_mirror_view(hostGrids[i].device_Delta_v);
                
                Grid tempGrid(m1, 8*K, S_0, K, K/5, m2, 5.0, V_0, 5.0/500);
                
                for (int j = 0; j <= m1; j++) h_Vec_s(j) = tempGrid.Vec_s[j];
                for (int j = 0; j <= m2; j++) h_Vec_v(j) = tempGrid.Vec_v[j];
                for (int j = 0; j < m1; j++) h_Delta_s(j) = tempGrid.Delta_s[j];
                for (int j = 0; j < m2; j++) h_Delta_v(j) = tempGrid.Delta_v[j];
                
                Kokkos::deep_copy(hostGrids[i].device_Vec_s, h_Vec_s);
                Kokkos::deep_copy(hostGrids[i].device_Vec_v, h_Vec_v);
                Kokkos::deep_copy(hostGrids[i].device_Delta_s, h_Delta_s);
                Kokkos::deep_copy(hostGrids[i].device_Delta_v, h_Delta_v);
            }
            
            Kokkos::View<GridViews*> deviceGrids("deviceGrids", nInstances);
            auto h_deviceGrids = Kokkos::create_mirror_view(deviceGrids);
            for (int i = 0; i < nInstances; ++i) h_deviceGrids(i) = hostGrids[i];
            Kokkos::deep_copy(deviceGrids, h_deviceGrids);
            
            // Create workspace
            DO_Workspace<Device> workspace(nInstances, total_size);

            for (int run = 0; run < NUM_RUNS; ++run) {
                
                // Initialize initial conditions U_0
                Kokkos::View<double**> U_0("U_0", nInstances, total_size);
                auto h_U_0 = Kokkos::create_mirror_view(U_0);
                
                // Fill initial conditions on host
                for (int inst = 0; inst < nInstances; ++inst) {
                    auto grid = hostGrids[inst];
                    auto h_Vec_s = Kokkos::create_mirror_view(grid.device_Vec_s);
                    Kokkos::deep_copy(h_Vec_s, grid.device_Vec_s);
                    double K = strikes[inst];
                    
                    for (int j = 0; j <= m2; j++) {
                        for (int i = 0; i <= m1; i++) {
                            h_U_0(inst, i + j*(m1+1)) = std::max(h_Vec_s(i) - K, 0.0);
                        }
                    }
                }
                Kokkos::deep_copy(U_0, h_U_0);
                Kokkos::deep_copy(workspace.U, U_0);
                
                using team_policy = Kokkos::TeamPolicy<>;
                team_policy policy(nInstances, Kokkos::AUTO);
                
                // Start timer
                auto t_start = timer::now();
                
                // Main kernel launch
                Kokkos::parallel_for("DO_scheme_benchmark", policy,
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
                });
                Kokkos::fence();
                
                auto t_end = timer::now();
                double runtime = std::chrono::duration<double>(t_end - t_start).count();
                timings.push_back(runtime);
            }
            
            // Calculate average runtime and throughput metrics
            double avg_runtime = std::accumulate(timings.begin(), timings.end(), 0.0) / timings.size();
            double runtime_per_instance = avg_runtime / nInstances;
            double throughput = nInstances / avg_runtime; // instances per second
            
            // Write to CSV
            outfile << m1 << "," << m2 << "," << nInstances << "," << total_size 
                    << "," << avg_runtime << "," << runtime_per_instance << "," << throughput << "\n";
            outfile.flush(); // Ensure data is written even if we crash later
            
            std::cout << "  Average runtime: " << avg_runtime << " seconds" << std::endl;
            std::cout << "  Throughput: " << throughput << " instances/second" << std::endl;
        }
    }
    
    outfile.close();
    std::cout << "Benchmark results written to " << output_filename << std::endl;
}

//Benchmark for dividend handling
void benchmark_DO_solver_performance_dividends(const std::string& output_filename = "kokkos_performance_dividends.csv") {
    using timer = std::chrono::high_resolution_clock;
    using Device = Kokkos::DefaultExecutionSpace;
    
    // Open CSV file for output
    std::cout << "Writing benchmark results to: " << output_filename << std::endl;
    std::ofstream outfile(output_filename);
    if (!outfile.is_open()) {
        std::cerr << "Error: Cannot open file for writing: " << output_filename << std::endl;
        return;
    }
    
    // Write header row
    outfile << "m1,m2,instances,total_size,total_runtime,runtime_per_instance,throughput\n";
    
    
    // Fixed parameters
    const double S_0 = 100.0;
    const double V_0 = 0.04;

    const double T = 1.0;

    const double r_d = 0.025;
    const double r_f = 0.0;

    const double rho = -0.9;
    const double sigma = 0.3;
    const double kappa = 1.5;
    const double eta = 0.04;

    const int N = 20;
    const double theta = 0.8;
    
    // Vary grid sizes and instance counts
    std::vector<int> grid_sizes_m2 = {25, 50, 100, 150};
    std::vector<int> instance_counts = {5, 10, 20, 50, 100, 200, 300};
    
    // Number of runs for each configuration to average
    const int NUM_RUNS = 10;

    //Init Dividend data once
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
    
    for (int m2 : grid_sizes_m2) {
        int m1 = 2*m2;
        const int total_size = (m1+1) * (m2+1);
        
        for (int nInstances : instance_counts) {
            // Skip very large configurations that might be too slow or memory-intensive
            if (static_cast<size_t>(total_size) * nInstances > 100000000){
                std::cout << "skipped " << nInstances << " with total size " << m1 << ", " << m2 << std::endl;
                continue;
            }
            
            std::cout << "Testing m1=" << m1 << ", m2=" << m2 
                        << ", instances=" << nInstances << std::endl;
            
            std::vector<double> timings;
            
            // Generate strike prices
            std::vector<double> strikes(nInstances, 0.0);
            for (int i = 0; i < nInstances; ++i) {
                strikes[i] = 85.0;
            }
            
            
            // Reset Kokkos for clean state in each run
            // Create solver arrays
            Kokkos::View<Device_A0_heston<Device>*> A0_solvers("A0_solvers", nInstances);
            Kokkos::View<Device_A1_heston<Device>*> A1_solvers("A1_solvers", nInstances);
            Kokkos::View<Device_A2_shuffled_heston<Device>*> A2_solvers("A2_solvers", nInstances);
            
            // Initialize solvers
            auto h_A0 = Kokkos::create_mirror_view(A0_solvers);
            auto h_A1 = Kokkos::create_mirror_view(A1_solvers);
            auto h_A2 = Kokkos::create_mirror_view(A2_solvers);
            
            for (int i = 0; i < nInstances; i++) {
                h_A0(i) = Device_A0_heston<Device>(m1, m2);
                h_A1(i) = Device_A1_heston<Device>(m1, m2);
                h_A2(i) = Device_A2_shuffled_heston<Device>(m1, m2);
            }
            Kokkos::deep_copy(A0_solvers, h_A0);
            Kokkos::deep_copy(A1_solvers, h_A1);
            Kokkos::deep_copy(A2_solvers, h_A2);
            
            // Time step size
            const double delta_t = T / N;
            
            // Create boundary conditions array
            Kokkos::View<Device_BoundaryConditions<Device>*> bounds_d("bounds_d", nInstances);
            auto h_bounds = Kokkos::create_mirror_view(bounds_d);
            for (int i = 0; i < nInstances; ++i) {
                h_bounds(i) = Device_BoundaryConditions<Device>(m1, m2, r_d, r_f, N, delta_t);
            }
            Kokkos::deep_copy(bounds_d, h_bounds);
            
            // Initialize grid views
            std::vector<GridViews> hostGrids;
            buildMultipleGridViews(hostGrids, nInstances, m1, m2);
            for (int i = 0; i < nInstances; ++i) {
                double K = strikes[i];
                auto h_Vec_s = Kokkos::create_mirror_view(hostGrids[i].device_Vec_s);
                auto h_Vec_v = Kokkos::create_mirror_view(hostGrids[i].device_Vec_v);
                auto h_Delta_s = Kokkos::create_mirror_view(hostGrids[i].device_Delta_s);
                auto h_Delta_v = Kokkos::create_mirror_view(hostGrids[i].device_Delta_v);
                
                Grid tempGrid(m1, 8*K, S_0, K, K/5, m2, 5.0, V_0, 5.0/500);
                
                for (int j = 0; j <= m1; j++) h_Vec_s(j) = tempGrid.Vec_s[j];
                for (int j = 0; j <= m2; j++) h_Vec_v(j) = tempGrid.Vec_v[j];
                for (int j = 0; j < m1; j++) h_Delta_s(j) = tempGrid.Delta_s[j];
                for (int j = 0; j < m2; j++) h_Delta_v(j) = tempGrid.Delta_v[j];
                
                Kokkos::deep_copy(hostGrids[i].device_Vec_s, h_Vec_s);
                Kokkos::deep_copy(hostGrids[i].device_Vec_v, h_Vec_v);
                Kokkos::deep_copy(hostGrids[i].device_Delta_s, h_Delta_s);
                Kokkos::deep_copy(hostGrids[i].device_Delta_v, h_Delta_v);
            }
            
            Kokkos::View<GridViews*> deviceGrids("deviceGrids", nInstances);
            auto h_deviceGrids = Kokkos::create_mirror_view(deviceGrids);
            for (int i = 0; i < nInstances; ++i) h_deviceGrids(i) = hostGrids[i];
            Kokkos::deep_copy(deviceGrids, h_deviceGrids);
            
            // Create workspace
            DO_Workspace<Device> workspace(nInstances, total_size);



            for (int run = 0; run < NUM_RUNS; ++run) {
                
                // Initialize initial conditions U_0
                Kokkos::View<double**> U_0("U_0", nInstances, total_size);
                auto h_U_0 = Kokkos::create_mirror_view(U_0);
                
                // Fill initial conditions on host
                for (int inst = 0; inst < nInstances; ++inst) {
                    auto grid = hostGrids[inst];
                    auto h_Vec_s = Kokkos::create_mirror_view(grid.device_Vec_s);
                    Kokkos::deep_copy(h_Vec_s, grid.device_Vec_s);
                    double K = strikes[inst];
                    
                    for (int j = 0; j <= m2; j++) {
                        for (int i = 0; i <= m1; i++) {
                            h_U_0(inst, i + j*(m1+1)) = std::max(h_Vec_s(i) - K, 0.0);
                        }
                    }
                }
                Kokkos::deep_copy(U_0, h_U_0);
                Kokkos::deep_copy(workspace.U, U_0);
                
                using team_policy = Kokkos::TeamPolicy<>;
                team_policy policy(nInstances, Kokkos::AUTO);
                
                // Start timer
                auto t_start = timer::now();
                
                // Main kernel launch
                Kokkos::parallel_for("Benchmark_DO_scheme_divid", policy,
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
            
                        //Initialize Grid views
                        GridViews grid_i = deviceGrids(instance);
            
                        //Dividend specifics
                        auto U_temp_i = Kokkos::subview(workspace.U_temp, instance, Kokkos::ALL);
                        auto device_Vec_s_i = grid_i.device_Vec_s;
                        
                        // Initialize boundaries 
                        bounds_d(instance).initialize(grid_i, team);
                        auto bounds = bounds_d(instance);
                        
                        // Build matrices
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
                            d_dividend_dates,     // Use device view instead of vector
                            d_dividend_amounts,   // Use device view instead of vector
                            d_dividend_percentages, // Use device view instead of vector
                            device_Vec_s_i,        
                            U_temp_i,  
                            team
                        );
                });
                Kokkos::fence();
                
                auto t_end = timer::now();
                double runtime = std::chrono::duration<double>(t_end - t_start).count();
                timings.push_back(runtime);
            }
            
            // Calculate average runtime and throughput metrics
            double avg_runtime = std::accumulate(timings.begin(), timings.end(), 0.0) / timings.size();
            double runtime_per_instance = avg_runtime / nInstances;
            double throughput = nInstances / avg_runtime; // instances per second
            
            // Write to CSV
            outfile << m1 << "," << m2 << "," << nInstances << "," << total_size 
                    << "," << avg_runtime << "," << runtime_per_instance << "," << throughput << "\n";
            outfile.flush(); // Ensure data is written even if we crash later
            
            std::cout << "  Average runtime: " << avg_runtime << " seconds" << std::endl;
            std::cout << "  Throughput: " << throughput << " instances/second" << std::endl;
        }
    }
    
    outfile.close();
    std::cout << "Benchmark results written to " << output_filename << std::endl;
}




/*

Simple methods which call the above tests

*/
//european call
void test_european_call_performance(){

    // Get architecture information
    std::string arch_name;
    
    #ifdef KOKKOS_ENABLE_CUDA
    if (std::is_same<Kokkos::DefaultExecutionSpace, Kokkos::Cuda>::value) {
        // Get CUDA device properties for the current device
        cudaDeviceProp props;
        int device;
        cudaGetDevice(&device);
        cudaGetDeviceProperties(&props, device);
        arch_name = "cuda_" + std::string(props.name);
    }
    #endif

    #ifdef KOKKOS_ENABLE_OPENMP
    if (std::is_same<Kokkos::DefaultExecutionSpace, Kokkos::OpenMP>::value) {
        arch_name = "openmp_" + std::to_string(Kokkos::OpenMP::impl_thread_pool_size());
    }
    #endif

    // Fallback if no specific architecture is detected
    if (arch_name.empty()) {
        arch_name = Kokkos::DefaultExecutionSpace::name();
    }

    // Replace any spaces or special characters in the architecture name with underscores
    std::replace(arch_name.begin(), arch_name.end(), ' ', '_');
    std::replace(arch_name.begin(), arch_name.end(), '/', '_');
    std::replace(arch_name.begin(), arch_name.end(), '\\', '_');

    std::string filename = base_dir + "25_performance_european_call_" + arch_name + ".csv";


    benchmark_DO_solver_performance_european(filename);
}

//european call wiht dividends
void test_dividend_performance(){

    // Get architecture information
    std::string arch_name;
    
    #ifdef KOKKOS_ENABLE_CUDA
    if (std::is_same<Kokkos::DefaultExecutionSpace, Kokkos::Cuda>::value) {
        // Get CUDA device properties for the current device
        cudaDeviceProp props;
        int device;
        cudaGetDevice(&device);
        cudaGetDeviceProperties(&props, device);
        arch_name = "cuda_" + std::string(props.name);
    }
    #endif

    #ifdef KOKKOS_ENABLE_OPENMP
    if (std::is_same<Kokkos::DefaultExecutionSpace, Kokkos::OpenMP>::value) {
        arch_name = "openmp_" + std::to_string(Kokkos::OpenMP::impl_thread_pool_size());
    }
    #endif

    // Fallback if no specific architecture is detected
    if (arch_name.empty()) {
        arch_name = Kokkos::DefaultExecutionSpace::name();
    }

    // Replace any spaces or special characters in the architecture name with underscores
    std::replace(arch_name.begin(), arch_name.end(), ' ', '_');
    std::replace(arch_name.begin(), arch_name.end(), '/', '_');
    std::replace(arch_name.begin(), arch_name.end(), '\\', '_');

    std::string filename = base_dir + "performance_dividend_" + arch_name + ".csv";

    benchmark_DO_solver_performance_dividends(filename);
}


void test_perfomance_Tests(){
    Kokkos::initialize();
    {
      test_european_call_performance();
      //test_dividend_performance();
    }
    Kokkos::finalize();
}