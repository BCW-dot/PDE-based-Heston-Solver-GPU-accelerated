#include "hes_a0_kernels.hpp"
#include <iomanip>
#include <iostream>
#include "grid_pod.hpp"
#include <numeric>

//Test using the class
void test_device_a0_multiple_instances() {
    using timer = std::chrono::high_resolution_clock;
    using Device = Kokkos::DefaultExecutionSpace;

    // Test parameters
    const int m1 = 100;
    const int m2 = 75;
    const double rho = -0.9;
    const double sigma = 0.3;
    const int nInstances = 100;

    std::cout << "A0 Dimension StockxVariance: " << m1+1 << "x" << m2+1 << std::endl;
    std::cout << "Instances: " << nInstances << std::endl;

    // Create solvers array on device
    Kokkos::View<Device_A0_heston<Device>*> solvers_d("solvers_d", nInstances);
    auto solvers_h = Kokkos::create_mirror_view(solvers_d);
    for(int i = 0; i < nInstances; ++i) {
        solvers_h(i) = Device_A0_heston<Device>(m1, m2);
    }
    Kokkos::deep_copy(solvers_d, solvers_h);

    // Initialize grid views
    std::vector<GridViews> hostGrids;
    buildMultipleGridViews(hostGrids, nInstances, m1, m2);
    for(int i = 0; i < nInstances; ++i) {
        auto h_Vec_s = Kokkos::create_mirror_view(hostGrids[i].device_Vec_s);
        auto h_Vec_v = Kokkos::create_mirror_view(hostGrids[i].device_Vec_v);
        auto h_Delta_s = Kokkos::create_mirror_view(hostGrids[i].device_Delta_s);
        auto h_Delta_v = Kokkos::create_mirror_view(hostGrids[i].device_Delta_v);

        Grid tempGrid = create_test_grid(m1, m2);
        
        for(int j = 0; j <= m1; j++) h_Vec_s(j) = tempGrid.Vec_s[j];
        for(int j = 0; j <= m2; j++) h_Vec_v(j) = tempGrid.Vec_v[j];
        for(int j = 0; j < m1; j++) h_Delta_s(j) = tempGrid.Delta_s[j];
        for(int j = 0; j < m2; j++) h_Delta_v(j) = tempGrid.Delta_v[j];

        Kokkos::deep_copy(hostGrids[i].device_Vec_s, h_Vec_s);
        Kokkos::deep_copy(hostGrids[i].device_Vec_v, h_Vec_v);
        Kokkos::deep_copy(hostGrids[i].device_Delta_s, h_Delta_s);
        Kokkos::deep_copy(hostGrids[i].device_Delta_v, h_Delta_v);
    }

    Kokkos::View<GridViews*> deviceGrids("deviceGrids", nInstances);
    auto h_deviceGrids = Kokkos::create_mirror_view(deviceGrids);
    for(int i = 0; i < nInstances; ++i) h_deviceGrids(i) = hostGrids[i];
    Kokkos::deep_copy(deviceGrids, h_deviceGrids);

    // Create test vectors
    const int total_size = (m1+1)*(m2+1);
    Kokkos::View<double**> x("x", nInstances, total_size);
    Kokkos::View<double**> result("result", nInstances, total_size);

    auto h_x = Kokkos::create_mirror_view(x);
    for(int inst = 0; inst < nInstances; ++inst) {
        for(int idx = 0; idx < total_size; ++idx) {
            h_x(inst, idx) = 1.0;
        }
    }
    Kokkos::deep_copy(x, h_x);
    Kokkos::deep_copy(result, 0.0);

    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;
    team_policy policy(nInstances, Kokkos::AUTO);

    const int NUM_RUNS = 5;
    std::vector<double> timings(NUM_RUNS);

    for(int run = 0; run < NUM_RUNS; run++) {
        auto t_start = timer::now();
        
        Kokkos::parallel_for("solve_all_instances", policy,
            KOKKOS_LAMBDA(const member_type& team) {
                const int instance = team.league_rank();
                
                Device_A0_heston<Device>& solver = solvers_d(instance);
                auto x_i = Kokkos::subview(x, instance, Kokkos::ALL);
                auto result_i = Kokkos::subview(result, instance, Kokkos::ALL);
                GridViews grid_i = deviceGrids(instance);
                
                solver.build_matrix(grid_i, rho, sigma, team);
                solver.multiply_parallel_v(x_i, result_i, team);
        });
        Kokkos::fence();

        auto t_end = timer::now();
        timings[run] = std::chrono::duration<double>(t_end - t_start).count();
    }

    double avg_time = std::accumulate(timings.begin(), timings.end(), 0.0) / NUM_RUNS;
    double variance = 0.0;
    for(const auto& t : timings) {
        variance += (t - avg_time) * (t - avg_time);
    }
    double std_dev = std::sqrt(variance);

    std::cout << "Average time: " << avg_time << " seconds\n";
    std::cout << "Standard deviation: " << std_dev << " seconds\n";

    // Print results for first few instances
    auto h_result = Kokkos::create_mirror_view(result);
    Kokkos::deep_copy(h_result, result);

    for(int inst = 0; inst < std::min(5, nInstances); ++inst) {
        std::cout << "Instance " << inst << " result:\n";
        std::cout << "  result[0..9] = ";
        for(int i = 0; i < std::min(10,total_size); i++) {
            std::cout << h_result(inst, i) << " ";
        }
        std::cout << "\n------------------------------------\n";
    }
}

void test_a0_kernel(){
    Kokkos::initialize();
        {
            try{
                test_device_a0_multiple_instances();
            }
            catch (std::exception& e) {
                std::cout << "Error: " << e.what() << std::endl;
            }
        } // All test objects destroyed here
    Kokkos::finalize();
}