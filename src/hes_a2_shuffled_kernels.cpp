#include "hes_a2_shuffled_kernels.hpp"
#include <iomanip>
#include <iostream>
//for accumulate
#include <numeric>


/*

Class test

*/
void test_device_a2_shuffled_multiple_instances() {
    using timer = std::chrono::high_resolution_clock;
    using Device = Kokkos::DefaultExecutionSpace;

    const int m1 = 150;
    const int m2 = 75;
    std::cout << "A2 Dimension StockxVariance: " << m1+1 << "x" << m2+1 << std::endl;

    const double theta = 0.8;
    const double delta_t = 1.0/40.0;

    const double r_d = 0.025;
    const double kappa = 1.5;
    const double eta = 0.04;
    const double sigma = 0.3;

    const int nInstances = 100;
    std::cout << "Instances: " << nInstances << std::endl;

    // Create solvers array
    Kokkos::View<Device_A2_shuffled_heston<Device>*> solvers_d("solvers_d", nInstances);
    auto solvers_h = Kokkos::create_mirror_view(solvers_d);
    for(int i = 0; i < nInstances; ++i) {
        solvers_h(i) = Device_A2_shuffled_heston<Device>(m1, m2);
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

    const int total_size = (m1+1)*(m2+1);
    Kokkos::View<double**> x("x", nInstances, total_size);
    Kokkos::View<double**> b("b", nInstances, total_size);
    Kokkos::View<double**> result("result", nInstances, total_size);

    auto h_x = Kokkos::create_mirror_view(x);
    auto h_b = Kokkos::create_mirror_view(b);
    for(int inst = 0; inst < nInstances; ++inst) {
        for(int idx = 0; idx < total_size; ++idx) {
            h_x(inst, idx) = 1.0;
            h_b(inst, idx) = 2.0;
        }
    }
    Kokkos::deep_copy(x, h_x);
    Kokkos::deep_copy(b, h_b);
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
                
                Device_A2_shuffled_heston<Device>& solver = solvers_d(instance);
                auto x_i = Kokkos::subview(x, instance, Kokkos::ALL);
                auto b_i = Kokkos::subview(b, instance, Kokkos::ALL);
                auto result_i = Kokkos::subview(result, instance, Kokkos::ALL);
                GridViews grid_i = deviceGrids(instance);
                
                solver.build_matrix(grid_i, r_d, kappa, eta, sigma, theta, delta_t, team);
                solver.multiply_parallel_s(x_i, result_i, team);
                solver.solve_implicit_parallel_s(x_i, b_i, team);
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

    // Verify results
    Kokkos::View<double**> verify("verify", nInstances, total_size);
    Kokkos::parallel_for("verify_multiply", policy,
        KOKKOS_LAMBDA(const member_type& team) {
            const int instance = team.league_rank();
            solvers_d(instance).multiply_parallel_s(
                Kokkos::subview(x, instance, Kokkos::ALL),
                Kokkos::subview(verify, instance, Kokkos::ALL),
                team
            );
    });
    Kokkos::fence();

    auto h_verify = Kokkos::create_mirror_view(verify);
    Kokkos::deep_copy(h_verify, verify);
    Kokkos::deep_copy(h_x, x);
    Kokkos::deep_copy(h_b, b);

    for(int inst = 0; inst < std::min(5, nInstances); ++inst) {
        double residual_sum = 0.0;
        for(int idx = 0; idx < total_size; idx++) {
            double res = h_x(inst, idx) - theta * delta_t * h_verify(inst, idx) - h_b(inst, idx);
            residual_sum += res * res;
        }
        double residual = std::sqrt(residual_sum);
        
        std::cout << "Instance " << inst << " => residual norm = " << residual << std::endl;
        std::cout << "  x[0..4] = ";
        for(int i = 0; i < std::min(10,total_size); i++) {
            std::cout << h_x(inst, i) << " ";
        }
        std::cout << "\n------------------------------------\n";
    }
}

//comparing to test_heston_A2_shuffled() -> works
void test_device_a2_shuffled_comparison() {
    using timer = std::chrono::high_resolution_clock;
    using Device = Kokkos::DefaultExecutionSpace;

    const int m1 = 2;
    const int m2 = 14;
    const int nInstances = 1;  // Single instance for comparison
    const double theta = 0.8;
    const double delta_t = 1.0/40;
    const double r_d = 0.025;
    const double kappa = 1.5;
    const double eta = 0.04;
    const double sigma = 0.3;

    Kokkos::View<Device_A2_shuffled_heston<Device>*> solvers_d("solvers_d", nInstances);
    auto solvers_h = Kokkos::create_mirror_view(solvers_d);
    for(int i = 0; i < nInstances; ++i) {
        solvers_h(i) = Device_A2_shuffled_heston<Device>(m1, m2);
    }
    Kokkos::deep_copy(solvers_d, solvers_h);

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

    const int total_size = (m1+1)*(m2+1);
    Kokkos::View<double**> x("x", nInstances, total_size);
    Kokkos::View<double**> b("b", nInstances, total_size);
    Kokkos::View<double**> result("result", nInstances, total_size);

    auto h_x = Kokkos::create_mirror_view(x);
    auto h_b = Kokkos::create_mirror_view(b);
    for(int inst = 0; inst < nInstances; ++inst) {
        for(int idx = 0; idx < total_size; ++idx) {
            h_x(inst, idx) = idx + 1;
            h_b(inst, idx) = idx + 1;
        }
    }
    Kokkos::deep_copy(x, h_x);
    Kokkos::deep_copy(b, h_b);

    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;
    team_policy policy(nInstances, Kokkos::AUTO);

    // Test implicit solve timing
    auto t_start = timer::now();
    
    Kokkos::parallel_for("solve_implicit", policy,
        KOKKOS_LAMBDA(const member_type& team) {
            const int instance = team.league_rank();
            Device_A2_shuffled_heston<Device>& solver = solvers_d(instance);
            auto x_i = Kokkos::subview(x, instance, Kokkos::ALL);
            auto b_i = Kokkos::subview(b, instance, Kokkos::ALL);
            GridViews grid_i = deviceGrids(instance);
            
            solver.build_matrix(grid_i, r_d, kappa, eta, sigma, theta, delta_t, team);
            solver.solve_implicit_parallel_s(x_i, b_i, team);
    });
    Kokkos::fence();

    auto t_end = timer::now();
    std::cout << "Implicit solve time: "
              << std::chrono::duration<double>(t_end - t_start).count()
              << " seconds" << std::endl;

    // Test multiply timing
    t_start = timer::now();
    
    Kokkos::parallel_for("multiply", policy,
        KOKKOS_LAMBDA(const member_type& team) {
            const int instance = team.league_rank();
            Device_A2_shuffled_heston<Device>& solver = solvers_d(instance);
            auto x_i = Kokkos::subview(x, instance, Kokkos::ALL);
            auto result_i = Kokkos::subview(result, instance, Kokkos::ALL);
            
            solver.multiply_parallel_s(x_i, result_i, team);
    });
    Kokkos::fence();

    t_end = timer::now();
    std::cout << "Explicit solve time: "
              << std::chrono::duration<double>(t_end - t_start).count()
              << " seconds" << std::endl;

    // Compute residual
    auto h_result = Kokkos::create_mirror_view(result);
    Kokkos::deep_copy(h_result, result);
    Kokkos::deep_copy(h_x, x);

    double residual = 0.0;
    for(int i = 0; i < total_size; i++) {
        double res = h_x(0, i) - theta * delta_t * h_result(0, i) - h_b(0, i);
        residual += res * res;
    }
    residual = std::sqrt(residual);
    std::cout << "Residual norm: " << residual << std::endl;

    std::cout << "First few x values: ";
    for(int i = 0; i < std::min(5, total_size); i++) {
        std::cout << h_x(0, i) << " ";
    }
    std::cout << std::endl;
}

void test_a2_shuffled_kernel(){
    Kokkos::initialize();
        {
            try{
                test_device_a2_shuffled_multiple_instances();
                //test_device_a2_shuffled_comparison();
            }
            catch (std::exception& e) {
                std::cout << "Error: " << e.what() << std::endl;
            }
        } // All test objects destroyed here
    Kokkos::finalize();
}
