#include <Kokkos_Core.hpp>
#include "device_solver.hpp"
#include <iostream>
#include <numeric>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <string>
#include <vector>

/*

This is a basic example on hwo to use class instances in device code

*/
/*
template<class DeviceType>
struct SimpleDevice {
    typedef DeviceType execution_space;
    typedef typename DeviceType::memory_space memory_space;
    
    Kokkos::View<double**, DeviceType> data;      // Matrix
    Kokkos::View<double*, DeviceType> vector;     // Vector for multiplication
    Kokkos::View<double*, DeviceType> result;     // Result vector
    int rows, cols;
    
    KOKKOS_FUNCTION
    SimpleDevice() : rows(0), cols(0) {}
    
    // Constructor for initialization
    SimpleDevice(int r, int c) 
        : data(Kokkos::View<double**, DeviceType>("matrix", r, c)),
          vector(Kokkos::View<double*, DeviceType>("vector", c)),
          result(Kokkos::View<double*, DeviceType>("result", r)),
          rows(r), cols(c) {}
    
    KOKKOS_FUNCTION
    void fill_matrix(const int id) {
        // Fill matrix with instance id
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                data(i,j) = static_cast<double>(id);
            }
        }
        // Fill vector with 1's for testing
        for(int j = 0; j < cols; j++) {
            vector(j) = 1.0;
        }
    }

    KOKKOS_FUNCTION
    void matrix_vector_product() {
        // Compute matrix-vector product
        for(int i = 0; i < rows; i++) {
            double sum = 0.0;
            for(int j = 0; j < cols; j++) {
                sum += data(i,j) * vector(j);
            }
            result(i) = sum;
        }
    }
};

void run_device_solver_example() {
    using Device = Kokkos::DefaultExecutionSpace;
    
    const int num_instances = 3;
    const int matrix_size = 10;  // 10x10 matrices
    
    // Create device and host views of our class
    Kokkos::View<SimpleDevice<Device>*> devices_d("devices_d", num_instances);
    auto devices_h = Kokkos::create_mirror_view(devices_d);
    
    // Initialize instances
    for (int i = 0; i < num_instances; ++i) {
        devices_h(i) = SimpleDevice<Device>(matrix_size, matrix_size);
    }
    
    // Copy to device
    Kokkos::deep_copy(devices_d, devices_h);
    
    // Fill matrices in parallel
    Kokkos::parallel_for("fill_matrices", num_instances, 
        KOKKOS_LAMBDA(const int i) {
            devices_d(i).fill_matrix(i);
        }
    );

    // Compute matrix-vector products in parallel
    Kokkos::parallel_for("matrix_vector_products", num_instances,
        KOKKOS_LAMBDA(const int i) {
            devices_d(i).matrix_vector_product();
        }
    );
    
    // Copy back results and print first entries
    Kokkos::deep_copy(devices_h, devices_d);
    
    std::cout << "Results (first entry of result vector for each instance):\n";
    for (int i = 0; i < num_instances; ++i) {
        auto result_h = Kokkos::create_mirror_view(devices_h(i).result);
        Kokkos::deep_copy(result_h, devices_h(i).result);
        std::cout << "Instance " << i << ": " << result_h(0) << "\n";
    }
}
*/



/*

The following implements the the DO scheme in a parallel fashion. It is hardocded inside this test.

*/
void test_DEVICE_parallel_DO_scheme() {
    using timer = std::chrono::high_resolution_clock;
    using Device = Kokkos::DefaultExecutionSpace;

    // Test parameters
    //const double K = 90.0;
    const double S_0 = 100.0;
    const double V_0 = 0.04;
    const double T = 1.0;


    const double r_d = 0.025;
    const double r_f = 0.0;

    const double rho = -0.9;
    const double sigma = 0.3;
    const double kappa = 1.5;
    const double eta = 0.04;
    
    // Parameters
    const int m1 = 50;
    const int m2 = 25;

    const int nInstances = 20;

    //each instance gets its own strike. So we compute the Optioin price to nInstances of strikes in parallel
    //this is accounted for in the different grids (non uniform around strike) as well as the initial condition
    std::vector<double> strikes(nInstances,0.0);
    for(int i = 0; i < nInstances; ++i) {
        strikes[i] = 90 + i;
    }
    
    // Solver parameters
    const int N = 20;
    const double theta = 0.8;
    const double delta_t = T/N;


    std::cout << "Number of Instances: " << nInstances << std::endl;
    std::cout << "Stock S0 = " << S_0 << ", Dimensions: m1 = " << m1 << ", m2 = " << m2 << ", time steps = " << N << std::endl;


    // Create solver arrays
    Kokkos::View<Device_A0_heston<Device>*> A0_solvers("A0_solvers", nInstances);
    Kokkos::View<Device_A1_heston<Device>*> A1_solvers("A1_solvers", nInstances);
    Kokkos::View<Device_A2_shuffled_heston<Device>*> A2_solvers("A2_solvers", nInstances);
    
    // Initialize solvers
    auto h_A0 = Kokkos::create_mirror_view(A0_solvers);
    auto h_A1 = Kokkos::create_mirror_view(A1_solvers);
    auto h_A2 = Kokkos::create_mirror_view(A2_solvers);
    
    for(int i = 0; i < nInstances; i++) {
        h_A0(i) = Device_A0_heston<Device>(m1, m2);
        h_A1(i) = Device_A1_heston<Device>(m1, m2);
        h_A2(i) = Device_A2_shuffled_heston<Device>(m1, m2);
    }
    Kokkos::deep_copy(A0_solvers, h_A0);
    Kokkos::deep_copy(A1_solvers, h_A1);
    Kokkos::deep_copy(A2_solvers, h_A2);

    // Create boundary conditions array
    Kokkos::View<Device_BoundaryConditions<Device>*> bounds_d("bounds_d", nInstances);
    auto h_bounds = Kokkos::create_mirror_view(bounds_d);
    for(int i = 0; i < nInstances; ++i) {
        h_bounds(i) = Device_BoundaryConditions<Device>(m1, m2, r_d, r_f, N, delta_t);
    }
    Kokkos::deep_copy(bounds_d, h_bounds);


    // Initialize grid views
    std::vector<GridViews> hostGrids;
    buildMultipleGridViews(hostGrids, nInstances, m1, m2);
    for(int i = 0; i < nInstances; ++i) {
        double K = strikes[i];
        auto h_Vec_s = Kokkos::create_mirror_view(hostGrids[i].device_Vec_s);
        auto h_Vec_v = Kokkos::create_mirror_view(hostGrids[i].device_Vec_v);
        auto h_Delta_s = Kokkos::create_mirror_view(hostGrids[i].device_Delta_s);
        auto h_Delta_v = Kokkos::create_mirror_view(hostGrids[i].device_Delta_v);

        //Grid tempGrid = create_test_grid(m1, m2);
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

    Kokkos::View<GridViews*> deviceGrids("deviceGrids", nInstances);
    auto h_deviceGrids = Kokkos::create_mirror_view(deviceGrids);
    for(int i = 0; i < nInstances; ++i) h_deviceGrids(i) = hostGrids[i];
    Kokkos::deep_copy(deviceGrids, h_deviceGrids);

    const int total_size = (m1+1)*(m2+1);
    
    // Main solution arrays
    Kokkos::View<double**> U("U", nInstances, total_size);
    Kokkos::View<double**> Y_0("Y_0", nInstances, total_size);
    Kokkos::View<double**> Y_1("Y_1", nInstances, total_size);
    
    // Results arrays
    Kokkos::View<double**> A0_result("A0_result", nInstances, total_size);
    Kokkos::View<double**> A1_result("A1_result", nInstances, total_size);
    Kokkos::View<double**> A2_result_unshuf("A2_result_unshuf", nInstances, total_size);
    
    // Shuffled arrays
    Kokkos::View<double**> U_shuffled("U_shuffled", nInstances, total_size);
    Kokkos::View<double**> Y_1_shuffled("Y_1_shuffled", nInstances, total_size);
    Kokkos::View<double**> A2_result_shuffled("A2_result_shuffled", nInstances, total_size);
    Kokkos::View<double**> U_next_shuffled("U_next_shuffled", nInstances, total_size);


    // Initialize initial conditions U_0
    Kokkos::View<double**> U_0("U_0", nInstances, total_size);
    auto h_U_0 = Kokkos::create_mirror_view(U_0);

    // Fill initial conditions on host
    for(int inst = 0; inst < nInstances; ++inst) {
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
    Kokkos::deep_copy(U, U_0); // Copy to solution vector

    using team_policy = Kokkos::TeamPolicy<>;
    team_policy policy(nInstances, Kokkos::AUTO);


    auto t_start = timer::now();

   

    Kokkos::parallel_for("DO_scheme", policy,
        KOKKOS_LAMBDA(const team_policy::member_type& team) {
            const int instance = team.league_rank();
            
            auto U_i = Kokkos::subview(U, instance, Kokkos::ALL);
            auto Y_0_i = Kokkos::subview(Y_0, instance, Kokkos::ALL);
            auto Y_1_i = Kokkos::subview(Y_1, instance, Kokkos::ALL);
            auto A0_result_i = Kokkos::subview(A0_result, instance, Kokkos::ALL);
            auto A1_result_i = Kokkos::subview(A1_result, instance, Kokkos::ALL);
            auto A2_result_unshuf_i = Kokkos::subview(A2_result_unshuf, instance, Kokkos::ALL);
            
            auto U_shuffled_i = Kokkos::subview(U_shuffled, instance, Kokkos::ALL);
            auto Y_1_shuffled_i = Kokkos::subview(Y_1_shuffled, instance, Kokkos::ALL);
            auto A2_result_shuffled_i = Kokkos::subview(A2_result_shuffled, instance, Kokkos::ALL);
            auto U_next_shuffled_i = Kokkos::subview(U_next_shuffled, instance, Kokkos::ALL);

    
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

    auto t_end = timer::now();
    std::cout << "Parallel DO time: "
              << std::chrono::duration<double>(t_end - t_start).count()
              << " seconds" << std::endl;

    // Create host mirrors to access the data
    const double reference_price = 8.8948693600540167;

    auto h_U = Kokkos::create_mirror_view(U);
    Kokkos::deep_copy(h_U, U);

    for(int inst = 0; inst < min(5,nInstances); ++inst) {
        // Create host mirrors for the grid views
        auto h_Vec_s = Kokkos::create_mirror_view(hostGrids[inst].device_Vec_s);
        auto h_Vec_v = Kokkos::create_mirror_view(hostGrids[inst].device_Vec_v);
        Kokkos::deep_copy(h_Vec_s, hostGrids[inst].device_Vec_s);
        Kokkos::deep_copy(h_Vec_v, hostGrids[inst].device_Vec_v);

        // Find indices (assuming S_0 and V_0 are defined)
        int index_s = -1;
        int index_v = -1;
        
        for(int i = 0; i <= m1; i++) {
            if(std::abs(h_Vec_s(i) - S_0) < 1e-10) {
                index_s = i;
                break;
            }
        }
        
        for(int i = 0; i <= m2; i++) {
            if(std::abs(h_Vec_v(i) - V_0) < 1e-10) {
                index_v = i;
                break;
            }
        }

        double price = h_U(inst, index_s + index_v*(m1+1));
        double rel_error = std::abs(price - reference_price)/reference_price;
        
        std::cout << "Instance " << inst 
                  << " Strike " << strikes[inst] 
                << ": Price = " << std::setprecision(16) << price << "\n";
                //<< ", Relative Error = " << rel_error << "\n";
    }
}


/*

This tests the implementation of a method of the the parallel_do_solver

*/
void test_parallel_DO_method() {
    using timer = std::chrono::high_resolution_clock;
    using Device = Kokkos::DefaultExecutionSpace;

    const double S_0 = 100.0;
    const double V_0 = 0.04;
    const double T = 1.0;


    const double r_d = 0.025;
    const double r_f = 0.0;
    const double rho = -0.9;
    const double sigma = 0.3;
    const double kappa = 1.5;
    const double eta = 0.04;
    
    // Parameters
    const int m1 = 50;
    const int m2 = 25;

    const int nInstances = 5;

    //each instance gets its own strike. So we compute the Optioin price to nInstances of strikes in parallel
    //this is accounted for in the different grids (non uniform around strike) as well as the initial condition
    std::vector<double> strikes(nInstances,0.0);
    for(int i = 0; i < nInstances; ++i) {
        strikes[i] = 90 + i;
    }
    
    // Solver parameters
    const int N = 20;
    const double theta = 0.8;
    const double delta_t = T/N;

    std::cout << "Number of Instances: " << nInstances << std::endl;
    std::cout << "Stock S0 = " << S_0 << ", Dimensions: m1 = " << m1 << ", m2 = " << m2 << ", time steps = " << N << std::endl;


    // Create solver arrays
    Kokkos::View<Device_A0_heston<Device>*> A0_solvers("A0_solvers", nInstances);
    Kokkos::View<Device_A1_heston<Device>*> A1_solvers("A1_solvers", nInstances);
    Kokkos::View<Device_A2_shuffled_heston<Device>*> A2_solvers("A2_solvers", nInstances);
    
    // Initialize solvers
    auto h_A0 = Kokkos::create_mirror_view(A0_solvers);
    auto h_A1 = Kokkos::create_mirror_view(A1_solvers);
    auto h_A2 = Kokkos::create_mirror_view(A2_solvers);
    
    for(int i = 0; i < nInstances; i++) {
        h_A0(i) = Device_A0_heston<Device>(m1, m2);
        h_A1(i) = Device_A1_heston<Device>(m1, m2);
        h_A2(i) = Device_A2_shuffled_heston<Device>(m1, m2);
    }
    Kokkos::deep_copy(A0_solvers, h_A0);
    Kokkos::deep_copy(A1_solvers, h_A1);
    Kokkos::deep_copy(A2_solvers, h_A2);

    // Create boundary conditions array
    Kokkos::View<Device_BoundaryConditions<Device>*> bounds_d("bounds_d", nInstances);
    auto h_bounds = Kokkos::create_mirror_view(bounds_d);
    for(int i = 0; i < nInstances; ++i) {
        h_bounds(i) = Device_BoundaryConditions<Device>(m1, m2, r_d, r_f, N, delta_t);
    }
    Kokkos::deep_copy(bounds_d, h_bounds);


    // Initialize grid views
    std::vector<GridViews> hostGrids;
    buildMultipleGridViews(hostGrids, nInstances, m1, m2);
    for(int i = 0; i < nInstances; ++i) {
        double K = strikes[i];
        auto h_Vec_s = Kokkos::create_mirror_view(hostGrids[i].device_Vec_s);
        auto h_Vec_v = Kokkos::create_mirror_view(hostGrids[i].device_Vec_v);
        auto h_Delta_s = Kokkos::create_mirror_view(hostGrids[i].device_Delta_s);
        auto h_Delta_v = Kokkos::create_mirror_view(hostGrids[i].device_Delta_v);

        //Grid tempGrid = create_test_grid(m1, m2);
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

    Kokkos::View<GridViews*> deviceGrids("deviceGrids", nInstances);
    auto h_deviceGrids = Kokkos::create_mirror_view(deviceGrids);
    for(int i = 0; i < nInstances; ++i) h_deviceGrids(i) = hostGrids[i];
    Kokkos::deep_copy(deviceGrids, h_deviceGrids);
    
    const int total_size = (m1+1)*(m2+1);
    
    // Create workspace instead of individual arrays
    DO_Workspace<Device> workspace(nInstances, total_size);

    // Initialize initial conditions U_0
    Kokkos::View<double**> U_0("U_0", nInstances, total_size);
    auto h_U_0 = Kokkos::create_mirror_view(U_0);

    // Fill initial conditions on host
    for(int inst = 0; inst < nInstances; ++inst) {
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

    auto t_start = timer::now();

    // Call solver with workspace
    for(int i = 0; i<5; i++){
        
        Kokkos::deep_copy(workspace.U, U_0);

        parallel_DO_solve(
        nInstances, m1, m2, N, T, delta_t, theta,
        r_d, r_f, rho, sigma, kappa, eta,
        A0_solvers, A1_solvers, A2_solvers,
        bounds_d, deviceGrids,
        workspace);
    }

    auto t_end = timer::now();
    std::cout << "Parallel DO time: "
              << std::chrono::duration<double>(t_end - t_start).count()
              << " seconds" << std::endl;

    //Results processing uses workspace.U instead of U
    auto h_U = Kokkos::create_mirror_view(workspace.U);
    Kokkos::deep_copy(h_U, workspace.U);

    for(int inst = 0; inst < min(5,nInstances); ++inst) {
        // Create host mirrors for the grid views
        auto h_Vec_s = Kokkos::create_mirror_view(hostGrids[inst].device_Vec_s);
        auto h_Vec_v = Kokkos::create_mirror_view(hostGrids[inst].device_Vec_v);
        Kokkos::deep_copy(h_Vec_s, hostGrids[inst].device_Vec_s);
        Kokkos::deep_copy(h_Vec_v, hostGrids[inst].device_Vec_v);

        // Find indices (assuming S_0 and V_0 are defined)
        int index_s = -1;
        int index_v = -1;
        
        for(int i = 0; i <= m1; i++) {
            if(std::abs(h_Vec_s(i) - S_0) < 1e-10) {
                index_s = i;
                break;
            }
        }
        
        for(int i = 0; i <= m2; i++) {
            if(std::abs(h_Vec_v(i) - V_0) < 1e-10) {
                index_v = i;
                break;
            }
        }

        double price = h_U(inst, index_s + index_v*(m1+1));
        //double rel_error = std::abs(price - reference_price)/reference_price;
        
        std::cout << "Instance " << inst 
                  << " Strike " << strikes[inst] 
                << ": Price = " << std::setprecision(16) << price << "\n";
                //<< ", Relative Error = " << rel_error << "\n";
    }
}


/*

This is a test for how we could compute the jacobian in parallel.

*/
void test_deviceCallable_Do_solver() {
    using timer = std::chrono::high_resolution_clock;
    using Device = Kokkos::DefaultExecutionSpace;

    const double S_0 = 100.0;
    const double V_0 = 0.04;
    const double T = 1.0;


    const double r_d = 0.025;
    const double r_f = 0.0;

    const double rho = -0.9;
    const double sigma = 0.3;
    const double kappa = 1.5;
    const double eta = 0.04;
    
    // Parameters
    const int m1 = 50;
    const int m2 = 25;

    const int nInstances = 5;

    //each instance gets its own strike. So we compute the Optioin price to nInstances of strikes in parallel
    //this is accounted for in the different grids (non uniform around strike) as well as the initial condition
    std::vector<double> strikes(nInstances,0.0);
    for(int i = 0; i < nInstances; ++i) {
        strikes[i] = 90 + i;
    }
    
    // Solver parameters
    const int N = 20;
    const double theta = 0.8;
    const double delta_t = T/N;

    std::cout << "Number of Instances: " << nInstances << std::endl;
    std::cout << "Stock S0 = " << S_0 << ", Dimensions: m1 = " << m1 << ", m2 = " << m2 << ", time steps = " << N << std::endl;


    // Create solver arrays
    Kokkos::View<Device_A0_heston<Device>*> A0_solvers("A0_solvers", nInstances);
    Kokkos::View<Device_A1_heston<Device>*> A1_solvers("A1_solvers", nInstances);
    Kokkos::View<Device_A2_shuffled_heston<Device>*> A2_solvers("A2_solvers", nInstances);
    
    // Initialize solvers
    auto h_A0 = Kokkos::create_mirror_view(A0_solvers);
    auto h_A1 = Kokkos::create_mirror_view(A1_solvers);
    auto h_A2 = Kokkos::create_mirror_view(A2_solvers);
    
    for(int i = 0; i < nInstances; i++) {
        h_A0(i) = Device_A0_heston<Device>(m1, m2);
        h_A1(i) = Device_A1_heston<Device>(m1, m2);
        h_A2(i) = Device_A2_shuffled_heston<Device>(m1, m2);
    }
    Kokkos::deep_copy(A0_solvers, h_A0);
    Kokkos::deep_copy(A1_solvers, h_A1);
    Kokkos::deep_copy(A2_solvers, h_A2);

    // Create boundary conditions array
    Kokkos::View<Device_BoundaryConditions<Device>*> bounds_d("bounds_d", nInstances);
    auto h_bounds = Kokkos::create_mirror_view(bounds_d);
    for(int i = 0; i < nInstances; ++i) {
        h_bounds(i) = Device_BoundaryConditions<Device>(m1, m2, r_d, r_f, N, delta_t);
    }
    Kokkos::deep_copy(bounds_d, h_bounds);


    // Initialize grid views
    std::vector<GridViews> hostGrids;
    buildMultipleGridViews(hostGrids, nInstances, m1, m2);
    for(int i = 0; i < nInstances; ++i) {
        double K = strikes[i];
        auto h_Vec_s = Kokkos::create_mirror_view(hostGrids[i].device_Vec_s);
        auto h_Vec_v = Kokkos::create_mirror_view(hostGrids[i].device_Vec_v);
        auto h_Delta_s = Kokkos::create_mirror_view(hostGrids[i].device_Delta_s);
        auto h_Delta_v = Kokkos::create_mirror_view(hostGrids[i].device_Delta_v);

        //Grid tempGrid = create_test_grid(m1, m2);
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

    Kokkos::View<GridViews*> deviceGrids("deviceGrids", nInstances);
    auto h_deviceGrids = Kokkos::create_mirror_view(deviceGrids);
    for(int i = 0; i < nInstances; ++i) h_deviceGrids(i) = hostGrids[i];
    Kokkos::deep_copy(deviceGrids, h_deviceGrids);
    
    const int total_size = (m1+1)*(m2+1);
    
    // Create workspace instead of individual arrays
    DO_Workspace<Device> workspace(nInstances, total_size);

    // Initialize initial conditions U_0
    Kokkos::View<double**> U_0("U_0", nInstances, total_size);
    auto h_U_0 = Kokkos::create_mirror_view(U_0);

    // Fill initial conditions on host
    for(int inst = 0; inst < nInstances; ++inst) {
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

    using team_policy = Kokkos::TeamPolicy<>;
    team_policy policy(nInstances, Kokkos::AUTO);

    auto t_start = timer::now();

    // Main kernel launch with modified internals
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

            //Initialize Grid views
            GridViews grid_i = deviceGrids(instance);
            
            // Initialize boundaries 
            bounds_d(instance).initialize(grid_i, team);
            auto bounds = bounds_d(instance);
            
            // Build matrices
            A0_solvers(instance).build_matrix(grid_i, rho, sigma, team);
            A1_solvers(instance).build_matrix(grid_i, r_d, r_f, theta, delta_t, team);
            A2_solvers(instance).build_matrix(grid_i, r_d, kappa, eta, sigma, theta, delta_t, team);

            // Call device timestepping
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
    std::cout << "Parallel DO time: "
              << std::chrono::duration<double>(t_end - t_start).count()
              << " seconds" << std::endl;

    // Results processing uses workspace.U instead of U
    auto h_U = Kokkos::create_mirror_view(workspace.U);
    Kokkos::deep_copy(h_U, workspace.U);

    for(int inst = 0; inst < min(5,nInstances); ++inst) {
        // Create host mirrors for the grid views
        auto h_Vec_s = Kokkos::create_mirror_view(hostGrids[inst].device_Vec_s);
        auto h_Vec_v = Kokkos::create_mirror_view(hostGrids[inst].device_Vec_v);
        Kokkos::deep_copy(h_Vec_s, hostGrids[inst].device_Vec_s);
        Kokkos::deep_copy(h_Vec_v, hostGrids[inst].device_Vec_v);

        // Find indices (assuming S_0 and V_0 are defined)
        int index_s = -1;
        int index_v = -1;
        
        for(int i = 0; i <= m1; i++) {
            if(std::abs(h_Vec_s(i) - S_0) < 1e-10) {
                index_s = i;
                break;
            }
        }
        
        for(int i = 0; i <= m2; i++) {
            if(std::abs(h_Vec_v(i) - V_0) < 1e-10) {
                index_v = i;
                break;
            }
        }

        double price = h_U(inst, index_s + index_v*(m1+1));
        //double rel_error = std::abs(price - reference_price)/reference_price;
        
        std::cout << "Instance " << inst 
                  << " Strike " << strikes[inst] 
                << ": Price = " << std::setprecision(16) << price << "\n";
                //<< ", Relative Error = " << rel_error << "\n";
    }
}


/*

First for loop test for jacobian

*/
//works! Only v0 will need to do a "trick" to get the FD approximation
void test_jacobian_computation() {
    using timer = std::chrono::high_resolution_clock;
    using Device = Kokkos::DefaultExecutionSpace;

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
    const int num_strikes = 500;
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
    

    using team_policy = Kokkos::TeamPolicy<>;
    team_policy policy(num_strikes, Kokkos::AUTO);

    // First compute base prices
    auto t_start = timer::now();

    // Single kernel call that handles both base price and parameter perturbations
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

            // Loop over parameters for finite differences
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
                    pert_prices(instance, param) = pert_price;
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
                    pert_prices(instance, param) = pert_price;
                    J(instance, param) = (pert_price - base_price) / eps;
                }
            }
        });
        Kokkos::fence();

    auto t_end = timer::now();
    std::cout << "Jacobian computation time: "
              << std::chrono::duration<double>(t_end - t_start).count()
              << " seconds" << std::endl;

    /*
    // Print Jacobian matrix
    auto h_J = Kokkos::create_mirror_view(J);
    Kokkos::deep_copy(h_J, J);
    
    
    // Column headers
    int precision = 6;
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
    
   //Printing prices
   /*
    auto h_base_prices = Kokkos::create_mirror_view(base_prices);
    auto h_pert_prices = Kokkos::create_mirror_view(pert_prices);
    Kokkos::deep_copy(h_base_prices, base_prices);
    Kokkos::deep_copy(h_pert_prices, pert_prices);

    
    std::cout << "\nBase and perturbed prices:\n";
    std::cout << std::setw(10) << "Strike" << std::setw(15) << "Base Price" 
            << std::setw(15) << "κ pert" << std::setw(15) << "η pert" 
            << std::setw(15) << "σ pert" << std::setw(15) << "ρ pert" 
            << std::setw(15) << "v0 pert\n";

    for(int i = 0; i < num_strikes; i++) {
        std::cout << std::setw(10) << strikes[i] << std::setw(15) << h_base_prices(i);
        for(int j = 0; j < 5; j++) {
            std::cout << std::setw(15) << h_pert_prices(i,j);
        }
        std::cout << "\n";
    }

    // Compute and verify Jacobian entries from prices
    std::cout << "\nJacobian computed from prices:\n";
    std::cout << "Strike   κ       η       σ       ρ       v0\n";
    for(int i = 0; i < num_strikes; i++) {
        std::cout << std::fixed << std::setprecision(6);
        std::cout << strikes[i] << "  ";
        
        // For each parameter
        for(int j = 0; j < 5; j++) {
            double finite_diff = (h_pert_prices(i,j) - h_base_prices(i)) / eps;
            std::cout << finite_diff << "  ";
        }
        std::cout << "\n";
    }
    */
    
    
}


/*

Parallising the entire matrix

*/
//wrong
void compute_heston_jacobian(
    // Market parameters
    const double S_0,              // Initial stock price
    const double V_0,              // Initial variance
    const double T,                // Time to maturity
    const double r_d,              // Domestic interest rate
    const double r_f,              // Foreign interest rate
    // Current parameters to compute Jacobian at
    const std::vector<double>& params,  // [kappa, eta, sigma, rho, v0]
    const std::vector<double>& strikes, // Market strikes
    // Grid parameters
    const int m1,                  // Stock price grid points
    const int m2,                  // Variance grid points
    const int N,                   // Time steps
    const double eps,              // Parameter perturbation size
    // Output
    Kokkos::View<double**>& jacobian) {  // Output Jacobian matrix
    
    using timer = std::chrono::high_resolution_clock;
    using Device = Kokkos::DefaultExecutionSpace;

    const int num_strikes = strikes.size();
    const int num_params = 5;  // Heston has 5 parameters
    const int total_solvers = num_strikes * (num_params + 1);  // +1 for base computation
    const double theta = 0.8;  // ADI scheme parameter
    const double delta_t = T/N;

    std::cout << "Computing Jacobian for " << num_strikes << " strikes\n";
    std::cout << "Total solvers: " << total_solvers << "\n";
    std::cout << "Grid: m1 = " << m1 << ", m2 = " << m2 << ", N = " << N << "\n";

    // Create solver arrays
    Kokkos::View<Device_A0_heston<Device>*> A0_solvers("A0_solvers", total_solvers);
    Kokkos::View<Device_A1_heston<Device>*> A1_solvers("A1_solvers", total_solvers);
    Kokkos::View<Device_A2_shuffled_heston<Device>*> A2_solvers("A2_solvers", total_solvers);
    
    // Initialize solvers
    auto h_A0 = Kokkos::create_mirror_view(A0_solvers);
    auto h_A1 = Kokkos::create_mirror_view(A1_solvers);
    auto h_A2 = Kokkos::create_mirror_view(A2_solvers);
    
    for(int i = 0; i < total_solvers; i++) {
        h_A0(i) = Device_A0_heston<Device>(m1, m2);
        h_A1(i) = Device_A1_heston<Device>(m1, m2);
        h_A2(i) = Device_A2_shuffled_heston<Device>(m1, m2);
    }
    Kokkos::deep_copy(A0_solvers, h_A0);
    Kokkos::deep_copy(A1_solvers, h_A1);
    Kokkos::deep_copy(A2_solvers, h_A2);

    // Create boundary conditions array
    Kokkos::View<Device_BoundaryConditions<Device>*> bounds_d("bounds_d", total_solvers);
    auto h_bounds = Kokkos::create_mirror_view(bounds_d);
    for(int i = 0; i < total_solvers; ++i) {
        h_bounds(i) = Device_BoundaryConditions<Device>(m1, m2, r_d, r_f, N, delta_t);
    }
    Kokkos::deep_copy(bounds_d, h_bounds);

    // Initialize grid views for each solver
    std::vector<GridViews> hostGrids;
    buildMultipleGridViews(hostGrids, total_solvers, m1, m2);

    // Set up grids - each strike gets num_params + 1 identical grids
    for(int i = 0; i < num_strikes; ++i) {
        double K = strikes[i];
        // Create base grid for this strike
        Grid tempGrid(m1, 8*K, S_0, K, K/5, m2, 5.0, V_0, 5.0/500);
        
        // Copy this grid to all solvers for this strike
        for(int j = 0; j <= num_params; ++j) {
            int solver_idx = i * (num_params + 1) + j;
            
            auto h_Vec_s = Kokkos::create_mirror_view(hostGrids[solver_idx].device_Vec_s);
            auto h_Vec_v = Kokkos::create_mirror_view(hostGrids[solver_idx].device_Vec_v);
            auto h_Delta_s = Kokkos::create_mirror_view(hostGrids[solver_idx].device_Delta_s);
            auto h_Delta_v = Kokkos::create_mirror_view(hostGrids[solver_idx].device_Delta_v);

            for(int k = 0; k <= m1; k++) h_Vec_s(k) = tempGrid.Vec_s[k];
            for(int k = 0; k <= m2; k++) h_Vec_v(k) = tempGrid.Vec_v[k];
            for(int k = 0; k < m1; k++) h_Delta_s(k) = tempGrid.Delta_s[k];
            for(int k = 0; k < m2; k++) h_Delta_v(k) = tempGrid.Delta_v[k];

            Kokkos::deep_copy(hostGrids[solver_idx].device_Vec_s, h_Vec_s);
            Kokkos::deep_copy(hostGrids[solver_idx].device_Vec_v, h_Vec_v);
            Kokkos::deep_copy(hostGrids[solver_idx].device_Delta_s, h_Delta_s);
            Kokkos::deep_copy(hostGrids[solver_idx].device_Delta_v, h_Delta_v);
        }
    }

    Kokkos::View<GridViews*> deviceGrids("deviceGrids", total_solvers);
    auto h_deviceGrids = Kokkos::create_mirror_view(deviceGrids);
    for(int i = 0; i < total_solvers; ++i) h_deviceGrids(i) = hostGrids[i];
    Kokkos::deep_copy(deviceGrids, h_deviceGrids);
    
    const int total_size = (m1+1)*(m2+1);
    
    // Create workspace
    DO_Workspace<Device> workspace(total_solvers, total_size);

    // Initialize initial conditions
    Kokkos::View<double**> U_0("U_0", total_solvers, total_size);
    auto h_U_0 = Kokkos::create_mirror_view(U_0);

    // Set initial conditions for each solver
    for(int i = 0; i < num_strikes; ++i) {
        double K = strikes[i];
        auto h_Vec_s = Kokkos::create_mirror_view(hostGrids[i*(num_params+1)].device_Vec_s);
        Kokkos::deep_copy(h_Vec_s, hostGrids[i*(num_params+1)].device_Vec_s);
        
        // Set same initial condition for all solvers for this strike
        for(int j = 0; j <= num_params; ++j) {
            int solver_idx = i * (num_params + 1) + j;
            for(int k = 0; k <= m2; k++) {
                for(int l = 0; l <= m1; l++) {
                    h_U_0(solver_idx, l + k*(m1+1)) = std::max(h_Vec_s(l) - K, 0.0);
                }
            }
        }
    }
    Kokkos::deep_copy(U_0, h_U_0);
    Kokkos::deep_copy(workspace.U, U_0);

    // Create team policy
    using team_policy = Kokkos::TeamPolicy<>;
    team_policy policy(total_solvers, Kokkos::AUTO);

    // Create device Views for parameters and strikes
    Kokkos::View<double*> d_params("params", 5);
    Kokkos::View<double*> d_strikes("strikes", num_strikes);

    // Copy parameters and strikes to device
    auto h_params = Kokkos::create_mirror_view(d_params);
    auto h_strikes = Kokkos::create_mirror_view(d_strikes);

    for(int i = 0; i < 5; i++) {
        h_params(i) = params[i];
    }
    for(int i = 0; i < num_strikes; i++) {
        h_strikes(i) = strikes[i];
    }

    Kokkos::deep_copy(d_params, h_params);
    Kokkos::deep_copy(d_strikes, h_strikes);

    auto t_start = timer::now();

    // Launch all solvers in parallel
    Kokkos::parallel_for("Jacobian_computation", policy,
        KOKKOS_LAMBDA(const team_policy::member_type& team) {
            const int instance = team.league_rank();
            const int strike_idx = instance / (num_params + 1); //maybe get rid of the 1 here
            const int param_idx = instance % (num_params + 1);
            
            // Get current parameters for this solver
            double kappa = d_params(0);
            double eta = d_params(1);
            double sigma = d_params(2);
            double rho = d_params(3);
            double v0 = d_params(4);
            
            // If not base solve, perturb appropriate parameter
            if(param_idx > 0) {
                switch(param_idx - 1) {  // -1 because param_idx starts at 0
                    case 0: kappa += eps; break;
                    case 1: eta += eps; break;
                    case 2: sigma += eps; break;
                    case 3: rho += eps; break;
                    case 4: v0 += eps; break;
                }
            }
            
            // Get workspace views
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
            
            // Initialize boundaries
            bounds_d(instance).initialize(grid_i, team);
            auto bounds = bounds_d(instance);
            
            // Build matrices with current parameters
            A0_solvers(instance).build_matrix(grid_i, rho, sigma, team);
            A1_solvers(instance).build_matrix(grid_i, r_d, r_f, theta, delta_t, team);
            A2_solvers(instance).build_matrix(grid_i, r_d, kappa, eta, sigma, theta, delta_t, team);

            // Solve PDE
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
    std::cout << "Parallel Jacobian computation time: "
            << std::chrono::duration<double>(t_end - t_start).count()
            << " seconds\n";

    // Extract results and build Jacobian
    auto h_U = Kokkos::create_mirror_view(workspace.U);
    Kokkos::deep_copy(h_U, workspace.U);

    // Find S_0 and V_0 indices
    auto h_Vec_s = Kokkos::create_mirror_view(hostGrids[0].device_Vec_s);
    auto h_Vec_v = Kokkos::create_mirror_view(hostGrids[0].device_Vec_v);
    Kokkos::deep_copy(h_Vec_s, hostGrids[0].device_Vec_s);
    Kokkos::deep_copy(h_Vec_v, hostGrids[0].device_Vec_v);

    int index_s = -1, index_v = -1;
    for(int i = 0; i <= m1; i++) {
        if(std::abs(h_Vec_s(i) - S_0) < 1e-10) {
            index_s = i;
            break;
        }
    }
    for(int i = 0; i <= m2; i++) {
        if(std::abs(h_Vec_v(i) - V_0) < 1e-10) {
            index_v = i;
            break;
        }
    }

    std::cout << "\nExtracted indices: index_s = " << index_s << ", index_v = " << index_v << "\n";

    // Build Jacobian matrix with debug output
    auto h_jacobian = Kokkos::create_mirror_view(jacobian);
    for(int i = 0; i < num_strikes; ++i) {
        int base_idx = i * (num_params + 1);
        double base_price = h_U(base_idx, index_s + index_v*(m1+1));
        
        //std::cout << "\nStrike " << strikes[i] << " base price: " << base_price << "\n";
        
        for(int j = 0; j < num_params; ++j) {
            int perturb_idx = base_idx + j + 1;
            double perturbed_price = h_U(perturb_idx, index_s + index_v*(m1+1));
            
            h_jacobian(i,j) = (perturbed_price - base_price) / eps;
            
            //std::cout << "Param " << j << ": perturbed price = " << perturbed_price 
                    //<< ", diff = " << perturbed_price - base_price 
                    //<< ", derivative = " << h_jacobian(i,j) << "\n";
        }
    }

    // Copy Jacobian back to device
    Kokkos::deep_copy(jacobian, h_jacobian);
}

//this is wrong
void test_heston_jacobian() {
    // Market parameters
    const double S_0 = 100.0;  // Initial stock price
    const double V_0 = 0.04;   // Initial variance
    const double T = 1.0;      // Time to maturity
    const double r_d = 0.025;  // Domestic interest rate
    const double r_f = 0.0;    // Foreign interest rate

    // Initial parameter guess
    std::vector<double> params = {
        1.5,   // kappa: mean reversion rate
        0.04,  // eta: long-term mean of variance
        0.3,   // sigma: volatility of variance
        -0.9,  // rho: correlation
        0.04   // v0: initial variance (same as V_0 for initial guess)
    };

    // Create array of strikes around S_0
    const int num_strikes = 5;
    std::vector<double> strikes(num_strikes);
    for(int i = 0; i < num_strikes; ++i) {
        strikes[i] = 90.0 + i;  // Strikes 
    }

    // Grid and solver parameters
    const int m1 = 50;          // Stock grid points
    const int m2 = 25;          // Variance grid points
    const int N = 20;           // Time steps
    const double eps = 1e-6;     // Parameter perturbation size

    // Create Jacobian matrix
    Kokkos::View<double**> jacobian("jacobian", num_strikes, 5);

    std::cout << "Starting Heston Jacobian computation\n";
    std::cout << "Number of strikes: " << num_strikes << "\n";
    std::cout << "Grid: m1 = " << m1 << ", m2 = " << m2 << ", N = " << N << "\n";
    std::cout << "Current parameters: kappa = " << params[0] 
                << ", eta = " << params[1]
                << ", sigma = " << params[2]
                << ", rho = " << params[3]
                << ", v0 = " << params[4] << "\n\n";

    // Compute Jacobian
    compute_heston_jacobian(
        S_0, V_0, T, r_d, r_f,
        params, strikes,
        m1, m2, N, eps,
        jacobian
    );

    // Print Jacobian matrix
    auto h_jacobian = Kokkos::create_mirror_view(jacobian);
    Kokkos::deep_copy(h_jacobian, jacobian);

    std::cout << "\nJacobian matrix:\n";
    std::cout << "Format: Row = strike, Columns = [kappa, eta, sigma, rho, v0]\n\n";
    /*
    for(int i = 0; i < num_strikes; ++i) {
        std::cout << "Strike " << strikes[i] << ": ";
        for(int j = 0; j < 5; ++j) {
            std::cout << std::setw(10) << std::setprecision(4) << h_jacobian(i,j) << " ";
        }
        std::cout << "\n";
    }
    */
    
}


/*

sequential J

*/
void test_sequential_J() {
    using timer = std::chrono::high_resolution_clock;
    using Device = Kokkos::DefaultExecutionSpace;

    const double S_0 = 100.0;
    const double V_0 = 0.04;
    const double T = 1.0;

    const double rho = -0.9;
    const double sigma = 0.3;
    const double kappa = 1.5;
    const double eta = 0.04;

    const double r_d = 0.025;
    const double r_f = 0.0;

    std::vector<double> base_params = {kappa, eta, sigma, rho, V_0};
    const std::vector<std::string> param_names = {"kappa", "eta", "sigma", "rho", "V_0"};
    const double eps = 1e-6;  // Perturbation size

    std::cout << "Base parameters:\n";
    for(int i = 0; i < 5; i++) {
        std::cout << param_names[i] << " = " << base_params[i] << "\n";
    }
    
    // Parameters
    const int m1 = 50;
    const int m2 = 25;

    const int nInstances = 10;

    //each instance gets its own strike. So we compute the Optioin price to nInstances of strikes in parallel
    //this is accounted for in the different grids (non uniform around strike) as well as the initial condition
    std::vector<double> strikes(nInstances,0.0);
    for(int i = 0; i < nInstances; ++i) {
        strikes[i] = 90 + i;
    }
    
    // Solver parameters
    const int N = 20;
    const double theta = 0.8;
    const double delta_t = T/N;

    std::cout << "Number of Instances: " << nInstances << std::endl;
    std::cout << "Stock S0 = " << S_0 << ", Dimensions: m1 = " << m1 << ", m2 = " << m2 << ", time steps = " << N << std::endl;


    // Create solver arrays
    Kokkos::View<Device_A0_heston<Device>*> A0_solvers("A0_solvers", nInstances);
    Kokkos::View<Device_A1_heston<Device>*> A1_solvers("A1_solvers", nInstances);
    Kokkos::View<Device_A2_shuffled_heston<Device>*> A2_solvers("A2_solvers", nInstances);
    
    // Initialize solvers
    auto h_A0 = Kokkos::create_mirror_view(A0_solvers);
    auto h_A1 = Kokkos::create_mirror_view(A1_solvers);
    auto h_A2 = Kokkos::create_mirror_view(A2_solvers);
    
    for(int i = 0; i < nInstances; i++) {
        h_A0(i) = Device_A0_heston<Device>(m1, m2);
        h_A1(i) = Device_A1_heston<Device>(m1, m2);
        h_A2(i) = Device_A2_shuffled_heston<Device>(m1, m2);
    }
    Kokkos::deep_copy(A0_solvers, h_A0);
    Kokkos::deep_copy(A1_solvers, h_A1);
    Kokkos::deep_copy(A2_solvers, h_A2);

    // Create boundary conditions array
    Kokkos::View<Device_BoundaryConditions<Device>*> bounds_d("bounds_d", nInstances);
    auto h_bounds = Kokkos::create_mirror_view(bounds_d);
    for(int i = 0; i < nInstances; ++i) {
        h_bounds(i) = Device_BoundaryConditions<Device>(m1, m2, r_d, r_f, N, delta_t);
    }
    Kokkos::deep_copy(bounds_d, h_bounds);


    // Initialize grid views
    std::vector<GridViews> hostGrids;
    buildMultipleGridViews(hostGrids, nInstances, m1, m2);
    for(int i = 0; i < nInstances; ++i) {
        double K = strikes[i];
        auto h_Vec_s = Kokkos::create_mirror_view(hostGrids[i].device_Vec_s);
        auto h_Vec_v = Kokkos::create_mirror_view(hostGrids[i].device_Vec_v);
        auto h_Delta_s = Kokkos::create_mirror_view(hostGrids[i].device_Delta_s);
        auto h_Delta_v = Kokkos::create_mirror_view(hostGrids[i].device_Delta_v);

        //Grid tempGrid = create_test_grid(m1, m2);
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

    Kokkos::View<GridViews*> deviceGrids("deviceGrids", nInstances);
    auto h_deviceGrids = Kokkos::create_mirror_view(deviceGrids);
    for(int i = 0; i < nInstances; ++i) h_deviceGrids(i) = hostGrids[i];
    Kokkos::deep_copy(deviceGrids, h_deviceGrids);
    
    const int total_size = (m1+1)*(m2+1);
    
    // Create workspace instead of individual arrays
    DO_Workspace<Device> workspace(nInstances, total_size);

    // Initialize initial conditions U_0
    Kokkos::View<double**> U_0("U_0", nInstances, total_size);
    auto h_U_0 = Kokkos::create_mirror_view(U_0);

    // Fill initial conditions on host
    for(int inst = 0; inst < nInstances; ++inst) {
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

    // Storage for perturbed prices
    Kokkos::View<double**> perturbed_prices("perturbed_prices", 5, nInstances);
    auto h_perturbed_prices = Kokkos::create_mirror_view(perturbed_prices);

    // Create device storage for parameters
    Kokkos::View<double*> d_params("params", 5);
    auto h_params = Kokkos::create_mirror_view(d_params);

    using team_policy = Kokkos::TeamPolicy<>;
    team_policy policy(nInstances, Kokkos::AUTO);

    // Loop over parameters for perturbation
    std::cout << "\nComputing prices for parameter " << " perturbation\n";
    auto t_start = timer::now();
    for(int param_idx = 0; param_idx < 5; param_idx++) {
    //for(int param_idx = 4; param_idx >= 0; param_idx--) {
    //std::vector<int> param_sequence = {4, 0, 1, 2, 3};  // sigma, kappa, rho, eta, V_0
    //for(int i = 0; i < 5; i++) {
        //int param_idx = param_sequence[i];
        std::cout << "\n========= Parameter " << param_names[param_idx] << " iteration =========\n";
        
        // Reset ALL parameters to base values
        for(int i = 0; i < 5; i++) {
            h_params(i) = base_params[i];
        }
        
        // "Perturb" current parameter (though eps = 0)
        h_params(param_idx) += eps;
        
        // Copy to device
        Kokkos::deep_copy(d_params, h_params);
    

        // Main kernel launch with perturbed parameter
        Kokkos::parallel_for("DO_scheme_perturbed", policy,
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
                
                // Initialize boundaries 
                bounds_d(instance).initialize(grid_i, team);
                auto bounds = bounds_d(instance);
                
                // Build matrices with perturbed parameters
                A0_solvers(instance).build_matrix(grid_i, d_params(3), d_params(2), team);
                A1_solvers(instance).build_matrix(grid_i, r_d, r_f, theta, delta_t, team);
                A2_solvers(instance).build_matrix(grid_i, r_d, d_params(0), d_params(1), 
                                            d_params(2), theta, delta_t, team);

                // Call device timestepping
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

        // Get results for this parameter
        auto h_U = Kokkos::create_mirror_view(workspace.U);
        Kokkos::deep_copy(h_U, workspace.U);

        // Extract and store prices
        for(int inst = 0; inst < nInstances; inst++) {
            auto h_Vec_s = Kokkos::create_mirror_view(hostGrids[inst].device_Vec_s);
            auto h_Vec_v = Kokkos::create_mirror_view(hostGrids[inst].device_Vec_v);
            Kokkos::deep_copy(h_Vec_s, hostGrids[inst].device_Vec_s);
            Kokkos::deep_copy(h_Vec_v, hostGrids[inst].device_Vec_v);

            int index_s = -1;
            int index_v = -1;
            
            for(int i = 0; i <= m1; i++) {
                if(std::abs(h_Vec_s(i) - S_0) < 1e-10) {
                    index_s = i;
                    break;
                }
            }
            
            for(int i = 0; i <= m2; i++) {
                if(std::abs(h_Vec_v(i) - V_0) < 1e-10) {
                    index_v = i;
                    break;
                }
            }

            double price = h_U(inst, index_s + index_v*(m1+1));
            h_perturbed_prices(param_idx, inst) = price;
        }

        // Reset workspace to initial condition
        Kokkos::deep_copy(workspace.U, U_0);

    }
    auto t_end = timer::now();
    std::cout << "This is a tentative time, there is device host copying done in between" << std::endl;
    std::cout << "Parameter " << " solve time: "
             << std::chrono::duration<double>(t_end - t_start).count()
            << " seconds\n";

    // Print final results
    
    std::cout << "\nPerturbed prices for each parameter:\n";
    for(int param_idx = 0; param_idx < 5; param_idx++) {
        std::cout << "\nParameter " <<  param_names[param_idx] << " perturbation:\n";
        for(int inst = 0; inst < nInstances; inst++) {
            std::cout << "Strike " << strikes[inst] << ": " 
                    << std::setprecision(16) << h_perturbed_prices(param_idx, inst) << "\n";
        }
    }
    
}





void test_device_class() {
  Kokkos::initialize();
  {
    //run_device_solver_example();

    //test_DEVICE_parallel_DO_scheme();  
    //test_parallel_DO_method();  
    //test_deviceCallable_Do_solver();

    test_jacobian_computation();
    //test_heston_jacobian();
    //test_sequential_J();
  }
  Kokkos::finalize();
 
}








