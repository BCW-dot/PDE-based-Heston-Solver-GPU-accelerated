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

The following implements the the DO scheme in a parallel fashion

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
    parallel_DO_solve(
        nInstances, m1, m2, N, T, delta_t, theta,
        r_d, r_f, rho, sigma, kappa, eta,
        A0_solvers, A1_solvers, A2_solvers,
        bounds_d, deviceGrids,
        workspace);

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

This is a test for how we can compute the jacobian in parallel

*/
void test_parallel_DO_solve_params()
{
    using timer = std::chrono::high_resolution_clock;
    using Device = Kokkos::DefaultExecutionSpace;

    // ----------------------------
    // 1) Basic PDE / Heston setup
    // ----------------------------
    const double S_0 = 100.0;
    const double V_0 = 0.04;
    const double T   = 1.0;
    const int    N   = 20;
    const double theta   = 0.8;
    const double delta_t = T / N;

    // Grid dimensions
    const int m1 = 50;
    const int m2 = 25;

    // We will test multiple strikes *and* multiple parameter sets
    const int nStrikes = 5;           // e.g. 5 different options
    std::vector<double> strikes(nStrikes);
    for(int i = 0; i < nStrikes; ++i) {
        strikes[i] = 90 + i;  // e.g. 90, 91, 92, 93, 94
    }

    // Suppose we have 2 different sets of Heston parameters to test
    std::vector<HestonParams> hostParamVec;
    // Param set #0 (like your defaults)
    hostParamVec.push_back(HestonParams(0.025, 0.0, -0.9, 0.3, 1.5, 0.04));
    // Param set #1 (some variation)
    hostParamVec.push_back(HestonParams(0.01,  0.0, -0.7, 0.4, 2.0, 0.05));

    const int nParamSets = static_cast<int>(hostParamVec.size());  // = 2
    const int nInstances = nStrikes * nParamSets;  // e.g. 5 * 2 = 10 PDE solves

    std::cout << "Number of Instances: " << nInstances << " ("
              << nStrikes << " strikes x " << nParamSets << " parameter sets)\n";
    std::cout << "Stock S0 = " << S_0 << ", Dimensions: m1 = " 
              << m1 << ", m2 = " << m2 << ", time steps = " << N << std::endl;

    // ----------------------------
    // 2) Allocate PDE solver arrays for nInstances
    // ----------------------------
    Kokkos::View<Device_A0_heston<Device>*> A0_solvers("A0_solvers", nInstances);
    Kokkos::View<Device_A1_heston<Device>*> A1_solvers("A1_solvers", nInstances);
    Kokkos::View<Device_A2_shuffled_heston<Device>*> A2_solvers("A2_solvers", nInstances);

    auto h_A0 = Kokkos::create_mirror_view(A0_solvers);
    auto h_A1 = Kokkos::create_mirror_view(A1_solvers);
    auto h_A2 = Kokkos::create_mirror_view(A2_solvers);

    // Initialize them on host
    for(int i = 0; i < nInstances; i++) {
        h_A0(i) = Device_A0_heston<Device>(m1, m2);
        h_A1(i) = Device_A1_heston<Device>(m1, m2);
        h_A2(i) = Device_A2_shuffled_heston<Device>(m1, m2);
    }
    // Copy to device
    Kokkos::deep_copy(A0_solvers, h_A0);
    Kokkos::deep_copy(A1_solvers, h_A1);
    Kokkos::deep_copy(A2_solvers, h_A2);

    // ----------------------------
    // 3) Allocate boundary conditions array (size = nInstances)
    //    Each instance will have the relevant r_d, r_f from its parameter set
    // ----------------------------
    Kokkos::View<Device_BoundaryConditions<Device>*> bounds_d("bounds_d", nInstances);
    auto h_bounds = Kokkos::create_mirror_view(bounds_d);

    // ----------------------------
    // 4) Create a Kokkos::View of HestonParams for all nInstances
    //    So each PDE instance has its own param set
    // ----------------------------
    Kokkos::View<HestonParams*, Device> paramsView("paramsView", nInstances);
    auto h_paramsView = Kokkos::create_mirror_view(paramsView);

    // ----------------------------
    // 5) Build the device grids for each instance
    //    BUT the grid depends only on strike => we will build one grid per strike,
    //    then assign the same grid to all param sets of that strike.
    // ----------------------------
    std::vector<GridViews> hostGrids(nInstances);
    Kokkos::View<GridViews*> deviceGrids("deviceGrids", nInstances);
    auto h_deviceGrids = Kokkos::create_mirror_view(deviceGrids);

    // We'll use your helper: buildMultipleGridViews() for the *unique* strikes.
    // That returns 'nStrikes' grids. But we have nInstances because each strike
    // is replicated for each param set. So let's do it in two steps:
    std::vector<GridViews> uniqueStrikeGrids;
    buildMultipleGridViews(uniqueStrikeGrids, nStrikes, m1, m2);

    // Now fill out the PDE data instance by instance
    int counter = 0;
    for(int iStrike = 0; iStrike < nStrikes; ++iStrike) {
        double K = strikes[iStrike];

        // For each param set, create PDE instance with that param set + that strike
        for(int p = 0; p < nParamSets; ++p) {
            // 1) Fill the param set
            h_paramsView(counter) = hostParamVec[p];

            // 2) Fill boundary conditions object
            const double r_d   = hostParamVec[p].r_d;
            const double r_f   = hostParamVec[p].r_f;
            h_bounds(counter) = 
               Device_BoundaryConditions<Device>(m1, m2, r_d, r_f, N, delta_t);

            // 3) Assign the same grid used for strike iStrike
            //    We just copy that struct into hostGrids[counter]
            hostGrids[counter] = uniqueStrikeGrids[iStrike];

            // 4) Store that into deviceGrids mirror
            h_deviceGrids(counter) = hostGrids[counter];

            counter++;
        }
    }

    // Now we deep-copy the boundary conditions and param sets
    Kokkos::deep_copy(bounds_d, h_bounds);
    Kokkos::deep_copy(paramsView, h_paramsView);
    Kokkos::deep_copy(deviceGrids, h_deviceGrids);

    // ----------------------------
    // 6) Create a workspace for all PDE instances
    // ----------------------------
    const int total_size = (m1+1)*(m2+1);
    DO_Workspace<Device> workspace(nInstances, total_size);

    // ----------------------------
    // 7) Initialize the initial condition (payoff) for each PDE instance
    //    payoff = max(S - K, 0).
    //    But each PDE instance has a "strike" we can read from the host vector.
    //    We'll do it exactly as you do in your code.
    // ----------------------------
    {
        auto h_U_0 = Kokkos::create_mirror_view(workspace.U);

        // We'll loop over instances. Each instance belongs to a certain iStrike, paramSet.
        // But let's figure out the strike from iStrike = instance // nParamSets 
        counter = 0;
        for(int iStrike = 0; iStrike < nStrikes; ++iStrike) {
            double K = strikes[iStrike];

            for(int p = 0; p < nParamSets; ++p) {
                // fetch the grid
                auto grid = hostGrids[counter];
                // mirror to access S-values
                auto h_Vec_s = Kokkos::create_mirror_view(grid.device_Vec_s);
                Kokkos::deep_copy(h_Vec_s, grid.device_Vec_s);

                for(int j = 0; j <= m2; j++) {
                    for(int i = 0; i <= m1; i++) {
                        int idx = i + j*(m1+1);
                        h_U_0(counter, idx) = std::max(h_Vec_s(i) - K, 0.0);
                    }
                }
                counter++;
            }
        }
        // Copy initial condition to device
        Kokkos::deep_copy(workspace.U, h_U_0);
    }

    // ----------------------------
    // 8) Run the PDE solver in parallel for all param sets & strikes
    // ----------------------------
    auto t_start = timer::now();

    parallel_DO_solve_params(
        // PDE sizes
        nInstances, m1, m2, N, T, delta_t, theta,
        // array of parameters
        paramsView,
        // PDE data
        A0_solvers, A1_solvers, A2_solvers,
        bounds_d, deviceGrids,
        // workspace
        workspace);

    auto t_end = timer::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();
    std::cout << "parallel_DO_solve_params completed in " 
              << elapsed << " seconds.\n";

    // ----------------------------
    // 9) Retrieve final results and print a few
    // ----------------------------
    auto h_U = Kokkos::create_mirror_view(workspace.U);
    Kokkos::deep_copy(h_U, workspace.U);

    // We'll show results for the first min(5,nInstances) PDE instances
    int toPrint = std::min(5, nInstances);
    for(int inst = 0; inst < toPrint; ++inst) {
        // figure out which strike + paramSet
        int iStrike  = inst / nParamSets;  // integer division
        int p        = inst % nParamSets;
        double K     = strikes[iStrike];

        // find grid indices for S_0=100, V_0=0.04
        auto & grid = hostGrids[inst];
        auto h_Vec_s = Kokkos::create_mirror_view(grid.device_Vec_s);
        auto h_Vec_v = Kokkos::create_mirror_view(grid.device_Vec_v);
        Kokkos::deep_copy(h_Vec_s, grid.device_Vec_s);
        Kokkos::deep_copy(h_Vec_v, grid.device_Vec_v);

        int idx_s = -1, idx_v = -1;
        for(int i=0; i<=m1; i++){
            if(std::fabs(h_Vec_s(i) - S_0) < 1e-10) {
                idx_s = i;
                break;
            }
        }
        for(int j=0; j<=m2; j++){
            if(std::fabs(h_Vec_v(j) - V_0) < 1e-10) {
                idx_v = j;
                break;
            }
        }
        int finalIndex = idx_s + idx_v*(m1+1);
        double price   = h_U(inst, finalIndex);

        // Print
        std::cout << "Instance " << inst
                  << " (Strike=" << K
                  << ", ParamSet=" << p << ") -> Price = "
                  << std::setprecision(10) << price << "\n";
    }

    std::cout << "test_parallel_DO_solve_params finished.\n\n";
}


void test_device_class() {
  Kokkos::initialize();
  {
    //run_device_solver_example();

    //test_DEVICE_parallel_DO_scheme();  
    //test_parallel_DO_method();
    test_parallel_DO_solve_params();   
  }
  Kokkos::finalize();
 
}









//this is an abomenation of code. I thought classes are not the option since they are tough, 
//but looking at this i think they are the only option
/*
void solve_parallel_heston_do(const std::vector<HestonSolverParams>& params,std::vector<double>& results) {

    using timer = std::chrono::high_resolution_clock;

    // Get dimensions from first instance (all same)
    const int nInstances = params.size();
    const int m1 = params[0].m1;
    const int m2 = params[0].m2;
    const int total_size = (m1 + 1) * (m2 + 1);

    std::cout << "Solving " << nInstances << " PDEs in parallel\n";
    std::cout << "Dimensions: m1=" << m1 << ", m2=" << m2 << "\n";

    // Initialize grid views
    std::vector<GridViews> hostGrids;
    buildMultipleGridViews(hostGrids, nInstances, m1, m2);

    // Fill grid values for each instance
    for(int i = 0; i < nInstances; ++i) {
        // Create host mirrors
        auto h_Vec_s = Kokkos::create_mirror_view(hostGrids[i].device_Vec_s);
        auto h_Vec_v = Kokkos::create_mirror_view(hostGrids[i].device_Vec_v);
        auto h_Delta_s = Kokkos::create_mirror_view(hostGrids[i].device_Delta_s);
        auto h_Delta_v = Kokkos::create_mirror_view(hostGrids[i].device_Delta_v);
        
        // Create temporary Grid object
        Grid tempGrid = create_test_grid(m1, m2);
        
        // Copy values to host mirrors
        for(int j = 0; j <= m1; j++) {
            h_Vec_s(j) = tempGrid.Vec_s[j];
        }
        for(int j = 0; j <= m2; j++) {
            h_Vec_v(j) = tempGrid.Vec_v[j];
        }
        for(int j = 0; j < m1; j++) {
            h_Delta_s(j) = tempGrid.Delta_s[j];
        }
        for(int j = 0; j < m2; j++) {
            h_Delta_v(j) = tempGrid.Delta_v[j];
        }
        
        // Copy to device
        Kokkos::deep_copy(hostGrids[i].device_Vec_s, h_Vec_s);
        Kokkos::deep_copy(hostGrids[i].device_Vec_v, h_Vec_v);
        Kokkos::deep_copy(hostGrids[i].device_Delta_s, h_Delta_s);
        Kokkos::deep_copy(hostGrids[i].device_Delta_v, h_Delta_v);
    }

    // Create device view of GridViews
    Kokkos::View<GridViews*> deviceGrids("deviceGrids", nInstances);
    auto h_deviceGrids = Kokkos::create_mirror_view(deviceGrids);
    for(int i = 0; i < nInstances; ++i) {
        h_deviceGrids(i) = hostGrids[i];
    }
    Kokkos::deep_copy(deviceGrids, h_deviceGrids);

    // Create matrix storage for all instances
    // A0 matrices
    Kokkos::View<double***> A0_values("A0_values", nInstances, m2-1, (m1-1)*9);

    // A1 matrices
    Kokkos::View<double***> A1_main("A1_main", nInstances, m2+1, m1+1);
    Kokkos::View<double***> A1_lower("A1_lower", nInstances, m2+1, m1);
    Kokkos::View<double***> A1_upper("A1_upper", nInstances, m2+1, m1);
    
    Kokkos::View<double***> A1_impl_main("A1_impl_main", nInstances, m2+1, m1+1);
    Kokkos::View<double***> A1_impl_lower("A1_impl_lower", nInstances, m2+1, m1);
    Kokkos::View<double***> A1_impl_upper("A1_impl_upper", nInstances, m2+1, m1);

    Kokkos::View<double***> temp("temp",nInstances, (m2+1), (m1+1)); 

    //A2 matrices
    Kokkos::View<double***> main_diag("main_diag", nInstances, m1+1, m2+1);
    Kokkos::View<double***> lower_diag("lower_diag", nInstances, m1+1, m2);
    Kokkos::View<double***> lower2_diag("lower2_diag", nInstances, m1+1, m2-1);
    Kokkos::View<double***> upper_diag("upper_diag", nInstances, m1+1, m2);
    Kokkos::View<double***> upper2_diag("upper2_diag", nInstances, m1+1, m2-1);

    Kokkos::View<double***> impl_main_diag("impl_main_diag", nInstances, m1+1, m2+1);
    Kokkos::View<double***> impl_lower_diag("impl_lower_diag", nInstances, m1+1, m2);
    Kokkos::View<double***> impl_lower2_diag("impl_lower2_diag", nInstances, m1+1, m2-1);
    Kokkos::View<double***> impl_upper_diag("impl_upper_diag", nInstances, m1+1, m2);
    Kokkos::View<double***> impl_upper2_diag("impl_upper2_diag", nInstances, m1+1, m2-1);

    Kokkos::View<double***> c_prime("c_prime", nInstances, m1+1, m2+1);
    Kokkos::View<double***> c2_prime("c2_prime", nInstances, m1+1, m2+1);
    Kokkos::View<double***> d_prime("d_prime", nInstances, m1+1, m2+1);

    // Solution and temporary vectors
    Kokkos::View<double**> U("U", nInstances, total_size);
    Kokkos::View<double**> Y_0("Y_0", nInstances, total_size);
    Kokkos::View<double**> Y_1("Y_1", nInstances, total_size);

    Kokkos::View<double**> U_shuffled("U_shuffled", total_size);
    Kokkos::View<double**> Y_1_shuffled("Y_1_shuffled", total_size);
    Kokkos::View<double**> A2_result_shuffled("A2_result_shuffled", total_size);
    Kokkos::View<double**> A2_result_unshuf("A2_result_unshuf", total_size);

    // Initialize U with payoff
    auto h_U = Kokkos::create_mirror_view(U);
    for(int inst = 0; inst < nInstances; ++inst) {
        for(int j = 0; j <= m2; j++) {
            for(int i = 0; i <= m1; i++) {
                double s = tempGrid.Vec_s[i];
                h_U(inst, i + j*(m1+1)) = std::max(s - params[inst].K, 0.0);
            }
        }
    }
    Kokkos::deep_copy(U, h_U);

    // Set up team policy
    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;
    team_policy policy(nInstances, Kokkos::AUTO);

    // Main solver kernel
    auto t_start = timer::now();

    Kokkos::parallel_for("DO_scheme_solve", policy,
        KOKKOS_LAMBDA(const member_type& team) {
            const int inst = team.league_rank();
            
            // Get parameters for this instance
            const double theta = params[inst].theta;
            const double delta_t = params[inst].delta_t;
            const int N = params[inst].N;

            // Get grid for this instance
            GridViews grid_i = deviceGrids(inst);


            // Get subviews for solution vectors
            auto U_i = Kokkos::subview(U, inst, Kokkos::ALL);
            auto Y_0_i = Kokkos::subview(Y_0, inst, Kokkos::ALL);
            auto Y_1_i = Kokkos::subview(Y_1, inst, Kokkos::ALL);

            // Create a shuffled view for A2 operations
            auto U_i_shuffled = Kokkos::subview(U_shuffled, inst, Kokkos::ALL);
            auto Y_1_i_shuffled = Kokkos::subview(Y_1_shuffled, inst, Kokkos::ALL);
            auto A2_i_result_shuffled = Kokkos::subview(A2_result_shuffled, inst, Kokkos::ALL);
            auto A2_i_result_unshuf = Kokkos::subview(A2_result_unshuf, inst, Kokkos::ALL);


            // Get subviews for matrices
            //A0
            auto A0_values_i = Kokkos::subview(A0_values, inst, Kokkos::ALL, Kokkos::ALL);

            //A1
            auto A1_main_i = Kokkos::subview(A1_main, inst, Kokkos::ALL, Kokkos::ALL);
            auto A1_lower_i = Kokkos::subview(A1_lower, inst, Kokkos::ALL, Kokkos::ALL);
            auto A1_upper_i = Kokkos::subview(A1_upper, inst, Kokkos::ALL, Kokkos::ALL);

            auto A1_impl_main_i = Kokkos::subview(impl_main_diag, instance, Kokkos::ALL, Kokkos::ALL);
            auto A1_impl_lower_i = Kokkos::subview(impl_lower_diag, instance, Kokkos::ALL, Kokkos::ALL);
            auto A1_impl_upper_i = Kokkos::subview(impl_upper_diag, instance, Kokkos::ALL, Kokkos::ALL);

            //A2 shuffled
            auto A2_main_i = Kokkos::subview(A2_main, inst, Kokkos::ALL, Kokkos::ALL);
            auto A2_lower_i = Kokkos::subview(A2_lower, inst, Kokkos::ALL, Kokkos::ALL);
            auto A2_lower2_i = Kokkos::subview(A2_lower2, inst, Kokkos::ALL, Kokkos::ALL);
            auto A2_upper_i = Kokkos::subview(A2_upper, inst, Kokkos::ALL, Kokkos::ALL);
            auto A2_upper2_i = Kokkos::subview(A2_upper2, inst, Kokkos::ALL, Kokkos::ALL);

            auto A2_impl_main_i = Kokkos::subview(A2_impl_main, inst, Kokkos::ALL, Kokkos::ALL);
            auto A2_impl_lower_i = Kokkos::subview(A2_impl_lower, inst, Kokkos::ALL, Kokkos::ALL);
            auto A2_impl_lower2_i = Kokkos::subview(A2_impl_lower2, inst, Kokkos::ALL, Kokkos::ALL);
            auto A2_impl_upper_i = Kokkos::subview(A2_impl_upper, inst, Kokkos::ALL, Kokkos::ALL);
            auto A2_impl_upper2_i = Kokkos::subview(A2_impl_upper2, inst, Kokkos::ALL, Kokkos::ALL);

            // Build A0, A1, A2 matrices
            build_a0_values(A0_values_i,grid_i,params[inst].rho, params[inst].sigma,team);

            build_a1_diagonals(A1_main_i, A1_lower_i, A1_upper_i,A1_impl_main_i, A1_impl_lower_i, A1_impl_upper_i,grid_i,
                theta, delta_t, params[inst].r_d, params[inst].r_f,team);

            build_a2_diagonals_shuffled(A2_main_i, A2_lower_i, A2_lower2_i, A2_upper_i, A2_upper2_i,
                A2_impl_main_i, A2_impl_lower_i, A2_impl_lower2_i, A2_impl_upper_i, A2_impl_upper2_i,
                grid_i,theta, delta_t, params[inst].r_d, params[inst].kappa, params[inst].eta, params[inst].sigma,team);

            //A1 temp storage
            auto temp_i = Kokkos::subview(temp, instance, Kokkos::ALL, Kokkos::ALL);

            // Get A2 temp storage
            auto c_prime_i = Kokkos::subview(A2_c_prime, inst, Kokkos::ALL, Kokkos::ALL);
            auto c2_prime_i = Kokkos::subview(A2_c2_prime, inst, Kokkos::ALL, Kokkos::ALL);
            auto d_prime_i = Kokkos::subview(A2_d_prime, inst, Kokkos::ALL, Kokkos::ALL);

            // Time stepping loop
            for(int n = 1; n <= N; n++) {
                // Step 1: Y_0 = U + dt*(A0 + A1 + A2)U
                // A0 multiply
                a0_device_multiply_parallel_s_and_v(A0_values_i,U_i, Y_0_i,team);
                
                // A1 multiply
                a1_device_multiply_parallel_v(A1_main_i, A1_lower_i, A1_upper_i,U_i, Y_0_i,team);
                
                // A2 multiply (with shuffling)
                device_shuffle_vector(U_i, U_shuffled, m1, m2, team);
                a2_device_multiply_shuffled(A2_main_i, A2_lower_i, A2_lower2_i, A2_upper_i, A2_upper2_i,
                    U_shuffled, A2_result_shuffled,team);
                device_unshuffle_vector(A2_result_shuffled, A2_result_unshuf, m1, m2, team);
                
                // Combine results for Y_0
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, total_size),
                    [&](const int i) {
                        double exp_factor = std::exp(params[inst].r_f * delta_t * (n-1));
                        Y_0_i(i) = U_i(i) + delta_t * (Y_0_i(i) + A2_result_unshuf(i) + bounds.get_b(i) * exp_factor);
                });

                // Step 2: (I - theta*dt*A1)Y1 = Y0
                a1_device_solve_implicit_parallel_v(A1_impl_main_i, A1_impl_lower_i, A1_impl_upper_i,Y_1_i, temp_i, Y_0_i,team);

                // Step 3: (I - theta*dt*A2)U = Y1 (with shuffling)
                device_shuffle_vector(Y_1_i, Y_1_shuffled, m1, m2, team);
                a2_device_solve_implicit_shuffled(A2_impl_main_i, A2_impl_lower_i, A2_impl_lower2_i, 
                    A2_impl_upper_i, A2_impl_upper2_i,U_shuffled, c_prime_i, c2_prime_i, d_prime_i, Y_1_shuffled,team);
                device_unshuffle_vector(U_shuffled, U_i, m1, m2, team);

                team.team_barrier();
            }
    });
    Kokkos::fence();

    auto t_end = timer::now();
    std::cout << "Solve time: "
              << std::chrono::duration<double>(t_end - t_start).count()
              << " seconds" << std::endl;

    // Copy results back
    auto h_result = Kokkos::create_mirror_view(U);
    Kokkos::deep_copy(h_result, U);
    
    // Extract option prices at (S_0, V_0) for each instance
    for(int inst = 0; i < nInstances; ++i) {
        // Find indices for S_0, V_0
        double S_0 = params[inst].S_0;
        double V_0 = params[inst].V_0;
        
        int index_s = binary_search(grid.Vec_s, S_0);
        int index_v = binary_search(grid.Vec_v, V_0);
        
        results[inst] = h_result(inst, index_s + index_v*(m1+1));
    }
}

*/