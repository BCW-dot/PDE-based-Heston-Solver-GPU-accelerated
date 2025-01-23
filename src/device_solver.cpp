#include <Kokkos_Core.hpp>
#include "device_solver.hpp"
#include <iostream>

/*

This is a basic example on hwo to use class instances in device code

*/

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







void test_device_class() {
  Kokkos::initialize();
  {
    // All your tests, calls, etc. happen here
    //run_device_solver_example();       // or runTest(), etc.
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