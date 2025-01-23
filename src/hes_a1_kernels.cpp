// In hes_a1_kernels.cpp
#include "hes_a1_kernels.hpp"
#include <iomanip>
#include <iostream>
#include <numeric>   // For std::accumulate
#include "grid_pod.hpp"


/*
  3) A test function that:
     - builds an array of GridViews on host,
     - copies that array into deviceGridViews,
     - launches a kernel that prints device_Vec_s(5) for each instance.
*/
void testMultipleGridViews()
{
  // Example: 5 PDE instances, each with m1=10, m2=8, etc.
  int nInstances = 5;
  int m1 = 10;
  int m2 = 8;

  // 3a) Build the array of GridViews on host
  std::vector<GridViews> hostGrids;
  buildMultipleGridViews(hostGrids, nInstances, m1, m2);

  // 3b) Create a device array of GridViews
  Kokkos::View<GridViews*> deviceGridViews("deviceGridViews", nInstances);
  // Mirror
  auto h_deviceGridViews = Kokkos::create_mirror_view(deviceGridViews);

  // 3c) Copy the struct handles to device
  for(int i=0; i<nInstances; i++){
    h_deviceGridViews(i) = hostGrids[i];
  }
  Kokkos::deep_copy(deviceGridViews, h_deviceGridViews);

  // 3d) Kernel that prints device_Vec_s(5) for each PDE instance
  Kokkos::parallel_for("print_grid_s", Kokkos::RangePolicy<>(0,nInstances),
    KOKKOS_LAMBDA(const int idx)
  {
    GridViews gv = deviceGridViews(idx);

    double val = gv.device_Vec_s(5);
    int length = gv.device_Vec_s.extent(0);

    printf("Instance %d => device_Vec_s(5)=%.2f, extent=%d\n",
           idx, val, length);
  });
  Kokkos::fence();
}

void test_myGrids()
{
  // Initialize vectors with grid views and diagonals
    int nInstances = 5;
    int m1 = 10;
    int m2 = 8;
    double S_0 = 100;
    double V_0 = 0.04;

    std::vector<GridViews> hostGrids;
    buildMultipleGridViews(hostGrids, nInstances, m1, m2);  // This creates empty views

    // Now we need to actually fill them with proper grid values
    for(int i = 0; i < nInstances; ++i) {
        double K = 90.0 + 10.0 * i;
        
        // Create host mirrors of the views
        auto h_Vec_s = Kokkos::create_mirror_view(hostGrids[i].device_Vec_s);
        auto h_Vec_v = Kokkos::create_mirror_view(hostGrids[i].device_Vec_v);
        auto h_Delta_s = Kokkos::create_mirror_view(hostGrids[i].device_Delta_s);
        auto h_Delta_v = Kokkos::create_mirror_view(hostGrids[i].device_Delta_v);
        
        // Create temporary Grid object to get the values
        Grid tempGrid(m1, 8*K, S_0, K, K/5, m2, 5.0, V_0, 5.0/500);
        
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

    // Create device view of GridViews array
    Kokkos::View<GridViews*> deviceGrids("deviceGrids", nInstances);
    auto h_deviceGrids = Kokkos::create_mirror_view(deviceGrids);

    // Copy GridViews to device
    for(int i = 0; i < nInstances; ++i) {
        h_deviceGrids(i) = hostGrids[i];
    }
    Kokkos::deep_copy(deviceGrids, h_deviceGrids);

  // 3d) Kernel that prints device_Vec_s(5) for each PDE instance
  Kokkos::parallel_for("print_grid_s", Kokkos::RangePolicy<>(0,nInstances),
    KOKKOS_LAMBDA(const int idx)
  {
    GridViews gv = deviceGrids(idx);

    double val = gv.device_Vec_s(5);
    int length = gv.device_Vec_s.extent(0);

    printf("Instance %d => device_Vec_s(5)=%.2f, extent=%d\n",
           idx, val, length);
  });
  Kokkos::fence();
}


/*

Class test

*/
//this works
void debugging_test_device_adi_multiple_instances() {
   using timer = std::chrono::high_resolution_clock;
   using Device = Kokkos::DefaultExecutionSpace;

    double K = 100.0;
    double S_0 = K;
    double V_0 = 0.04;

   // Test parameters 
   const int m1 = 150;
   const int m2 = 75;
   std::cout << "A1 Dimension StockxVariance: " << m1+1 << "x" << m2+1 << std::endl;

   const double theta = 0.8;
   const double delta_t = 1.0/40.0;
   const double r_d = 0.025;
   const double r_f = 0.0;

   const int nInstances = 100;
   std::cout << "Instances: " << nInstances << std::endl;

   // Create solvers array on device
   Kokkos::View<Device_A1_heston<Device>*> solvers_d("solvers_d", nInstances);
   auto solvers_h = Kokkos::create_mirror_view(solvers_d);
   for(int i = 0; i < nInstances; ++i) {
       solvers_h(i) = Device_A1_heston<Device>(m1, m2);
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

       //Grid tempGrid(m1, 8*K, S_0, K, K/5, m2, 5.0, V_0, 5.0/500);
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
   Kokkos::View<double**> b("b", nInstances, total_size);
   Kokkos::View<double**> result("result", nInstances, total_size);

   auto h_x = Kokkos::create_mirror_view(x);
   auto h_b = Kokkos::create_mirror_view(b);
   for(int inst = 0; inst < nInstances; ++inst) {
       for(int idx = 0; idx < total_size; ++idx) {
           h_x(inst, idx) = 1.0;//(double)std::rand() / RAND_MAX;
           h_b(inst, idx) = 2.0;//(double)std::rand() / RAND_MAX;
       }
   }
   Kokkos::deep_copy(x, h_x);
   Kokkos::deep_copy(b, h_b);

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
               
               Device_A1_heston<Device>& solver = solvers_d(instance);
               auto x_i = Kokkos::subview(x, instance, Kokkos::ALL);
               auto b_i = Kokkos::subview(b, instance, Kokkos::ALL);
               auto result_i = Kokkos::subview(result, instance, Kokkos::ALL);
               GridViews grid_i = deviceGrids(instance);
               
               solver.build_matrix(grid_i, r_d, r_f, theta, delta_t, team);
               solver.multiply_parallel_v(x_i, result_i, team);
               solver.solve_implicit_parallel_v(x_i, b_i, team);
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
           solvers_d(instance).multiply_parallel_v(
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




void test_a1_kernel(){
    Kokkos::initialize();
    {
        try{
            //testMultipleGridViews();
            //test_myGrids();

            debugging_test_device_adi_multiple_instances();
        }
        catch (std::exception& e) {
            std::cout << "Error: " << e.what() << std::endl;
        }
    } // All test objects destroyed here
    Kokkos::finalize();
}