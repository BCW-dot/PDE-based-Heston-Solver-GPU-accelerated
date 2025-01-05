#include <iostream>
#include <iomanip>
#include "BoundaryConditions.hpp"

#include "BoundaryConditions.hpp"
#include "grid.hpp"

void test_boundary_conditions(){
    Kokkos::initialize();
    {
        // Create test grid using helper function
        int m1 = 5;  // Small grid for testing
        int m2 = 3;
        Grid grid = create_test_grid(m1, m2);

        // Create Kokkos View for Vec_s and copy data
        Kokkos::View<double*> d_Vec_s("Vec_s", m1 + 1);
        auto h_Vec_s = Kokkos::create_mirror_view(d_Vec_s);
        
        for(int i = 0; i <= m1; i++) {
            h_Vec_s(i) = grid.Vec_s[i];
        }
        Kokkos::deep_copy(d_Vec_s, h_Vec_s);

        // Create boundary conditions with test parameters
        double r_d = 0.025;  // Domestic interest rate
        double r_f = 0.0;    // Foreign interest rate
        int N = 20;          // Time steps
        double delta_t = 1.0 / N;

        BoundaryConditions bc(m1, m2, r_d, r_f, N, delta_t);
        bc.initialize(d_Vec_s);

        // Get results back to host for printing
        auto b0 = bc.get_b0();
        auto b1 = bc.get_b1();
        auto b2 = bc.get_b2();
        auto b = bc.get_b();

        auto h_b0 = Kokkos::create_mirror_view(b0);
        auto h_b1 = Kokkos::create_mirror_view(b1);
        auto h_b2 = Kokkos::create_mirror_view(b2);
        auto h_b = Kokkos::create_mirror_view(b);

        Kokkos::deep_copy(h_b0, b0);
        Kokkos::deep_copy(h_b1, b1);
        Kokkos::deep_copy(h_b2, b2);
        Kokkos::deep_copy(h_b, b);

        // Print grid information
        std::cout << "Grid Parameters:\n";
        std::cout << "Stock price grid (Vec_s):\n";
        for(int i = 0; i <= m1; i++) {
            std::cout << "[" << i << "] = " << grid.Vec_s[i] << "\n";
        }
        std::cout << "\nVariance grid (Vec_v):\n";
        for(int i = 0; i <= m2; i++) {
            std::cout << "[" << i << "] = " << grid.Vec_v[i] << "\n";
        }
        
        // Print boundary values
        std::cout << "\nBoundary Values:\n";
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Grid dimensions: m1 = " << m1 << ", m2 = " << m2 << "\n";
        std::cout << "Total size: " << bc.get_size() << "\n\n";

        int print_size = std::min(24, bc.get_size());
        
        std::cout << "First " << print_size << " values of boundary arrays:\n";
        std::cout << "Index\tb0\t\tb1\t\tb2\t\tb\n";
        std::cout << std::string(60, '-') << "\n";
        
        for(int i = 0; i < print_size; i++) {
            std::cout << i << "\t"
                     << h_b0(i) << "\t"
                     << h_b1(i) << "\t" 
                     << h_b2(i) << "\t"
                     << h_b(i) << "\n";
        }

        // Print some specific boundary values for verification
        std::cout << "\nKey boundary values:\n";
        // b1 boundary (S direction) at j=0
        std::cout << "b1 at S_max, j=0: " << h_b1(m1) << "\n";
        // b2 boundary (V direction) for first few points
        std::cout << "b2 at V_max, first few i values:\n";
        for(int i = 1; i <= std::min(3, m1); i++) {
            std::cout << "i=" << i << ": " << h_b2[bc.get_size() - m1 - 1 + i] << "\n";
        }
    }
    Kokkos::finalize();
}
