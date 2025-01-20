#pragma once
#include <Kokkos_Core.hpp>

struct Grid {
    std::vector<double> Vec_s;
    std::vector<double> Delta_s;
    std::vector<double> Vec_v;
    std::vector<double> Delta_v;

    // Device Views (added)
    Kokkos::View<double*> device_Vec_s;
    Kokkos::View<double*> device_Vec_v;
    Kokkos::View<double*> device_Delta_s;
    Kokkos::View<double*> device_Delta_v;

    static double Map_s(double xi, double K, double c);
    static double Map_v(double xi, double d);

    Grid(int m1, double S, double S_0, double K, double c, 
         int m2, double V, double V_0, double d);

    Grid() = default;  // <-- "do-nothing" constructor for device callable grids, this is on host side since we build them there
};

// Helper function to create test grid
Grid create_test_grid(int m1, int m2);

void test_grid();