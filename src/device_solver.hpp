#pragma once
#include <Kokkos_Core.hpp>
#include "hes_a0_kernels.hpp"
#include "hes_a1_kernels.hpp"
#include "hes_a2_shuffled_kernels.hpp"


template<class DeviceType>
struct Device_DO_solver {
    typedef DeviceType execution_space;
    typedef typename DeviceType::memory_space memory_space;
    
    Device_A0_heston<DeviceType> a0_solver;
    Device_A1_heston<DeviceType> a1_solver;
    Device_A2_shuffled_heston<DeviceType> a2_solver;
    
    // Vectors for each solver instance
    Kokkos::View<double*, DeviceType> Y_0;
    Kokkos::View<double*, DeviceType> Y_1;
    Kokkos::View<double*, DeviceType> temp;
    Kokkos::View<double*, DeviceType> U;
    Kokkos::View<double*, DeviceType> U_init;
    
    int m1, m2, total_size, N;
    double theta, dt, r_d, r_f;
    double rho, sigma, kappa, eta;
    
    KOKKOS_FUNCTION Device_DO_solver() = default;
    
    // Constructor now also takes number of timesteps N
    Device_DO_solver(int m1_in, int m2_in, int N_in, const double params[8]) : 
        m1(m1_in), m2(m2_in), N(N_in),
        total_size((m1+1)*(m2+1)),
        a0_solver(m1, m2),
        a1_solver(m1, m2),
        a2_solver(m1, m2) {
        
        theta = params[0]; dt = params[1];
        r_d = params[2]; r_f = params[3]; 
        rho = params[4]; sigma = params[5];
        kappa = params[6]; eta = params[7];

        Y_0 = Kokkos::View<double*>("Y_0", total_size);
        Y_1 = Kokkos::View<double*>("Y_1", total_size);
        temp = Kokkos::View<double*>("temp", total_size); 
        U = Kokkos::View<double*>("U", total_size);
        U_init = Kokkos::View<double*>("U_init", total_size);
    }
    
    KOKKOS_FUNCTION
    void solve(const GridViews& grid, const Kokkos::View<double*>& bounds,
               const Kokkos::TeamPolicy<>::member_type& team) {
        
        // Copy initial condition
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, total_size),
            [&](const int i) { U(i) = U_init(i); });
        team.team_barrier();

        // Build matrices once
        a0_solver.build_matrix(grid, rho, sigma, team);
        a1_solver.build_matrix(grid, r_d, r_f, theta, dt, team);
        a2_solver.build_matrix(grid, r_d, kappa, eta, sigma, theta, dt, team);
        
        // Run all timesteps for this solver
        for(int n = 1; n <= N; n++) {
            // Y_0 computation
            a0_solver.multiply_parallel_s_and_v(U, temp, team);
            a1_solver.multiply_parallel_v(U, Y_0, team);  
            a2_solver.multiply_parallel_s(U, Y_1, team);
            
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, total_size),
                [&](const int i) {
                    Y_0(i) = U(i) + dt * (temp(i) + Y_0(i) + Y_1(i) + bounds(i));
                });
            team.team_barrier();
            
            // Implicit solves
            a1_solver.solve_implicit_parallel_v(Y_1, Y_0, team);
            a2_solver.solve_implicit_parallel_s(U, Y_1, team);
        }
    }
};


void test_device_class();