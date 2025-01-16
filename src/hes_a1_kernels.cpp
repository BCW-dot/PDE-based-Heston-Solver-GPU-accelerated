// In hes_a1_kernels.cpp
#include "hes_a1_kernels.hpp"
#include <iomanip>
#include <iostream>
#include "coeff.hpp"

/*
KOKKOS_FUNCTION
void build_a1_diagonals(
    const Kokkos::View<double**>& main_diag,
    const Kokkos::View<double**>& lower_diag,
    const Kokkos::View<double**>& upper_diag,
    const Kokkos::View<double**>& impl_main_diag,
    const Kokkos::View<double**>& impl_lower_diag,
    const Kokkos::View<double**>& impl_upper_diag,
    const Grid& grid,
    const double theta,
    const double dt,
    const double r_d,
    const double r_f)
{
    // Get dimensions
    const int m1 = main_diag.extent(1) - 1;
    const int m2 = main_diag.extent(0) - 1;

    // Parallel over variance levels
    Kokkos::parallel_for(m2 + 1, KOKKOS_LAMBDA(const int j) {
        // First point (i=0) is boundary
        main_diag(j,0) = 0.0;
        impl_main_diag(j,0) = 1.0;
        if(j < m2) {
            upper_diag(j,0) = 0.0;
            impl_upper_diag(j,0) = 0.0;
        }

        // Interior points
        for(int i = 1; i < m1; i++) {
            // Compute PDE coefficients
            const double s = grid.device_Vec_s[i];
            const double v = grid.device_Vec_v[j];
            
            // a = 0.5*s^2*v (diffusion)
            // b = (r_d - r_f)*s (drift)
            const double a = 0.5 * s * s * v;
            const double b = (r_d - r_f) * s;

            // Build explicit diagonals using central differences
            // PDE coefficients

            // Build tridiagonal system for this level
            // Lower diagonal
            lower_diag(j,i-1) = a * device_delta_s(i-1, -1, grid.device_Delta_s) + 
                                b * device_beta_s(i-1, -1, grid.device_Delta_s);
            
            // Main diagonal
            main_diag(j,i) = a * device_delta_s(i-1, 0, grid.device_Delta_s) + 
                                b * device_beta_s(i-1, 0, grid.device_Delta_s) - 0.5 * r_d;
            
            // Upper diagonal
            upper_diag(j,i) = a * device_delta_s(i-1, 1, grid.device_Delta_s) + 
                                b * device_beta_s(i-1, 1, grid.device_Delta_s);

            // Build implicit diagonals: (I - theta*dt*A)
            impl_lower_diag(j,i-1) = -theta * dt * lower_diag(j,i-1);
            impl_main_diag(j,i) = 1.0 - theta * dt * main_diag(j,i);
            impl_upper_diag(j,i) = -theta * dt * upper_diag(j,i);
        }

        // Last point (i=m1)
        main_diag(j,m1) = -0.5 * r_d;
        impl_main_diag(j,m1) = 1.0 - theta * dt * main_diag(j,m1);
        lower_diag(j,m1-1) = 0.0;
        impl_lower_diag(j,m1-1) = 0.0;
    });
}
*/

KOKKOS_FUNCTION
void build_a1_diagonals(
    const Kokkos::View<double**>& main_diag,
    const Kokkos::View<double**>& lower_diag,
    const Kokkos::View<double**>& upper_diag,
    const Kokkos::View<double**>& impl_main_diag,
    const Kokkos::View<double**>& impl_lower_diag,
    const Kokkos::View<double**>& impl_upper_diag,
    const Grid& grid,
    const double theta,
    const double dt,
    const double r_d,
    const double r_f,
    const Kokkos::TeamPolicy<>::member_type& team)  // Add team parameter
{
    // Get dimensions
    const int m1 = main_diag.extent(1) - 1;
    const int m2 = main_diag.extent(0) - 1;

    // Parallel over variance levels using team
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, m2 + 1), 
        [&](const int j) {
            // First point (i=0) is boundary
            main_diag(j,0) = 0.0;
            impl_main_diag(j,0) = 1.0;
            if(j < m2) {
                upper_diag(j,0) = 0.0;
                impl_upper_diag(j,0) = 0.0;
            }

            // Interior points
            for(int i = 1; i < m1; i++) {
                // Compute PDE coefficients
                const double s = grid.device_Vec_s[i];
                const double v = grid.device_Vec_v[j];
                
                // a = 0.5*s^2*v (diffusion)
                // b = (r_d - r_f)*s (drift)
                const double a = 0.5 * s * s * v;
                const double b = (r_d - r_f) * s;

                // Build explicit diagonals using central differences
                // PDE coefficients

                // Build tridiagonal system for this level
                // Lower diagonal
                lower_diag(j,i-1) = a * device_delta_s(i-1, -1, grid.device_Delta_s) + 
                                    b * device_beta_s(i-1, -1, grid.device_Delta_s);
                
                // Main diagonal
                main_diag(j,i) = a * device_delta_s(i-1, 0, grid.device_Delta_s) + 
                                    b * device_beta_s(i-1, 0, grid.device_Delta_s) - 0.5 * r_d;
                
                // Upper diagonal
                upper_diag(j,i) = a * device_delta_s(i-1, 1, grid.device_Delta_s) + 
                                    b * device_beta_s(i-1, 1, grid.device_Delta_s);

                // Build implicit diagonals: (I - theta*dt*A)
                impl_lower_diag(j,i-1) = -theta * dt * lower_diag(j,i-1);
                impl_main_diag(j,i) = 1.0 - theta * dt * main_diag(j,i);
                impl_upper_diag(j,i) = -theta * dt * upper_diag(j,i);
            }

            // Last point (i=m1)
            main_diag(j,m1) = -0.5 * r_d;
            impl_main_diag(j,m1) = 1.0 - theta * dt * main_diag(j,m1);
            lower_diag(j,m1-1) = 0.0;
            impl_lower_diag(j,m1-1) = 0.0;
        });
    team.team_barrier();
}


void test_a1_build() {
    // Test dimensions
    const int m1 = 5;  // Stock price points
    const int m2 = 2;  // Variance points
    
    // Create grid
    Grid grid = create_test_grid(m1, m2);
    
    // Create Views for diagonals
    Kokkos::View<double**> main_diag("main_diag", m2+1, m1+1);
    Kokkos::View<double**> lower_diag("lower_diag", m2+1, m1);
    Kokkos::View<double**> upper_diag("upper_diag", m2+1, m1);
    
    Kokkos::View<double**> impl_main_diag("impl_main_diag", m2+1, m1+1);
    Kokkos::View<double**> impl_lower_diag("impl_lower_diag", m2+1, m1);
    Kokkos::View<double**> impl_upper_diag("impl_upper_diag", m2+1, m1);
    
    // Test parameters
    const double theta = 0.5;
    const double dt = 0.05;
    const double r_d = 0.025;
    const double r_f = 0.0;

    // Build diagonals
    // And then for testing:
    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;

    team_policy policy(1, Kokkos::AUTO);  // One team for testing
    Kokkos::parallel_for("test_device_call", policy,
        KOKKOS_LAMBDA(const member_type& team) {
            build_a1_diagonals(
                main_diag, lower_diag, upper_diag,
                impl_main_diag, impl_lower_diag, impl_upper_diag,
                grid, theta, dt, r_d, r_f,
                team);
        });

    // Create host mirrors
    auto h_main = Kokkos::create_mirror_view(main_diag);
    auto h_lower = Kokkos::create_mirror_view(lower_diag);
    auto h_upper = Kokkos::create_mirror_view(upper_diag);
    auto h_impl_main = Kokkos::create_mirror_view(impl_main_diag);
    auto h_impl_lower = Kokkos::create_mirror_view(impl_lower_diag);
    auto h_impl_upper = Kokkos::create_mirror_view(impl_upper_diag);

    // Copy to host
    Kokkos::deep_copy(h_main, main_diag);
    Kokkos::deep_copy(h_lower, lower_diag);
    Kokkos::deep_copy(h_upper, upper_diag);
    Kokkos::deep_copy(h_impl_main, impl_main_diag);
    Kokkos::deep_copy(h_impl_lower, impl_lower_diag);
    Kokkos::deep_copy(h_impl_upper, impl_upper_diag);

    // Print results
    std::cout << std::fixed << std::setprecision(6);
    
    for(int j = 0; j <= m2; j++) {
        std::cout << "\nVariance level j=" << j << ":\n";
        std::cout << "Explicit diagonals:\n";
        
        std::cout << "Lower:  ";
        for(int i = 0; i < m1; i++) {
            std::cout << h_lower(j,i) << " ";
        }
        std::cout << "\n";
        
        std::cout << "Main:   ";
        for(int i = 0; i <= m1; i++) {
            std::cout << h_main(j,i) << " ";
        }
        std::cout << "\n";
        
        std::cout << "Upper:  ";
        for(int i = 0; i < m1; i++) {
            std::cout << h_upper(j,i) << " ";
        }
        std::cout << "\n\n";
        
        std::cout << "Implicit diagonals:\n";
        std::cout << "Lower:  ";
        for(int i = 0; i < m1; i++) {
            std::cout << h_impl_lower(j,i) << " ";
        }
        std::cout << "\n";
        
        std::cout << "Main:   ";
        for(int i = 0; i <= m1; i++) {
            std::cout << h_impl_main(j,i) << " ";
        }
        std::cout << "\n";
        
        std::cout << "Upper:  ";
        for(int i = 0; i < m1; i++) {
            std::cout << h_impl_upper(j,i) << " ";
        }
        std::cout << "\n";
        std::cout << "----------------------------------------\n";
    }
}

void test_a1_kernel(){
Kokkos::initialize();
    {
        try{
            test_a1_build();
        }
        catch (std::exception& e) {
            std::cout << "Error: " << e.what() << std::endl;
        }
    } // All test objects destroyed here
    Kokkos::finalize();
}