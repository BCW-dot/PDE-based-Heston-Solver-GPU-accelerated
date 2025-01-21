// In hes_a1_kernels.cpp
#include "hes_a1_kernels.hpp"
#include <iomanip>
#include <iostream>
#include "coeff.hpp"

void buildMultipleGridViews(
    std::vector<GridViews> &hostGrids,
    int nInstances, int m1, int m2);

//runs on the cpu for testing
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


//storage
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
    const double r_f,
    const Kokkos::TeamPolicy<>::member_type& team)

    KOKKOS_FUNCTION
void device_multiply_parallel_s_and_v(
    const Kokkos::View<const double**>& main_diag,
    const Kokkos::View<const double**>& lower_diag,
    const Kokkos::View<const double**>& upper_diag,
    const Kokkos::View<double*>& x,
    const Kokkos::View<double*>& result,  // Changed to const
    const Kokkos::TeamPolicy<>::member_type& team)

    KOKKOS_FUNCTION
void device_solve_implicit_parallel_v(
    const Kokkos::View<const double**>& impl_main,
    const Kokkos::View<const double**>& impl_lower,
    const Kokkos::View<const double**>& impl_upper,
    const Kokkos::View<double*>& x,         // Changed to const -> means that the memory address will not be changed, but the values can be modified
    const Kokkos::View<double**>& temp,     // Changed to const
    const Kokkos::View<double*>& b,
    const Kokkos::TeamPolicy<>::member_type& team)

*/

//test build for parallel device

template <class MDView, class LDView, class UDView,
          class IMDView, class ILDView, class IUDView,
          class GridType>  // New template parameter for Grid type
KOKKOS_FUNCTION
void build_a1_diagonals(
    const MDView& main_diag,
    const LDView& lower_diag,
    const UDView& upper_diag,
    const IMDView& impl_main_diag,
    const ILDView& impl_lower_diag,
    const IUDView& impl_upper_diag,
    const GridType& grid,    // Now accepts either Grid or GridViews
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



//name is missleading, we are just parallising in v
template<
    class View2D_const_main,  // e.g. Kokkos::View<const double**, LayoutStride, ...>
    class View2D_const_lower,
    class View2D_const_upper,
    class View1D_x,           // e.g. Kokkos::View<double*, ...>
    class View1D_result
>
KOKKOS_FUNCTION
void device_multiply_parallel_s_and_v(
    const View2D_const_main&  main_diag,
    const View2D_const_lower& lower_diag,
    const View2D_const_upper& upper_diag,
    const View1D_x&           x,
    const View1D_result&      result,
    const Kokkos::TeamPolicy<>::member_type& team)
{
    // Infer matrix dimensions: 
    //    m2 = number of variance lines - 1
    //    m1 = number of stock points    - 1
    // For example, if main_diag is shape [m2+1, m1+1],
    // then main_diag.extent(0) = m2+1, main_diag.extent(1) = m1+1.

    //maybe i can optimize this, by passing it into the function itself
    const int m2 = main_diag.extent(0) - 1;
    const int m1 = main_diag.extent(1) - 1;

    // TeamThreadRange: parallelize over j in [0..m2].
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, m2 + 1),
        [&](const int j)
        {
            // The row offset in the 1D vectors x/result
            const int offset = j * (m1 + 1);

            // Option A: Do a simple for-loop over i
            for (int i = 0; i <= m1; i++) {
                double sum = main_diag(j, i) * x(offset + i);

                if (i > 0) {
                    sum += lower_diag(j, i - 1) * x(offset + i - 1);
                }
                if (i < m1) {
                    sum += upper_diag(j, i) * x(offset + i + 1);
                }
                result(offset + i) = sum;
            }

            // Option B: Another (often more GPU-friendly) approach is
            // to add a nested parallel_for with ThreadVectorRange here:
            //
            // Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, m1 + 1),
            //     [&](const int i) {
            //         double sum = main_diag(j, i) * x(offset + i);
            //         ...
            //         result(offset + i) = sum;
            //     });
        });

    // Barrier to ensure all threads in this team are done
    team.team_barrier();
}



template<
    class View2D_const_main,
    class View2D_const_lower,
    class View2D_const_upper,
    class View1D_x,     // x is 1D
    class View2D_temp,  // temp is rank-2
    class View1D_b
>
KOKKOS_FUNCTION
void device_solve_implicit_parallel_v(
    const View2D_const_main&  impl_main,
    const View2D_const_lower& impl_lower,
    const View2D_const_upper& impl_upper,
    const View1D_x&           x,
    const View2D_temp&        temp,
    const View1D_b&           b,
    const Kokkos::TeamPolicy<>::member_type& team)
{
    // Determine grid sizes
    const int m2 = impl_main.extent(0) - 1;
    const int m1 = impl_main.extent(1) - 1;

    // Parallelize over j in [0..m2], each team handles one or more j's
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, m2 + 1),
        [&](const int j)
        {
            // For each j, we do a forward and backward sweep in i
            const int offset = j * (m1 + 1);

            // Forward sweep
            temp(j, 0) = impl_main(j, 0);
            x(offset)  = b(offset);

            for (int i = 1; i <= m1; i++) {
                const double m = impl_lower(j, i - 1) / temp(j, i - 1);
                temp(j, i) = impl_main(j, i) - m * impl_upper(j, i - 1);
                x(offset + i) = b(offset + i) - m * x(offset + i - 1);
            }

            // Backward sweep
            x(offset + m1) /= temp(j, m1);

            for (int i = m1 - 1; i >= 0; i--) {
                x(offset + i) = (x(offset + i)
                                 - impl_upper(j, i) * x(offset + i + 1))
                                / temp(j, i);
            }
        });

    // Team-level barrier if subsequent code in the same kernel depends on the result
    team.team_barrier();
}


//first test which prints out the Residual and the Diagonals as well as the implicict diagoanls
void test_a1_build() {
    using timer = std::chrono::high_resolution_clock;
    // Test dimensions
    const int m1 = 100;  // Stock price points
    const int m2 = 75;  // Variance points
    
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
    const double theta = 0.8;
    const double dt = 1.0/40.0;
    const double r_d = 0.025;
    const double r_f = 0.0;

    // Build diagonals
    // And then for testing:
    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;

    team_policy policy(1, Kokkos::AUTO);  // One team for testing, with "Kokkos::Auto" number of threads
    
    //
    auto t_start = timer::now();

    Kokkos::parallel_for("test_device_call", policy,
        KOKKOS_LAMBDA(const member_type& team) {
            build_a1_diagonals(
                main_diag, lower_diag, upper_diag,
                impl_main_diag, impl_lower_diag, impl_upper_diag,
                grid, theta, dt, r_d, r_f,
                team);
    });
    Kokkos::fence();

    auto t_end = timer::now();
    std::cout << "Build matrix time: "
              << std::chrono::duration<double>(t_end - t_start).count()
              << " seconds" << std::endl;
    
    auto total_time = std::chrono::duration<double>(t_end - t_start).count();
    /*
    
    Testing multiply
    
    */
    //building rhs and multiply vectors
    const int total_size = (m1 + 1) * (m2 + 1);
    Kokkos::View<double*> x("x", (m2+1)*(m1+1));
    Kokkos::View<double*> b("b", (m2+1)*(m1+1));
    Kokkos::View<double*> result("result", total_size);

    auto h_b = Kokkos::create_mirror_view(b);
    auto h_x = Kokkos::create_mirror_view(x);
        for (int i = 0; i < total_size; ++i) {
            h_b(i) = total_size - i;//(double)std::rand() / RAND_MAX;
            h_x(i) = 1.0 + i;//(double)std::rand() / RAND_MAX;
        }
    Kokkos::deep_copy(b, h_b);
    Kokkos::deep_copy(x, h_x);
    Kokkos::deep_copy(result, 0.0);

    t_start = timer::now();

    //auto x_tmp = x;  // Create non-const copy
    //auto result_tmp = result;

    Kokkos::parallel_for("test_multiply", policy,
        KOKKOS_LAMBDA(const member_type& team)
        {
            device_multiply_parallel_s_and_v(
                main_diag, lower_diag, upper_diag,
                x, 
                result,  // input x, output result
                team);
        });
    Kokkos::fence();

    t_end = timer::now();
    std::cout << "Multiply matrix time: "
              << std::chrono::duration<double>(t_end - t_start).count()
              << " seconds" << std::endl;

    total_time += std::chrono::duration<double>(t_end - t_start).count();
    // Get results back to host
    auto h_result = Kokkos::create_mirror_view(result);


    /*
    
    Testing implcicit call
    
    */
    // For 'temp', we definitely need to write into it:
    Kokkos::View<double**> temp("temp", m2+1, m1+1);

    //auto temp_tmp = temp;
    //auto b_tmp = b;

    t_start = timer::now();
    Kokkos::parallel_for("test_implicit_solve", policy,
        KOKKOS_LAMBDA(const member_type& team) {
        device_solve_implicit_parallel_v(
            impl_main_diag, impl_lower_diag, impl_upper_diag,
            x, temp, b,  // Use the non-const copies
            team);
    }); 
    Kokkos::fence();
    t_end = timer::now();

    // After implicit solve:
    t_end = timer::now();
    std::cout << "Implicit solve time: "
            << std::chrono::duration<double>(t_end - t_start).count()
            << " seconds" << std::endl;
    
    total_time += std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "TOTAL TIME: "
            << total_time
            << " seconds" << std::endl;

    // Need to do one more multiply to verify the solution
    Kokkos::deep_copy(result, 0.0);  // Clear previous result
    //auto result_verify = result;
    Kokkos::parallel_for("verify_implicit", policy,
        KOKKOS_LAMBDA(const member_type& team)
        {
            device_multiply_parallel_s_and_v(
                main_diag, lower_diag, upper_diag,
                x, result,
                team);
        });
    Kokkos::fence();

    // Get results and compute residual on host
    Kokkos::deep_copy(h_x, x);
    Kokkos::deep_copy(h_result, result);
    Kokkos::deep_copy(h_b, b);

    double implicit_residual = 0.0;
    for(int i = 0; i < total_size; i++) {
        double res = h_x(i) - theta * dt * h_result(i) - h_b(i);
        implicit_residual += res * res;
    }
    implicit_residual = std::sqrt(implicit_residual);
    std::cout << "Implicit solve residual norm: " << implicit_residual << std::endl;

    // Add this after computing implicit_residual:
    /*
    std::cout << "Dimesnion StockxVariance: " << m1+1 << "x" << m2+1 << "\n";
    std::cout << "Instance 0 => residual norm = " << implicit_residual << std::endl;
    std::cout << "  x[0..4] = ";
    for(int i = 0; i < std::min(20, total_size); i++) {
        std::cout << h_x(i) << " ";
    }
    std::cout << "\n------------------------------------\n";
    */

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
    /*
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
    */
}

//This test compares the explicit and implicit output to the test in the A1 class. To make sure we are doing exactly the same
void test_a1_structure_function() {
    // Test dimensions - matching your test
    int m1 = 4;  
    int m2 = 3;
    
    Grid grid = create_test_grid(m1, m2);
    
    // Create Views for diagonals
    Kokkos::View<double**> main_diag("main_diag", m2+1, m1+1);
    Kokkos::View<double**> lower_diag("lower_diag", m2+1, m1);
    Kokkos::View<double**> upper_diag("upper_diag", m2+1, m1);
    
    Kokkos::View<double**> impl_main_diag("impl_main_diag", m2+1, m1+1);
    Kokkos::View<double**> impl_lower_diag("impl_lower_diag", m2+1, m1);
    Kokkos::View<double**> impl_upper_diag("impl_upper_diag", m2+1, m1);
    
    // Same parameters as your test
    double rho = -0.9;
    double sigma = 0.3;
    double r_d = 0.025;
    double r_f = 0.0;
    double theta = 0.8;
    double delta_t = 1.0/20;

    // Build matrices using team policy
    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;
    team_policy policy(1, Kokkos::AUTO);

    // Build the matrices
    Kokkos::parallel_for("build_matrices", policy,
        KOKKOS_LAMBDA(const member_type& team) {
            build_a1_diagonals(
                main_diag, lower_diag, upper_diag,
                impl_main_diag, impl_lower_diag, impl_upper_diag,
                grid, theta, delta_t, r_d, r_f,
                team);
    });
    Kokkos::fence();

    // Print matrix structure - matching your output format
    const int total_size = (m1 + 1) * (m2 + 1);
    std::cout << "\nA1 Matrix Structure (Function Version):";
    std::cout << "\nShape: [" << total_size << ", " << total_size << "]" << std::endl;

    auto h_main = Kokkos::create_mirror_view(main_diag);
    auto h_lower = Kokkos::create_mirror_view(lower_diag);
    auto h_upper = Kokkos::create_mirror_view(upper_diag);
    
    Kokkos::deep_copy(h_main, main_diag);
    Kokkos::deep_copy(h_lower, lower_diag);
    Kokkos::deep_copy(h_upper, upper_diag);

    // Print block structure
    for(int j = 0; j <= m2; j++) {
        std::cout << "\nBlock j=" << j << ":" << std::endl;
        
        std::cout << "\n  Lower diagonal for block " << j << ":" << std::endl;
        for(int i = 0; i < m1; i++) {
            double val = h_lower(j,i);
            if(std::abs(val) > 1e-10) {
                std::cout << "    [" << i+1 << "," << i << "] = " 
                         << std::fixed << std::setprecision(6) << val << std::endl;
            }
        }
        
        std::cout << "\n  Main diagonal for block " << j << ":" << std::endl;
        for(int i = 0; i <= m1; i++) {
            double val = h_main(j,i);
            if(std::abs(val) > 1e-10) {
                std::cout << "    [" << i << "," << i << "] = " 
                         << std::fixed << std::setprecision(6) << val << std::endl;
            }
        }
        
        std::cout << "\n  Upper diagonal for block " << j << ":" << std::endl;
        for(int i = 0; i < m1; i++) {
            double val = h_upper(j,i);
            if(std::abs(val) > 1e-10) {
                std::cout << "    [" << i << "," << i+1 << "] = " 
                         << std::fixed << std::setprecision(6) << val << std::endl;
            }
        }
    }

    // Create and initialize test vectors exactly as in your test
    Kokkos::View<double*> x("x", total_size);
    Kokkos::View<double*> b("b", total_size);
    Kokkos::View<double*> result("result", total_size);
    
    auto h_x = Kokkos::create_mirror_view(x);
    auto h_b = Kokkos::create_mirror_view(b);
    
    for(int i = 0; i < total_size; i++) {
        h_x(i) = i + 1.0;
        h_b(i) = total_size - i;
    }
    
    std::cout << "\nTest vector x first 10 values:" << std::endl;
    for(int i = 0; i < std::min(10, total_size); i++) {
        std::cout << "x[" << i << "] = " << std::fixed 
                 << std::setprecision(6) << h_x(i) << std::endl;
    }
    
    std::cout << "\nTest vector b first 10 values:" << std::endl;
    for(int i = 0; i < std::min(10, total_size); i++) {
        std::cout << "b[" << i << "] = " << std::fixed 
                 << std::setprecision(6) << h_b(i) << std::endl;
    }
    
    Kokkos::deep_copy(x, h_x);
    Kokkos::deep_copy(b, h_b);
    Kokkos::deep_copy(result, 0.0);

    // Test multiply
    //auto x_tmp = x;
    //auto result_tmp = result;
    
    Kokkos::parallel_for("test_multiply", policy,
        KOKKOS_LAMBDA(const member_type& team)
        {
            device_multiply_parallel_s_and_v(
                main_diag, lower_diag, upper_diag,
                x, result,
                team);
        });
    Kokkos::fence();

    auto h_result = Kokkos::create_mirror_view(result);
    Kokkos::deep_copy(h_result, result);
    
    std::cout << "\nExplicit multiplication first 30 results:" << std::endl;
    for(int i = 0; i < std::min(30, total_size); i++) {
        std::cout << "result[" << i << "] = " << std::fixed 
                 << std::setprecision(6) << h_result(i) << std::endl;
    }

    // Test implicit solve
    Kokkos::View<double**> temp("temp", m2+1, m1+1);
    //auto temp_tmp = temp;
    //auto b_tmp = b;

    Kokkos::parallel_for("test_implicit_solve", policy,
        KOKKOS_LAMBDA(const member_type& team) {
            device_solve_implicit_parallel_v(
                impl_main_diag, impl_lower_diag, impl_upper_diag,
                x, temp, b,
                team);
        });
    Kokkos::fence();

    auto h_implicit = Kokkos::create_mirror_view(x);
    Kokkos::deep_copy(h_implicit, x);
    
    std::cout << "\nImplicit solve first 30 results:" << std::endl;
    for(int i = 0; i < std::min(30, total_size); i++) {
        std::cout << "implicit_result[" << i << "] = " << std::fixed 
                 << std::setprecision(6) << h_implicit(i) << std::endl;
    }
}

//this tests calling multiple solver instances on the GPU
void test_a1_multiple_instances(){
    using timer = std::chrono::high_resolution_clock;

    // Test parameters
    double K = 100.0;
    double S_0 = K;
    double V_0 = 0.04;
    
    const int m1 = 100;
    const int m2 = 75;
    std::cout << "Dimesnion StockxVariance: " << m1+1 << "x" << m2+1 << std::endl;

    double theta = 0.8;
    double delta_t = 1.0/40.0;

    int nInstances = 5;          

    /*
    
    New Grid construction
    
    */
    // Initialize vectors with grid views and diagonals
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
        //Grid tempGrid(m1, 8*K, S_0, K, K/5, m2, 5.0, V_0, 5.0/500);
        Grid tempGrid = create_test_grid(m1,m2);
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

    


    // 3D arrays for the diagonals: dimension [nInstances, (m2+1), (m1+1)]
    Kokkos::View<double***> main_diag("main_diag", nInstances, m2+1, m1+1);
    Kokkos::View<double***> lower_diag("lower_diag", nInstances, m2+1, m1);
    Kokkos::View<double***> upper_diag("upper_diag", nInstances, m2+1, m1);

    Kokkos::View<double***> impl_main_diag("impl_main_diag", nInstances, m2+1, m1+1);
    Kokkos::View<double***> impl_lower_diag("impl_lower_diag", nInstances, m2+1, m1);
    Kokkos::View<double***> impl_upper_diag("impl_upper_diag", nInstances, m2+1, m1);

    int total_size = (m1+1)*(m2+1);

    Kokkos::View<double**> x   ("x",   nInstances, total_size);
    Kokkos::View<double**> b   ("b",   nInstances, total_size);
    Kokkos::View<double***> temp("temp",nInstances, (m2+1), (m1+1)); 

    ///////////////////////////////////////////////
    // 1) Create host mirrors for x and b
    ///////////////////////////////////////////////
    auto h_x = Kokkos::create_mirror_view(x);
    auto h_b = Kokkos::create_mirror_view(b);

    // 2) Fill them with values
    //    Here we do a simple pattern: x = (inst + 1.0), b = (inst + 2.0), etc.
    for(int inst = 0; inst < nInstances; ++inst) {
        for(int idx = 0; idx < total_size; ++idx) {
            // Example: each instance has a unique offset so we can see it's not all zero
            double val_x = 1.0+idx;//(inst + 1.0) + 0.001 * idx;
            double val_b = total_size - idx;//(inst + 2.0) + 0.002 * idx;

            h_x(inst, idx) = val_x;
            h_b(inst, idx) = val_b;
        }
    }

    // 3) Copy the host data back to device
    Kokkos::deep_copy(x, h_x);
    Kokkos::deep_copy(b, h_b);

    // or 2D if you prefer [nInstances, total_size], 
    // but your solver code uses (m2+1, m1+1).

    Kokkos::View<double**> result("result", nInstances, total_size);

    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;

    // Each team handles ONE instance. 
    team_policy policy(nInstances, Kokkos::AUTO);

    auto t_start = timer::now();

    Kokkos::parallel_for("build_and_solve_all", policy,
    KOKKOS_LAMBDA(const member_type& team)
    {
        const int instance = team.league_rank();  // which PDE instance are we?

        // 1) Subview the diagonals for this instance
        auto mainDiag_i = Kokkos::subview(main_diag, instance, Kokkos::ALL, Kokkos::ALL);
        auto lowerDiag_i = Kokkos::subview(lower_diag, instance, Kokkos::ALL, Kokkos::ALL);
        auto upperDiag_i = Kokkos::subview(upper_diag, instance, Kokkos::ALL, Kokkos::ALL);

        auto implMain_i = Kokkos::subview(impl_main_diag, instance, Kokkos::ALL, Kokkos::ALL);
        auto implLower_i = Kokkos::subview(impl_lower_diag, instance, Kokkos::ALL, Kokkos::ALL);
        auto implUpper_i = Kokkos::subview(impl_upper_diag, instance, Kokkos::ALL, Kokkos::ALL);

        // 2) Subview the solution and RHS for this instance
        auto x_i      = Kokkos::subview(x, instance, Kokkos::ALL);
        auto b_i      = Kokkos::subview(b, instance, Kokkos::ALL);
        auto result_i = Kokkos::subview(result, instance, Kokkos::ALL);

        // Possibly we need a 2D subview for 'temp'
        auto temp_i = Kokkos::subview(temp, instance, Kokkos::ALL, Kokkos::ALL);

        // Retrieve the grid for this instance
        GridViews grid_i = deviceGrids(instance);

        /*
        
        Testing Grid pod inside kernel call
        
        */
        

        // PDE parameters (could vary per instance)
        double r_d   = 0.025;
        double r_f   = 0.0;

        
        // 3) Now build the diagonals for this instance
        build_a1_diagonals(
            mainDiag_i, lowerDiag_i, upperDiag_i,
            implMain_i, implLower_i, implUpper_i,
            grid_i, theta, delta_t, r_d, r_f,
            team
        );
        
        
        // 4) Multiply: A * x_i = result_i
        //    (Each instance does the same PDE steps, but with its own data)
        device_multiply_parallel_s_and_v(
            mainDiag_i, lowerDiag_i, upperDiag_i,
            x_i, result_i,
            team
        );

        
        // 5) Solve: (I - theta*dt*A)*x_i = b_i
        device_solve_implicit_parallel_v(
            implMain_i, implLower_i, implUpper_i,
            x_i, temp_i, b_i,
            team
        );
        
        // 6) Optionally compute a local residual for debugging
        team.team_barrier();
        // If you want a local check, you can do another multiply or gather results:
        // ...
    });
    Kokkos::fence();

    auto t_end = timer::now();
    std::cout << "ENTIRE KERNEL: "
              << std::chrono::duration<double>(t_end - t_start).count()
              << " seconds" << std::endl;


    ////////////////////////////////////////////////////////////
    // 1) Multiply again: result_i = A_i * x_i
    ////////////////////////////////////////////////////////////
    Kokkos::parallel_for("final_check_multiply", policy,
    KOKKOS_LAMBDA(const member_type& team)
    {
    const int instance = team.league_rank();

    // Subview the diagonals and x/result for this instance
    auto mainDiag_i   = Kokkos::subview(main_diag,   instance, Kokkos::ALL, Kokkos::ALL);
    auto lowerDiag_i  = Kokkos::subview(lower_diag,  instance, Kokkos::ALL, Kokkos::ALL);
    auto upperDiag_i  = Kokkos::subview(upper_diag,  instance, Kokkos::ALL, Kokkos::ALL);

    auto x_i      = Kokkos::subview(x, instance, Kokkos::ALL);
    auto result_i = Kokkos::subview(result, instance, Kokkos::ALL);

    // Reuse your multiply function
    device_multiply_parallel_s_and_v(
        mainDiag_i, lowerDiag_i, upperDiag_i,
        x_i, result_i,
        team
    );
    });
    Kokkos::fence();

    ////////////////////////////////////////////////////////////
    // 2) Copy x, b, and result back to host
    ////////////////////////////////////////////////////////////
    //h_x      = Kokkos::create_mirror_view(x);
    //h_b      = Kokkos::create_mirror_view(b);
    auto h_result = Kokkos::create_mirror_view(result);

    Kokkos::deep_copy(h_x, x);
    Kokkos::deep_copy(h_b, b);
    Kokkos::deep_copy(h_result, result);

    ////////////////////////////////////////////////////////////
    // 3) Compute the residual on the host for each instance
    ////////////////////////////////////////////////////////////
    for(int inst = 0; inst < nInstances; ++inst) {

    double residual_sum = 0.0;
    for(int idx = 0; idx < total_size; idx++) {
        double lhs = h_x(inst, idx);
        double rhs = theta * delta_t * h_result(inst, idx) + h_b(inst, idx);

        double diff = lhs - rhs;  // x - (θ⋅Δt⋅A x + b)
        residual_sum += diff * diff;
    }

    double residual_norm = std::sqrt(residual_sum);

    // Print the residual for this instance
    std::cout << "Instance " << inst
                << " => residual norm = " << residual_norm << std::endl;

    // (Optional) print first ~5 solution values
    /*
    std::cout << "  x[0..4] = ";
    for(int i = 0; i < std::min(20,total_size); i++){
        std::cout << h_x(inst, i) << " ";
    }
    */
    std::cout << "\n------------------------------------\n";
    }
}


/**
 * 
 * Test grid usage on device
 * 
 */
//this function is the "constructor" for the struct GridViews
void buildMultipleGridViews(
    std::vector<GridViews> &hostGrids,
    int nInstances, int m1, int m2)
{
  // Resize to hold nInstances
  hostGrids.resize(nInstances);

  for(int i = 0; i < nInstances; i++) {
    // 1) Allocate device arrays for each PDE dimension
    hostGrids[i].device_Vec_s = Kokkos::View<double*>("vec_s",    m1+1);
    hostGrids[i].device_Vec_v = Kokkos::View<double*>("vec_v",    m2+1);
    hostGrids[i].device_Delta_s = Kokkos::View<double*>("delta_s",m1);
    hostGrids[i].device_Delta_v = Kokkos::View<double*>("delta_v",m2);

    hostGrids[i].m1 = m1;
    hostGrids[i].m2 = m2;

    /*
    // 2) For demo, just fill device_Vec_s with: (i * 100) + j
    //    (Pretend it's your PDE logic.)
    auto mirror_s = Kokkos::create_mirror_view(hostGrids[i].device_Vec_s);
    for(int j = 0; j <= m1; j++){
      mirror_s(j) = 100.0 * i + j;  
      // e.g. instance 0 => [0,1,2,...], instance 1 => [100,101,...]
    }
    Kokkos::deep_copy(hostGrids[i].device_Vec_s, mirror_s);

    // For brevity, we won't fill device_Vec_v, device_Delta_s, etc. 
    // but in real code, you'd do the same pattern with mirrors + deep_copy.
    */
  }
}

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




void test_a1_kernel(){
    Kokkos::initialize();
    {
        try{
            //test_a1_build();
            //test_a1_structure_function();

            test_a1_multiple_instances();
            //testMultipleGridViews();
            //test_myGrids();
        }
        catch (std::exception& e) {
            std::cout << "Error: " << e.what() << std::endl;
        }
    } // All test objects destroyed here
    Kokkos::finalize();
}