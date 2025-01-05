/*
// Base Kokkos includes
#include <Kokkos_Core.hpp>
#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_spmv.hpp>

#include <KokkosSparse_sptrsv.hpp>
#include <KokkosKernels_default_types.hpp>

#include <Kokkos_Core.hpp>
#include <KokkosKernels_default_types.hpp>
#include <KokkosKernels_Handle.hpp>
#include <KokkosSparse_IOUtils.hpp>
#include <KokkosSparse_spmv.hpp>
#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_gauss_seidel.hpp>
#include <KokkosBlas1_nrm2.hpp>

// Type definitions
using scalar_t = double;
using lno_t = int;    // Local indices
using size_type = int;  // Global indices
using layout_t = Kokkos::LayoutLeft;
using exec_space = Kokkos::DefaultExecutionSpace;
using memory_space = exec_space::memory_space;
using device_t = Kokkos::Device<exec_space, memory_space>;
using handle_t = KokkosKernels::Experimental::KokkosKernelsHandle
    <size_type, lno_t, scalar_t, exec_space, memory_space, memory_space>;



// Type definitions that will be used across the project
using ViewVectorType = Kokkos::View<scalar_t*, layout_t, memory_space>;



void heston_adi() {
    // Initialize Kokkos
    Kokkos::initialize();
    {
        // Matrix size
        const size_type n_rows = 5;
        const size_type n_cols = 5;
        const size_type n_entries = 13; // Number of non-zero elements

        // Host arrays for matrix construction
        Kokkos::View<size_type*>("row_map_h", n_rows + 1);
        Kokkos::View<lno_t*>("entries_h", n_entries);
        Kokkos::View<scalar_t*>("values_h", n_entries);

        // Create host mirrors for initialization
        auto row_map_h = Kokkos::create_mirror_view(Kokkos::View<size_type*>("row_map_h", n_rows + 1));
        auto entries_h = Kokkos::create_mirror_view(Kokkos::View<lno_t*>("entries_h", n_entries));
        auto values_h = Kokkos::create_mirror_view(Kokkos::View<scalar_t*>("values_h", n_entries));

        // Example sparse matrix (tridiagonal)
        // [2 -1  0  0  0]
        // [-1 2 -1  0  0]
        // [0 -1  2 -1  0]
        // [0  0 -1  2 -1]
        // [0  0  0 -1  2]

        // Fill row_map (CSR format row pointers)
        row_map_h(0) = 0;
        row_map_h(1) = 2;
        row_map_h(2) = 5;
        row_map_h(3) = 8;
        row_map_h(4) = 11;
        row_map_h(5) = 13;

        // Fill column indices
        int idx = 0;
        // First row
        entries_h(idx) = 0; values_h(idx++) = 2;
        entries_h(idx) = 1; values_h(idx++) = -1;
        // Second row
        entries_h(idx) = 0; values_h(idx++) = -1;
        entries_h(idx) = 1; values_h(idx++) = 2;
        entries_h(idx) = 2; values_h(idx++) = -1;
        // Third row
        entries_h(idx) = 1; values_h(idx++) = -1;
        entries_h(idx) = 2; values_h(idx++) = 2;
        entries_h(idx) = 3; values_h(idx++) = -1;
        // Fourth row
        entries_h(idx) = 2; values_h(idx++) = -1;
        entries_h(idx) = 3; values_h(idx++) = 2;
        entries_h(idx) = 4; values_h(idx++) = -1;
        // Fifth row
        entries_h(idx) = 3; values_h(idx++) = -1;
        entries_h(idx) = 4; values_h(idx++) = 2;

        // Create device views
        Kokkos::View<size_type*> row_map_d("row_map_d", n_rows + 1);
        Kokkos::View<lno_t*> entries_d("entries_d", n_entries);
        Kokkos::View<scalar_t*> values_d("values_d", n_entries);

        // Copy data to device
        Kokkos::deep_copy(row_map_d, row_map_h);
        Kokkos::deep_copy(entries_d, entries_h);
        Kokkos::deep_copy(values_d, values_h);

        // Create sparse matrix
        KokkosSparse::CrsMatrix<scalar_t, lno_t, exec_space, void, size_type> A(
            "A", n_rows, n_cols, n_entries, values_d, row_map_d, entries_d);

        // Create exact solution vector x = [1, 1, 1, 1, 1]
        Kokkos::View<scalar_t*> x_exact("x_exact", n_cols);
        auto x_exact_h = Kokkos::create_mirror_view(x_exact);
        for(size_type i = 0; i < n_cols; ++i) {
            x_exact_h(i) = 1.0;
        }
        Kokkos::deep_copy(x_exact, x_exact_h);

        // Create right-hand side vector b = A*x_exact
        Kokkos::View<scalar_t*> b("b", n_rows);
        KokkosSparse::spmv("N", 1.0, A, x_exact, 0.0, b);

        // TEST LINEAR SOLVER
        std::cout << "\nTesting Gauss-Seidel Linear Solver\n";
        std::cout << "===================================\n";

        // Create solution vector (initialized to zero)
        Kokkos::View<scalar_t*> x("x", n_rows, 0.0);
        
        // Create residual vector
        Kokkos::View<scalar_t*> res("res", n_rows);

        // Create handle and its GS subhandle
        handle_t handle;
        handle.create_gs_handle(KokkosSparse::GS_DEFAULT);

        // Symbolic phase
        KokkosSparse::Experimental::gauss_seidel_symbolic(
            &handle, n_rows, n_cols, 
            A.graph.row_map, A.graph.entries, 
            true  // matrix is symmetric
        );

        // Numeric phase
        KokkosSparse::Experimental::gauss_seidel_numeric(
            &handle, n_rows, n_cols,
            A.graph.row_map, A.graph.entries, A.values,
            true  // matrix is symmetric
        );

        // Solve iterations
        const scalar_t tolerance = 1e-6;
        const scalar_t omega = 1.0;  // relaxation parameter
        bool first_iter = true;
        
        // Get initial residual norm
        Kokkos::deep_copy(res, b);
        scalar_t initial_res = KokkosBlas::nrm2(res);
        scalar_t scaled_res_norm = 1.0;
        
        int num_iters = 0;
        const int max_iters = 100;

        std::cout << "Starting iterations...\n";
        while(scaled_res_norm > tolerance && num_iters < max_iters) {
            // Apply one forward sweep
            KokkosSparse::Experimental::forward_sweep_gauss_seidel_apply(
                &handle, n_rows, n_cols,
                A.graph.row_map, A.graph.entries, A.values,
                x, b, first_iter, first_iter, omega, 1
            );
            
            first_iter = false;

            // Compute new residual
            Kokkos::deep_copy(res, b);
            KokkosSparse::spmv("N", 1.0, A, x, -1.0, res);
            
            scaled_res_norm = KokkosBlas::nrm2(res) / initial_res;
            num_iters++;
            
            if(num_iters % 10 == 0) {
                std::cout << "Iteration " << num_iters 
                         << " scaled residual: " << scaled_res_norm << std::endl;
            }
        }

        // Clean up
        handle.destroy_gs_handle();

        // Check solution
        auto solution_h = Kokkos::create_mirror_view(x);
        Kokkos::deep_copy(solution_h, x);

        std::cout << "\nSolver Results:\n";
        std::cout << "Completed in " << num_iters << " iterations\n";
        std::cout << "Final scaled residual: " << scaled_res_norm << "\n";
        std::cout << "Solution vector:\n";
        for(size_type i = 0; i < n_rows; ++i) {
            std::cout << solution_h(i) << " ";
        }
        std::cout << "\n\nExpected solution:\n";
        for(size_type i = 0; i < n_rows; ++i) {
            std::cout << x_exact_h(i) << " ";
        }
        std::cout << std::endl;
    }
    // Finalize Kokkos
    Kokkos::finalize();
}
*/

#include "stdfax.hpp" 

using matrix_t     = KokkosSparse::CrsMatrix<scalar_t, lno_t, exec_space, void, size_type>;
using mv_t         = Kokkos::View<scalar_t*, device_t>;

//for gmres_ilu()
#include <KokkosSparse_spiluk.hpp>
#include <KokkosSparse_gmres.hpp>


struct BenchmarkResult {
    double spmv_time;
    double solver_time;
    int solver_iterations;
    double final_residual;
};

BenchmarkResult run_benchmark(size_type n) {
    BenchmarkResult result{};
    
    // Number of non-zeros per row (tridiagonal + diagonal = 3)
    const size_type nnz_per_row = 3;
    const size_type total_nnz = n * nnz_per_row - 2; // Subtract 2 for first and last rows
    
    // Create host arrays for matrix construction
    Kokkos::View<size_type*> row_map_d("row_map", n + 1);
    Kokkos::View<lno_t*> entries_d("entries", total_nnz);
    Kokkos::View<scalar_t*> values_d("values", total_nnz);
    
    auto row_map_h = Kokkos::create_mirror_view(row_map_d);
    auto entries_h = Kokkos::create_mirror_view(entries_d);
    auto values_h = Kokkos::create_mirror_view(values_d);
    
    // Fill matrix data (tridiagonal matrix)
    size_type nnz_count = 0;
    for(size_type i = 0; i < n; i++) {
        row_map_h(i) = nnz_count;
        
        if(i > 0) {
            entries_h(nnz_count) = i-1;
            values_h(nnz_count++) = -1.0;
        }
        
        entries_h(nnz_count) = i;
        values_h(nnz_count++) = 2.0;
        
        if(i < n-1) {
            entries_h(nnz_count) = i+1;
            values_h(nnz_count++) = -1.0;
        }
    }
    row_map_h(n) = nnz_count;
    
    // Copy to device
    Kokkos::deep_copy(row_map_d, row_map_h);
    Kokkos::deep_copy(entries_d, entries_h);
    Kokkos::deep_copy(values_d, values_h);
    
    // Create matrix
    KokkosSparse::CrsMatrix<scalar_t, lno_t, exec_space, void, size_type> A(
        "A", n, n, total_nnz, values_d, row_map_d, entries_d);
    
    // Create vectors
    Kokkos::View<scalar_t*> x("x", n);
    Kokkos::View<scalar_t*> b("b", n);
    
    // Initialize x with ones for SpMV test
    Kokkos::deep_copy(x, 1.0);
    
    // Benchmark SpMV
    Kokkos::fence();
    auto spmv_start = std::chrono::high_resolution_clock::now();
    
    for(int i = 0; i < 100; i++) { // Run multiple times for better timing
        KokkosSparse::spmv("N", 1.0, A, x, 0.0, b);
    }
    
    Kokkos::fence();
    auto spmv_end = std::chrono::high_resolution_clock::now();
    result.spmv_time = std::chrono::duration<double>(spmv_end - spmv_start).count() / 100.0;
    
    // Benchmark Solver
    handle_t handle;
    handle.create_gs_handle(KokkosSparse::GS_DEFAULT);
    
    // Reset x to zero for solver test
    Kokkos::deep_copy(x, 0.0);
    
    // Create residual vector
    Kokkos::View<scalar_t*> res("res", n);
    
    // Symbolic and numeric phase
    KokkosSparse::Experimental::gauss_seidel_symbolic(
        &handle, n, n, A.graph.row_map, A.graph.entries, true);
        
    KokkosSparse::Experimental::gauss_seidel_numeric(
        &handle, n, n, A.graph.row_map, A.graph.entries, A.values, true);
    
    // Solve timing
    Kokkos::fence();
    auto solve_start = std::chrono::high_resolution_clock::now();
    
    const scalar_t tolerance = 1e-6;
    const scalar_t omega = 1.0;
    bool first_iter = true;
    
    Kokkos::deep_copy(res, b);
    scalar_t initial_res = KokkosBlas::nrm2(res);
    scalar_t scaled_res_norm = 1.0;
    int num_iters = 0;
    const int max_iters = 1000;
    
    while(scaled_res_norm > tolerance && num_iters < max_iters) {
        KokkosSparse::Experimental::forward_sweep_gauss_seidel_apply(
            &handle, n, n, A.graph.row_map, A.graph.entries, A.values,
            x, b, first_iter, first_iter, omega, 1);
        
        first_iter = false;
        
        Kokkos::deep_copy(res, b);
        KokkosSparse::spmv("N", 1.0, A, x, -1.0, res);
        
        scaled_res_norm = KokkosBlas::nrm2(res) / initial_res;
        num_iters++;
    }
    
    Kokkos::fence();
    auto solve_end = std::chrono::high_resolution_clock::now();
    result.solver_time = std::chrono::duration<double>(solve_end - solve_start).count();
    
    result.solver_iterations = num_iters;
    result.final_residual = scaled_res_norm;
    
    handle.destroy_gs_handle();
    
    return result;
}

void test_benchmark(){
        Kokkos::initialize();
    {
        std::cout << "Running benchmarks on: " << typeid(exec_space).name() << "\n\n";
        
        std::vector<size_type> problem_sizes = {1000, 5000, 10000, 50000, 100000};
        
        std::cout << std::setw(10) << "Size" 
                  << std::setw(15) << "SpMV (ms)"
                  << std::setw(15) << "Solve (ms)"
                  << std::setw(15) << "Iterations"
                  << std::setw(15) << "Final Res"
                  << "\n";
        std::cout << std::string(70, '-') << "\n";
        
        for(auto n : problem_sizes) {
            auto result = run_benchmark(n);
            
            std::cout << std::setw(10) << n
                      << std::setw(15) << std::fixed << std::setprecision(3) << result.spmv_time * 1000
                      << std::setw(15) << result.solver_time * 1000
                      << std::setw(15) << result.solver_iterations
                      << std::setw(15) << std::scientific << std::setprecision(3) << result.final_residual
                      << "\n";
        }
    }
    Kokkos::finalize();
}

void test_scheme() {
    Kokkos::initialize();
    {
        // Simple problem size
        const int N = 1000;
        //const int timesteps = 10;
        
        // Create a simple tridiagonal matrix
        const size_type nnz = 3*N - 2;
        Kokkos::View<size_type*> row_map("row_map", N + 1);
        Kokkos::View<lno_t*> entries("entries", nnz);
        Kokkos::View<scalar_t*> values("values", nnz);
        
        auto row_map_h = Kokkos::create_mirror_view(row_map);
        auto entries_h = Kokkos::create_mirror_view(entries);
        auto values_h = Kokkos::create_mirror_view(values);
        
        // Fill matrix on host
        size_type nnz_count = 0;
        for(int i = 0; i < N; i++) {
            row_map_h(i) = nnz_count;
            if(i > 0) { entries_h(nnz_count) = i-1; values_h(nnz_count++) = -1.0; }
            entries_h(nnz_count) = i; values_h(nnz_count++) = 2.0;
            if(i < N-1) { entries_h(nnz_count) = i+1; values_h(nnz_count++) = -1.0; }
        }
        row_map_h(N) = nnz_count;
        
        // Copy to device
        Kokkos::deep_copy(row_map, row_map_h);
        Kokkos::deep_copy(entries, entries_h);
        Kokkos::deep_copy(values, values_h);
        
        // Create matrix
        auto A = KokkosSparse::CrsMatrix<scalar_t, lno_t, exec_space>
            ("A", N, N, nnz, values, row_map, entries);
        
        // Create vectors
        Kokkos::View<scalar_t*> x("x", N);
        Kokkos::View<scalar_t*> b("b", N);
        Kokkos::View<scalar_t*> residual("residual", N);

        // Initialize vectors
        Kokkos::deep_copy(x, 1.0);  // Initial guess
        Kokkos::deep_copy(b, 0.0);  // Right hand side
        
        // Initialize solver handle
        handle_t handle;
        handle.create_gs_handle(KokkosSparse::GS_DEFAULT);
        
        // Preconditioner setup
        KokkosSparse::Experimental::gauss_seidel_symbolic
            (&handle, N, N, A.graph.row_map, A.graph.entries, false);
        KokkosSparse::Experimental::gauss_seidel_numeric
            (&handle, N, N, A.graph.row_map, A.graph.entries, A.values, false);
        
        // Calculate initial residual norm
        Kokkos::deep_copy(residual, b);
        KokkosSparse::spmv("N", 1.0, A, x, -1.0, residual);  // residual = b - Ax
        scalar_t initial_res = KokkosBlas::nrm2(residual);
        scalar_t prev_res = initial_res;

        std::cout << "Initial residual norm: " << initial_res << std::endl;

        /*
        Kokkos::Timer timer;
        // Time stepping loop
        for(int t = 0; t < 2*timesteps; t++) {
            // SpMV operation
            KokkosSparse::spmv("N", 1.0, A, x, 0.0, b);
            
            // Solve step
            bool is_first = true;
            KokkosSparse::Experimental::forward_sweep_gauss_seidel_apply
                (&handle, N, N, A.graph.row_map, A.graph.entries, A.values,
                 x, b, is_first, is_first, 1.0, 1);
            Kokkos::deep_copy(residual, b);
            KokkosSparse::spmv("N", 1.0, A, x, -1.0, residual);  // residual = b - Ax
            
            // Calculate and print relative residual norm
            scalar_t current_res = KokkosBlas::nrm2(residual);
            scalar_t rel_res = current_res / initial_res;
            
            std::cout << "Timestep " << t << ", Relative residual: " << rel_res << std::endl;
        }
        
        double time = timer.seconds();
        std::cout << "Total solve time: " << time << " seconds" << std::endl;
        */
        const scalar_t tolerance = 1e-6;     // Convergence tolerance
        const int max_iterations = 1000;     // Safety limit
        int iter = 0;                        // Iteration counter

        scalar_t omega = 1.2;
        Kokkos::Timer timer;

        while(iter < max_iterations) {
            // SpMV operation
            KokkosSparse::spmv("N", 1.0, A, x, 0.0, b);
            
            // Solve step
            bool is_first = (iter == 0);
            KokkosSparse::Experimental::forward_sweep_gauss_seidel_apply
                (&handle, N, N, A.graph.row_map, A.graph.entries, A.values,
                x, b, is_first, is_first, omega, 1);

            // Backward sweep
            KokkosSparse::Experimental::backward_sweep_gauss_seidel_apply
                (&handle, N, N, A.graph.row_map, A.graph.entries, A.values,
                x, b, false, false, omega, 1);
            
            // Calculate residual
            Kokkos::deep_copy(residual, b);
            KokkosSparse::spmv("N", 1.0, A, x, -1.0, residual);
            
            scalar_t current_res = KokkosBlas::nrm2(residual);
            scalar_t rel_res = current_res / initial_res;
            
            std::cout << "Iteration " << iter << ", Relative residual: " << rel_res << std::endl;
            
            if(rel_res < tolerance) {
                std::cout << "Converged to tolerance " << tolerance << " after " << iter+1 << " iterations" << std::endl;
                break;
            }

            
            if(iter > 0) {
                scalar_t conv_rate = log(rel_res/prev_res);
                std::cout << "Convergence rate: " << conv_rate << std::endl;
            }
            
            scalar_t prev_res = rel_res;

            iter++;
        }

        double time = timer.seconds();

        if(iter == max_iterations) {
            std::cout << "Warning: Failed to converge after " << max_iterations << " iterations" << std::endl;
        }

        std::cout << "Total solve time: " << time << " seconds" << std::endl;
        handle.destroy_gs_handle();
    }
    Kokkos::finalize();
}

struct SchemeBenchmarkResult {
    size_type N;
    int timesteps;
    double total_time;
    double assembly_time;
    double spmv_time;
    double solve_time;
};

SchemeBenchmarkResult run_scheme_benchmark(size_type N, int timesteps) {
    SchemeBenchmarkResult result{};
    result.N = N;
    result.timesteps = timesteps;

    // Start total timer
    Kokkos::fence();
    auto total_start = std::chrono::high_resolution_clock::now();

    // Start assembly timer
    Kokkos::fence();
    auto assembly_start = std::chrono::high_resolution_clock::now();

    // Create a simple tridiagonal matrix
    const size_type nnz = 3 * N - 2;
    Kokkos::View<size_type*> row_map("row_map", N + 1);
    Kokkos::View<lno_t*> entries("entries", nnz);
    Kokkos::View<scalar_t*> values("values", nnz);

    auto row_map_h = Kokkos::create_mirror_view(row_map);
    auto entries_h = Kokkos::create_mirror_view(entries);
    auto values_h = Kokkos::create_mirror_view(values);

    // Fill matrix on host
    size_type nnz_count = 0;
    for (int i = 0; i < N; i++) {
        row_map_h(i) = nnz_count;
        if (i > 0) { entries_h(nnz_count) = i - 1; values_h(nnz_count++) = -1.0; }
        entries_h(nnz_count) = i; values_h(nnz_count++) = 2.0;
        if (i < N - 1) { entries_h(nnz_count) = i + 1; values_h(nnz_count++) = -1.0; }
    }
    row_map_h(N) = nnz_count;

    // Copy to device
    Kokkos::deep_copy(row_map, row_map_h);
    Kokkos::deep_copy(entries, entries_h);
    Kokkos::deep_copy(values, values_h);

    // Create matrix
    auto A = KokkosSparse::CrsMatrix<scalar_t, lno_t, exec_space>
        ("A", N, N, nnz, values, row_map, entries);

    // Create vectors
    Kokkos::View<scalar_t*> x("x", N);
    Kokkos::View<scalar_t*> b("b", N);

    // Initialize solver handle
    handle_t handle;
    handle.create_gs_handle(KokkosSparse::GS_DEFAULT);

    // Preconditioner setup
    KokkosSparse::Experimental::gauss_seidel_symbolic
        (&handle, N, N, A.graph.row_map, A.graph.entries, false);
    KokkosSparse::Experimental::gauss_seidel_numeric
        (&handle, N, N, A.graph.row_map, A.graph.entries, A.values, false);

    Kokkos::fence();
    auto assembly_end = std::chrono::high_resolution_clock::now();
    result.assembly_time = std::chrono::duration<double>(assembly_end - assembly_start).count();

    // Time stepping loop
    Kokkos::fence();
    auto timestep_start = std::chrono::high_resolution_clock::now();

    double total_spmv_time = 0.0;
    double total_solve_time = 0.0;

    for (int t = 0; t < timesteps; t++) {
        // SpMV operation
        Kokkos::fence();
        auto spmv_start = std::chrono::high_resolution_clock::now();

        KokkosSparse::spmv("N", 1.0, A, x, 0.0, b);

        Kokkos::fence();
        auto spmv_end = std::chrono::high_resolution_clock::now();
        total_spmv_time += std::chrono::duration<double>(spmv_end - spmv_start).count();

        // Solve step
        Kokkos::fence();
        auto solve_start = std::chrono::high_resolution_clock::now();

        bool is_first = (t == 0);
        KokkosSparse::Experimental::forward_sweep_gauss_seidel_apply
            (&handle, N, N, A.graph.row_map, A.graph.entries, A.values,
             x, b, is_first, is_first, 1.0, 1);

        Kokkos::fence();
        auto solve_end = std::chrono::high_resolution_clock::now();
        total_solve_time += std::chrono::duration<double>(solve_end - solve_start).count();
    }

    Kokkos::fence();
    auto timestep_end = std::chrono::high_resolution_clock::now();

    handle.destroy_gs_handle();

    Kokkos::fence();
    auto total_end = std::chrono::high_resolution_clock::now();
    result.total_time = std::chrono::duration<double>(total_end - total_start).count();
    result.spmv_time = total_spmv_time;
    result.solve_time = total_solve_time;

    return result;
}

void test_scheme_benchmark() {
    Kokkos::initialize();
    {
        std::cout << "Running scheme benchmarks on: " << typeid(exec_space).name() << "\n\n";

        // Define problem sizes and timesteps
        std::vector<size_type> problem_sizes = {1000, 5000, 10000, 50000, 100000};
        std::vector<int> timesteps_list = {10, 50, 100};

        // Print table header
        std::cout << std::setw(10) << "Size"
                  << std::setw(10) << "Timesteps"
                  << std::setw(15) << "Total Time (s)"
                  << std::setw(15) << "Assembly (s)"
                  << std::setw(15) << "SpMV (s)"
                  << std::setw(15) << "Solve (s)"
                  << "\n";
        std::cout << std::string(80, '-') << "\n";

        for (auto N : problem_sizes) {
            for (auto timesteps : timesteps_list) {
                auto result = run_scheme_benchmark(N, timesteps);

                std::cout << std::setw(10) << N
                          << std::setw(10) << timesteps
                          << std::setw(15) << std::fixed << std::setprecision(3) << result.total_time
                          << std::setw(15) << result.assembly_time
                          << std::setw(15) << result.spmv_time
                          << std::setw(15) << result.solve_time
                          << "\n";
            }
        }
    }
    Kokkos::finalize();
}

void heston_adi() {
    test_benchmark();
    //test_scheme();
    //test_scheme_benchmark();
    //test_simple_tridiagonal_solver();
}


