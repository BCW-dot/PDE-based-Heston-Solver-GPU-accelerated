#include "mat_factory.hpp"
#include "coeff.hpp"


//this shoudl be a sequential implemntaion but somehow it is rly fast
void test_tridiagonal_matrixfree() {
    using timer = std::chrono::high_resolution_clock;

    // Problem size
    const int N = 30000;
    std::cout<< "D simsize: " << N << std::endl;
    
    // Create vectors for diagonal structure instead of full matrix
    Kokkos::View<double*> diag("diagonal", N);        // Main diagonal (2.0)
    Kokkos::View<double*> lower("lower_diag", N-1);   // Lower diagonal (-1.0)
    Kokkos::View<double*> upper("upper_diag", N-1);   // Upper diagonal (-1.0)
    Kokkos::View<double*> x("solution", N);
    Kokkos::View<double*> b("rhs", N);
    Kokkos::View<double*> temp("temp", N);
    
    // Initialize diagonals
    Kokkos::parallel_for("init_diags", N, KOKKOS_LAMBDA(const int i) {
        diag(i) = 2;
        if(i < N-1) {
            lower(i) = -1;
            upper(i) = -1;
        }
    });

    // Initialize with random values
    auto h_b = Kokkos::create_mirror_view(b);
    for (int i = 0; i < N; ++i) {
        h_b(i) = std::rand() / (RAND_MAX + 1.0);
    }
    Kokkos::deep_copy(b, h_b);

    Kokkos::deep_copy(x, 0.0);

    // Copy b to b_copy for residual calculation later
    Kokkos::View<double*> b_copy("b_copy", N);
    Kokkos::deep_copy(b_copy, b);

    // In test_tridiagonal_matrixfree:
    for(int i = 0; i < 5; i++) {
        auto t_start = timer::now();
        // Sequential Thomas algorithm on device
        Kokkos::parallel_for("tridiagonal_solve", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,1), KOKKOS_LAMBDA(const int) {
            // Forward sweep
            temp(0) = diag(0);
            x(0) = b(0);
            for(int i = 1; i < N; ++i) {
                double w = lower(i-1) / temp(i-1);
                temp(i) = diag(i) - w * upper(i-1);
                x(i) = b(i) - w * x(i-1);
            }

            // Backward substitution
            x(N-1) = x(N-1) / temp(N-1);
            for(int i = N-2; i >= 0; --i) {
                x(i) = (x(i) - upper(i) * x(i+1)) / temp(i);
            }
        });
        auto t_end = timer::now();
        std::cout << "Run " << i << " solve time: "
                << std::chrono::duration<double>(t_end - t_start).count()
                << " seconds\n";
    }

    // Verify solution by computing residual matrix-free
    double residual = 0.0;
    Kokkos::parallel_reduce("residual", N, KOKKOS_LAMBDA(const int i, double& update) {
        double r_i = b_copy(i) - (diag(i) * x(i) + 
                              (i > 0 ? lower(i-1) * x(i-1) : 0.0) +
                              (i < N-1 ? upper(i) * x(i+1) : 0.0));
        update += r_i * r_i;
    }, residual);

    std::cout << "Residual norm: " << std::sqrt(residual) << std::endl;
}

//this test the above but with a random b as inpout
void test_tridiagonal_matrixfree_with_random_b() {
    using timer = std::chrono::high_resolution_clock;

    // Problem size
    const int N = 1326;

    // Create vectors
    Kokkos::View<double*> diag("diag", N);
    Kokkos::View<double*> lower("lower", N - 1);
    Kokkos::View<double*> upper("upper", N - 1);
    Kokkos::View<double*> x("x", N);
    Kokkos::View<double*> b("b", N);
    Kokkos::View<double*> temp("temp", N);

    // Initialize diagonals
    Kokkos::parallel_for("init_diagonals", N, KOKKOS_LAMBDA(const int i) {
        diag(i) = 2.0;
        if (i < N - 1) {
            lower(i) = -1.0;
            upper(i) = -1.0;
        }
    });

    // Initialize b with random values
    auto h_b = Kokkos::create_mirror_view(b);
    for (int i = 0; i < N; ++i) {
        h_b(i) = std::rand() / (RAND_MAX + 1.0);
    }
    Kokkos::deep_copy(b, h_b);

    // Copy b to b_copy for residual calculation
    Kokkos::View<double*> b_copy("b_copy", N);
    Kokkos::deep_copy(b_copy, b);
    
    auto t_start = timer::now();

    // Sequential Thomas algorithm on device
    Kokkos::parallel_for("tridiagonal_solve", Kokkos::RangePolicy<>(0, 1), KOKKOS_LAMBDA(const int) {
        // Forward sweep
        temp(0) = diag(0);
        x(0) = b(0);
        for (int i = 1; i < N; ++i) {
            double m = lower(i - 1) / temp(i - 1);
            temp(i) = diag(i) - m * upper(i - 1);
            x(i) = b(i) - m * x(i - 1);
        }
        // Backward substitution
        x(N - 1) /= temp(N - 1);
        for (int i = N - 2; i >= 0; --i) {
            x(i) = (x(i) - upper(i) * x(i + 1)) / temp(i);
        }
    });

    
    auto t_end = timer::now();

    std::cout << "Total solve time: "
              << std::chrono::duration<double>(t_end - t_start).count()
              << " seconds" << std::endl;

    // Compute Residual
    double residual = 0.0;
    Kokkos::parallel_reduce("compute_residual", N, KOKKOS_LAMBDA(const int i, double& update) {
        double sum = diag(i) * x(i);
        if (i > 0) sum += lower(i - 1) * x(i - 1);
        if (i < N - 1) sum += upper(i) * x(i + 1);
        double r_i = b_copy(i) - sum;
        update += r_i * r_i;
    }, residual);

    std::cout << "Residual norm: " << std::sqrt(residual) << std::endl;
}

//this aims at a quick parallel tridigonal solver, but it isnt working. The dimesnion for PCR to be faster than se
//sequential thomas needs to be a lot bigger
void kokkos_cyclic_reduction_solver() {
    const int N = 1024;  // Must be 2^k + 1 for some k
    using timer = std::chrono::high_resolution_clock;
    using Layout = Kokkos::LayoutRight;
    
    // Allocate arrays - note we use N-1 for lower/upper diagonals
    Kokkos::View<double*, Layout> diag("diag", N);
    Kokkos::View<double*, Layout> lower("lower", N-1);
    Kokkos::View<double*, Layout> upper("upper", N-1);
    Kokkos::View<double*, Layout> b("b", N);
    Kokkos::View<double*, Layout> x("x", N);

    // Initialize matrix - classic -1, 2, -1 pattern
    Kokkos::parallel_for("init", N, KOKKOS_LAMBDA(const int i) {
        diag(i) = 2.0;
        b(i) = 1.0;
        x(i) = 0.0;
        
        if(i < N-1) {
            lower(i) = -1.0;
            upper(i) = -1.0;
        }
    });

    // Print diagonal dominance check
    double max_ratio = 0.0;
    Kokkos::parallel_reduce("check_diags", N-1, KOKKOS_LAMBDA(const int i, double& max_val) {
        double sum = std::abs(lower(i)) + std::abs(upper(i));
        double ratio = sum / std::abs(diag(i));
        max_val = max(max_val, ratio);
    }, Kokkos::Max<double>(max_ratio));
    printf("Diagonal dominance ratio: %f (should be < 1.0)\n", max_ratio);

    // Allocate reduction levels
    int num_levels = static_cast<int>(std::log2(N-1));
    Kokkos::View<double**, Layout> level_diag("level_diag", num_levels+1, N);
    Kokkos::View<double**, Layout> level_lower("level_lower", num_levels+1, N);
    Kokkos::View<double**, Layout> level_upper("level_upper", num_levels+1, N);
    Kokkos::View<double**, Layout> level_rhs("level_rhs", num_levels+1, N);

    // Initialize level 0
    Kokkos::parallel_for("init_level0", N, KOKKOS_LAMBDA(const int i) {
        level_diag(0,i) = diag(i);
        level_rhs(0,i) = b(i);
        if(i < N-1) {
            level_lower(0,i) = lower(i);
            level_upper(0,i) = upper(i);
        }
    });

    auto t_start = timer::now();

    // Forward reduction
    for(int level = 0; level < num_levels; level++) {
        int stride = 1 << level;
        int num_elements = (N-1) / (2 * stride);
        
        // Print diagnostic for first level
        if(level == 0) {
            double max_diag = 0.0;
            Kokkos::parallel_reduce("check_level0", N, KOKKOS_LAMBDA(const int i, double& max_val) {
                max_val = max(max_val, std::abs(level_diag(0,i)));
            }, Kokkos::Max<double>(max_diag));
            printf("Level 0 max diagonal: %e\n", max_diag);
        }

        Kokkos::parallel_for("reduction", num_elements, KOKKOS_LAMBDA(const int idx) {
            const int i = (2 * idx + 1) * stride;
            if(i < N-1) {  // Important boundary check
                const double pivot_m = level_diag(level,i-stride);
                const double pivot_p = level_diag(level,i+stride);
                
                if(std::abs(pivot_m) > 1e-14 && std::abs(pivot_p) > 1e-14) {  // Avoid division by zero
                    const double alpha = level_lower(level,i) / pivot_m;
                    const double beta = level_upper(level,i) / pivot_p;
                    
                    level_diag(level+1,i) = level_diag(level,i) - 
                        alpha * level_upper(level,i-stride) - 
                        beta * level_lower(level,i+stride);
                    
                    level_rhs(level+1,i) = level_rhs(level,i) - 
                        alpha * level_rhs(level,i-stride) - 
                        beta * level_rhs(level,i+stride);
                    
                    level_lower(level+1,i) = -alpha * level_lower(level,i-stride);
                    level_upper(level+1,i) = -beta * level_upper(level,i+stride);
                }
            }
        });
    }

    // Solve reduced system
    Kokkos::parallel_for("solve_root", 1, KOKKOS_LAMBDA(const int) {
        const int mid = N/2;
        if(std::abs(level_diag(num_levels,mid)) > 1e-14) {
            x(mid) = level_rhs(num_levels,mid) / level_diag(num_levels,mid);
        }
    });

    // Back substitution
    for(int level = num_levels-1; level >= 0; level--) {
        int stride = 1 << level;
        int num_elements = (N-1) / (2 * stride);
        
        Kokkos::parallel_for("back_sub", num_elements, KOKKOS_LAMBDA(const int idx) {
            const int i = (2 * idx + 1) * stride;
            if(i < N-1) {
                const double denom = level_diag(level,i);
                if(std::abs(denom) > 1e-14) {
                    x(i) = (level_rhs(level,i) - 
                           level_lower(level,i) * x(i-stride) - 
                           level_upper(level,i) * x(i+stride)) / denom;
                }
            }
        });
    }

    auto t_end = timer::now();
    double solve_time = std::chrono::duration<double>(t_end - t_start).count();
    
    // Compute residual
    double residual = 0.0;
    double b_norm = 0.0;

    Kokkos::parallel_reduce("b_norm", N, KOKKOS_LAMBDA(const int i, double& update) {
        update += b(i) * b(i);
    }, b_norm);

    Kokkos::parallel_reduce("residual", N, KOKKOS_LAMBDA(const int i, double& update) {
        double Ax_i = diag(i) * x(i);
        if(i > 0) Ax_i += lower(i-1) * x(i-1);
        if(i < N-1) Ax_i += upper(i) * x(i+1);
        const double diff = b(i) - Ax_i;
        update += diff * diff;
    }, residual);

    double relative_residual = std::sqrt(residual) / std::sqrt(b_norm);

    printf("Solver Statistics:\n");
    printf("Matrix size: %d x %d\n", N, N);
    printf("Solve time: %f seconds\n", solve_time);
    printf("Absolute residual: %e\n", std::sqrt(residual));
    printf("Relative residual: %e\n", relative_residual);
}

//this is a try to "parallize" the thomas algorithm. Where we are solving
//multiple tridigonal systems in parallel. This will become important for
//calibration the heston model
void solve_multiple_tridiagonal_systems() {
    using timer = std::chrono::high_resolution_clock;

    // Problem size and number of systems to solve
    const int N = 30000;          // Size of each system
    const int NUM_SYSTEMS = 1024; //best chosen as multipls of warp sizes (32)    // Number of systems to solve simultaneously

    // Create vectors - Note the additional dimension for multiple systems
    Kokkos::View<double*> diag("diag", N);                    // Shared diagonal
    Kokkos::View<double*> lower("lower", N-1);               // Shared lower diagonal
    Kokkos::View<double*> upper("upper", N-1);               // Shared upper diagonal
    
    // Arrays for multiple systems
    Kokkos::View<double**> x("x", NUM_SYSTEMS, N);           // Solutions
    Kokkos::View<double**> b("b", NUM_SYSTEMS, N);           // Right-hand sides
    Kokkos::View<double**> temp("temp", NUM_SYSTEMS, N);     // Temporary storage

    // Initialize shared diagonals
    Kokkos::parallel_for("init_diagonals", N, KOKKOS_LAMBDA(const int i) {
        diag(i) = 2.0;
        if (i < N-1) {
            lower(i) = -1.0;
            upper(i) = -1.0;
        }
    });

    // Initialize b with different random values for each system
    auto h_b = Kokkos::create_mirror_view(b);
    for(int sys = 0; sys < NUM_SYSTEMS; ++sys) {
        for(int i = 0; i < N; ++i) {
            h_b(sys, i) = std::rand() / (RAND_MAX + 1.0);
        }
    }
    Kokkos::deep_copy(b, h_b);

    // Copy b to b_copy for residual calculation
    Kokkos::View<double**> b_copy("b_copy", NUM_SYSTEMS, N);
    Kokkos::deep_copy(b_copy, b);

    auto t_start = timer::now();

    // Solve multiple systems in parallel
    // Each system is solved sequentially, but multiple systems are solved simultaneously
    Kokkos::parallel_for("solve_multiple_systems", NUM_SYSTEMS, KOKKOS_LAMBDA(const int sys) {
        // Forward sweep for system 'sys'
        temp(sys, 0) = diag(0);
        x(sys, 0) = b(sys, 0);
        
        for(int i = 1; i < N; ++i) {
            double m = lower(i-1) / temp(sys, i-1);
            temp(sys, i) = diag(i) - m * upper(i-1);
            x(sys, i) = b(sys, i) - m * x(sys, i-1);
        }

        // Backward substitution for system 'sys'
        x(sys, N-1) /= temp(sys, N-1);
        for(int i = N-2; i >= 0; --i) {
            x(sys, i) = (x(sys, i) - upper(i) * x(sys, i+1)) / temp(sys, i);
        }
    });

    auto t_end = timer::now();

    // Compute residuals for all systems
    Kokkos::View<double*> residuals("residuals", NUM_SYSTEMS);
    Kokkos::parallel_for("compute_residuals", 
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {NUM_SYSTEMS,N}),
        KOKKOS_LAMBDA(const int sys, const int i) {
            double sum = diag(i) * x(sys, i);
            if(i > 0) sum += lower(i-1) * x(sys, i-1);
            if(i < N-1) sum += upper(i) * x(sys, i+1);
            double r_i = b_copy(sys, i) - sum;
            Kokkos::atomic_add(&residuals(sys), r_i * r_i);
        }
    );

    // Output results
    auto h_residuals = Kokkos::create_mirror_view(residuals);
    Kokkos::deep_copy(h_residuals, residuals);
    
    std::cout << "Total solve time for " << NUM_SYSTEMS << " systems: "
              << std::chrono::duration<double>(t_end - t_start).count()
              << " seconds" << std::endl;
              
    for(int sys = 0; 10*sys < NUM_SYSTEMS; ++sys) {
        std::cout << "System " << 10*sys << " residual norm: " 
                  << std::sqrt(h_residuals(sys)) << std::endl;
    }
}

//this implements a tridiagonla matrix vector product sequantially on the 
//gpu using one thread
void test_tridiagonal_matvec_with_random_x() {
    using timer = std::chrono::high_resolution_clock;

    // Problem size
    const int N = 30000;

    // Create vectors
    Kokkos::View<double*> diag("diag", N);
    Kokkos::View<double*> lower("lower", N - 1);
    Kokkos::View<double*> upper("upper", N - 1);
    Kokkos::View<double*> x("x", N);
    Kokkos::View<double*> y("y", N);

    // Initialize diagonals
    Kokkos::parallel_for("init_diagonals", N, KOKKOS_LAMBDA(const int i) {
        diag(i) = 2.0;
        if (i < N - 1) {
            lower(i) = -1.0;
            upper(i) = -1.0;
        }
    });

    // Initialize x with random values
    auto h_x = Kokkos::create_mirror_view(x);
    for (int i = 0; i < N; ++i) {
        h_x(i) = std::rand() / (RAND_MAX + 1.0);
    }
    Kokkos::deep_copy(x, h_x);

    auto t_start = timer::now();

    
    // Version 1: Sequential on GPU (single thread)
    Kokkos::parallel_for("tridiagonal_matvec_seq", Kokkos::RangePolicy<>(0, 1), KOKKOS_LAMBDA(const int) {
        // First element
        y(0) = diag(0) * x(0) + upper(0) * x(1);
        
        // Middle elements
        for (int i = 1; i < N-1; ++i) {
            y(i) = lower(i-1) * x(i-1) + diag(i) * x(i) + upper(i) * x(i+1);
        }
        
        // Last element
        y(N-1) = lower(N-2) * x(N-2) + diag(N-1) * x(N-1);
    });
    

    /*
    // Version 2: Parallel version using parallel_for
    Kokkos::parallel_for("tridiagonal_matvec_par", 32, KOKKOS_LAMBDA(const int i) {
        double sum = diag(i) * x(i);
        if (i > 0) sum += lower(i-1) * x(i-1);
        if (i < N-1) sum += upper(i) * x(i+1);
        y(i) = sum;
    });
    */

    Kokkos::fence();
    auto t_end = timer::now();

    std::cout << "Total matrix-vector time: "
              << std::chrono::duration<double>(t_end - t_start).count()
              << " seconds" << std::endl;

    // Calculate norm of result
    double norm = 0.0;
    Kokkos::parallel_reduce("compute_norm", N, KOKKOS_LAMBDA(const int i, double& update) {
        update += y(i) * y(i);
    }, norm);

    std::cout << "Norm of result vector: " << std::sqrt(norm) << std::endl;
}



/*

Here begins the translation of python to Kokkos code. The following has to be done:
For the matrices A1 and A2 implicit and explicit solvers need to be written
For the matrices A0 and A explicit solvers need to be writtten

For all matrices a struct and a constructor/memeroy allocation system needs to be developed

*/



/*

This is a first try at the A0 matrix, it is workign now. I have decided on a 2D flattend storage 
format. so we have a 2D array of dim m2-1x(m1-1)*9 the 9 comes form the l and k loops. Since for each stock price 
level we get 9 values. This memeory layout is also good for parallisaing wrt variance levels since the memeroy is 
continuously stored. The same idea could and should be applied to A1, where i first started wiht the implementation,
but stopped since i ran into a few errors. 
The below function implements 3 multiply methods, one sequential, one 2D parallisation (varaince and stock) and one with 
1D parallisation (varaince)

*/
struct A0Storage_gpu {
    int m1, m2;
    Kokkos::View<double**> values; // [m2 - 1][(m1 - 1) * 9]

    A0Storage_gpu(int m1_in, int m2_in) : m1(m1_in), m2(m2_in) {
        // Allocate the values View
        values = Kokkos::View<double**>("A0_values", m2 - 1, (m1 - 1) * 9);
    }

    // Initialize the matrix with the simple pattern: lower = 1, main = 1, upper = 1
    void initialize_simple_pattern() {
        auto values_host = Kokkos::create_mirror_view(values);

        for (int j = 0; j < m2 - 1; ++j) {
            for (int i = 0; i < m1 - 1; ++i) {
                // Each of the three tridiagonal matrices
                for (int diag = 0; diag < 3; ++diag) {
                    int val_idx_base = i * 9 + diag * 3;
                    values_host(j, val_idx_base + 0) = 1.0; // Lower diagonal
                    values_host(j, val_idx_base + 1) = 1.0; // Main diagonal
                    values_host(j, val_idx_base + 2) = 1.0; // Upper diagonal
                }
            }
        }

        // Copy to device
        Kokkos::deep_copy(values, values_host);
    }

    // Sequential multiply method (runs on one thread on the GPU)
    void multiply_seq(const Kokkos::View<double*>& x, Kokkos::View<double*>& result) const {
        int total_size = x.size();
        int m1_ = m1;
        int m2_ = m2;
        auto values_ = values;
        auto x_ = x;
        auto result_ = result;

        // Run on a single thread
        Kokkos::parallel_for("A0_multiply_seq", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, 1), KOKKOS_LAMBDA(const int&) {
            for (int j = 0; j < m2_ - 1; ++j) {
                int row_offset = (j + 1) * (m1_ + 1);
                for (int i = 0; i < m1_ - 1; ++i) {
                    double sum = 0.0;
                    for (int l = -1; l <= 1; ++l) {
                        for (int k = -1; k <= 1; ++k) {
                            int val_idx = i * 9 + (l + 1) * 3 + (k + 1);
                            int col_idx = (i + 1 + k) + (j + 1 + l) * (m1_ + 1);
                            if (col_idx >= 0 && col_idx < total_size) {
                                sum += values_(j, val_idx) * x_(col_idx);
                            }
                        }
                    }
                    result_(row_offset + i + 1) += sum;
                }
            }
        });
        Kokkos::fence();
    }

    // Parallel multiply method (parallelizes over variance levels and stock price levels)
    void multiply_parallel(const Kokkos::View<double*>& x, Kokkos::View<double*>& result) const {
        int total_size = x.size();
        int m1_ = m1;
        int m2_ = m2;
        auto values_ = values;
        auto x_ = x;
        auto result_ = result;

        Kokkos::parallel_for("A0_multiply_parallel", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {m2_ - 1, m1_ - 1}), KOKKOS_LAMBDA(const int j, const int i) {
            int row_offset = (j + 1) * (m1_ + 1);
            double sum = 0.0;
            for (int l = -1; l <= 1; ++l) {
                for (int k = -1; k <= 1; ++k) {
                    int val_idx = i * 9 + (l + 1) * 3 + (k + 1);
                    int col_idx = (i + 1 + k) + (j + 1 + l) * (m1_ + 1);
                    if (col_idx >= 0 && col_idx < total_size) {
                        sum += values_(j, val_idx) * x_(col_idx);
                    }
                }
            }
            result_(row_offset + i + 1) += sum;
        });
        Kokkos::fence();
    }

    void multiply_parallel_over_j(const Kokkos::View<double*>& x, Kokkos::View<double*>& result) const {
        int total_size = x.size();
        int m1_ = m1;
        int m2_ = m2;
        auto values_ = values;
        auto x_ = x;
        auto result_ = result;

        // Parallelize over variance levels (j)
        Kokkos::parallel_for("A0_multiply_parallel_over_j", Kokkos::RangePolicy<>(0, m2_ - 1), KOKKOS_LAMBDA(const int j) {
            int row_offset = (j + 1) * (m1_ + 1);
            for (int i = 0; i < m1_ - 1; ++i) {
                double sum = 0.0;
                for (int l = -1; l <= 1; ++l) {
                    for (int k = -1; k <= 1; ++k) {
                        int val_idx = i * 9 + (l + 1) * 3 + (k + 1);
                        int col_idx = (i + 1 + k) + (j + 1 + l) * (m1_ + 1);
                        if (col_idx >= 0 && col_idx < total_size) {
                            sum += values_(j, val_idx) * x_(col_idx);
                        }
                    }
                }
                result_(row_offset + i + 1) += sum;
            }
        });
        Kokkos::fence();
    }

};

void test_A0_gpu() {
    using timer = std::chrono::high_resolution_clock;

    int m1 = 300; // Number of stock price steps
    int m2 = 100; // Number of variance steps
    int total_size = (m1 + 1) * (m2 + 1);
    std::cout << "Total size: " << total_size << std::endl;

    // Initialize A0Storage_gpu
    A0Storage_gpu A0(m1, m2);
    A0.initialize_simple_pattern();

    // Initialize x with its indices
    Kokkos::View<double*> x("x", total_size);
    Kokkos::parallel_for("init_x", total_size, KOKKOS_LAMBDA(const int idx) {
        x(idx) = static_cast<double>(idx);
    });

    // Initialize result vector with zeros
    Kokkos::View<double*> result("result", total_size);
    Kokkos::deep_copy(result, 0.0);

    // Perform matrix-vector multiplication (choose sequential or parallel)
    // Sequential version
    /*
    auto t_start_multip = timer::now();
    A0.multiply_seq(x, result);
    auto t_end_multip = timer::now();

    std::cout << "A0 Explicit sequential: "
              << std::chrono::duration<double>(t_end_multip - t_start_multip).count()
              << " seconds" << std::endl;
    */

    // Parallel version (2D parallel, in variance and stock prices)
    
    
    auto t_start_multip = timer::now();
    A0.multiply_parallel(x, result);;
    auto t_end_multip = timer::now();

    std::cout << "A0 Explicit 2D parallel: "
              << std::chrono::duration<double>(t_end_multip - t_start_multip).count()
              << " seconds" << std::endl;
    

    /*
    auto t_start_multip = timer::now();
    A0.multiply_parallel_over_j(x, result);;
    auto t_end_multip = timer::now();

    std::cout << "A0 Explicit 1D parallel: "
              << std::chrono::duration<double>(t_end_multip - t_start_multip).count()
              << " seconds" << std::endl;
    */
    // Copy result back to host and print
    
    auto h_result = Kokkos::create_mirror_view(result);
    Kokkos::deep_copy(h_result, result);

    /*
    std::cout << "Result of A0 * x:" << std::endl;
    for (int idx = 0; idx < total_size; ++idx) {
        std::cout << "result[" << idx << "] = " << h_result(idx) << std::endl;
    }
    */
}


/*

This is a cpu test version of the A0 struct. It works perfectly. Tests were done with simple diagonal values
1. all ones vector
2. unique x vector to check indexing

*/


struct A0Storage_cpu {
    int m1, m2;
    std::vector<std::vector<double>> values; // [m2 - 1][(m1 - 1) * 9]

    A0Storage_cpu(int m1_in, int m2_in) : m1(m1_in), m2(m2_in) {
        values.resize(m2 - 1, std::vector<double>((m1 - 1) * 9, 0.0));
    }

    
    void initialize(const std::vector<double>& Vec_s, const std::vector<double>& Vec_v,
                    const std::vector<double>& Delta_s, const std::vector<double>& Delta_v,
                    double rho, double sigma) {
        for (int j = 1; j < m2; ++j) {
            for (int i = 1; i < m1; ++i) {
                double c = rho * sigma * Vec_s[i] * Vec_v[j];
                for (int k = -1; k <= 1; ++k) {
                    for (int l = -1; l <= 1; ++l) {
                        int val_idx = (i - 1) * 9 + (l + 1) * 3 + (k + 1);
                        double beta_s_val = beta_s(i - 1, k, Delta_s);
                        double beta_v_val = beta_v(j - 1, l, Delta_v);
                        values[j - 1][val_idx] = c * beta_s_val * beta_v_val;

                        // Debug print
                        std::cout << "values[" << j - 1 << "][" << val_idx << "] = " << values[j - 1][val_idx] << std::endl;
                    }
                }
            }
        }
    }
    

    // Initialize the matrix with the simple pattern: lower = -1, main = 2, upper = -1
    void initialize_simple_pattern() {
        for (int j = 0; j < m2 - 1; ++j) {
            for (int i = 0; i < m1 - 1; ++i) {
                /*
                // Each of the three tridiagonal matrices
                if(j==0){
                    for (int diag = 0; diag < 3; ++diag) {
                        int val_idx_base = i * 9 + diag * 3;
                        values[j][val_idx_base + 0] = 1.0; // Lower diagonal
                        values[j][val_idx_base + 1] = 1.0;  // Main diagonal
                        values[j][val_idx_base + 2] = 1.0; // Upper diagonal
                    }
                }
                if(j==1){
                    for (int diag = 0; diag < 3; ++diag) {
                        int val_idx_base = i * 9 + diag * 3;
                        values[j][val_idx_base + 0] = 2.0; // Lower diagonal
                        values[j][val_idx_base + 1] = 2.0;  // Main diagonal
                        values[j][val_idx_base + 2] = 2.0; // Upper diagonal
                    }
                }
                if(j==2){
                    for (int diag = 0; diag < 3; ++diag) {
                        int val_idx_base = i * 9 + diag * 3;
                        values[j][val_idx_base + 0] = 3.0; // Lower diagonal
                        values[j][val_idx_base + 1] = 3.0;  // Main diagonal
                        values[j][val_idx_base + 2] = 3.0; // Upper diagonal
                    }
                }
                if(j==3){
                    for (int diag = 0; diag < 3; ++diag) {
                        int val_idx_base = i * 9 + diag * 3;
                        values[j][val_idx_base + 0] = 4.0; // Lower diagonal
                        values[j][val_idx_base + 1] = 4.0;  // Main diagonal
                        values[j][val_idx_base + 2] = 4.0; // Upper diagonal
                    }
                }
                */
                for (int diag = 0; diag < 3; ++diag) {
                    int val_idx_base = i * 9 + diag * 3;
                    values[j][val_idx_base + 0] = 1.0; // Lower diagonal
                    values[j][val_idx_base + 1] = 1.0; // Main diagonal
                    values[j][val_idx_base + 2] = 1.0; // Upper diagonal
                }
            }
        }
    }

    /*
    void multiply(const std::vector<double>& x, std::vector<double>& result) const {
        for (int j = 0; j < m2 - 1; ++j) {
            int row_offset = (j + 1) * (m1 + 1);
            for (int i = 0; i < m1 - 1; ++i) {
                double sum = 0.0;
                for (int l = -1; l <= 1; ++l) {
                    for (int k = -1; k <= 1; ++k) {
                        int val_idx = i * 9 + (l + 1) * 3 + (k + 1);
                        int col_idx = (i + 1 + k) + (j + 1 + l) * (m1 + 1);
                        sum += values[j][val_idx] * x[col_idx];
                    }
                }
                result[row_offset + i + 1] += sum;

                // Debug print
                std::cout << "result[" << row_offset + i + 1 << "] += " << sum << std::endl;
            }
        }
    }
    */
    void multiply(const std::vector<double>& x, std::vector<double>& result) const {
        int total_size = x.size();
        for (int j = 0; j < m2 - 1; ++j) {
            int row_offset = (j + 1) * (m1 + 1);
            for (int i = 0; i < m1 - 1; ++i) {
                double sum = 0.0;
                std::cout << "Computing result[" << row_offset + i + 1 << "]:" << std::endl;
                for (int l = -1; l <= 1; ++l) {
                    for (int k = -1; k <= 1; ++k) {
                        int val_idx = i * 9 + (l + 1) * 3 + (k + 1);
                        int col_idx = (i + 1 + k) + (j + 1 + l) * (m1 + 1);
                        if (col_idx >= 0 && col_idx < total_size) {
                            double val = values[j][val_idx];
                            double x_val = x[col_idx];
                            sum += val * x_val;
                            // Debug print
                            std::cout << "  values[" << j << "][" << val_idx << "] * x[" << col_idx << "] = "
                                    << val << " * " << x_val << " = " << val * x_val << std::endl;
                        }
                    }
                }
                result[row_offset + i + 1] += sum;
                std::cout << "Sum for result[" << row_offset + i + 1 << "] = " << sum << "\n" << std::endl;
            }
        }
    }
};

// Test function to initialize the matrix and perform multiplication
void test_A0_cpu() {
    int m1 = 7; // Number of stock price steps
    int m2 = 7; // Number of variance steps
    int total_size = (m1 + 1) * (m2 + 1);

    // Initialize A0StorageSeq
    A0Storage_cpu A0(m1, m2);
    A0.initialize_simple_pattern();

    
    std::cout << "\nInitilized values:\n";
    std::cout << "";
    for(int j = 0; j < m2-1; j++){
        std::cout << "Super Block " << j << std::endl;
        for(int i = 0; i < (m1-1)*9; i++) std::cout << "[" << i << "] = " << A0.values[j][i] << ",";
        std::cout << "\n";
    }
    

    // Create a vector x filled with ones
    //std::vector<double> x(total_size, 1.0);

    // Initialize x with its indices
    std::vector<double> x(total_size);
    for (int idx = 0; idx < total_size; ++idx) {
        x[idx] = static_cast<double>(idx);
    }


    // Initialize result vector with zeros
    std::vector<double> result(total_size, 0.0);

    
    // Perform matrix-vector multiplication
    A0.multiply(x, result);

    // Print the result
    std::cout << "Result of A0 * x:" << std::endl;
    for (int idx = 0; idx < total_size; ++idx) {
        std::cout << "result[" << idx << "] = " << result[idx] << ", ";
    }
    
}


/*

This is a A2 oszillation test on the cpu. The first A2 test struct is right below. You should start reading
the below version first. This one is a test struct to start applying a forward scheme to reduce oszilatory 
behavior for larger variance levels and small vol of vol terms. For this a fourth diagonal appears at the end


THIS TEST WAS FIRST ONLY Written for dim sizes m1=5, m2=20. To make sure we have at least two variance levels
of stock prices appearing in the lower2 diagonal.
*/

struct A2_oszilation_cpu {
    int m1, m2;
    
    // Explicit system diagonals
    std::vector<double> main_diag;     // (m2-1)*(m1+1)
    std::vector<double> lower_diag;    // (m2-2)*(m1+1)
    std::vector<double> upper_diag;    // (m2-1)*(m1+1)
    std::vector<double> upper2_diag;   // m1+1 (for j=0 special case)

    // New optional diagonal
    std::vector<double> lower2_diag;   // (for vec_v[j]>1.0 special case)

    // Implicit system diagonals
    std::vector<double> implicit_main_diag;   // (m2+1)*(m1+1)
    std::vector<double> implicit_lower_diag;  // (m2-2)*(m1+1)
    std::vector<double> implicit_upper_diag;  // (m2-1)*(m1+1)
    std::vector<double> implicit_upper2_diag; // m1+1

    // New implicit diagonal for lower2
    std::vector<double> implicit_lower2_diag;

    A2_oszilation_cpu(int m1_, int m2_) : m1(m1_), m2(m2_) {
        main_diag.resize((m2-1)*(m1+1));
        lower_diag.resize((m2-2)*(m1+1));
        upper_diag.resize((m2-1)*(m1+1));
        upper2_diag.resize(m1+1);
        // lower2_diag should be empty by default, for debugging this was changed
        lower2_diag.resize(2*(m1+1));

        implicit_main_diag.resize((m2+1)*(m1+1));
        implicit_lower_diag.resize((m2-2)*(m1+1));
        implicit_upper_diag.resize((m2-1)*(m1+1));
        implicit_upper2_diag.resize(m1+1);
        // implicit_lower2_diag will be resized after we know how many entries we have
        implicit_lower2_diag.resize(2*(m1+1));

    }

    void build_implicit(const double theta, const double delta_t) {
        // Init implicit_main_diag with identity
        std::fill(implicit_main_diag.begin(), implicit_main_diag.end(), 1.0);

        // Modify diagonals where explicit matrix is defined
        for(int i = 0; i < (m2-1)*(m1+1); i++) {
            implicit_main_diag[i] -= theta * delta_t * main_diag[i];
        }
        
        for(int i = 0; i < (m2-2)*(m1+1); i++) {
            implicit_lower_diag[i] = -theta * delta_t * lower_diag[i];
        }
        
        for(int i = 0; i < (m2-1)*(m1+1); i++) {
            implicit_upper_diag[i] = -theta * delta_t * upper_diag[i];
        }
        
        for(int i = 0; i < m1+1; i++) {
            implicit_upper2_diag[i] = -theta * delta_t * upper2_diag[i];
        }

        for(int i = 0; i < lower2_diag.size(); i++) {
            implicit_lower2_diag[i] = -theta * delta_t * lower2_diag[i];
        }
    }

    void multiply(const std::vector<double>& x, std::vector<double>& result) {
        //result.resize(x.size(), 0.0);
        const int spacing = m1 + 1;
        
        // First block (j=0)
        for(int i = 0; i < spacing; i++) {
            result[i] = main_diag[i] * x[i];
            result[i] += upper_diag[i] * x[i + spacing];
            result[i] += upper2_diag[i] * x[i + 2*spacing];
            //if(i < spacing) {
                //result[i] += upper_diag[i] * x[i + spacing];
                //result[i] += upper2_diag[i] * x[i + 2*spacing];
            //}
        }

        // Middle blocks
        for(int j = 1; j < m2-1 - 2; j++) {
            for(int i = 0; i < spacing; i++) {
                int idx = j*spacing + i;
                result[idx] = main_diag[idx] * x[idx];
                result[idx] += lower_diag[idx-spacing] * x[idx-spacing];
                result[idx] += upper_diag[idx] * x[idx+spacing];
            }
        }

        int k = 0;
        // End Blocks
        //the two works only for this example here. It would need to change. 
        for(int j = m2-1 - 2; j < m2-1; j++) {
            for(int i = 0; i < spacing; i++) {
                int idx = j*spacing + i;
                int kidx = k*spacing + i;
                result[idx] = main_diag[idx] * x[idx];
                result[idx] += lower_diag[idx-spacing] * x[idx-spacing];
                result[idx] += lower2_diag[kidx] * x[idx-2*(spacing)];
                result[idx] += upper_diag[idx] * x[idx+spacing];
            }
            k +=1;
        }
    }

   
    // CPU implementation
    void solve_implicit(std::vector<double>& x, const std::vector<double>& d) {
        const int spacing = m1 + 1;
        const int num_rows = (m2-1)*spacing;
        const int total_size = (m2+1)*spacing;

        // Temp storage
        std::vector<double> c_star(num_rows); 
        std::vector<double> c2_star(spacing);
        std::vector<double> d_star(total_size);

        //this will be used to account for the changed lower_diag values when eliminating lower2_diag
        //this can be optimized, by setting a_star(lower2_diag.size()), then you need to account for 
        //indexing in the forward sweep where we handle the fourth diagonal
        std::vector<double> a_star(lower_diag.size());

        for(int i = 0; i< lower_diag.size();i++){
            a_star[i] = implicit_lower_diag[i];
        }


        // Identity block 
        for(int i = num_rows; i < total_size; i++) {
            d_star[i] = d[i];
        }
        /*
        std::cout << "\nAfter handling identity block:\n";
        std::cout << "d_star = ";
        for(int i = 0; i < total_size; i++) std::cout << "[" << i << "] = " << d_star[i] << ",";
        std::cout << "\n";
        */

        //normalize the first m1+1 rows. Corresponds to Thomas step c'[0] = c[0]/b[0]
        //also the upper2_diagonal 
        for(int i = 0; i < spacing; i++){
            c_star[i] = implicit_upper_diag[i] / implicit_main_diag[i];
            c2_star[i] = implicit_upper2_diag[i] / implicit_main_diag[i];
            d_star[i] = d[i] / implicit_main_diag[i];
        }

        //First block forward sweep (handle upper2_diag)
        //here spacing needs to be accounted for
        double c_upper = 0; //since we have upper2_diag, we need to account for this in updating
                            //c_star. Usually it would be c_star=c/m, now it is c_star=c'/m, where c'
                            //is changed by upper2_diag. This is done in the first line c_upper = ...
        
        for(int i = 0; i < spacing; i++) { 
            c_upper = implicit_upper_diag[i+spacing] - c2_star[i]*implicit_lower_diag[i];
            double m = 1.0 / 
                        (implicit_main_diag[i+spacing] - c_star[i]*implicit_lower_diag[i]);
            c_star[i+spacing] = c_upper * m;
            d_star[i+spacing] = (d[i+spacing] - implicit_lower_diag[i] * d_star[i]) * m;
        }
        
         /*
        std::cout << "\nD_STAR After first forward:\n";
        std::cout << "d_star = ";
        for(int i = 0; i < total_size; i++) std::cout << "[" << i << "] = " << d_star[i] << ",";
        std::cout << "\n";
        
        std::cout << "\nC_STAR After first forward:\n";
        std::cout << "c_star = ";
        for(int i = 0; i < num_rows; i++) std::cout << "[" << i << "] = " << c_star[i] << ",";
        std::cout << "\n";
        */
        

        //taking care one block in lower_diag. Right above lower2_diag
        for(int i = spacing; i < num_rows - lower2_diag.size() - spacing; i++) {
            double m = 1.0 / 
                        (implicit_main_diag[i+spacing] - c_star[i]*implicit_lower_diag[i]);
            c_star[i+spacing] = implicit_upper_diag[i+spacing] * m;
            d_star[i+spacing] = (d[i+spacing] - implicit_lower_diag[i] * d_star[i]) * m;
        }

        //taking care of the fourth diagonal, needs to be done first. After that we eliminate lower_diag
        int k;
        std::vector<double> new_main;
        for(int i = num_rows - lower2_diag.size() - 2*spacing; i < num_rows - 2*spacing; i++) {
            k = i - (num_rows - lower2_diag.size() - 2*spacing); //starts at zero and increases by 1
            double m = 1.0 / 
                        (implicit_main_diag[i+2*spacing] - c_star[i+spacing]*implicit_lower2_diag[k]);
            new_main.push_back(m);
            //std::cout << m;
            c_star[i+2*spacing] = implicit_upper_diag[i+2*spacing] * m;

            //here we account for the change in the lower_diag values
            //a_star[i+spacing] = a_star[i+spacing] - implicit_lower2_diag[k] * implicit_main_diag[i+spacing];

            //we need to account for the fact that we dont want to overwrite here!
            //we will access d_star index two times since we have four diagonals now
            d_star[i+2*spacing] = (d[i+2*spacing] - implicit_lower2_diag[k] * d_star[i]) * m;
            std::cout << d_star[i+2*spacing] << ", ";
        }
            
        //compute entire "normal" forward sweep
        // Middle blocks forward sweep
        //"i<num_rows - spacing" since we pre backward eliminated the last m1+1 entries of upper diag
        k=0;
        for(int i = num_rows - lower2_diag.size() - spacing; i < num_rows - spacing; i++) {
            double m = 1.0 / 
                        (new_main[k] - c_star[i]*a_star[i]); //implicit_main_diag[i+spacing]
            c_star[i+spacing] = implicit_upper_diag[i+spacing] * m;
            d_star[i+spacing] = (d_star[i+spacing] - a_star[i] * d_star[i]) * m;
            k +=1;
        }

        //then comes backward sweep as usual i think
        /*
        std::cout << "\nD_STAR After second forward:\n";
        std::cout << "d_star = ";
        for(int i = 0; i < total_size; i++) std::cout << "[" << i << "] = " << d_star[i] << ",";
        std::cout << "\n";

        std::cout << "\nC_STAR After second forward:\n";
        std::cout << "c_star = ";
        for(int i = 0; i < num_rows; i++) std::cout << "[" << i << "] = " << c_star[i] << ",";
        std::cout << "\n";
        */

        //pre-backward sweep to eliminate m1+1 last entries of upper diag. Eliminated by
        //identity block
        //here d_star[i+spacing] should be equal to d, since of the identity block pre set
        //at the beginning fo this function

        //correct
        for(int i = num_rows - spacing; i < num_rows; i++){
            d_star[i] = d_star[i] - d_star[i+spacing]*c_star[i];
        }

        /*
        std::cout << "\nD_STAR After pre backward sweep:\n";
        std::cout << "d_star = ";
        for(int i = 0; i < total_size; i++) std::cout << "[" << i << "] = " << d_star[i] << ",";
        std::cout << "\n";
        */

        //backward substitution on last m1+1 rows (only main diagonal present)
        // Last m1+1 rows only have main diagonal
        //this might be wrong with doing this with implic_main_diag
        //maybe the main was changed by the above forward
        //or it is right and the "change" is considered in d_star

        //maybe this shouldnt be devided
        //should be correct, tested with -1 on diago so implicit has 2 on diagonal, rest zero
        for(int i = num_rows - spacing; i < num_rows; i++) {
            x[i] = d_star[i];///implicit_main_diag[i];
        }

        /*
        std::cout << "\nX: After m1+1 row devide:\n";
        std::cout << "x = ";
        for(int i = 0; i < total_size; i++) std::cout << "[" << i << "] = " << x[i] << ",";
        std::cout << "\n";
        */

        //backward sweep until the upper2_diag appears. 
        //is correct
        for(int i = num_rows-1; i >= 3*spacing; i--) {
            x[i-spacing] = d_star[i-spacing] - c_star[i-spacing] * x[i];
        }
        
        /*
        std::cout << "\nX After first backward sweep:\n";
        std::cout << "x = ";
        for(int i = 0; i < total_size; i++) std::cout << "[" << i << "] = " << x[i] << ",";
        std::cout << "\n";
        */

        //First block back substitution with upper2_diag
        //in the second line it is NOT x[i-2*spacing] since this entry will be "changed" again in the last backward 
        //sweep, therefore it has to be d_star which keeps track of changed right hand side values
        for(int i = 3*spacing-1; i >= 2*spacing; i--) {
            x[i-spacing] = d_star[i-spacing] - c_star[i-spacing] * x[i];
            d_star[i-2*spacing] = d_star[i-2*spacing] - c2_star[i-2*spacing] * x[i];
        }

        /*
        std::cout << "\nX After second backward sweep:\n";
        std::cout << "x = ";
        for(int i = 0; i < total_size; i++) std::cout << "[" << i << "] = " << x[i] << ",";
        std::cout << "\n";
        */

        //last backwar substitution after upp2 is gone
        for(int i = 2*spacing-1; i >= spacing; i--) {
            x[i-spacing] = d_star[i-spacing] - c_star[i-spacing] * x[i];
        }

        /*
        std::cout << "\nX After third backward sweep:\n";
        std::cout << "x = ";
        for(int i = 0; i < total_size; i++) std::cout << "[" << i << "] = " << x[i] << ",";
        std::cout << "\n";
        */

        //this shouldnt be done, look at thomas algorithm. The above loop already accesses
        //x[0]
        /*
        //last m1+1 rows are left 
        for(int i = spacing-1; i >= 0; i--) {
            x[i] = d_star[i];
        }

        std::cout << "\nX After first rows:\n";
        std::cout << "x = ";
        for(int i = 0; i < total_size; i++) std::cout << "[" << i << "] = " << x[i] << ",";
        std::cout << "\n";
        */
        
        // Identity block 
        for(int i = num_rows; i < total_size; i++) {
            x[i] = d_star[i];
        }

    }
    

    void solve_implicit2(std::vector<double>& x, const std::vector<double>& d) {
    const int spacing = m1 + 1;
    const int num_rows = (m2 - 1)*spacing;
    const int total_size = (m2 + 1)*spacing;

    // Temporary arrays
    std::vector<double> c_star(num_rows, 0.0); 
    std::vector<double> c2_star(spacing, 0.0);
    std::vector<double> d_star(total_size, 0.0);

    // a_star is a copy of implicit_lower_diag, which may be modified when eliminating lower2_diag
    std::vector<double> a_star(implicit_lower_diag.begin(), implicit_lower_diag.end());

    // Determine where lower2_diag applies. This depends on your logic for when Vec_v[j]>1.0.
    // For example, if lower2_diag covers L rows at the end:
    int L = (int)lower2_diag.size(); // number of rows with lower2_diag
    // Let's assume lower2_diag starts at:
    int start_lower2 = num_rows - L - 2*spacing; 
    // and ends at start_lower2 + L.
    // You must adjust this logic to your actual matrix structure.

    // Identity block (for rows >= num_rows)
    for(int i = num_rows; i < total_size; i++) {
        d_star[i] = d[i];
    }

    // Normalize first block with upper2_diag
    for(int i = 0; i < spacing; i++){
        double denom = implicit_main_diag[i];
        c_star[i] = implicit_upper_diag[i] / denom;
        c2_star[i] = implicit_upper2_diag[i] / denom;
        d_star[i] = d[i] / denom;
    }

    // First block forward elimination (handle upper2_diag)
    for(int i = 0; i < spacing; i++) {
        double c_upper = implicit_upper_diag[i+spacing] - c2_star[i]*implicit_lower_diag[i];
        double m = 1.0 / (implicit_main_diag[i+spacing] - c_star[i]*implicit_lower_diag[i]);
        c_star[i+spacing] = c_upper * m;
        d_star[i+spacing] = (d[i+spacing] - implicit_lower_diag[i]*d_star[i]) * m;
    }

    // Forward elimination in normal region (before lower2_diag region)
    // This goes from i=spacing up to start_lower2 (since below that lower2 applies)
    for(int i = spacing; i < start_lower2; i++) {
        double denom = implicit_main_diag[i+spacing] - c_star[i]*a_star[i];
        double m = 1.0 / denom;
        c_star[i+spacing] = implicit_upper_diag[i+spacing] * m;
        d_star[i+spacing] = (d[i+spacing] - a_star[i]*d_star[i]) * m;
    }

    // Now handle the region with lower2_diag
    // In this region, each row i in [start_lower2, start_lower2+L) has a lower2 connection:
    // main_diag[i+2*spacing], lower2_diag[k], etc.
    // We must first eliminate the lower2 diagonal entries.
    // We'll do a two-stage elimination:
    // 1) Eliminate lower2_diag similar to how we handled upper2_diag at the start.

    for (int offset = 0; offset < L; offset++) {
        int i = start_lower2 + offset;   // current row with lower2
        // lower2_diag[offset] corresponds to row i connecting i to i-2*spacing
        // Adjust the row i+2*spacing (the "pivot" row) similarly:
        
        double denom = implicit_main_diag[i+2*spacing] - c_star[i+spacing]*implicit_lower2_diag[offset];
        double m = 1.0/denom;

        // Update c_star for i+2*spacing using upper_diag
        // If needed, also update something for a potential upper2 in this region if it exists
        c_star[i+2*spacing] = implicit_upper_diag[i+2*spacing]*m;

        // Update d_star for that row
        d_star[i+2*spacing] = (d[i+2*spacing] - implicit_lower2_diag[offset]*d_star[i])*m;

        // Now eliminate lower2 from a_star:
        // a_star is at index i+spacing for the next row referencing i
        // After removing lower2 effect:
        a_star[i+spacing] = a_star[i+spacing] - implicit_lower2_diag[offset]*c_star[i+spacing];
    }

    // After handling lower2_diag elimination, proceed with normal forward elimination again:
    // from i = start_lower2 up to num_rows - spacing (just like normal Thomas)
    for(int i = start_lower2; i < num_rows - spacing; i++) {
        double denom = implicit_main_diag[i+spacing] - c_star[i]*a_star[i];
        double m = 1.0 / denom;
        c_star[i+spacing] = implicit_upper_diag[i+spacing] * m;
        d_star[i+spacing] = (d_star[i+spacing] - a_star[i]*d_star[i]) * m;
    }

    // Pre-backward sweep for last m1+1 rows (identity block)
    for(int i = num_rows - spacing; i < num_rows; i++){
        d_star[i] = d_star[i] - d_star[i+spacing]*c_star[i];
    }

    // Back substitution on last m1+1 rows
    for(int i = num_rows - spacing; i < num_rows; i++) {
        x[i] = d_star[i]; 
    }

    // Backward sweep until upper2_diag appears
    for(int i = num_rows - 1; i >= 3*spacing; i--) {
        x[i - spacing] = d_star[i - spacing] - c_star[i - spacing]*x[i];
    }

    // First block back substitution with upper2_diag
    for(int i = 3*spacing - 1; i >= 2*spacing; i--) {
        x[i - spacing] = d_star[i - spacing] - c_star[i - spacing]*x[i];
        d_star[i - 2*spacing] = d_star[i - 2*spacing] - c2_star[i - 2*spacing]*x[i];
    }

    // Last backward substitution after upper2 is gone
    for(int i = 2*spacing - 1; i >= spacing; i--) {
        x[i - spacing] = d_star[i - spacing] - c_star[i - spacing]*x[i];
    }

    // Identity block
    for(int i = num_rows; i < total_size; i++) {
        x[i] = d_star[i];
    }
}

};

void test_A2_oszilation() {
    using timer = std::chrono::high_resolution_clock;
    // Test dimensions
    const int m1 = 5;
    const int m2 = 20;
    std::cout << "Testing A2 Oszillation CPU implementation with dimensions m1=" << m1 << ", m2=" << m2 << "\n";

    A2_oszilation_cpu A2(m1, m2);

    // Fill explicit diagonals with test values
    std::fill(A2.main_diag.begin(), A2.main_diag.end(), 2.0); //also bad
    std::fill(A2.upper_diag.begin(), A2.upper_diag.end(), 0.0); //upper_diag is bad right now, becuase it also changes the diagonal twice!
    std::fill(A2.upper2_diag.begin(), A2.upper2_diag.end(), 0.0);
    std::fill(A2.lower_diag.begin(), A2.lower_diag.end(), 0.0);
    std::fill(A2.lower2_diag.begin(), A2.lower2_diag.end(), 0.0);

    // Print explicit diagonals
    /*
    std::cout << "\nExplicit diagonals:\n";
    std::cout << "Main diagonal: ";
    for(auto val : A2.main_diag) std::cout << val << " ";
    std::cout << "\nLower diagonal: ";
    for(auto val : A2.lower_diag) std::cout << val << " ";
    std::cout << "\nUpper diagonal: ";
    for(auto val : A2.upper_diag) std::cout << val << " ";
    std::cout << "\nUpper2 diagonal: ";
    for(auto val : A2.upper2_diag) std::cout << val << " ";
    std::cout << "\nLower2 diagonal: ";
    for(auto val : A2.lower2_diag) std::cout << val << " ";
    std::cout << "\n";
    */

    // Build and test implicit system
    double theta = 1.0;
    double delta_t = 1.0;
    A2.build_implicit(theta, delta_t);

    
    // Print implicit diagonals
    /*
    std::cout << "\nImplicit diagonals:\n";
    std::cout << "Main diagonal: ";
    for(auto val : A2.implicit_main_diag) std::cout << val << " ";
    std::cout << "\nLower diagonal: ";
    for(auto val : A2.implicit_lower_diag) std::cout << val << " ";
    std::cout << "\nUpper diagonal: ";
    for(auto val : A2.implicit_upper_diag) std::cout << val << " ";
    std::cout << "\nUpper2 diagonal: ";
    for(auto val : A2.implicit_upper2_diag) std::cout << val << " ";
    std::cout << "\nLower2 diagonal: ";
    for(auto val : A2.implicit_lower2_diag) std::cout << val << " ";
    std::cout << "\n";
    */

    // Test multiply with vector of ones
    /*
    std::vector<double> x((m2+1)*(m1+1), 1.0);
    
    std::vector<double> result((m2+1)*(m1+1), 0.0);
    A2.multiply(x, result);

    std::cout << "\nResult of multiplying with ones:\n";
    for(int i = 0; i < result.size(); i++) {
        std::cout << "[" << i << "] = " << result[i] << ",";
    }
    */
    


    
    // Test implicit solve
    std::vector<double> x((m2+1)*(m1+1), 0); //initial guess
    std::vector<double> b((m2+1)*(m1+1), 5.0);  // RHS = ones

    /*
    std::cout << "\nTesting implicit solve\n";
    std::cout << "Initial x = ";
    for(const auto& val : x) std::cout << val << " ";
    std::cout << "\n";
    std::cout << "RHS b = ";
    for(const auto& val : b) std::cout << val << " ";
    std::cout << "\n";
    */

    auto t_start_multip = timer::now();
    A2.solve_implicit(x, b);
    auto t_end_multip = timer::now();

    std::cout << "Implicit cpu solve time: "
              << std::chrono::duration<double>(t_end_multip - t_start_multip).count()
              << " seconds" << std::endl;

    
    std::cout << "\nImplicit Final Result:\n";
    std::cout << "x = ";
    for(int i = 0; i < (m2+1)*(m1+1); i++) std::cout << "[" << i << "] = " << x[i] << ",";
    std::cout << "\n";
    

    // Verify solution by computing residual
    std::vector<double> temp((m2+1)*(m1+1));
    A2.multiply(x, temp);
    /*
    std::cout << "\nExplicit Final Result:\n";
    std::cout << "temp = ";
    for(int i = 0; i < (m2+1)*(m1+1); i++) std::cout << "[" << i << "] = " << temp[i] << ",";
    std::cout << "\n";
    */

    double residual_norm = 0.0;
    for(size_t i = 0; i < x.size(); i++) {
        double res = x[i] - theta * delta_t * temp[i] - b[i];
        //std::cout << "x " << x[i] << ", ";
        residual_norm += res * res;
    }
    residual_norm = std::sqrt(residual_norm);
    std::cout << "\nFinal residual norm: " << residual_norm << "\n";
}


/*

This tests how fast we could reorder a vector of given dimesnion size of our PDE problem.
This code is done on the GPU with two methods, one uses 1D parallism, the other team based parallism

*/

void test_reordering_performance() {
    // Test different grid sizes
    std::vector<std::pair<int, int>> grid_sizes = {
        {50, 25},   // Typical size
        {100, 50},  // Medium size
        {200, 100}, // Large size
        {300, 100}, // Problem Size
    };

    for (const auto& [m1, m2] : grid_sizes) {
        const int rows = m2 + 1;
        const int cols = m1 + 1;
        const int total_size = rows * cols;

        // Create source and destination views
        Kokkos::View<double*> v_flattened("v_flattened", total_size);
        Kokkos::View<double*> s_flattened("s_flattened", total_size);
        
        // Initialize with test data
        Kokkos::parallel_for("init", total_size, KOKKOS_LAMBDA(const int i) {
            v_flattened(i) = i;
        });

        // Method 1: Simple parallel_for approach
        auto start = std::chrono::high_resolution_clock::now();
        
        Kokkos::parallel_for("reorder", total_size, KOKKOS_LAMBDA(const int idx) {
            const int i = idx / rows;    // col index in original
            const int j = idx % rows;    // row index in original
            const int new_idx = i + j * cols;
            s_flattened(new_idx) = v_flattened(idx);
        });
        
        Kokkos::fence();
        auto end = std::chrono::high_resolution_clock::now();
        double time1 = std::chrono::duration<double>(end - start).count();

        // Method 2: Team approach
        const int team_size = 16;  // Tune for specific GPU architecture
        using team_policy = Kokkos::TeamPolicy<>;
        using member_type = team_policy::member_type;

        start = std::chrono::high_resolution_clock::now();
        
        Kokkos::parallel_for("reorder_team", 
            team_policy(cols, team_size),
            KOKKOS_LAMBDA(const member_type& team_member) {
                const int i = team_member.league_rank();
                
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, rows),
                    [&] (const int j) {
                        const int old_idx = i + j * cols;
                        const int new_idx = i * rows + j;
                        s_flattened(new_idx) = v_flattened(old_idx);
                    });
            });
        
        Kokkos::fence();
        end = std::chrono::high_resolution_clock::now();
        double time2 = std::chrono::duration<double>(end - start).count();

        // Print results
        std::cout << "\nGrid size: " << m1 << "x" << m2 << " (" << total_size << " elements)\n";
        std::cout << "Simple parallel_for time: " << time1 * 1000 << " ms\n";
        std::cout << "Team-based time: " << time2 * 1000 << " ms\n";
        
        // Optional: Verify correctness
        auto h_v = Kokkos::create_mirror_view(v_flattened);
        auto h_s = Kokkos::create_mirror_view(s_flattened);
        Kokkos::deep_copy(h_v, v_flattened);
        Kokkos::deep_copy(h_s, s_flattened);
        
        bool correct = true;
        for (int i = 0; i < cols && correct; ++i) {
            for (int j = 0; j < rows && correct; ++j) {
                const int old_idx = i + j * cols;
                const int new_idx = i * rows + j;
                if (h_s(new_idx) != h_v(old_idx)) {
                    correct = false;
                    break;
                }
            }
        }
        std::cout << "Correctness check: " << (correct ? "PASSED" : "FAILED") << "\n";
    }
}


/*

This test the A2 cpu version and is working correctly

*/

struct A2_cpu_storage {
    int m1, m2;
    
    // Explicit system diagonals
    std::vector<double> main_diag;     // (m2-1)*(m1+1)
    std::vector<double> lower_diag;    // (m2-2)*(m1+1)
    std::vector<double> upper_diag;    // (m2-1)*(m1+1)
    std::vector<double> upper2_diag;   // m1+1 (for j=0 special case)

    // Implicit system diagonals
    std::vector<double> implicit_main_diag;   // (m2+1)*(m1+1)
    std::vector<double> implicit_lower_diag;  // (m2-2)*(m1+1)
    std::vector<double> implicit_upper_diag;  // (m2-1)*(m1+1)
    std::vector<double> implicit_upper2_diag; // m1+1

    A2_cpu_storage(int m1_, int m2_) : m1(m1_), m2(m2_) {
        main_diag.resize((m2-1)*(m1+1));
        lower_diag.resize((m2-2)*(m1+1));
        upper_diag.resize((m2-1)*(m1+1));
        upper2_diag.resize(m1+1);

        implicit_main_diag.resize((m2+1)*(m1+1));
        implicit_lower_diag.resize((m2-2)*(m1+1));
        implicit_upper_diag.resize((m2-1)*(m1+1));
        implicit_upper2_diag.resize(m1+1);
    }

    void build_implicit(const double theta, const double delta_t) {
        // Init implicit_main_diag with identity
        std::fill(implicit_main_diag.begin(), implicit_main_diag.end(), 1.0);

        // Modify diagonals where explicit matrix is defined
        for(int i = 0; i < (m2-1)*(m1+1); i++) {
            implicit_main_diag[i] -= theta * delta_t * main_diag[i];
        }
        
        for(int i = 0; i < (m2-2)*(m1+1); i++) {
            implicit_lower_diag[i] = -theta * delta_t * lower_diag[i];
        }
        
        for(int i = 0; i < (m2-1)*(m1+1); i++) {
            implicit_upper_diag[i] = -theta * delta_t * upper_diag[i];
        }
        
        for(int i = 0; i < m1+1; i++) {
            implicit_upper2_diag[i] = -theta * delta_t * upper2_diag[i];
        }
    }

    void multiply(const std::vector<double>& x, std::vector<double>& result) {
        //result.resize(x.size(), 0.0);
        const int spacing = m1 + 1;
        
        // First block (j=0)
        for(int i = 0; i < spacing; i++) {
            result[i] = main_diag[i] * x[i];
            result[i] += upper_diag[i] * x[i + spacing];
            result[i] += upper2_diag[i] * x[i + 2*spacing];
            //if(i < spacing) {
                //result[i] += upper_diag[i] * x[i + spacing];
                //result[i] += upper2_diag[i] * x[i + 2*spacing];
            //}
        }

        // Middle blocks
        for(int j = 1; j < m2-1; j++) {
            for(int i = 0; i < spacing; i++) {
                int idx = j*spacing + i;
                result[idx] = main_diag[idx] * x[idx];
                result[idx] += lower_diag[idx-spacing] * x[idx-spacing];
                result[idx] += upper_diag[idx] * x[idx+spacing];
            }
        }
    }

    /*
    void solve_implicit(std::vector<double>& x, const std::vector<double>& b) {
        const int spacing = m1 + 1;
        const int num_rows = (m2-1)*spacing;
        const int total_size = (m2+1)*spacing;

        // Handle identity block first
        for(int i = num_rows; i < total_size; i++) {
            x[i] = b[i];
        }
        std::cout << "\nAfter handling identity block:\n";
        std::cout << "x = ";
        for(int i = 0; i < total_size; i++) std::cout << x[i] << " ";
        std::cout << "\n";

        // Pre-backward sweep to eliminate last m1+1 entries of upper_diag
        //std::cout << "Pre backward sweep" << std::endl;
        for(int i = num_rows - spacing; i < num_rows; i++) {
            //std::cout << "i " <<  i << ", i+spacing " << i+spacing << std::endl;
            x[i] -= implicit_upper_diag[i] * x[i + spacing];
        }
        std::cout << "\nAfter pre-backward sweep:\n";
        std::cout << "x = ";
        for(int i = 0; i < total_size; i++) std::cout << x[i] << " ";
        std::cout << "\n";

        // Temporary storage
        std::vector<double> temp_main(num_rows);
        std::vector<double> temp_upper(num_rows);

        // Initialize temp arrays with diagonal values
        for(int i = 0; i < num_rows; i++) {
            temp_main[i] = implicit_main_diag[i];
            temp_upper[i] = implicit_upper_diag[i];
        }

        // Forward elimination - First block
        std::cout << "First block forward sweep " << std::endl;
        //the indexing should be correct for the diagonals!
        for(int i = 0; i < spacing; i++) {
            x[i] = b[i]; // is this true? I think yes it is 
            if(i > 0) {
                std::cout << "i " <<  i << ", i+spacing-1 " << i+spacing-1 << std::endl; 
                double factor = implicit_lower_diag[i-1] / temp_main[i-1];
                temp_main[i+spacing-1] = implicit_main_diag[i+spacing-1] - factor * temp_upper[i-1];
                temp_upper[i+spacing-1] = implicit_upper_diag[i+spacing-1] - factor * implicit_upper2_diag[i-1];
                x[i+spacing-1] = b[i+spacing-1] - factor * x[i-1];
            }
        }
        std::cout << "\nAfter first block forward elimination:\n";
        std::cout << "temp_main = ";
        for(int i = 0; i < num_rows; i++) std::cout << temp_main[i] << " ";
        std::cout << "\ntemp_upper = ";
        for(int i = 0; i < num_rows; i++) std::cout << temp_upper[i] << " ";
        std::cout << "\nx = ";
        for(int i = 0; i < total_size; i++) std::cout << x[i] << " ";
        std::cout << "\n";

        // Middle blocks elimination
        for(int i = spacing; i < num_rows - spacing; i++) {  // Stop before last block
            double factor = implicit_lower_diag[i-spacing] / temp_main[i-spacing];
            temp_main[i+spacing] = implicit_main_diag[i+spacing] - factor * temp_upper[i-spacing];
            temp_upper[i+spacing] = implicit_upper_diag[i+spacing];  // No upper2_diag here
            x[i+spacing] = b[i+spacing] - factor * x[i-spacing];
        }

        std::cout << "\nAfter middle blocks elimination:\n";
        std::cout << "temp_main = ";
        for(int i = 0; i < num_rows; i++) std::cout << temp_main[i] << " ";
        std::cout << "\nx = ";
        for(int i = 0; i < total_size; i++) std::cout << x[i] << " ";
        std::cout << "\n";

        //THIS IS WRONG. WE now have m1+1 rows where we only have 1 main diagonal entry!!!!
        // Back substitution for middle blocks
        x[num_rows-1] /= temp_main[num_rows-1];
        // Middle blocks back substitution
        for(int i = num_rows - spacing - 1; i >= spacing; i--) {
            x[i] = (x[i] - temp_upper[i] * x[i + spacing]) / temp_main[i];
        }
        std::cout << "\nAfter middle blocks back substitution:\n";
        std::cout << "x = ";
        for(int i = 0; i < total_size; i++) std::cout << x[i] << " ";
        std::cout << "\n";

        // First block back substitution
        // First block back substitution (with upper2_diag)
        for(int i = spacing-1; i >= 0; i--) {
            x[i] = (x[i] - temp_upper[i] * x[i + spacing] 
                    - implicit_upper2_diag[i] * x[i + 2*spacing]) / temp_main[i];
        }
        std::cout << "\nAfter final back substitution:\n";
        std::cout << "x = ";
        for(int i = 0; i < total_size; i++) std::cout << x[i] << " ";
        std::cout << "\n";
        
    }
    */

    // CPU implementation
    void solve_implicit(std::vector<double>& x, const std::vector<double>& d) {
        const int spacing = m1 + 1;
        const int num_rows = (m2-1)*spacing;
        const int total_size = (m2+1)*spacing;

        // Temp storage
        std::vector<double> c_star(num_rows); 
        std::vector<double> c2_star(spacing);
        std::vector<double> d_star(total_size);


        // Identity block 
        for(int i = num_rows; i < total_size; i++) {
            d_star[i] = d[i];
        }
        /*
        std::cout << "\nAfter handling identity block:\n";
        std::cout << "d_star = ";
        for(int i = 0; i < total_size; i++) std::cout << "[" << i << "] = " << d_star[i] << ",";
        std::cout << "\n";
        */

        //normalize the first m1+1 rows. Corresponds to Thomas step c'[0] = c[0]/b[0]
        //also the upper2_diagonal 
        for(int i = 0; i < spacing; i++){
            c_star[i] = implicit_upper_diag[i] / implicit_main_diag[i];
            c2_star[i] = implicit_upper2_diag[i] / implicit_main_diag[i];
            d_star[i] = d[i] / implicit_main_diag[i];
        }

        //First block forward sweep (handle upper2_diag)
        //here spacing needs to be accounted for
        double c_upper = 0; //since we have upper2_diag, we need to account for this in updating
                            //c_star. Usually it would be c_star=c/m, now it is c_star=c'/m, where c'
                            //is changed by upper2_diag. This is done in the first line c_upper = ...
        
        for(int i = 0; i < spacing; i++) { 
            c_upper = implicit_upper_diag[i+spacing] - c2_star[i]*implicit_lower_diag[i];
            double m = 1.0 / 
                        (implicit_main_diag[i+spacing] - c_star[i]*implicit_lower_diag[i]);
            c_star[i+spacing] = c_upper * m;
            d_star[i+spacing] = (d[i+spacing] - implicit_lower_diag[i] * d_star[i]) * m;
        }
        
         /*
        std::cout << "\nD_STAR After first forward:\n";
        std::cout << "d_star = ";
        for(int i = 0; i < total_size; i++) std::cout << "[" << i << "] = " << d_star[i] << ",";
        std::cout << "\n";
        
        std::cout << "\nC_STAR After first forward:\n";
        std::cout << "c_star = ";
        for(int i = 0; i < num_rows; i++) std::cout << "[" << i << "] = " << c_star[i] << ",";
        std::cout << "\n";
        */
        

        // Middle blocks forward sweep
        //"i<num_rows - spacing" since we pre backward eliminated the last m1+1 entries of upper diag
        for(int i = spacing; i < num_rows - spacing; i++) {
            double m = 1.0 / 
                        (implicit_main_diag[i+spacing] - c_star[i]*implicit_lower_diag[i]);
            c_star[i+spacing] = implicit_upper_diag[i+spacing] * m;
            d_star[i+spacing] = (d[i+spacing] - implicit_lower_diag[i] * d_star[i]) * m;
            std::cout << d_star[i+spacing] << ", ";
        }

        /*
        std::cout << "\nD_STAR After second forward:\n";
        std::cout << "d_star = ";
        for(int i = 0; i < total_size; i++) std::cout << "[" << i << "] = " << d_star[i] << ",";
        std::cout << "\n";

        std::cout << "\nC_STAR After second forward:\n";
        std::cout << "c_star = ";
        for(int i = 0; i < num_rows; i++) std::cout << "[" << i << "] = " << c_star[i] << ",";
        std::cout << "\n";
        */

        //pre-backward sweep to eliminate m1+1 last entries of upper diag. Eliminated by
        //identity block
        //here d_star[i+spacing] should be equal to d, since of the identity block pre set
        //at the beginning fo this function

        //correct
        for(int i = num_rows - spacing; i < num_rows; i++){
            d_star[i] = d_star[i] - d_star[i+spacing]*c_star[i];
        }

        /*
        std::cout << "\nD_STAR After pre backward sweep:\n";
        std::cout << "d_star = ";
        for(int i = 0; i < total_size; i++) std::cout << "[" << i << "] = " << d_star[i] << ",";
        std::cout << "\n";
        */

        //backward substitution on last m1+1 rows (only main diagonal present)
        // Last m1+1 rows only have main diagonal
        //this might be wrong with doing this with implic_main_diag
        //maybe the main was changed by the above forward
        //or it is right and the "change" is considered in d_star

        //maybe this shouldnt be devided
        //should be correct, tested with -1 on diago so implicit has 2 on diagonal, rest zero
        for(int i = num_rows - spacing; i < num_rows; i++) {
            x[i] = d_star[i];///implicit_main_diag[i];
        }

        /*
        std::cout << "\nX: After m1+1 row devide:\n";
        std::cout << "x = ";
        for(int i = 0; i < total_size; i++) std::cout << "[" << i << "] = " << x[i] << ",";
        std::cout << "\n";
        */

        //backward sweep until the upper2_diag appears. 
        //is correct
        for(int i = num_rows-1; i >= 3*spacing; i--) {
            x[i-spacing] = d_star[i-spacing] - c_star[i-spacing] * x[i];
        }
        
        /*
        std::cout << "\nX After first backward sweep:\n";
        std::cout << "x = ";
        for(int i = 0; i < total_size; i++) std::cout << "[" << i << "] = " << x[i] << ",";
        std::cout << "\n";
        */

        //First block back substitution with upper2_diag
        //in the second line it is NOT x[i-2*spacing] since this entry will be "changed" again in the last backward 
        //sweep, therefore it has to be d_star which keeps track of changed right hand side values
        for(int i = 3*spacing-1; i >= 2*spacing; i--) {
            x[i-spacing] = d_star[i-spacing] - c_star[i-spacing] * x[i];
            d_star[i-2*spacing] = d_star[i-2*spacing] - c2_star[i-2*spacing] * x[i];
        }

        /*
        std::cout << "\nX After second backward sweep:\n";
        std::cout << "x = ";
        for(int i = 0; i < total_size; i++) std::cout << "[" << i << "] = " << x[i] << ",";
        std::cout << "\n";
        */

        //last backwar substitution after upp2 is gone
        for(int i = 2*spacing-1; i >= spacing; i--) {
            x[i-spacing] = d_star[i-spacing] - c_star[i-spacing] * x[i];
        }

        /*
        std::cout << "\nX After third backward sweep:\n";
        std::cout << "x = ";
        for(int i = 0; i < total_size; i++) std::cout << "[" << i << "] = " << x[i] << ",";
        std::cout << "\n";
        */

        //this shouldnt be done, look at thomas algorithm. The above loop already accesses
        //x[0]
        /*
        //last m1+1 rows are left 
        for(int i = spacing-1; i >= 0; i--) {
            x[i] = d_star[i];
        }

        std::cout << "\nX After first rows:\n";
        std::cout << "x = ";
        for(int i = 0; i < total_size; i++) std::cout << "[" << i << "] = " << x[i] << ",";
        std::cout << "\n";
        */
        
        // Identity block 
        for(int i = num_rows; i < total_size; i++) {
            x[i] = d_star[i];
        }

    }
};

void test_A2_cpu() {
    using timer = std::chrono::high_resolution_clock;
    // Test dimensions
    const int m1 = 5;
    const int m2 = 20;
    std::cout << "Testing A2 CPU implementation with dimensions m1=" << m1 << ", m2=" << m2 << "\n";

    A2_cpu_storage A2(m1, m2);

    // Fill explicit diagonals with test values
    std::fill(A2.main_diag.begin(), A2.main_diag.end(), 0.0);
    std::fill(A2.upper_diag.begin(), A2.upper_diag.end(), 0.0);
    std::fill(A2.upper2_diag.begin(), A2.upper2_diag.end(), 0.0);
    std::fill(A2.lower_diag.begin(), A2.lower_diag.end(), 1.0);

    // Print explicit diagonals
    /*
    std::cout << "\nExplicit diagonals:\n";
    std::cout << "Main diagonal: ";
    for(auto val : A2.main_diag) std::cout << val << " ";
    std::cout << "\nLower diagonal: ";
    for(auto val : A2.lower_diag) std::cout << val << " ";
    std::cout << "\nUpper diagonal: ";
    for(auto val : A2.upper_diag) std::cout << val << " ";
    std::cout << "\nUpper2 diagonal: ";
    for(auto val : A2.upper2_diag) std::cout << val << " ";
    std::cout << "\n";
    */
    // Build and test implicit system
    double theta = 1.0;
    double delta_t = 1.0;
    A2.build_implicit(theta, delta_t);

    // Print implicit diagonals
    /*
    std::cout << "\nImplicit diagonals:\n";
    std::cout << "Main diagonal: ";
    for(auto val : A2.implicit_main_diag) std::cout << val << " ";
    std::cout << "\nLower diagonal: ";
    for(auto val : A2.implicit_lower_diag) std::cout << val << " ";
    std::cout << "\nUpper diagonal: ";
    for(auto val : A2.implicit_upper_diag) std::cout << val << " ";
    std::cout << "\nUpper2 diagonal: ";
    for(auto val : A2.implicit_upper2_diag) std::cout << val << " ";
    std::cout << "\n";
    */

    // Test multiply with vector of ones
    /*
    std::vector<double> x((m2+1)*(m1+1), 1.0);
    
    std::vector<double> result((m2+1)*(m1+1), 0.0);
    A2.multiply(x, result);

    std::cout << "\nResult of multiplying with ones:\n";
    for(int i = 0; i < result.size(); i++) {
        std::cout << "[" << i << "] = " << result[i] << ",";
    }
    */
    


    
    // Test implicit solve
    std::vector<double> x((m2+1)*(m1+1), 0); //initial guess
    std::vector<double> b((m2+1)*(m1+1), 1.0);  // RHS = ones

    /*
    std::cout << "\nTesting implicit solve\n";
    std::cout << "Initial x = ";
    for(const auto& val : x) std::cout << val << " ";
    std::cout << "\n";
    std::cout << "RHS b = ";
    for(const auto& val : b) std::cout << val << " ";
    std::cout << "\n";
    */

    auto t_start_multip = timer::now();
    A2.solve_implicit(x, b);
    auto t_end_multip = timer::now();

    std::cout << "Implicit cpu solve time: "
              << std::chrono::duration<double>(t_end_multip - t_start_multip).count()
              << " seconds" << std::endl;

    
    std::cout << "\nImplicit Final Result:\n";
    std::cout << "x = ";
    for(int i = 0; i < (m2+1)*(m1+1); i++) std::cout << "[" << i << "] = " << x[i] << ",";
    std::cout << "\n";


    // Verify solution by computing residual
    std::vector<double> temp((m2+1)*(m1+1));
    A2.multiply(x, temp);
    /*
    std::cout << "\nExplicit Final Result:\n";
    std::cout << "temp = ";
    for(int i = 0; i < (m2+1)*(m1+1); i++) std::cout << "[" << i << "] = " << temp[i] << ",";
    std::cout << "\n";
    */

    double residual_norm = 0.0;
    for(size_t i = 0; i < x.size(); i++) {
        double res = x[i] - theta * delta_t * temp[i] - b[i];
        //std::cout << "x " << x[i] << ", ";
        residual_norm += res * res;
    }
    residual_norm = std::sqrt(residual_norm);
    std::cout << "\nFinal residual norm: " << residual_norm << "\n";
}

void test_complex_A2_cpu_random_b() {
    using timer = std::chrono::high_resolution_clock;
    // Test dimensions
    const int m1 = 300;
    const int m2 = 100;
    std::cout << "Testing A2 CPU implementation with dimensions m1=" << m1 << ", m2=" << m2 << "\n";

    A2_cpu_storage A2(m1, m2);

    // Fill explicit diagonals with test values
    std::fill(A2.main_diag.begin(), A2.main_diag.end(), 8.0);
    std::fill(A2.upper_diag.begin(), A2.upper_diag.end(), 1.0);
    std::fill(A2.upper2_diag.begin(), A2.upper2_diag.end(), 1.0);
    std::fill(A2.lower_diag.begin(), A2.lower_diag.end(), -1.0);

    // Print explicit diagonals
    /*
    std::cout << "\nExplicit diagonals:\n";
    std::cout << "Main diagonal: ";
    for(auto val : A2.main_diag) std::cout << val << " ";
    std::cout << "\nLower diagonal: ";
    for(auto val : A2.lower_diag) std::cout << val << " ";
    std::cout << "\nUpper diagonal: ";
    for(auto val : A2.upper_diag) std::cout << val << " ";
    std::cout << "\nUpper2 diagonal: ";
    for(auto val : A2.upper2_diag) std::cout << val << " ";
    std::cout << "\n";
    */
    // Build and test implicit system
    double theta = 0.8;
    double delta_t = 0.001;
    A2.build_implicit(theta, delta_t);

    // Print implicit diagonals
    /*
    std::cout << "\nImplicit diagonals:\n";
    std::cout << "Main diagonal: ";
    for(auto val : A2.implicit_main_diag) std::cout << val << " ";
    std::cout << "\nLower diagonal: ";
    for(auto val : A2.implicit_lower_diag) std::cout << val << " ";
    std::cout << "\nUpper diagonal: ";
    for(auto val : A2.implicit_upper_diag) std::cout << val << " ";
    std::cout << "\nUpper2 diagonal: ";
    for(auto val : A2.implicit_upper2_diag) std::cout << val << " ";
    std::cout << "\n";
    */

    // Test multiply with vector of ones
    /*
    std::vector<double> x((m2+1)*(m1+1), 1.0);
    
    std::vector<double> result((m2+1)*(m1+1), 0.0);
    A2.multiply(x, result);

    std::cout << "\nResult of multiplying with ones:\n";
    for(int i = 0; i < result.size(); i++) {
        std::cout << "[" << i << "] = " << result[i] << ",";
    }
    */
    


    
    // Test implicit solve
    std::vector<double> x((m2+1)*(m1+1), 0); //initial guess
    std::vector<double> b((m2+1)*(m1+1));  // RHS

    // Initialize b with random values
    std::srand(52); // Seed for reproducibility
    for(int i = 0; i < (m2+1)*(m1+1); i++) {
        b[i] = std::rand() / (RAND_MAX + 1.0);
    }

    /*
    std::cout << "\nTesting implicit solve\n";
    std::cout << "Initial x = ";
    for(const auto& val : x) std::cout << val << " ";
    std::cout << "\n";
    std::cout << "RHS b = ";
    for(const auto& val : b) std::cout << val << " ";
    std::cout << "\n";
    */

    auto t_start_multip = timer::now();
    A2.solve_implicit(x, b);
    auto t_end_multip = timer::now();

    std::cout << "Implicit cpu solve time: "
              << std::chrono::duration<double>(t_end_multip - t_start_multip).count()
              << " seconds" << std::endl;

    /*
    std::cout << "\nImplicit Final Result:\n";
    std::cout << "x = ";
    for(int i = 0; i < (m2+1)*(m1+1); i++) std::cout << "[" << i << "] = " << x[i] << ",";
    std::cout << "\n";
    */

    // Verify solution by computing residual
    std::vector<double> temp((m2+1)*(m1+1));
    A2.multiply(x, temp);
    /*
    std::cout << "\nExplicit Final Result:\n";
    std::cout << "temp = ";
    for(int i = 0; i < (m2+1)*(m1+1); i++) std::cout << "[" << i << "] = " << temp[i] << ",";
    std::cout << "\n";
    */

    double residual_norm = 0.0;
    for(size_t i = 0; i < x.size(); i++) {
        double res = x[i] - theta * delta_t * temp[i] - b[i];
        //std::cout << "x " << x[i] << ", ";
        residual_norm += res * res;
    }
    residual_norm = std::sqrt(residual_norm);
    std::cout << "\nFinal residual norm: " << residual_norm << "\n";
}


/*

This is the gpu translation of the above A2 cpu code. It runs everything in a single thread

*/

//Two print functions. Since the main diagonals do not have the same dimesion. 
template <typename ViewType>
void print_A2_matrix(const ViewType& lower, const ViewType& main, const ViewType& upper, 
                    const ViewType& upper2, int m1, int m2) {
    std::cout << "A2 matrix structure:\n";
    const int spacing = m1 + 1;
    const int total_size = (m2 + 1) * (m1 + 1);
    
    // Print column headers
    std::cout << "      ";
    for(int i = 0; i < std::min(20, total_size); i++) {
        std::cout << "Col_" << std::setw(2) << std::setfill('0') << i << "  ";
    }
    std::cout << "\n";
    
    // Print separator line
    std::cout << "-----------------------------------------------------------------------------------------------------------\n";
    
    // Print matrix rows
    for(int row = 0; row < std::min(20, total_size); row++) {
        // Print row label
        std::cout << "Row_" << std::setw(2) << std::setfill('0') << row << "  ";
        std::cout << std::setfill(' '); // Reset fill character
        
        for(int col = 0; col < std::min(20, total_size); col++) {
            bool printed = false;
            
            // Main diagonal
            if(row < (m2-1)*(m1+1) && row == col) {
                std::cout << std::setw(8) << std::fixed << std::setprecision(3) << main(row) << "  ";
                printed = true;
                continue;
            }
            
            // Lower diagonal
            if(row >= spacing && row < (m2-1)*(m1+1) && 
               col == row-spacing && (row/spacing) > 0) {
                std::cout << std::setw(8) << std::fixed << std::setprecision(3) << lower(row-spacing) << "  ";
                printed = true;
                continue;
            }
            
            // Upper diagonal
            if(row < (m2-1)*(m1+1) && 
               col == row+spacing) {
                std::cout << std::setw(8) << std::fixed << std::setprecision(3) << upper(row) << "  ";
                printed = true;
                continue;
            }
            
            // Upper2 diagonal (j=0 special case)
            if(row < m1+1 && col == row + 2*spacing) {
                std::cout << std::setw(8) << std::fixed << std::setprecision(3) << upper2(row) << "  ";
                printed = true;
                continue;
            }
            
            if(!printed) {
                std::cout << std::setw(8) << "0.000" << "  ";
            }
        }
        std::cout << "...\n";  // Add line break after each row
    }
    if(total_size > 20) std::cout << "...\n";
}

template <typename ViewType>
void print_A2_implicit_matrix(const ViewType& lower, const ViewType& main, const ViewType& upper, 
                    const ViewType& upper2, int m1, int m2) {
    std::cout << "A2 matrix structure:\n";
    const int spacing = m1 + 1;
    const int total_size = (m2 + 1) * (m1 + 1);
    
    // Print column headers
    std::cout << "      ";
    for(int i = 0; i < std::min(20, total_size); i++) {
        std::cout << "Col_" << std::setw(2) << std::setfill('0') << i << "  ";
    }
    std::cout << "\n";
    
    // Print separator line
    std::cout << "-----------------------------------------------------------------------------------------------------------\n";
    
    // Print matrix rows
    for(int row = 0; row < std::min(20, total_size); row++) {
        // Print row label
        std::cout << "Row_" << std::setw(2) << std::setfill('0') << row << "  ";
        std::cout << std::setfill(' '); // Reset fill character
        
        for(int col = 0; col < std::min(20, total_size); col++) {
            bool printed = false;
            
            // Main diagonal
            if(row < (m2+1)*(m1+1) && row == col) {
                std::cout << std::setw(8) << std::fixed << std::setprecision(3) << main(row) << "  ";
                printed = true;
                continue;
            }
            
            // Lower diagonal
            if(row >= spacing && row < (m2-1)*(m1+1) && 
               col == row-spacing && (row/spacing) > 0) {
                std::cout << std::setw(8) << std::fixed << std::setprecision(3) << lower(row-spacing) << "  ";
                printed = true;
                continue;
            }
            
            // Upper diagonal
            if(row < (m2-1)*(m1+1) && 
               col == row+spacing) {
                std::cout << std::setw(8) << std::fixed << std::setprecision(3) << upper(row) << "  ";
                printed = true;
                continue;
            }
            
            // Upper2 diagonal (j=0 special case)
            if(row < m1+1 && col == row + 2*spacing) {
                std::cout << std::setw(8) << std::fixed << std::setprecision(3) << upper2(row) << "  ";
                printed = true;
                continue;
            }
            
            if(!printed) {
                std::cout << std::setw(8) << "0.000" << "  ";
            }
        }
        std::cout << "...\n";  // Add line break after each row
    }
    if(total_size > 20) std::cout << "...\n";
}



struct A2Storage {
    int m1, m2;
    
    // Explicit system diagonals
    Kokkos::View<double*> main_diag;     // (m2-1)*(m1+1)
    Kokkos::View<double*> lower_diag;    // (m2-2)*(m1+1)
    Kokkos::View<double*> upper_diag;    // (m2-1)*(m1+1)
    Kokkos::View<double*> upper2_diag;   // m1+1

    // Implicit system diagonals
    Kokkos::View<double*> implicit_main_diag;   // (m2+1)*(m1+1)
    Kokkos::View<double*> implicit_lower_diag;  // (m2-2)*(m1+1)
    Kokkos::View<double*> implicit_upper_diag;  // (m2-1)*(m1+1)
    Kokkos::View<double*> implicit_upper2_diag; // m1+1

    A2Storage(int m1_, int m2_) : m1(m1_), m2(m2_) {
        main_diag = Kokkos::View<double*>("A2_main_diag", (m2-1)*(m1+1));
        lower_diag = Kokkos::View<double*>("A2_lower_diag", (m2-2)*(m1+1));
        upper_diag = Kokkos::View<double*>("A2_upper_diag", (m2-1)*(m1+1));
        upper2_diag = Kokkos::View<double*>("A2_upper2_diag", m1+1);

        implicit_main_diag = Kokkos::View<double*>("A2_impl_main_diag", (m2+1)*(m1+1));
        implicit_lower_diag = Kokkos::View<double*>("A2_impl_lower_diag", (m2-2)*(m1+1));
        implicit_upper_diag = Kokkos::View<double*>("A2_impl_upper_diag", (m2-1)*(m1+1));
        implicit_upper2_diag = Kokkos::View<double*>("A2_impl_upper2_diag", m1+1);
    }

    void build_implicit(const double theta, const double delta_t) {
        const int local_m1 = m1;
        const int local_m2 = m2;

        const auto local_main = main_diag;
        const auto local_lower = lower_diag;
        const auto local_upper = upper_diag;
        const auto local_upper2 = upper2_diag;

        const auto local_impl_main = implicit_main_diag;
        const auto local_impl_lower = implicit_lower_diag;
        const auto local_impl_upper = implicit_upper_diag;
        const auto local_impl_upper2 = implicit_upper2_diag;

        Kokkos::parallel_for("build_implicit", 1, KOKKOS_LAMBDA(const int) {
            // Initialize implicit_main_diag with identity
            for(int i = 0; i < (local_m2+1)*(local_m1+1); i++) {
                local_impl_main(i) = 1.0;
            }

            // Subtract theta*delta_t*A2 from main diagonal where A2 is defined
            for(int i = 0; i < (local_m2-1)*(local_m1+1); i++) {
                local_impl_main(i) -= theta * delta_t * local_main(i);
            }

            // Build the off-diagonal terms
            for(int i = 0; i < (local_m2-2)*(local_m1+1); i++) {
                local_impl_lower(i) = -theta * delta_t * local_lower(i);
            }

            for(int i = 0; i < (local_m2-1)*(local_m1+1); i++) {
                local_impl_upper(i) = -theta * delta_t * local_upper(i);
            }

            for(int i = 0; i < local_m1+1; i++) {
                local_impl_upper2(i) = -theta * delta_t * local_upper2(i);
            }
        });
        Kokkos::fence();
    }

    void multiply(const Kokkos::View<double*>& x, Kokkos::View<double*>& result) {
        const int local_m1 = m1;
        const int local_m2 = m2;

        const auto local_main = main_diag;
        const auto local_lower = lower_diag;
        const auto local_upper = upper_diag;
        const auto local_upper2 = upper2_diag;

        const int spacing = m1 + 1;
        
        // First set result to zero
        Kokkos::deep_copy(result, 0.0);
        
        Kokkos::parallel_for("multiply", 1, KOKKOS_LAMBDA(const int) {
            // First block (j=0)
            for(int i = 0; i < spacing; i++) {
                double temp = local_main(i) * x(i);
                temp += local_upper(i) * x(i + spacing);
                temp += local_upper2(i) * x(i + 2*spacing);
                result(i) = temp;
            }

            // Handle remaining blocks
            for(int i = spacing; i < (local_m2-1)*(local_m1+1); i++) {
                double temp = local_lower(i-spacing) * x(i-spacing);
                temp += local_main(i) * x(i);
                temp += local_upper(i) * x(i + spacing);
                result(i) = temp;
            }
        });
        Kokkos::fence();
    }

    void solve_implicit(Kokkos::View<double*>& x, const Kokkos::View<double*>& b) {
        //const int local_m1 = m1;
        const int local_m2 = m2;
        const int spacing = m1 + 1;
        const int num_rows = (local_m2-1)*spacing;
        const int total_size = (local_m2+1)*spacing;

        // Temporary storage on device
        Kokkos::View<double*> c_star("c_star", num_rows);
        Kokkos::View<double*> c2_star("c2_star", spacing);
        Kokkos::View<double*> d_star("d_star", total_size);

        const auto local_impl_main = implicit_main_diag;
        const auto local_impl_lower = implicit_lower_diag;
        const auto local_impl_upper = implicit_upper_diag;
        const auto local_impl_upper2 = implicit_upper2_diag;

        Kokkos::parallel_for("solve_implicit", 1, KOKKOS_LAMBDA(const int) {
            // Identity block
            for(int i = num_rows; i < total_size; i++) {
                d_star(i) = b(i);
            }

            // Normalize first m1+1 rows and upper2_diagonal
            for(int i = 0; i < spacing; i++) {
                c_star(i) = local_impl_upper(i) / local_impl_main(i);
                c2_star(i) = local_impl_upper2(i) / local_impl_main(i);
                d_star(i) = b(i) / local_impl_main(i);
            }

            // First block forward sweep (handle upper2_diag)
            for(int i = 0; i < spacing; i++) {
                double c_upper = local_impl_upper(i+spacing) - c2_star(i)*local_impl_lower(i);
                double m = 1.0 / (local_impl_main(i+spacing) - c_star(i)*local_impl_lower(i));
                c_star(i+spacing) = c_upper * m;
                d_star(i+spacing) = (b(i+spacing) - local_impl_lower(i) * d_star(i)) * m;
            }

            // Middle blocks forward sweep
            for(int i = spacing; i < num_rows - spacing; i++) {
                double m = 1.0 / (local_impl_main(i+spacing) - c_star(i)*local_impl_lower(i));
                c_star(i+spacing) = local_impl_upper(i+spacing) * m;
                d_star(i+spacing) = (b(i+spacing) - local_impl_lower(i) * d_star(i)) * m;
            }

            // Pre-backward sweep
            for(int i = num_rows - spacing; i < num_rows; i++) {
                d_star(i) -= d_star(i+spacing)*c_star(i);
            }

            // Last m1+1 rows
            for(int i = num_rows - spacing; i < num_rows; i++) {
                x(i) = d_star(i);
            }

            // Backward sweep until upper2_diag appears
            for(int i = num_rows-1; i >= 3*spacing; i--) {
                x(i-spacing) = d_star(i-spacing) - c_star(i-spacing) * x(i);
            }

            // First block back substitution with upper2_diag
            for(int i = 3*spacing-1; i >= 2*spacing; i--) {
                x(i-spacing) = d_star(i-spacing) - c_star(i-spacing) * x(i);
                d_star(i-2*spacing) = d_star(i-2*spacing) - c2_star(i-2*spacing) * x(i);
            }

            // Last backward substitution
            for(int i = 2*spacing-1; i >= spacing; i--) {
                x(i-spacing) = d_star(i-spacing) - c_star(i-spacing) * x(i);
            }

            // Identity block
            for(int i = num_rows; i < total_size; i++) {
                x(i) = d_star(i);
            }
        });
        Kokkos::fence();
    }

    //This is a 1D parallism over the stock prices. So we loop sequentially over all variance levels and for each 
    //variance level we perfom the modified thomas algorithm steps in parallel. This is because the spacing of m1+1
    //between each of the variance levels. This however is not resulting in the expected speed up.
    //for dim m1=300,m2=100 we go form: sequential time: 0.02, paralle time: 0.015
    //my idea for this is because we do not have a good memory layout for the GPU. Each thread needs to access
    //the four diagonals and get one value out of them and then update two vector entries accordingly. 

    //suggestens would be to:
    //1. change to a continuous memeroy layout
    //2. look at an optimized thomas algorithm which overwrites already given vectors. 
    //I think option 1. will yield better results, since option 2. will be done anyways when moving the 
    //matrices to a "real" finite difference implementation
    void solve_implicit_parallel(Kokkos::View<double*>& x, const Kokkos::View<double*>& b) {
        //const int local_m1 = m1;
        const int local_m2 = m2;
        const int spacing = m1 + 1;
        const int num_rows = (local_m2-1)*spacing;
        const int total_size = (local_m2+1)*spacing;

        // Temporary storage on device
        Kokkos::View<double*> c_star("c_star", num_rows);
        Kokkos::View<double*> c2_star("c2_star", spacing);
        Kokkos::View<double*> d_star("d_star", total_size);

        const auto local_impl_main = implicit_main_diag;
        const auto local_impl_lower = implicit_lower_diag;
        const auto local_impl_upper = implicit_upper_diag;
        const auto local_impl_upper2 = implicit_upper2_diag;

        using timer = std::chrono::high_resolution_clock;
        auto t_start_solve = timer::now();
        

        // Identity block (parallel)
        Kokkos::parallel_for("identity_block", total_size - num_rows, KOKKOS_LAMBDA(const int i) {
            d_star(num_rows + i) = b(num_rows + i);
        });
        

        // First block normalization (parallel)
        Kokkos::parallel_for("normalize_first_block", spacing, KOKKOS_LAMBDA(const int i) {
            c_star(i) = local_impl_upper(i) / local_impl_main(i);
            c2_star(i) = local_impl_upper2(i) / local_impl_main(i);
            d_star(i) = b(i) / local_impl_main(i);
        });
        

        // First block forward sweep (parallel)
        Kokkos::parallel_for("first_forward_sweep", spacing, KOKKOS_LAMBDA(const int i) {
            double c_upper = local_impl_upper(i+spacing) - c2_star(i)*local_impl_lower(i);
            double m = 1.0 / (local_impl_main(i+spacing) - c_star(i)*local_impl_lower(i));
            c_star(i+spacing) = c_upper * m;
            d_star(i+spacing) = (b(i+spacing) - local_impl_lower(i) * d_star(i)) * m;
        });
        

        // Middle blocks forward sweep (sequential over variance levels, parallel within each level)
        //start at the first varaince level j=1 and end 1 before! the last varaince level. Look at picture,
        //the forward sweep should only go until the last m1+1 entry of main diag.
        for(int j = 1; j < local_m2-1-1; j++) {
            const int level_offset = j * spacing;
            
            Kokkos::parallel_for("middle_forward_sweep", spacing, KOKKOS_LAMBDA(const int i) {
                int idx = level_offset + i;
                double m = 1.0 / (local_impl_main(idx+spacing) - c_star(idx)*local_impl_lower(idx));
                c_star(idx+spacing) = local_impl_upper(idx+spacing) * m;
                d_star(idx+spacing) = (b(idx+spacing) - local_impl_lower(idx) * d_star(idx)) * m;
            });
            Kokkos::fence();  // Ensure level completion before moving to next
        }
        
        // Pre-backward sweep and last rows (can be combined, parallel)
        Kokkos::parallel_for("pre_backward_sweep", spacing, KOKKOS_LAMBDA(const int i) {
            int idx = num_rows - spacing + i;
            d_star(idx) = d_star(idx) - d_star(idx+spacing)*c_star(idx);
            x(idx) = d_star(idx);
        });
        

        // Backward sweep until upper2_diag appears (sequential over levels, parallel within)
        for(int j = local_m2-1-1; j >= 2; j--) {
            const int level_offset = j * spacing;
            
            Kokkos::parallel_for("backward_sweep_1", spacing, KOKKOS_LAMBDA(const int i) {
                int idx = level_offset + i;
                x(idx) = d_star(idx) - c_star(idx) * x(idx+spacing);
            });
            Kokkos::fence();
        }

        // First block back substitution with upper2_diag (parallel)
        Kokkos::parallel_for("backward_sweep_2", spacing, KOKKOS_LAMBDA(const int i) {
            int idx = 3*spacing - 1 - i;  // Correct indexing to match sequential version
            x(idx-spacing) = d_star(idx-spacing) - c_star(idx-spacing) * x(idx);
            d_star(idx-2*spacing) = d_star(idx-2*spacing) - c2_star(idx-2*spacing) * x(idx);
        });

        // Last backward substitution (parallel)
        Kokkos::parallel_for("last_backward", spacing, KOKKOS_LAMBDA(const int i) {
            int idx = 2*spacing - 1 - i;
            x(idx-spacing) = d_star(idx-spacing) - c_star(idx-spacing) * x(idx);
        });

        // Copy identity block (parallel)
        Kokkos::parallel_for("copy_identity", total_size - num_rows, KOKKOS_LAMBDA(const int i) {
            x(num_rows + i) = d_star(num_rows + i);
        });
        
        auto t_end_solve = timer::now();

        std::cout << "True implicict solve time: "
                << std::chrono::duration<double>(t_end_solve - t_start_solve).count()
                << " seconds" << std::endl;
    }

};

void test_A2_gpu() {
    using timer = std::chrono::high_resolution_clock;
    
    // Test dimensions
    const int m1 = 300;
    const int m2 = 100;
    std::cout << "Testing A2 GPU implementation with dimensions m1=" << m1 << ", m2=" << m2 << "\n";

    // Create A2 storage
    A2Storage A2(m1, m2);

    // Create host mirrors for initialization
    auto h_main = Kokkos::create_mirror_view(A2.main_diag);
    auto h_lower = Kokkos::create_mirror_view(A2.lower_diag);
    auto h_upper = Kokkos::create_mirror_view(A2.upper_diag);
    auto h_upper2 = Kokkos::create_mirror_view(A2.upper2_diag);

    // Fill with test values on host
    std::fill(h_main.data(), h_main.data() + h_main.size(), 8.0);
    std::fill(h_lower.data(), h_lower.data() + h_lower.size(), -1.0);
    std::fill(h_upper.data(), h_upper.data() + h_upper.size(), 1.0);
    std::fill(h_upper2.data(), h_upper2.data() + h_upper2.size(), 1.0);

    // Copy to device
    Kokkos::deep_copy(A2.main_diag, h_main);
    Kokkos::deep_copy(A2.lower_diag, h_lower);
    Kokkos::deep_copy(A2.upper_diag, h_upper);
    Kokkos::deep_copy(A2.upper2_diag, h_upper2);

    // Build implicit system
    double theta = 0.8;
    double delta_t = 0.001;
    A2.build_implicit(theta, delta_t);

    // Create test vectors
    const int total_size = (m2+1)*(m1+1);
    Kokkos::View<double*> x("x", total_size);
    Kokkos::View<double*> b("b", total_size);
    Kokkos::View<double*> result("result", total_size);

    // Initialize b with random values
    auto h_b = Kokkos::create_mirror_view(b);
    std::srand(52); // Same seed as CPU test
    for(int i = 0; i < total_size; i++) {
        h_b(i) = std::rand() / (RAND_MAX + 1.0);
    }
    Kokkos::deep_copy(b, h_b);
    Kokkos::deep_copy(x, 0.0); // Initialize x with zeros

    
    // Test implicit solve
    std::cout << "\nTesting implicit solve...\n";

    auto t_start_solve = timer::now();
    A2.solve_implicit_parallel(x, b);
    auto t_end_solve = timer::now();

    std::cout << "Implicit gpu solve time: "
              << std::chrono::duration<double>(t_end_solve - t_start_solve).count()
              << " seconds" << std::endl;
    
    // Verify solution by computing residual
    // (I - θΔt*A2)*x - b
    t_start_solve = timer::now();
    A2.multiply(x, result);

    t_end_solve = timer::now();

    std::cout << "Explicit gpu solve time: "
              << std::chrono::duration<double>(t_end_solve - t_start_solve).count()
              << " seconds" << std::endl;
    
    // Compute residual on host
    auto h_x = Kokkos::create_mirror_view(x);
    auto h_result = Kokkos::create_mirror_view(result);
    Kokkos::deep_copy(h_x, x);
    Kokkos::deep_copy(h_result, result);

    /*
    std::cout << "\nExplicit Final Result:\n";
    std::cout << "result = ";
    for(int i = 0; i < (m2+1)*(m1+1); i++) std::cout << "[" << i << "] = " << h_result[i] << ",";
    std::cout << "\n";

    std::cout << "\nimplicit Final Result:\n";
    std::cout << "result = ";
    for(int i = 0; i < (m2+1)*(m1+1); i++) std::cout << "[" << i << "] = " << h_x[i] << ",";
    std::cout << "\n";
    */

    double residual_norm = 0.0;
    for(int i = 0; i < total_size; i++) {
        double res = h_x(i) - theta * delta_t * h_result(i) - h_b(i);
        residual_norm += res * res;
    }
    residual_norm = std::sqrt(residual_norm);
    std::cout << "\nFinal residual norm: " << residual_norm << "\n";
    
}

void test_A2_gpu_vizually() { 
    std::cout << "Starting A2 Storage test...\n";

    using timer = std::chrono::high_resolution_clock;

    
    // Problem dimensions (small for visualization)
    const int m1 = 5;  // stock price points
    const int m2 = 5;  // variance points
    
    std::cout<< "Total dimesnions: " << "(" << m1 << "+1)" << "*" << "(" << m2 << "+1)" << " = " << (m1+1)*(m2+1) << std::endl;
    // Create A2 storage
    A2Storage A2(m1, m2);
    
    // Fill with test values on host
    auto h_main = Kokkos::create_mirror_view(A2.main_diag);
    auto h_lower = Kokkos::create_mirror_view(A2.lower_diag);
    auto h_upper = Kokkos::create_mirror_view(A2.upper_diag);
    auto h_upper2 = Kokkos::create_mirror_view(A2.upper2_diag);
    
    // Initialize with recognizable pattern
    // Main diagonal
    for(int i = 0; i < (m2-1)*(m1+1); i++) {
        h_main(i) = -4.0;
    }
    
    // Lower diagonal
    for(int i = 0; i < (m2-2)*(m1+1); i++) {
        h_lower(i) = 1.0;
    }
    
    // Upper diagonal
    for(int i = 0; i < (m2-1)*(m1+1); i++) {
        h_upper(i) = 1.0;
    }
    
    // Upper2 diagonal (j=0 special case)
    for(int i = 0; i < m1+1; i++) {
        h_upper2(i) = 2.0;
    }
    
    // Copy to device
    Kokkos::deep_copy(A2.main_diag, h_main);
    Kokkos::deep_copy(A2.lower_diag, h_lower);
    Kokkos::deep_copy(A2.upper_diag, h_upper);
    Kokkos::deep_copy(A2.upper2_diag, h_upper2);
    
    // Print explicit matrix structure
    std::cout << "\nExplicit A2 matrix:\n";
    print_A2_matrix(h_lower, h_main, h_upper, h_upper2, m1, m2);
    
    // Build implicit system
    double theta = 0.5;
    double delta_t = 0.01;
    A2.build_implicit(theta, delta_t);
    
    // Get implicit matrix values
    auto h_impl_main = Kokkos::create_mirror_view(A2.implicit_main_diag);
    auto h_impl_lower = Kokkos::create_mirror_view(A2.implicit_lower_diag);
    auto h_impl_upper = Kokkos::create_mirror_view(A2.implicit_upper_diag);
    auto h_impl_upper2 = Kokkos::create_mirror_view(A2.implicit_upper2_diag);
    
    Kokkos::deep_copy(h_impl_main, A2.implicit_main_diag);
    Kokkos::deep_copy(h_impl_lower, A2.implicit_lower_diag);
    Kokkos::deep_copy(h_impl_upper, A2.implicit_upper_diag);
    Kokkos::deep_copy(h_impl_upper2, A2.implicit_upper2_diag);
    
    // Print implicit matrix structure
    std::cout << "\nImplicit A2 matrix (I - θΔt*A2):\n";
    print_A2_implicit_matrix(h_impl_lower, h_impl_main, h_impl_upper, h_impl_upper2, m1, m2);

    
    // Test explicit multiply
    std::cout << "\nTesting explicit multiply with vector of ones...\n";

    // Create test vectors
    Kokkos::View<double*> test_x("test_x", (m2+1)*(m1+1));
    Kokkos::View<double*> test_result("test_result", (m2+1)*(m1+1));

    // Initialize x with ones
    auto h_test_x = Kokkos::create_mirror_view(test_x);
    for(int i = 0; i < (m2+1)*(m1+1); i++) {
        h_test_x(i) = 1.0;
    }
    Kokkos::deep_copy(test_x, h_test_x);

    // Test explicit multiply
    std::cout << "\nTesting explicit multiply...\n";

    auto t_start_multip = timer::now();
    A2.multiply(test_x, test_result);
    auto t_end_multip = timer::now();

    std::cout << "Multiply solve time: "
              << std::chrono::duration<double>(t_end_multip - t_start_multip).count()
              << " seconds" << std::endl;

    // Get results on host
    auto h_test_result = Kokkos::create_mirror_view(test_result);
    Kokkos::deep_copy(h_test_result, test_result);
    

    // Print first few entries of result
    
    std::cout << "\nResult vector (first 20 entries):\n";
    std::cout << "Index    Value\n";
    std::cout << "----------------\n";
    for(int i = 0; i < std::min(20, (m2+1)*(m1+1)); i++) {
        std::cout << std::setw(5) << i << std::setw(12) << std::fixed 
                << std::setprecision(3) << h_test_result(i) << "\n";
    }
    if((m2+1)*(m1+1) > 20) std::cout << "...\n";
    

    // Verify result against matrix pattern
    // For example, first row should sum to: -4 + 1 + 2 = -1
    // Second row should sum to: -4 + 1 + 2 = -1
    
    std::cout << "\nVerification:\n";
    std::cout << "First row sum (should be -1.0): " << h_test_result(0) << "\n";
    std::cout << "Second row sum (should be -1.0): " << h_test_result(1) << "\n";
    std::cout << "Row " << m1+1 << " sum (should be -2.0): " << h_test_result(m1+1) << "\n";
    
}


/*

This implements an optimized memory layout for the A2 matrix. The above has strided memory access:

// For variance level j and stock price i:
main_diag[j * spacing + i]

// When threads access elements in parallel for stock prices:
Thread 0: main_diag[j * spacing + 0]
Thread 1: main_diag[j * spacing + 1]
Thread 2: main_diag[j * spacing + 2]
// These accesses are spacing=301 elements apart in memory

new layout (i think, in the below we have 2D arrays where we split the varaince levels and the 
correpsonding diagonals):
// Store as:
main_diag[i * m2 + j]

// When threads access elements in parallel for stock prices:
Thread 0: main_diag[0 * m2 + j]
Thread 1: main_diag[1 * m2 + j]
Thread 2: main_diag[2 * m2 + j]
// Now consecutive threads access consecutive memory locations


THIS DOES NOT WORK YET!
*/


struct A2Storage_optimized_memory {
    int m1, m2;
    
    // Explicit system diagonals - now stored with continuous memory layout
    // First index i (stock) varies fastest for coalesced access
    Kokkos::View<double**> main_diag;     // [m1+1][m2-1]  
    Kokkos::View<double**> lower_diag;    // [m1+1][m2-2]
    Kokkos::View<double**> upper_diag;    // [m1+1][m2-1]
    Kokkos::View<double*> upper2_diag;    // [m1+1] (special case for j=0)

    // Implicit system diagonals
    Kokkos::View<double**> implicit_main_diag;   // [m1+1][m2+1]
    Kokkos::View<double**> implicit_lower_diag;  // [m1+1][m2-2]
    Kokkos::View<double**> implicit_upper_diag;  // [m1+1][m2-1]
    Kokkos::View<double*> implicit_upper2_diag;  // [m1+1]

    A2Storage_optimized_memory(int m1_, int m2_) : m1(m1_), m2(m2_) {
        main_diag = Kokkos::View<double**>("A2_main_diag", m1+1, m2-1);
        lower_diag = Kokkos::View<double**>("A2_lower_diag", m1+1, m2-2);
        upper_diag = Kokkos::View<double**>("A2_upper_diag", m1+1, m2-1);
        upper2_diag = Kokkos::View<double*>("A2_upper2_diag", m1+1);

        implicit_main_diag = Kokkos::View<double**>("A2_impl_main_diag", m1+1, m2+1);
        implicit_lower_diag = Kokkos::View<double**>("A2_impl_lower_diag", m1+1, m2-2);
        implicit_upper_diag = Kokkos::View<double**>("A2_impl_upper_diag", m1+1, m2-1);
        implicit_upper2_diag = Kokkos::View<double*>("A2_impl_upper2_diag", m1+1);
    }

    void build_implicit(const double theta, const double delta_t) {
        const int local_m1 = m1;
        const int local_m2 = m2;

        const auto local_main = main_diag;
        const auto local_upper = upper_diag;
        const auto local_lower = lower_diag;
        const auto local_upper2 = upper2_diag;

        const auto local_impl_main = implicit_main_diag;
        const auto local_impl_upper = implicit_upper_diag;
        const auto local_impl_lower = implicit_lower_diag;
        const auto local_impl_upper2 = implicit_upper2_diag;

        // Initialize implicit_main_diag with identity and handle all diagonals
        Kokkos::parallel_for("build_implicit", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {m1+1, m2+1}), 
        KOKKOS_LAMBDA(const int i, const int j) {
            // Set identity
            local_impl_main(i,j) = 1.0;
            
            // Subtract theta*delta_t*A2 from main diagonal where A2 is defined
            if(j < m2-1) {
                local_impl_main(i,j) -= theta * delta_t * local_main(i,j);
            }

            // Build the off-diagonal terms
            if(j < m2-2) {
                local_impl_lower(i,j) = -theta * delta_t * local_lower(i,j);
            }
            if(j < m2-1) {
                local_impl_upper(i,j) = -theta * delta_t * local_upper(i,j);
            }
        });

        // Handle upper2 diagonal separately (only for j=0)
        Kokkos::parallel_for("build_implicit_upper2", m1+1, KOKKOS_LAMBDA(const int i) {
            local_impl_upper2(i) = -theta * delta_t * local_upper2(i);
        });
    }

    void multiply(const Kokkos::View<double*>& x, Kokkos::View<double*>& result) {
        const int local_m1 = m1;
        const int spacing = m1 + 1;
        const auto local_main = main_diag;
        const auto local_lower = lower_diag;
        const auto local_upper = upper_diag;
        const auto local_upper2 = upper2_diag;

        // First set result to zero
        Kokkos::deep_copy(result, 0.0);

        // First block (j=0, with upper2 diagonal)
        Kokkos::parallel_for("multiply_first_block", m1+1, KOKKOS_LAMBDA(const int i) {
            double temp = local_main(i,0) * x(i);
            temp += local_upper(i,0) * x(i + spacing);
            temp += local_upper2(i) * x(i + 2*spacing);
            result(i) = temp;
        });

        // Middle blocks
        Kokkos::parallel_for("multiply_middle", 
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 1}, {m1+1, m2-1}),
            KOKKOS_LAMBDA(const int i, const int j) {
                double temp = local_lower(i,j-1) * x((j-1)*spacing + i);
                temp += local_main(i,j) * x(j*spacing + i);
                temp += local_upper(i,j) * x((j+1)*spacing + i);
                result(j*spacing + i) = temp;
        });
    }

    void solve_implicit_parallel(Kokkos::View<double*>& x, const Kokkos::View<double*>& b) {
        const int local_m1 = m1;
        const int local_m2 = m2;
        const int spacing = m1 + 1;
        const int num_rows = (local_m2-1)*spacing;
        const int total_size = (local_m2+1)*spacing;

        // Temporary storage with optimized layout
        Kokkos::View<double**> c_star("c_star", m1+1, m2-1);  // [stock][variance]
        Kokkos::View<double*> c2_star("c2_star", m1+1);       // for j=0 special case
        Kokkos::View<double**> d_star("d_star", m1+1, m2+1);  // [stock][variance]

        const auto local_impl_main = implicit_main_diag;
        const auto local_impl_lower = implicit_lower_diag;
        const auto local_impl_upper = implicit_upper_diag;
        const auto local_impl_upper2 = implicit_upper2_diag;

        // Identity block (parallel over stock prices)
        Kokkos::parallel_for("identity_block", m1+1, KOKKOS_LAMBDA(const int i) {
            for(int j = m2-1; j <= m2; j++) {
                d_star(i,j) = b(j*spacing + i);
            }
        });

        // First block normalization (parallel over stock prices)
        Kokkos::parallel_for("normalize_first_block", m1+1, KOKKOS_LAMBDA(const int i) {
            c_star(i,0) = local_impl_upper(i,0) / local_impl_main(i,0);
            c2_star(i) = local_impl_upper2(i) / local_impl_main(i,0);
            d_star(i,0) = b(i) / local_impl_main(i,0);
        });

        // First block forward sweep (parallel over stock prices)
        Kokkos::parallel_for("first_forward_sweep", m1+1, KOKKOS_LAMBDA(const int i) {
            double c_upper = local_impl_upper(i,1) - c2_star(i)*local_impl_lower(i,0);
            double m = 1.0 / (local_impl_main(i,1) - c_star(i,0)*local_impl_lower(i,0));
            c_star(i,1) = c_upper * m;
            d_star(i,1) = (b(spacing + i) - local_impl_lower(i,0) * d_star(i,0)) * m;
        });

        // Middle blocks forward sweep (sequential over variance, parallel over stock)
        for(int j = 1; j < local_m2-2; j++) {
            Kokkos::parallel_for("middle_forward_sweep", m1+1, KOKKOS_LAMBDA(const int i) {
                double m = 1.0 / (local_impl_main(i,j+1) - c_star(i,j)*local_impl_lower(i,j));
                c_star(i,j+1) = local_impl_upper(i,j+1) * m;
                d_star(i,j+1) = (b((j+1)*spacing + i) - local_impl_lower(i,j) * d_star(i,j)) * m;
            });
            Kokkos::fence();
        }

        // Pre-backward sweep (parallel over stock prices)
        Kokkos::parallel_for("pre_backward_sweep", m1+1, KOKKOS_LAMBDA(const int i) {
            d_star(i,m2-2) = d_star(i,m2-2) - d_star(i,m2-1)*c_star(i,m2-2);
            x((m2-2)*spacing + i) = d_star(i,m2-2);
        });

        // Backward sweep until upper2_diag (sequential over variance, parallel over stock)
        for(int j = local_m2-3; j >= 1; j--) {
            Kokkos::parallel_for("backward_sweep_1", m1+1, KOKKOS_LAMBDA(const int i) {
                x(j*spacing + i) = d_star(i,j) - c_star(i,j) * x((j+1)*spacing + i);
            });
            Kokkos::fence();
        }

        // First block back substitution (parallel over stock prices)
        Kokkos::parallel_for("backward_sweep_2", m1+1, KOKKOS_LAMBDA(const int i) {
            x(i) = d_star(i,0) - c_star(i,0) * x(spacing + i) - c2_star(i) * x(2*spacing + i);
            d_star(i,0) = d_star(i,0) - c2_star(i) * x(2*spacing + i);
        });

        // Last backward substitution (parallel over stock prices)
        Kokkos::parallel_for("last_backward", m1+1, KOKKOS_LAMBDA(const int i) {
            x(i) = d_star(i,0) - c_star(i,0) * x(spacing + i);
        });

        // Copy identity block (parallel over stock prices)
        Kokkos::parallel_for("copy_identity", m1+1, KOKKOS_LAMBDA(const int i) {
            for(int j = m2-1; j <= m2; j++) {
                x(j*spacing + i) = d_star(i,j);
            }
        });
    }

};

void test_A2_gpu_optimized() {
   using timer = std::chrono::high_resolution_clock;
   // Test dimensions
   const int m1 = 300;
   const int m2 = 100;
   std::cout << "Testing A2 GPU optimized implementation with dimensions m1=" << m1 << ", m2=" << m2 << "\n";

   A2Storage_optimized_memory A2(m1, m2);

   // Fill with test values on host
   auto h_main = Kokkos::create_mirror_view(A2.main_diag);
   auto h_lower = Kokkos::create_mirror_view(A2.lower_diag);
   auto h_upper = Kokkos::create_mirror_view(A2.upper_diag);
   auto h_upper2 = Kokkos::create_mirror_view(A2.upper2_diag);

   // Fill explicit diagonals with test values
   for(int i = 0; i < m1+1; i++) {
       for(int j = 0; j < m2-1; j++) {
           h_main(i,j) = 8.0;
           if(j < m2-2) h_lower(i,j) = -1.0;
           h_upper(i,j) = 1.0;
       }
       h_upper2(i) = 1.0;
   }

   // Copy to device
   Kokkos::deep_copy(A2.main_diag, h_main);
   Kokkos::deep_copy(A2.lower_diag, h_lower);
   Kokkos::deep_copy(A2.upper_diag, h_upper);
   Kokkos::deep_copy(A2.upper2_diag, h_upper2);

   // Build and test implicit system
   double theta = 0.8;
   double delta_t = 0.001;
   A2.build_implicit(theta, delta_t);

   const int total_size = (m2+1)*(m1+1);

   // Create test vectors
   Kokkos::View<double*> x("x", total_size);
   Kokkos::View<double*> b("b", total_size);
   Kokkos::View<double*> result("result", total_size);

   // Initialize b with random values
   auto h_b = Kokkos::create_mirror_view(b);
   std::srand(52); // Same seed as previous tests
   for(int i = 0; i < total_size; i++) {
       h_b(i) = std::rand() / (RAND_MAX + 1.0);
   }
   Kokkos::deep_copy(b, h_b);
   Kokkos::deep_copy(x, 0.0);

   auto t_start_solve = timer::now();
   A2.solve_implicit_parallel(x, b);
   auto t_end_solve = timer::now();

   std::cout << "Implicit gpu optimized solve time: "
             << std::chrono::duration<double>(t_end_solve - t_start_solve).count()
             << " seconds" << std::endl;

   t_start_solve = timer::now();
   A2.multiply(x, result);
   t_end_solve = timer::now();

   std::cout << "Explicit gpu optimized solve time: "
             << std::chrono::duration<double>(t_end_solve - t_start_solve).count()
             << " seconds" << std::endl;

   // Compute residual
   auto h_x = Kokkos::create_mirror_view(x);
   auto h_result = Kokkos::create_mirror_view(result);
   Kokkos::deep_copy(h_x, x);
   Kokkos::deep_copy(h_result, result);

   double residual_norm = 0.0;
   for(size_t i = 0; i < total_size; i++) {
       double res = h_x(i) - theta * delta_t * h_result(i) - h_b(i);
       residual_norm += res * res;
   }
   residual_norm = std::sqrt(residual_norm);
   std::cout << "\nFinal residual norm: " << residual_norm << "\n";
}

/*

This implements the A1 struct and is working properly. There is at the end an idea of 
optimized memory access, but the speed up was not sufficient to put more time in it 
right now

*/

struct A1Storage {
    int m1, m2;
    
    Kokkos::View<double**> main_diags;
    Kokkos::View<double**> lower_diags;
    Kokkos::View<double**> upper_diags;

    Kokkos::View<double**> implicit_main_diags;
    Kokkos::View<double**> implicit_lower_diags;
    Kokkos::View<double**> implicit_upper_diags;

    A1Storage(int m1_, int m2_) : m1(m1_), m2(m2_) {
        //2D views. we have m2+1 Blocks of tridiagonal matrices
        //this memory layout is called strided memory access and is not optimized for the gpu. However
        //below called A1Storage_memory_optimized is s first impleemntation of an optimized version.
        //this version isnt entirely correct, but shows that further development isnt time effiecient right
        //now. For optimized memory acces we moved from one Option price takin 0.062 s to 0.012.
        /*
        //  Currently:
            main_diags(j,i)    // j-th block, i-th diagonal element   
            Thread 0 accesses main_diags(0,0), main_diags(0,1), ...
            Thread 1 accesses main_diags(1,0), main_diags(1,1), ...  
        */
        main_diags = Kokkos::View<double**>("A1_main_diags", m2+1, m1+1);
        lower_diags = Kokkos::View<double**>("A1_lower_diags", m2+1, m1);
        upper_diags = Kokkos::View<double**>("A1_upper_diags", m2+1, m1);

        implicit_main_diags = Kokkos::View<double**>("A1_impl_main_diags", m2+1, m1+1);
        implicit_lower_diags = Kokkos::View<double**>("A1_impl_lower_diags", m2+1, m1);
        implicit_upper_diags = Kokkos::View<double**>("A1_impl_upper_diags", m2+1, m1);
    }

    void build_implicit(const double theta, const double delta_t) {
        const int local_m1 = m1;
        const int local_m2 = m2;
        const auto local_main = main_diags;
        const auto local_lower = lower_diags;
        const auto local_upper = upper_diags;
        const auto local_impl_main = implicit_main_diags;
        const auto local_impl_lower = implicit_lower_diags;
        const auto local_impl_upper = implicit_upper_diags;

        Kokkos::parallel_for("build_implicit", 1, KOKKOS_LAMBDA(const int) {
            for(int j = 0; j <= local_m2; j++) {
                for(int i = 0; i <= local_m1; i++) {
                    local_impl_main(j,i) = 1.0 - theta * delta_t * local_main(j,i);
                }
                for(int i = 0; i < local_m1; i++) {
                    local_impl_lower(j,i) = -theta * delta_t * local_lower(j,i);
                    local_impl_upper(j,i) = -theta * delta_t * local_upper(j,i);
                }
            }
        });
        Kokkos::fence();
    }

    /*
    void multiply(const Kokkos::View<double*>& x, Kokkos::View<double*>& result) {
        const int local_m1 = m1;
        const int local_m2 = m2;
        const auto local_main = main_diags;
        const auto local_lower = lower_diags;
        const auto local_upper = upper_diags;

        Kokkos::parallel_for("multiply", Kokkos::RangePolicy<>(0, 1), KOKKOS_LAMBDA(const int) {
            for(int j = 0; j <= local_m2; j++) {
                const int offset = j * (local_m1 + 1);
                for(int i = 0; i <= local_m1; i++) {
                    double sum = local_main(j,i) * x(offset + i);
                    if(i > 0) {
                        sum += local_lower(j,i-1) * x(offset + i-1);
                    }
                    if(i < local_m1) {
                        sum += local_upper(j,i) * x(offset + i+1);
                    }
                    result(offset + i) = sum;
                }
            }
        });
        Kokkos::fence();
    }
    */

    //This is a sequential implicict solve. Only one thread is handling everything
    /*
    void solve_implicit(Kokkos::View<double*>& x, const Kokkos::View<double*>& b) {
        const int local_m1 = m1;
        const int local_m2 = m2;
        const auto local_impl_main = implicit_main_diags;
        const auto local_impl_lower = implicit_lower_diags;
        const auto local_impl_upper = implicit_upper_diags;
        
        Kokkos::View<double*> temp("temp", m1+1); //each block has dimesnion (m1+1)x(m1+1), A1_block x = d_block

        Kokkos::parallel_for("solve_implicit", Kokkos::RangePolicy<>(0, 1), KOKKOS_LAMBDA(const int) {
            for(int j = 0; j <= local_m2; j++) {
                const int offset = j * (local_m1 + 1); //index for first entry of b for each block
                
                // Forward sweep

                //first entry (0,0) of each block is 1 since A1 (0,0) is 0 (boundary condition at s=s_min=0 is 0)
                //so I - thata*delta_t*A1 is 1
                temp(0) = local_impl_main(j,0); //this is 1 for each j (see right above explanaiton)
                x(offset) = b(offset);
                
                for(int i = 1; i <= local_m1; i++) {
                    double m = local_impl_lower(j,i-1) / temp(i-1); // 
                    temp(i) = local_impl_main(j,i) - m * local_impl_upper(j,i-1); // (wikipedia notation) b_2-a2*c_1', this is the new value of the diagonal
                    x(offset + i) = b(offset + i) - m * x(offset + i-1); //updating solution with x_2 = x_2 - m * x_1
                }

                // Back substitution
                x(offset + local_m1) /= temp(local_m1);
                for(int i = local_m1-1; i >= 0; i--) {
                    x(offset + i) = (x(offset + i) - 
                        local_impl_upper(j,i) * x(offset + i+1)) / temp(i);
                }
            }
        });
        Kokkos::fence();
    }
    */
    
    //This is the explicit step but in parallel (at the outer level of blocks)! We parallize over the independent blocks of tridiagonal matrices
    /*
    void multiply(const Kokkos::View<double*>& x, Kokkos::View<double*>& result) {
        const int local_m1 = m1;
        const auto local_main = main_diags;
        const auto local_lower = lower_diags;
        const auto local_upper = upper_diags;

        // Parallelize over variance levels j
        Kokkos::parallel_for("multiply", m2+1, KOKKOS_LAMBDA(const int j) {
            const int offset = j * (local_m1 + 1);
            
            // Process block j
            for(int i = 0; i <= local_m1; i++) {
                double sum = local_main(j,i) * x(offset + i);
                if(i > 0) {
                    sum += local_lower(j,i-1) * x(offset + i-1);
                }
                if(i < local_m1) {
                    sum += local_upper(j,i) * x(offset + i+1);
                }
                result(offset + i) = sum;
            }
        });
        Kokkos::fence();
    }
    */
    
    //This is the implicit step but in parallel (at the outer level of blocks)! We parallize over the independent blocks of tridiagonal matrices
    
    void solve_implicit(Kokkos::View<double*>& x, const Kokkos::View<double*>& b) {
        const int local_m1 = m1;
        const auto local_impl_main = implicit_main_diags;
        const auto local_impl_lower = implicit_lower_diags;
        const auto local_impl_upper = implicit_upper_diags;
        
        // Create temporary storage for each thread
        // One temp array per tridiagonal system
        Kokkos::View<double**> temp("temp", m2+1, m1+1);

        // Parallelize over variance levels j
        Kokkos::parallel_for("solve_implicit", m2+1, KOKKOS_LAMBDA(const int j) {
            const int offset = j * (local_m1 + 1);
            
            // Forward sweep for block j
            temp(j,0) = local_impl_main(j,0);
            x(offset) = b(offset);
            
            for(int i = 1; i <= local_m1; i++) {
                double m = local_impl_lower(j,i-1) / temp(j,i-1);
                temp(j,i) = local_impl_main(j,i) - m * local_impl_upper(j,i-1);
                x(offset + i) = b(offset + i) - m * x(offset + i-1);
            }

            // Back substitution for block j
            x(offset + local_m1) /= temp(j,local_m1);
            for(int i = local_m1-1; i >= 0; i--) {
                x(offset + i) = (x(offset + i) - 
                    local_impl_upper(j,i) * x(offset + i+1)) / temp(j,i);
            }
        });
        Kokkos::fence();
    }
    
   
    // This is the explicit step using team parallelism.
    // We parallelize over variance levels (j) and stock price levels (i) within each team.
    void multiply(const Kokkos::View<double*>& x, Kokkos::View<double*>& result) {
        const int local_m1 = m1;
        const auto local_main = main_diags;
        const auto local_lower = lower_diags;
        const auto local_upper = upper_diags;

        // Define team policy: one team per variance level j
        using team_policy = Kokkos::TeamPolicy<>;
        using member_type = team_policy::member_type;

        // Parallelize over variance levels j
        Kokkos::parallel_for("multiply", team_policy(m2+1, Kokkos::AUTO), KOKKOS_LAMBDA(const member_type& team_member) {
            const int j = team_member.league_rank();
            const int offset = j * (local_m1 + 1);

            // Parallelize over i within each team
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, local_m1 + 1), [=](const int i) {
                double sum = local_main(j, i) * x(offset + i);
                if (i > 0) {
                    sum += local_lower(j, i - 1) * x(offset + i - 1);
                }
                if (i < local_m1) {
                    sum += local_upper(j, i) * x(offset + i + 1);
                }
                result(offset + i) = sum;
            });
        });
        Kokkos::fence();
    }

    /*
    // This is the implicit step using team parallelism.
    // We parallelize over variance levels (j) and stock price levels (i) within each team. The idea is to apply parallised 
    //cyclic reduction. The code doesnt run correctly (high residual)
    void solve_implicit(Kokkos::View<double*>& x, const Kokkos::View<double*>& b) {
        const int local_m1 = m1;
        const auto local_impl_main = implicit_main_diags;
        const auto local_impl_lower = implicit_lower_diags;
        const auto local_impl_upper = implicit_upper_diags;

        // Number of equations in each tridiagonal system
        const int n = local_m1 + 1;

        // Parallelize over variance levels j
        using team_policy = Kokkos::TeamPolicy<>;
        using member_type = team_policy::member_type;

        Kokkos::parallel_for("solve_implicit_pcr", team_policy(m2 + 1, Kokkos::AUTO), KOKKOS_LAMBDA(const member_type& team_member) {
            const int j = team_member.league_rank();
            const int offset = j * n;

            // Allocate shared memory for coefficients and right-hand side
            Kokkos::View<double*, Kokkos::MemoryUnmanaged> a(&local_impl_lower(j, 0), n - 1);
            Kokkos::View<double*, Kokkos::MemoryUnmanaged> b_diag(&local_impl_main(j, 0), n);
            Kokkos::View<double*, Kokkos::MemoryUnmanaged> c(&local_impl_upper(j, 0), n - 1);
            Kokkos::View<double*, Kokkos::MemoryUnmanaged> d(&x(offset), n);

            // Copy right-hand side into d
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, n), [=](int i) {
                d(i) = b(offset + i);
            });

            team_member.team_barrier();

            // Perform cyclic reduction
            int m = n;
            int log2n = 0;
            while (m > 1) {
                m = (m + 1) / 2;
                log2n++;
            }

            for (int s = 0; s < log2n; s++) {
                int stride = 1 << s;

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, n), [=](int i) {
                    if (i % (2 * stride) == stride) {
                        // Compute new coefficients
                        int i_left = i - stride;
                        int i_right = i + stride;

                        double alpha = a(i_left) / b_diag(i_left);
                        double gamma = c(i_right) / b_diag(i_right);

                        b_diag(i) = b_diag(i) - alpha * c(i_left) - gamma * a(i_right);
                        d(i) = d(i) - alpha * d(i_left) - gamma * d(i_right);

                        a(i) = -alpha * a(i_left);
                        c(i) = -gamma * c(i_right);
                    }
                });

                team_member.team_barrier();
            }

            // Solve the reduced system (should be small enough now)
            if (team_member.team_rank() == 0) {
                int mid = n / 2;
                x(offset + mid) = d(mid) / b_diag(mid);
            }

            team_member.team_barrier();

            // Back substitution phase
            for (int s = log2n - 1; s >= 0; s--) {
                int stride = 1 << s;

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, n), [=](int i) {
                    if (i % (2 * stride) == 0 && i + stride < n) {
                        int i_left = i;
                        int i_right = i + stride;

                        x(offset + i_right) = (d(i_right) - a(i_right) * x(offset + i_left) - c(i_right) * x(offset + i_right + stride)) / b_diag(i_right);
                    }
                });

                team_member.team_barrier();
            }
        });
        Kokkos::fence();
    }
    */
};

//this is the first initial test. It perfomas an explicit and implciit solver for the A1 struct
void test_A1_solve() {
    using timer = std::chrono::high_resolution_clock;
    std::cout << "Starting A1 Storage test...\n";
    
    // Problem dimensions
    const int m1 = 300;  // stock price points
    const int m2 = 100;  // variance points
    const int total_size = (m1 + 1) * (m2 + 1);

    std::cout << "Total size (m1 + 1) * (m2 + 1):" << (m1 + 1) * (m2 + 1) << std::endl;

    // Create A1 storage with dimensions
    A1Storage A1(m1, m2);

    // Fill with test values on host
    auto h_main = Kokkos::create_mirror_view(A1.main_diags);
    auto h_lower = Kokkos::create_mirror_view(A1.lower_diags);
    auto h_upper = Kokkos::create_mirror_view(A1.upper_diags);

    Kokkos::deep_copy(h_main, 0.0);
    Kokkos::deep_copy(h_lower, 0.0);
    Kokkos::deep_copy(h_upper, 0.0);

    for(int j = 0; j <= m2; j++) {
        for(int i = 0; i <= m1; i++) {
            if (i == 0) {
                // Boundary points
                h_main(j, i) = 0.0;
            }  
            else if (i == m1) {
                // Boundary points
                h_main(j, i) = -2;
            } else {
                // Internal points
                h_main(j, i) = -2;
                h_lower(j, i - 1) = 1.0;
                h_upper(j, i) = 1.0;
            }
        }
    }

    // Copy to device
    Kokkos::deep_copy(A1.main_diags, h_main);
    Kokkos::deep_copy(A1.lower_diags, h_lower);
    Kokkos::deep_copy(A1.upper_diags, h_upper);

    /*
    // Print structure of first block
    int block_j = 0;
    std::cout << "\nBlock (j=" << block_j << ") diagonals:\n";
    std::cout << "Main diagonal: ";
    for(int i = 0; i <= m1; i++) std::cout << h_main(block_j,i) << " ";
    std::cout << "\nLower diagonal: ";
    for(int i = 0; i < m1; i++) std::cout << h_lower(block_j,i) << " ";
    std::cout << "\nUpper diagonal: ";
    for(int i = 0; i < m1; i++) std::cout << h_upper(block_j,i) << " ";
    std::cout << "\n";
    */
    

    // Build implicit system
    double theta = 1.0;      // Crank-Nicolson
    double delta_t = 1.0;   // time step
    A1.build_implicit(theta, delta_t);

    // Create test vectors
    Kokkos::View<double*> x("x", total_size);
    Kokkos::View<double*> b("b", total_size);
    Kokkos::View<double*> result("result", total_size);

    // Initialize b with ones on host
    auto h_b = Kokkos::create_mirror_view(b);
    for(int i = 0; i < total_size; i++) {
        h_b(i) = 1.0;
    }

    //copy to device
    Kokkos::deep_copy(b, h_b);

    // Test explicit multiply
    std::cout << "\nTesting explicit multiply...\n";

    auto t_start_multip = timer::now();
    A1.multiply(b, result);
    auto t_end_multip = timer::now();

    std::cout << "Multiply solve time: "
              << std::chrono::duration<double>(t_end_multip - t_start_multip).count()
              << " seconds" << std::endl;

    // Check multiply result
    auto h_test = Kokkos::create_mirror_view(result);
    Kokkos::deep_copy(h_test, result);
    //std::cout << "After multiply, first few results: ";
    //for(int i = 0; i < total_size; i++) std::cout << h_test(i) << " ";
    //std::cout << "\n";

    // Test implicit solve
    std::cout << "\nTesting implicit solve...\n";

    auto t_start_solve = timer::now();
    A1.solve_implicit(x, b);
    auto t_end_solve = timer::now();

    std::cout << "implicit solve time: "
              << std::chrono::duration<double>(t_end_solve - t_start_solve).count()
              << " seconds" << std::endl;


    // Get results
    auto h_x = Kokkos::create_mirror_view(x);
    auto h_result = Kokkos::create_mirror_view(result);
    Kokkos::deep_copy(h_x, x);
    Kokkos::deep_copy(h_result, result);
    
    /*
    // Print results for first block
    std::cout << "\nResults for first block (j=0):\n";
    std::cout << "Explicit multiply result: ";
    for(int i = 0; i <= m1; i++) {
        std::cout << h_result(i) << " ";
    }
    std::cout << "\nImplicit solve result: ";
    for(int i = 0; i <= m1; i++) {
        std::cout << h_x(i) << " ";
    }
    std::cout << "\n";
    */

    // Verify implicit solve by computing residual
    // (I - θΔt*A1)x - b
    Kokkos::View<double*> residual("residual", total_size);
    Kokkos::View<double*> temp("temp", total_size);
    
    // First compute A1*x
    A1.multiply(x, temp);
    
    // Then compute residual on host
    auto h_temp = Kokkos::create_mirror_view(temp);
    auto h_residual = Kokkos::create_mirror_view(residual);
    Kokkos::deep_copy(h_temp, temp);
    
    double residual_norm = 0.0;
    for(int i = 0; i < total_size; i++) {
        h_residual(i) = h_x(i) - theta * delta_t * h_temp(i) - h_b(i);
        residual_norm += h_residual(i) * h_residual(i);
    }
    residual_norm = std::sqrt(residual_norm);

    std::cout << "\nResidual norm: " << residual_norm << "\n";
}

//this test prints out the matrix A1 struct. Therefore only a visual test
template <typename ViewType>
void print_A1_matrix(const ViewType& lower,const ViewType& main,const ViewType& upper,int m1, int j) {
    std::cout << "Tridiagonal matrix for block j=" << j << ":\n";
    for(int i = 0; i <= m1; i++) {
        for(int k = 0; k <= m1; k++) {
            if(k == i - 1 && i > 0) {
                std::cout << lower(j, i - 1) << "\t";
            } else if(k == i) {
                std::cout << main(j, i) << "\t";
            } else if(k == i + 1 && i < m1) {
                std::cout << upper(j, i) << "\t";
            } else {
                std::cout << "0\t";
            }
        }
        std::cout << "\n";
    }
}

//this test is implemented as an "advanced" explicit and implicit step. where the first vector
// A*v=b is formed randomly and then solve Ax=b to compare the residual from v and x
void test_A1_solve_for_random_vector() {
    using timer = std::chrono::high_resolution_clock;
    std::cout << "Starting A1 Storage test...\n";
    
    // Problem dimensions (small for visualization)
    const int m1 = 300;  // stock price points
    const int m2 = 100;  // variance points
    const int total_size = (m1 + 1) * (m2 + 1);

    std::cout << "Total size (m1 + 1) * (m2 + 1): " << total_size << std::endl;

    // Create A1 storage with dimensions
    A1Storage A1(m1, m2);

    // Fill with test values on host
    auto h_main = Kokkos::create_mirror_view(A1.main_diags);
    auto h_lower = Kokkos::create_mirror_view(A1.lower_diags);
    auto h_upper = Kokkos::create_mirror_view(A1.upper_diags);

    // Initialize all diagonals to zero
    //to follow python implementation
    Kokkos::deep_copy(h_main, 0.0);
    Kokkos::deep_copy(h_lower, 0.0);
    Kokkos::deep_copy(h_upper, 0.0);

    /*
    //THIS IS WRONG
    
    
     FOR THE SPECIFIC A1 structure. The below implements a TRUE tridiagonal matrix.
    //each block isnt one but one where the first row is zero and the last as well (boundary conditions)
    // Simple test pattern: -2 on main diagonal, 1 on off-diagonals
    for(int j = 0; j <= m2; j++) {
        for(int i = 1; i < m1; i++) {
            h_main(j,i) = -2.0;
            if(i < m1) {
                h_lower(j,i) = 1.0;
                h_upper(j,i) = 1.0;
            }
        }
        h_main(j,m1) = -2.0;
    }
    */

    for(int j = 0; j <= m2; j++) {
        for(int i = 0; i <= m1; i++) {
            if (i == 0) {
                // Boundary points
                h_main(j, i) = 0.0;
            }  
            else if (i == m1) {
                // Boundary points
                h_main(j, i) = -2.0;
            } else {
                // Internal points
                h_main(j, i) = -2.0;
                h_lower(j, i - 1) = 1.0;
                h_upper(j, i) = 1.0;
            }
        }
    }

    
    // Copy to device
    Kokkos::deep_copy(A1.main_diags, h_main);
    Kokkos::deep_copy(A1.lower_diags, h_lower);
    Kokkos::deep_copy(A1.upper_diags, h_upper);

    /*
    // Print structure of first block
    std::cout << "\nFirst block (j=0) diagonals:\n";
    std::cout << "Main diagonal: ";
    for(int i = 0; i <= m1; i++) std::cout << h_main(0,i) << " ";
    std::cout << "\nLower diagonal: ";
    for(int i = 0; i < m1; i++) std::cout << h_lower(0,i) << " ";
    std::cout << "\nUpper diagonal: ";
    for(int i = 0; i < m1; i++) std::cout << h_upper(0,i) << " ";
    std::cout << "\n";
    

    // Visualize tridiagonal matrix for the first block
    print_A1_matrix(h_lower, h_main, h_upper, m1, 0);
    */

    
    // Print structure of all blocks
    /*
    std::cout << "\nAll blocks' diagonals:\n";
    for(int j = 0; j <= m2; j++) {
        std::cout << "Block j=" << j << ":\n";
        std::cout << "Main diagonal: ";
        for(int i = 0; i <= m1; i++) std::cout << h_main(j,i) << " ";
        std::cout << "\nLower diagonal: ";
        for(int i = 0; i < m1; i++) std::cout << h_lower(j,i) << " ";
        std::cout << "\nUpper diagonal: ";
        for(int i = 0; i < m1; i++) std::cout << h_upper(j,i) << " ";
        std::cout << "\n";
    }
    */
    

    // Build implicit system
    double theta = 0.8;      // Crank-Nicolson
    double delta_t = 0.001;   // time step
    A1.build_implicit(theta, delta_t);

    /*
    // Visualize tridiagonal matrix for the first block of the implcict matrix I-theta*delta_t*A1
    // Create host mirrors for the implicit matrices
    auto h_impl_main = Kokkos::create_mirror_view(A1.implicit_main_diags);
    auto h_impl_lower = Kokkos::create_mirror_view(A1.implicit_lower_diags);
    auto h_impl_upper = Kokkos::create_mirror_view(A1.implicit_upper_diags);

    // Copy implicit matrices from device to host
    Kokkos::deep_copy(h_impl_main, A1.implicit_main_diags);
    Kokkos::deep_copy(h_impl_lower, A1.implicit_lower_diags);
    Kokkos::deep_copy(h_impl_upper, A1.implicit_upper_diags);

    // Print structure of the implicit matrix for the first block (j=0)
    std::cout << "\nImplicit matrix for first block (j=0):\n";
    print_A1_matrix(h_impl_lower, h_impl_main, h_impl_upper, m1, 0);
    */

    // Create test vectors
    Kokkos::View<double*> x("x", total_size);
    Kokkos::View<double*> b("b", total_size);
    Kokkos::View<double*> result("result", total_size);

    // Initialize b with random values
    auto h_b = Kokkos::create_mirror_view(b);
    std::srand(52); // Seed for reproducibility
    for(int i = 0; i < total_size; i++) {
        h_b(i) = std::rand() / (RAND_MAX + 1.0);
    }
    Kokkos::deep_copy(b, h_b);

    // Copy b to b_copy for residual calculation
    Kokkos::View<double*> b_copy("b_copy", total_size);
    Kokkos::deep_copy(b_copy, b);
    
    // Test explicit multiply
    std::cout << "\nTesting explicit multiply...\n";

    auto t_start_multip = timer::now();
    A1.multiply(b, result);
    auto t_end_multip = timer::now();

    std::cout << "Multiply solve time: "
              << std::chrono::duration<double>(t_end_multip - t_start_multip).count()
              << " seconds" << std::endl;

    // Test implicit solve
    std::cout << "\nTesting implicit solve...\n";

    auto t_start_solve = timer::now();
    A1.solve_implicit(x, b);
    auto t_end_solve = timer::now();

    std::cout << "Implicit solve time: "
              << std::chrono::duration<double>(t_end_solve - t_start_solve).count()
              << " seconds" << std::endl;

    // Get results
    auto h_x = Kokkos::create_mirror_view(x);
    auto h_result = Kokkos::create_mirror_view(result);
    Kokkos::deep_copy(h_x, x);
    Kokkos::deep_copy(h_result, result);

    
    // Print results for first block
    /*
    std::cout << "\nResults for first block (j=0):\n";
    std::cout << "b vector: ";
    for(int i = 0; i <= m1; i++) {
        int idx = i + 0 * (m1 + 1);
        std::cout << h_b(idx) << " ";
    }
    std::cout << "\nExplicit multiply result: ";
    for(int i = 0; i <= m1; i++) {
        int idx = i + 0 * (m1 + 1);
        std::cout << h_result(idx) << " ";
    }
    std::cout << "\nImplicit solve result: ";
    for(int i = 0; i <= m1; i++) {
        int idx = i + 0 * (m1 + 1);
        std::cout << h_x(idx) << " ";
    }
    std::cout << "\n";
    */
    

    // Verify implicit solve by computing residual
    // (I - θΔt*A1)*x - b_copy
    Kokkos::View<double*> residual("residual", total_size);
    Kokkos::View<double*> temp("temp", total_size);
    
    // First compute A1*x
    A1.multiply(x, temp);
    
    // Then compute residual on host
    auto h_temp = Kokkos::create_mirror_view(temp);
    auto h_residual = Kokkos::create_mirror_view(residual);
    Kokkos::deep_copy(h_temp, temp);
    auto h_b_copy = Kokkos::create_mirror_view(b_copy);
    Kokkos::deep_copy(h_b_copy, b_copy);
    
    double residual_norm = 0.0;
    for(int i = 0; i < total_size; i++) {
        h_residual(i) = h_x(i) - theta * delta_t * h_temp(i) - h_b_copy(i);
        residual_norm += h_residual(i) * h_residual(i);
    }
    residual_norm = std::sqrt(residual_norm);

    std::cout << "\nResidual norm: " << residual_norm << "\n";
}

//this is only a vizual test. It shows that the expliccit method accesses the right entries
void test_A1_visualization() {
    std::cout << "\nStarting A1 Storage Visualization Test...\n";
    
    // Small dimensions for clear visualization
    const int m1 = 5;  // stock price points 
    const int m2 = 5;  // variance points
    const int total_size = (m1 + 1) * (m2 + 1);

    std::cout << "Grid dimensions:\n";
    std::cout << "m1 (stock points) = " << m1 << " (giving " << m1+1 << " points including boundaries)\n";
    std::cout << "m2 (variance points) = " << m2 << " (giving " << m2+1 << " points including boundaries)\n";
    std::cout << "Total size = " << total_size << "\n\n";

    // Create A1 storage
    A1Storage A1(m1, m2);

    // Create host mirrors for visualization
    auto h_main = Kokkos::create_mirror_view(A1.main_diags);
    auto h_lower = Kokkos::create_mirror_view(A1.lower_diags);
    auto h_upper = Kokkos::create_mirror_view(A1.upper_diags);

    // Initialize with a clear pattern
    for(int j = 0; j <= m2; j++) {
        for(int i = 0; i <= m1; i++) {
            if(i == 0) {
                // First point in each block
                h_main(j, i) = 0.0;
            }
            else if(i == m1) {
                // Last point in each block
                h_main(j, i) = -2.0;
            }
            else {
                // Interior points
                h_main(j, i) = -2.0;
                h_lower(j, i-1) = 1.0;  // Lower diagonal
                h_upper(j, i) = 1.0;    // Upper diagonal
            }
        }
    }

    // Copy to device
    Kokkos::deep_copy(A1.main_diags, h_main);
    Kokkos::deep_copy(A1.lower_diags, h_lower);
    Kokkos::deep_copy(A1.upper_diags, h_upper);

    // Print the full structure for all blocks
    std::cout << "Full A1 matrix structure:\n";
    std::cout << "-------------------------\n";
    for(int j = 0; j <= m2; j++) {
        std::cout << "\nBlock j=" << j << ":\n";
        std::cout << "Main diagonal:  ";
        for(int i = 0; i <= m1; i++) {
            printf("%6.2f ", h_main(j,i));
        }
        std::cout << "\nLower diagonal: ";
        printf("%6s ", ""); // Spacing for first element
        for(int i = 0; i < m1; i++) {
            printf("%6.2f ", h_lower(j,i));
        }
        std::cout << "\nUpper diagonal: ";
        for(int i = 0; i < m1; i++) {
            printf("%6.2f ", h_upper(j,i));
        }
        printf("%6s ", ""); // Spacing for last element
        std::cout << "\n";
    }

    // Create and initialize test vector
    Kokkos::View<double*> x("x", total_size);
    auto h_x = Kokkos::create_mirror_view(x);
    
    // Initialize with increasing values for clear visualization
    for(int i = 0; i < total_size; i++) {
        h_x(i) = i + 1.0;
    }
    Kokkos::deep_copy(x, h_x);

    // Create result vector
    Kokkos::View<double*> result("result", total_size);
    auto h_result = Kokkos::create_mirror_view(result);

    // Perform multiply
    A1.multiply(x, result);
    Kokkos::deep_copy(h_result, result);

    // Print test vector and result block by block
    std::cout << "\nTest vector x and A1*x result:\n";
    std::cout << "------------------------------\n";
    for(int j = 0; j <= m2; j++) {
        std::cout << "\nBlock j=" << j << ":\n";
        std::cout << "x:      ";
        for(int i = 0; i <= m1; i++) {
            printf("%6.2f ", h_x(j*(m1+1) + i));
        }
        std::cout << "\nA1*x:   ";
        for(int i = 0; i <= m1; i++) {
            printf("%6.2f ", h_result(j*(m1+1) + i));
        }
        std::cout << "\n";
    }
}

//this struct tries to optimize memroy layout for the gpu. It is not fully working yet, but after
//some initial tests on performance the outcome was so little that i stopped working on it for now
struct A1Storage_memory_optimized {
    int m1, m2;
    
    Kokkos::View<double*> main_diags;
    Kokkos::View<double*> lower_diags;
    Kokkos::View<double*> upper_diags;
    Kokkos::View<double*> implicit_main_diags;
    Kokkos::View<double*> implicit_lower_diags;
    Kokkos::View<double*> implicit_upper_diags;

    A1Storage_memory_optimized(int m1_, int m2_) : m1(m1_), m2(m2_) {
        main_diags = Kokkos::View<double*>("A1_main_diags", (m2+1) * (m1+1));
        lower_diags = Kokkos::View<double*>("A1_lower_diags", (m2+1) * m1);
        upper_diags = Kokkos::View<double*>("A1_upper_diags", (m2+1) * m1);

        implicit_main_diags = Kokkos::View<double*>("A1_impl_main_diags", (m2+1) * (m1+1));
        implicit_lower_diags = Kokkos::View<double*>("A1_impl_lower_diags", (m2+1) * m1);
        implicit_upper_diags = Kokkos::View<double*>("A1_impl_upper_diags", (m2+1) * m1);
    }


    void build_implicit(const double theta, const double delta_t) {
        const int local_m1 = m1;
        const int local_m2 = m2;
        const int total_size_main = (local_m1 + 1) * (local_m2 + 1);
        const int total_size_off = local_m1 * (local_m2 + 1);

        const auto local_main = main_diags;
        const auto local_lower = lower_diags;
        const auto local_upper = upper_diags;
        auto local_impl_main = implicit_main_diags;
        auto local_impl_lower = implicit_lower_diags;
        auto local_impl_upper = implicit_upper_diags;

        // Update main diagonals
        Kokkos::parallel_for("build_implicit_main", total_size_main, KOKKOS_LAMBDA(const int idx) {
            local_impl_main(idx) = 1.0 - theta * delta_t * local_main(idx);
        });

        // Update lower diagonals
        Kokkos::parallel_for("build_implicit_lower", total_size_off, KOKKOS_LAMBDA(const int idx) {
            local_impl_lower(idx) = -theta * delta_t * local_lower(idx);
        });

        // Update upper diagonals
        Kokkos::parallel_for("build_implicit_upper", total_size_off, KOKKOS_LAMBDA(const int idx) {
            local_impl_upper(idx) = -theta * delta_t * local_upper(idx);
        });

        // No need for Kokkos::fence() here, as parallel_fors are synchronous
        Kokkos::fence();
    }

    void multiply(const Kokkos::View<double*>& x, Kokkos::View<double*>& result) {
        const int local_m1 = m1;
        const auto local_main = main_diags;
        const auto local_lower = lower_diags;
        const auto local_upper = upper_diags;

        Kokkos::parallel_for("multiply", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {m2+1, m1+1}), 
        KOKKOS_LAMBDA(const int j, const int i) {
            const int main_index = j*(local_m1 + 1) + i;
            const int lower_index = j*local_m1 + (i-1); // Offset for lower diagonal
            const int upper_index = j*local_m1 + i;     // Offset for upper diagonal
            
            double sum = local_main(main_index) * x(main_index);
            
            if (i > 0) {
                sum += local_lower(lower_index) * x(main_index - 1);
            }
            if (i < local_m1) {
                sum += local_upper(upper_index) * x(main_index + 1);
            }
            result(main_index) = sum;
        });
        Kokkos::fence();
    }

    void solve_implicit(Kokkos::View<double*>& x, const Kokkos::View<double*>& b) {
        const int local_m1 = m1;
        const auto local_impl_main = implicit_main_diags;
        const auto local_impl_lower = implicit_lower_diags;
        const auto local_impl_upper = implicit_upper_diags;

        Kokkos::parallel_for("solve_implicit", m2+1, KOKKOS_LAMBDA(const int j) {
            const int main_offset = j * (local_m1 + 1);
            const int off_offset = j * local_m1;
            
            // Forward sweep
            double temp = local_impl_main(main_offset);
            x(main_offset) = b(main_offset);
            
            for(int i = 1; i <= local_m1; i++) {
                const int curr_main_idx = main_offset + i;
                const int curr_lower_idx = off_offset + (i-1);
                const int curr_upper_idx = off_offset + (i-1);
                
                double m = local_impl_lower(curr_lower_idx) / temp;
                temp = local_impl_main(curr_main_idx) - m * local_impl_upper(curr_upper_idx);
                x(curr_main_idx) = b(curr_main_idx) - m * x(curr_main_idx - 1);
            }

            // Back substitution
            x(main_offset + local_m1) /= temp;
            for(int i = local_m1-1; i >= 0; i--) {
                const int curr_main_idx = main_offset + i;
                const int curr_upper_idx = off_offset + i;
                
                x(curr_main_idx) = (x(curr_main_idx) - 
                    local_impl_upper(curr_upper_idx) * x(curr_main_idx + 1)) / 
                    local_impl_main(curr_main_idx);
            }
        });
        Kokkos::fence();
    }
    /*

    //This is the implicit step but in parallel (at the outer level of blocks)! We parallize over the independent blocks of tridiagonal matrices
    void solve_implicit(Kokkos::View<double*>& x, const Kokkos::View<double*>& b) {
        const int local_m1 = m1;
        const auto local_impl_main = implicit_main_diags;
        const auto local_impl_lower = implicit_lower_diags;
        const auto local_impl_upper = implicit_upper_diags;

        Kokkos::parallel_for("solve_implicit", Kokkos::RangePolicy<>(0, m2+1), KOKKOS_LAMBDA(const int j) {
            const int offset = j * (local_m1 + 1);
            // Temporary arrays for coefficients and solution
            Kokkos::View<double*, Kokkos::MemoryUnmanaged> a(&local_impl_lower(offset), local_m1);
            Kokkos::View<double*, Kokkos::MemoryUnmanaged> b_diag(&local_impl_main(offset), local_m1 + 1);
            Kokkos::View<double*, Kokkos::MemoryUnmanaged> c(&local_impl_upper(offset), local_m1);
            Kokkos::View<double*, Kokkos::MemoryUnmanaged> d(&x(offset), local_m1 + 1);

            // Copy right-hand side into d
            for (int i = 0; i <= local_m1; i++) {
                d(i) = b(offset + i);
            }

            // Forward sweep
            for (int i = 1; i <= local_m1; i++) {
                double m = a(i - 1) / b_diag(i - 1);
                b_diag(i) -= m * c(i - 1);
                d(i) -= m * d(i - 1);
            }

            // Back substitution
            d(local_m1) /= b_diag(local_m1);
            for (int i = local_m1 - 1; i >= 0; i--) {
                d(i) = (d(i) - c(i) * d(i + 1)) / b_diag(i);
            }

            // Write the solution back to x
            for (int i = 0; i <= local_m1; i++) {
                x(offset + i) = d(i);
            }
        });
        Kokkos::fence();
    }

    
    // This is the explicit step using team parallelism.
    // We parallelize over variance levels (j) and stock price levels (i) within each team.
    void multiply(const Kokkos::View<double*>& x, Kokkos::View<double*>& result) {
        const int local_m1 = m1;
        const auto local_main = main_diags;
        const auto local_lower = lower_diags;
        const auto local_upper = upper_diags;

        // Parallelize over variance levels j and stock price levels i
        Kokkos::parallel_for("multiply", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {m2+1, m1+1}), KOKKOS_LAMBDA(const int j, const int i) {
            int index = i + j * (local_m1 + 1);
            double sum = local_main(index) * x(index);
            if (i > 0) {
                sum += local_lower(index) * x(index - 1);
            }
            if (i < local_m1) {
                sum += local_upper(index) * x(index + 1);
            }
            result(index) = sum;
        });
        Kokkos::fence();
    }
    */
};

//this test 
void test_A1_solve_memory_optimized() {
    using timer = std::chrono::high_resolution_clock;
    std::cout << "Starting A1 Storage Memory Optimized test...\n";
    
    // Problem dimensions
    const int m1 = 300;  // stock price points
    const int m2 = 100;  // variance points
    const int total_size = (m1 + 1) * (m2 + 1);

    std::cout << "Total size (m1 + 1) * (m2 + 1): " << total_size << std::endl;

    // Create A1 storage with dimensions
    A1Storage_memory_optimized A1(m1, m2);

    // Fill with test values on host
    auto h_main = Kokkos::create_mirror_view(A1.main_diags);
    auto h_lower = Kokkos::create_mirror_view(A1.lower_diags);
    auto h_upper = Kokkos::create_mirror_view(A1.upper_diags);

    // Simple test pattern: -2 on main diagonal, 1 on off-diagonals
    for(int j = 0; j <= m2; j++) {
        for(int i = 0; i <= m1; i++) {
            int main_index = i + j * (m1 + 1);
            h_main(main_index) = -2.0;
            if(i < m1) {
                int off_index = i + j * m1;
                h_lower(off_index) = 1.0;
                h_upper(off_index) = 1.0;
            }
        }
    }

    // Copy to device
    Kokkos::deep_copy(A1.main_diags, h_main);
    Kokkos::deep_copy(A1.lower_diags, h_lower);
    Kokkos::deep_copy(A1.upper_diags, h_upper);

    
    // Print structure of first block
    std::cout << "\nFirst block (j=0) diagonals:\n";
    std::cout << "Main diagonal: ";
    for(int i = 0; i <= m1; i++) {
        int main_index = i;
        std::cout << h_main(main_index) << " ";
    }
    std::cout << "\nLower diagonal: ";
    for(int i = 0; i < m1; i++) {
        int off_index = i;
        std::cout << h_lower(off_index) << " ";
    }
    std::cout << "\nUpper diagonal: ";
    for(int i = 0; i < m1; i++) {
        int off_index = i;
        std::cout << h_upper(off_index) << " ";
    }
    std::cout << "\n";
    

    // Build implicit system
    double theta = 0.5;      // Crank-Nicolson
    double delta_t = 0.01;   // time step
    A1.build_implicit(theta, delta_t);

    // Create test vectors
    Kokkos::View<double*> x("x", total_size);
    Kokkos::View<double*> b("b", total_size);
    Kokkos::View<double*> result("result", total_size);

    // Initialize b with ones
    auto h_b = Kokkos::create_mirror_view(b);
    for(int i = 0; i < total_size; i++) {
        h_b(i) = 1.0;
    }
    Kokkos::deep_copy(b, h_b);

    // Test explicit multiply
    std::cout << "\nTesting explicit multiply...\n";

    auto t_start_multip = timer::now();
    A1.multiply(b, result);
    auto t_end_multip = timer::now();

    std::cout << "Multiply solve time: "
              << std::chrono::duration<double>(t_end_multip - t_start_multip).count()
              << " seconds" << std::endl;


    // Test implicit solve
    std::cout << "\nTesting implicit solve...\n";

    auto t_start_solve = timer::now();
    A1.solve_implicit(x, b);
    auto t_end_solve = timer::now();

    std::cout << "Implicit solve time: "
              << std::chrono::duration<double>(t_end_solve - t_start_solve).count()
              << " seconds" << std::endl;


    // Get results
    auto h_x = Kokkos::create_mirror_view(x);
    auto h_result = Kokkos::create_mirror_view(result);
    Kokkos::deep_copy(h_x, x);
    Kokkos::deep_copy(h_result, result);

    
    // Print results for first block
    std::cout << "\nResults for first block (j=0):\n";
    std::cout << "Explicit multiply result: ";
    for(int i = 0; i <= m1; i++) {
        int index = i;
        std::cout << h_result(index) << " ";
    }
    std::cout << "\nImplicit solve result: ";
    for(int i = 0; i <= m1; i++) {
        int index = i;
        std::cout << h_x(index) << " ";
    }
    std::cout << "\n";
    

    // Verify implicit solve by computing residual
    // (I - θΔt*A1) * x - b
    Kokkos::View<double*> residual("residual", total_size);
    Kokkos::View<double*> temp("temp", total_size);
    
    // First compute A1 * x
    A1.multiply(x, temp);

    
    
    // Then compute residual on host
    auto h_temp = Kokkos::create_mirror_view(temp);
    auto h_residual = Kokkos::create_mirror_view(residual);
    Kokkos::deep_copy(h_temp, temp);
    
    bool nan_found = false;
    for(int i = 0; i < total_size; i++) {
        if (std::isnan(h_x(i)) || std::isnan(h_temp(i)) || std::isnan(h_b(i))) {
            std::cout << "NaN found at index " << i << "\n";
            nan_found = true;
            break;
        }
    }
    if (nan_found) {
        std::cout << "NaN detected in arrays before residual computation.\n";
    }


    double residual_norm = 0.0;
    for(int i = 0; i < total_size; i++) {
        double val = h_x(i) - theta * delta_t * h_temp(i) - h_b(i);
        h_residual(i) = val;
        residual_norm += val * val;
    }
    residual_norm = std::sqrt(residual_norm);

    std::cout << "\nResidual norm: " << residual_norm << "\n";
}





void mat_factory() {
    Kokkos::initialize();
    {
        std::cout << "Default execution space: " << Kokkos::DefaultExecutionSpace::name() << std::endl;
        test_tridiagonal_matrixfree();
        //test_tridiagonal_matrixfree_with_random_b();
        //kokkos_cyclic_reduction_solver();

        //solve_multiple_tridiagonal_systems();
        //test_tridiagonal_matvec_with_random_x();

        //test_A1_solve();
        //test_A1_solve_memory_optimized();
        //std::cout<< "\nTesting A1 solve:\n";
        //test_A1_solve_for_random_vector();
        //test_A1_visualization();

        //test_A2_solve();

        //std::cout<< "\nTesting A2 solve:\n";
        //test_A2_gpu();
        //test_A2_gpu_vizually();
        //test_A2_gpu_optimized();


        //std::cout<< "\nTesting A0 solve:\n";
        //test_A0_gpu();

        //test_reordering_performance();
    }
    Kokkos::finalize();
    //test_A2_cpu();
    //test_complex_A2_cpu_random_b();
    //test_A2_oszilation();

    //test_spaced_tridiagonal();

    //test_A0_cpu();
}

/*
// Main function to construct all matrices
MatrixSet make_matrices(
    int m1, int m2, scalar_t rho, scalar_t sigma, 
    scalar_t r_d, scalar_t r_f, scalar_t kappa, scalar_t eta,
    const vector_view_t& d_Vec_s, const vector_view_t& d_Vec_v,
    const vector_view_t& d_Delta_s, const vector_view_t& d_Delta_v) 
{
    const int m = (m1 + 1) * (m2 + 1);  // Total matrix size

    // Create host mirrors and copy data
    auto h_Vec_s = Kokkos::create_mirror_view(d_Vec_s);
    auto h_Vec_v = Kokkos::create_mirror_view(d_Vec_v);
    auto h_Delta_s = Kokkos::create_mirror_view(d_Delta_s);
    auto h_Delta_v = Kokkos::create_mirror_view(d_Delta_v);
    
    Kokkos::deep_copy(h_Vec_s, d_Vec_s);
    Kokkos::deep_copy(h_Vec_v, d_Vec_v);
    Kokkos::deep_copy(h_Delta_s, d_Delta_s);
    Kokkos::deep_copy(h_Delta_v, d_Delta_v);

    // Convert to std::vector for helper functions
    std::vector<scalar_t> Delta_s(h_Delta_s.extent(0));
    std::vector<scalar_t> Delta_v(h_Delta_v.extent(0));
    for(size_t i = 0; i < Delta_s.size(); i++) Delta_s[i] = h_Delta_s(i);
    for(size_t i = 0; i < Delta_v.size(); i++) Delta_v[i] = h_Delta_v(i);

    // Count entries for each matrix
    std::vector<size_type> nnz_per_row_A0(m, 0);
    std::vector<size_type> nnz_per_row_A1(m, 0);
    std::vector<size_type> nnz_per_row_A2(m, 0);

    // Count for A0 (cross derivatives)
    for(int j = 1; j < m2; j++) {
        for(int i = 1; i < m1; i++) {
            const int row = i + j * (m1 + 1);
            nnz_per_row_A0[row] = 4;  // diagonal and neighbors
        }
    }

    // Count for A1 (s-direction derivatives)
    for(int j = 0; j < m2 + 1; j++) {
        for(int i = 1; i < m1; i++) {
            const int row = i + j * (m1 + 1);
            if(i > 0) nnz_per_row_A1[row]++;
            nnz_per_row_A1[row]++;
            if(i < m1) nnz_per_row_A1[row]++;
        }
    }

    // Count for A2 (v-direction derivatives)
    for(int j = 0; j < m2 - 1; j++) {
        for(int i = 0; i < m1 + 1; i++) {
            const int row = i + j * (m1 + 1);
            if(j == 0) nnz_per_row_A2[row] = 3;  // boundary case
            else nnz_per_row_A2[row] = (h_Vec_v[j] > 1.0) ? 3 : 3;
        }
    }

    // Create rowmaps
    auto create_rowmap = [m](const std::vector<size_type>& nnz_per_row) {
        std::vector<size_type> rowmap(m + 1, 0);
        for(int i = 0; i < m; i++) {
            rowmap[i + 1] = rowmap[i] + nnz_per_row[i];
        }
        return rowmap;
    };

    auto rowmap_A0 = create_rowmap(nnz_per_row_A0);
    auto rowmap_A1 = create_rowmap(nnz_per_row_A1);
    auto rowmap_A2 = create_rowmap(nnz_per_row_A2);

    // Allocate arrays for each matrix
    std::vector<ordinal_t> colinds_A0(rowmap_A0.back());
    std::vector<ordinal_t> colinds_A1(rowmap_A1.back());
    std::vector<ordinal_t> colinds_A2(rowmap_A2.back());
    std::vector<scalar_t> values_A0(rowmap_A0.back());
    std::vector<scalar_t> values_A1(rowmap_A1.back());
    std::vector<scalar_t> values_A2(rowmap_A2.back());

    // Fill A0 (cross derivatives)
    {
        std::vector<size_type> current_pos = rowmap_A0;
        for(int j = 1; j < m2; j++) {
            for(int i = 1; i < m1; i++) {
                const int row = i + j * (m1 + 1);
                size_type& pos = current_pos[row];
                scalar_t c = rho * sigma * h_Vec_s[i] * h_Vec_v[j];

                // Add diagonal first (was missing in original)
                colinds_A0[pos] = row;
                values_A0[pos] = 0.0;  // Will accumulate diagonal contribution
                pos++;

                // Add cross terms with correct signs
                int corner_indices[4][2] = {
                    {-1, -1}, {-1, 1}, {1, -1}, {1, 1}
                };
                scalar_t corner_signs[4] = {1.0, -1.0, -1.0, 1.0};

                for(int idx = 0; idx < 4; idx++) {
                    int di = corner_indices[idx][0];
                    int dj = corner_indices[idx][1];
                    int col = (i + di) + (j + dj) * (m1 + 1);
                    
                    colinds_A0[pos] = col;
                    values_A0[pos] = c * corner_signs[idx] * 
                        (1.0 / (Delta_s[i-1] * Delta_v[j-1]));
                    
                    // Accumulate diagonal contribution
                    values_A0[pos-4] -= values_A0[pos];
                    pos++;
                }
            }
        }
    }
    // Fill A1 (already implemented in your code)
    {
        std::vector<size_type> current_pos = rowmap_A1;
        for(int j = 0; j < m2 + 1; j++) {
            for(int i = 1; i < m1; i++) {
                const int row = i + j * (m1 + 1);
                size_type& pos = current_pos[row];

                scalar_t a = 0.5 * h_Vec_s[i] * h_Vec_s[i] * h_Vec_v[j];
                scalar_t b = (r_d - r_f) * h_Vec_s[i];

                if(i > 0) {
                    colinds_A1[pos] = row - 1;
                    values_A1[pos] = a * delta_s(i - 1, -1, Delta_s) + 
                                   b * beta_s(i - 1, -1, Delta_s);
                    pos++;
                }

                colinds_A1[pos] = row;
                values_A1[pos] = a * delta_s(i - 1, 0, Delta_s) + 
                               b * beta_s(i - 1, 0, Delta_s) - 
                               0.5 * r_d;
                pos++;

                if(i < m1) {
                    colinds_A1[pos] = row + 1;
                    values_A1[pos] = a * delta_s(i - 1, 1, Delta_s) + 
                                   b * beta_s(i - 1, 1, Delta_s);
                    pos++;
                }
            }
        }
    }

    // Fill A2 (v-direction derivatives)
    {
        std::vector<size_type> current_pos = rowmap_A2;
        for(int j = 0; j < m2 - 1; j++) {
            for(int i = 0; i < m1 + 1; i++) {
                const int row = i + j * (m1 + 1);
                size_type& pos = current_pos[row];

                scalar_t temp = kappa * (eta - h_Vec_v[j]);
                scalar_t temp2 = 0.5 * sigma * sigma * h_Vec_v[j];

                // Always include diagonal term first
                colinds_A2[pos] = row;
                values_A2[pos] = -0.5 * r_d;  // Base diagonal contribution
                pos++;

                if(j == 0) {
                    // Special handling for first row
                    for(int k = 0; k <= 2; k++) {
                        if(k == 0) continue;  // Skip diagonal (already added)
                        colinds_A2[pos] = i + (m1 + 1) * k;
                        values_A2[pos] = temp * gamma_v(j, k, Delta_v);
                        values_A2[pos-2] -= values_A2[pos];  // Update diagonal
                        pos++;
                    }
                } else {
                    scalar_t central_coeff = temp * beta_v(j - 1, 0, Delta_v) + 
                                        temp2 * delta_v(j - 1, 0, Delta_v);
                    values_A2[pos-1] += central_coeff;  // Add to diagonal

                    for(int k = -1; k <= 1; k++) {
                        if(k == 0) continue;  // Skip diagonal
                        colinds_A2[pos] = i + (m1 + 1) * (j + k);
                        values_A2[pos] = temp * beta_v(j - 1, k, Delta_v) + 
                                    temp2 * delta_v(j - 1, k, Delta_v);
                        pos++;
                    }
                }
            }
        }
    }

    // Create device views and copy data for all matrices
    auto create_matrix = [m](const std::vector<size_type>& rowmap,
                           const std::vector<ordinal_t>& colinds,
                           const std::vector<scalar_t>& values,
                           const char* name) {
        const size_type total_nnz = rowmap.back();
        
        Kokkos::View<size_type*> d_rowmap(name + std::string("_rowmap"), m + 1);
        Kokkos::View<ordinal_t*> d_colinds(name + std::string("_colinds"), total_nnz);
        Kokkos::View<scalar_t*> d_values(name + std::string("_values"), total_nnz);

        auto h_d_rowmap = Kokkos::create_mirror_view(d_rowmap);
        auto h_d_colinds = Kokkos::create_mirror_view(d_colinds);
        auto h_d_values = Kokkos::create_mirror_view(d_values);

        for(size_t i = 0; i < rowmap.size(); i++) h_d_rowmap(i) = rowmap[i];
        for(size_t i = 0; i < total_nnz; i++) {
            h_d_colinds(i) = colinds[i];
            h_d_values(i) = values[i];
        }

        Kokkos::deep_copy(d_rowmap, h_d_rowmap);
        Kokkos::deep_copy(d_colinds, h_d_colinds);
        Kokkos::deep_copy(d_values, h_d_values);

        return matrix_t(name, m, m, total_nnz, d_values, d_rowmap, d_colinds);
    };

    // Near the end of make_matrices function, replace the last few lines with:
    // Create matrices
    matrix_t A0 = create_matrix(rowmap_A0, colinds_A0, values_A0, "A0");
    matrix_t A1 = create_matrix(rowmap_A1, colinds_A1, values_A1, "A1");
    matrix_t A2 = create_matrix(rowmap_A2, colinds_A2, values_A2, "A2");

    // Compute the sum A = A0 + A1 + A2
    matrix_t A = add_sparse_matrices(A0, A1, A2, "A");

    // Create and return all matrices
    return MatrixSet(A0, A1, A2, A);
}
*/


/*
void test_A1_construction() {
    Kokkos::initialize();
    {
        // Define dimensions
        const int m1 = 50;  // S grid points
        const int m2 = 25;  // V grid points
        const scalar_t r_d = 0.025;  // domestic interest rate
        const scalar_t r_f = 0.0;    // foreign interest rate

        // Create grid vectors on device
        vector_view_t d_Vec_s("Vec_s", m1 + 1);
        vector_view_t d_Vec_v("Vec_v", m2 + 1);
        vector_view_t d_Delta_s("Delta_s", m1);

        // Create host mirrors and fill with test data
        auto h_Vec_s = Kokkos::create_mirror_view(d_Vec_s);
        auto h_Vec_v = Kokkos::create_mirror_view(d_Vec_v);
        auto h_Delta_s = Kokkos::create_mirror_view(d_Delta_s);

        // Fill with simple test values
        for(int i = 0; i <= m1; i++) {
            h_Vec_s(i) = 50.0 + i * 2.0;  // S values from 50 to 150
        }
        for(int i = 0; i <= m2; i++) {
            h_Vec_v(i) = 0.04 + i * 0.04;  // V values from 0.04 to 1.04
        }
        for(int i = 0; i < m1; i++) {
            h_Delta_s(i) = h_Vec_s(i+1) - h_Vec_s(i);
        }

        // Copy to device
        Kokkos::deep_copy(d_Vec_s, h_Vec_s);
        Kokkos::deep_copy(d_Vec_v, h_Vec_v);
        Kokkos::deep_copy(d_Delta_s, h_Delta_s);

        // Construct A1 matrix
        std::cout << "Constructing A1 matrix..." << std::endl;
        auto A1 = construct_A1(m1, m2, r_d, r_f, d_Vec_s, d_Vec_v, d_Delta_s);
        const int matrix_size = (m1 + 1) * (m2 + 1);

        // Create test vectors for SpMV
        Kokkos::View<scalar_t*> x("x", matrix_size);
        Kokkos::View<scalar_t*> b("b", matrix_size);

        // Initialize x with ones
        Kokkos::deep_copy(x, 1.0);

        // Perform SpMV operation: b = A1*x
        std::cout << "Performing SpMV..." << std::endl;
        KokkosSparse::spmv("N", 1.0, A1, x, 0.0, b);

        // Setup and solve linear system using Gauss-Seidel
        std::cout << "Setting up linear solver..." << std::endl;
        handle_t handle;
        handle.create_gs_handle(KokkosSparse::GS_DEFAULT);

        // Setup preconditioner
        KokkosSparse::Experimental::gauss_seidel_symbolic
            (&handle, matrix_size, matrix_size, A1.graph.row_map, A1.graph.entries, false);
        KokkosSparse::Experimental::gauss_seidel_numeric
            (&handle, matrix_size, matrix_size, A1.graph.row_map, A1.graph.entries, A1.values, false);

        // Solve system A1*x = b
        std::cout << "Solving linear system..." << std::endl;
        Kokkos::View<scalar_t*> res("residual", matrix_size);
        Kokkos::deep_copy(x, 0.0);  // Reset x to zero

        bool is_first_iter = true;
        const scalar_t omega = 1.0;
        const int max_iters = 100;

        for(int iter = 0; iter < max_iters; iter++) {
            KokkosSparse::Experimental::forward_sweep_gauss_seidel_apply
                (&handle, matrix_size, matrix_size, A1.graph.row_map, A1.graph.entries, A1.values,
                 x, b, is_first_iter, is_first_iter, omega, 1);
            
            // Check convergence
            Kokkos::deep_copy(res, b);
            KokkosSparse::spmv("N", 1.0, A1, x, -1.0, res);
            scalar_t res_norm = KokkosBlas::nrm2(res);
            
            if(iter % 10 == 0) {
                std::cout << "Iteration " << iter << ", residual = " << res_norm << std::endl;
            }
            
            if(res_norm < 1e-6) {
                std::cout << "Converged after " << iter + 1 << " iterations" << std::endl;
                break;
            }
            is_first_iter = false;
        }

        // Cleanup
        handle.destroy_gs_handle();
    }
    Kokkos::finalize();
}
*/


