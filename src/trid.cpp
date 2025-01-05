#include "trid.hpp"
#include <vector>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <stdexcept>

/*

This shoudl compare the speed between a standard thomasa lgotithm (three arrays for the diagonals)
and a memory speed up version (only one array).

The speed up was negligable on the cpu

*/
void thomas_algorithm(const std::vector<double>& a, const std::vector<double>& b, const std::vector<double>& c,
                      const std::vector<double>& d, std::vector<double>& x) {
    int n = b.size();
    std::vector<double> c_prime(n, 0.0);
    std::vector<double> d_prime(n, 0.0);

    // First row
    if (b[0] == 0.0) {
        throw std::runtime_error("Zero pivot encountered at index 0");
    }
    c_prime[0] = c[0] / b[0];
    d_prime[0] = d[0] / b[0];

    // Forward sweep
    for (int i = 1; i < n; ++i) {
        double denom = b[i] - a[i - 1] * c_prime[i - 1];
        if (denom == 0.0) {
            throw std::runtime_error("Zero pivot encountered at index " + std::to_string(i));
        }
        if (i < n - 1) {
            c_prime[i] = c[i] / denom;
        }
        d_prime[i] = (d[i] - a[i - 1] * d_prime[i - 1]) / denom;
    }

    // Back substitution
    x.resize(n);
    x[n - 1] = d_prime[n - 1];
    for (int i = n - 2; i >= 0; --i) {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }
}

void test_standard_tridiagonal() {
    using timer = std::chrono::high_resolution_clock;
    int n = 50000;  // Adjust the dimension as needed
    std::cout << "\nStandard Thomas Dimension: " << n << std::endl;

    // Create diagonals
    std::vector<double> a(n - 1, -1.0);  // Lower diagonal (a[1] to a[n-1])
    std::vector<double> b(n, 2.0);       // Main diagonal (b[0] to b[n-1])
    std::vector<double> c(n - 1, -1.0);  // Upper diagonal (c[0] to c[n-2])

    // Create random right-hand side
    std::vector<double> d(n);
    std::srand(52);  // Seed for reproducibility
    for (int i = 0; i < n; ++i) {
        d[i] = std::rand() / (RAND_MAX + 1.0);
    }

    // Solve system
    std::vector<double> x;

    auto t_start_solve = timer::now();
    thomas_algorithm(a, b, c, d, x);
    auto t_end_solve = timer::now();

    std::cout << "\nSolve time: "
              << std::chrono::duration<double>(t_end_solve - t_start_solve).count()
              << " seconds" << std::endl;

    // Verify solution by computing residual
    std::vector<double> Ax(n, 0.0);

    auto t_start_multip = timer::now();
    // Multiply Ax = A * x
    Ax[0] = b[0] * x[0] + c[0] * x[1];
    for (int i = 1; i < n - 1; ++i) {
        Ax[i] = a[i - 1] * x[i - 1] + b[i] * x[i] + c[i] * x[i + 1];
    }
    Ax[n - 1] = a[n - 2] * x[n - 2] + b[n - 1] * x[n - 1];
    auto t_end_multip = timer::now();

    std::cout << "Explicit multiplication time: "
              << std::chrono::duration<double>(t_end_multip - t_start_multip).count()
              << " seconds" << std::endl;

    // Compute residual norm ||Ax - d||
    double residual_norm = 0.0;
    for (int i = 0; i < n; ++i) {
        double res = Ax[i] - d[i];
        residual_norm += res * res;
    }
    residual_norm = std::sqrt(residual_norm);

    std::cout << "Residual norm ||Ax - d||: " << residual_norm << std::endl;
}

struct GridBasedTridiagonal {
    std::vector<double> values;
    const int point_stride = 5;
    const int n;

    GridBasedTridiagonal(int size) : n(size) {
        values.resize(size * point_stride, 0.0);
    }

    void multiply(std::vector<double>& result, const std::vector<double>& vec) {
        if (result.size() != n) {
            result.resize(n);
        }

        // First point
        result[0] = values[0 * point_stride + 1] * vec[0] +      // b(0)
                    values[0 * point_stride + 2] * vec[1];       // c(0)

        // Middle points
        for (int i = 1; i < n - 1; i++) {
            const int idx = i * point_stride;
            result[i] = values[idx + 0] * vec[i - 1] +           // a(i)
                        values[idx + 1] * vec[i] +               // b(i)
                        values[idx + 2] * vec[i + 1];            // c(i)
        }

        // Last point
        const int last_idx = (n - 1) * point_stride;
        result[n - 1] = values[last_idx + 0] * vec[n - 2] +      // a(n-1)
                        values[last_idx + 1] * vec[n - 1];       // b(n-1)
    }

    void solve(std::vector<double>& solution) {
        if (solution.size() != n) {
            solution.resize(n);
        }

        std::vector<double> c_prime(n, 0.0);
        std::vector<double> d_prime(n, 0.0);

        // Forward sweep
        const int idx0 = 0 * point_stride;
        double w = values[idx0 + 1];  // b(0)
        if (w == 0.0) {
            throw std::runtime_error("Zero pivot encountered at first point");
        }
        c_prime[0] = values[idx0 + 2] / w;
        d_prime[0] = values[idx0 + 3] / w;

        for (int i = 1; i < n; i++) {
            const int idx = i * point_stride;
            double a_i = values[idx + 0];  // a(i)
            double b_i = values[idx + 1];  // b(i)
            double c_i = (i < n - 1) ? values[idx + 2] : 0.0;  // c(i), zero if last point
            double d_i = values[idx + 3];  // d(i)

            w = b_i - a_i * c_prime[i - 1];
            if (w == 0.0) {
                throw std::runtime_error("Zero pivot encountered at index " + std::to_string(i));
            }
            c_prime[i] = (i < n - 1) ? c_i / w : 0.0;
            d_prime[i] = (d_i - a_i * d_prime[i - 1]) / w;
        }

        // Back substitution
        solution[n - 1] = d_prime[n - 1];
        for (int i = n - 2; i >= 0; i--) {
            solution[i] = d_prime[i] - c_prime[i] * solution[i + 1];
        }
    }
};

void test_tridiagonal_residual() {
    using timer = std::chrono::high_resolution_clock;
    const int n = 50000;  // Small system for testing
    std::cout<< "\nMemory Optimized Dimension: " << n << std::endl;
    GridBasedTridiagonal system(n);

    // Set up test pattern (-1, 2, -1)
    // First point special case
    system.values[0 * system.point_stride + 1] = 2.0;     // b(0)
    system.values[0 * system.point_stride + 2] = -1.0;    // c(0)

    // Middle points
    for (int i = 1; i < n - 1; i++) {
        int idx = i * system.point_stride;
        system.values[idx + 0] = -1.0;      // a(i)
        system.values[idx + 1] = 2.0;       // b(i)
        system.values[idx + 2] = -1.0;      // c(i)
    }

    // Last point special case
    int last_idx = (n - 1) * system.point_stride;
    system.values[last_idx + 0] = -1.0;     // a(n-1)
    system.values[last_idx + 1] = 2.0;      // b(n-1)
    // c(n-1) is not used

    // Create random right-hand side
    std::vector<double> b(n);
    std::srand(52); // Seed for reproducibility
    for (int i = 0; i < n; i++) {
        b[i] = std::rand() / (RAND_MAX + 1.0);
        system.values[i * system.point_stride + 3] = b[i];  // Set d(i)
    }

    /*
    // Print matrix pattern for verification
    std::cout << "Matrix structure:\n";
    for (int i = 0; i < n; i++) {
        int idx = i * system.point_stride;
        if (i > 0) std::cout << "a[" << i << "] = " << system.values[idx + 0] << ", ";
        std::cout << "b[" << i << "] = " << system.values[idx + 1];
        if (i < n - 1) std::cout << ", c[" << i << "] = " << system.values[idx + 2];
        std::cout << std::endl;
    }
    */

    // Solve system
    std::vector<double> solution(n);

    auto t_start_solve = timer::now();
    system.solve(solution);
    auto t_end_solve = timer::now();

    std::cout << "\nSolve time: "
              << std::chrono::duration<double>(t_end_solve - t_start_solve).count()
              << " seconds" << std::endl;

    // Verify solution by computing residual
    std::vector<double> Ax(n);

    auto t_start_multip = timer::now();
    system.multiply(Ax, solution);
    auto t_end_multip = timer::now();

    std::cout << "Explicit time: "
              << std::chrono::duration<double>(t_end_multip - t_start_multip).count()
              << " seconds" << std::endl;

    // Compute residual norm ||Ax - b||
    double residual_norm = 0.0;
    for (int i = 0; i < n; i++) {
        double res = Ax[i] - b[i];
        residual_norm += res * res;
    }
    residual_norm = std::sqrt(residual_norm);

    std::cout << "Residual norm ||Ax - b||: " << residual_norm << std::endl;

    /*
    std::cout << "\nOriginal RHS b:\n";
    for (int i = 0; i < n; i++) {
        std::cout << "b[" << i << "] = " << b[i] << std::endl;
    }

    std::cout << "\nComputed solution x:\n";
    for (int i = 0; i < n; i++) {
        std::cout << "x[" << i << "] = " << solution[i] << std::endl;
    }

    std::cout << "\nVerification Ax:\n";
    for (int i = 0; i < n; i++) {
        std::cout << "Ax[" << i << "] = " << Ax[i] << std::endl;
    }
    */
}


#include <Kokkos_Core.hpp>

struct GridBasedTridiagonalKokkos {
    Kokkos::View<double*> values;
    const int point_stride = 5;
    const int n;

    GridBasedTridiagonalKokkos(int size) : n(size) {
        values = Kokkos::View<double*>("values", size * point_stride);
        Kokkos::deep_copy(values, 0.0);
    }

    void multiply(Kokkos::View<double*>& result, const Kokkos::View<double*>& vec) {
        // Ensure result has the correct size
        if (result.size() != n) {
            result = Kokkos::View<double*>("result", n);
        }

        auto values_ = values;
        int n_ = n;
        int point_stride_ = point_stride;

        // Run on a single thread
        Kokkos::parallel_for("multiply", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, 1), KOKKOS_LAMBDA(const int&) {
            // First point
            result(0) = values_(0 * point_stride_ + 1) * vec(0) +      // b(0)
                        values_(0 * point_stride_ + 2) * vec(1);       // c(0)

            // Middle points
            for (int i = 1; i < n_ - 1; i++) {
                const int idx = i * point_stride_;
                result(i) = values_(idx + 0) * vec(i - 1) +           // a(i)
                            values_(idx + 1) * vec(i) +               // b(i)
                            values_(idx + 2) * vec(i + 1);            // c(i)
            }

            // Last point
            const int last_idx = (n_ - 1) * point_stride_;
            result(n_ - 1) = values_(last_idx + 0) * vec(n_ - 2) +     // a(n-1)
                             values_(last_idx + 1) * vec(n_ - 1);      // b(n-1)
        });
        Kokkos::fence();
    }

    void solve(Kokkos::View<double*>& solution) {
        // Ensure solution has the correct size
        if (solution.size() != n) {
            solution = Kokkos::View<double*>("solution", n);
        }

        auto values_ = values;
        int n_ = n;
        int point_stride_ = point_stride;

        Kokkos::View<double*> c_prime("c_prime", n_);
        Kokkos::View<double*> d_prime("d_prime", n_);

        // Run on a single thread
        Kokkos::parallel_for("solve", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, 1), KOKKOS_LAMBDA(const int&) {
            // First point
            const int idx0 = 0 * point_stride_;
            double w = values_(idx0 + 1);  // b(0)
            if (w == 0.0) {
                printf("Zero pivot encountered at first point\n");
                return;
            }
            c_prime(0) = values_(idx0 + 2) / w;
            d_prime(0) = values_(idx0 + 3) / w;

            // Forward sweep
            for (int i = 1; i < n_; ++i) {
                const int idx = i * point_stride_;
                double a_i = values_(idx + 0);  // a(i)
                double b_i = values_(idx + 1);  // b(i)
                double c_i = (i < n_ - 1) ? values_(idx + 2) : 0.0;  // c(i)
                double d_i = values_(idx + 3);  // d(i)

                w = b_i - a_i * c_prime(i - 1);
                if (w == 0.0) {
                    printf("Zero pivot encountered at index %d\n", i);
                    return;
                }
                c_prime(i) = (i < n_ - 1) ? c_i / w : 0.0;
                d_prime(i) = (d_i - a_i * d_prime(i - 1)) / w;
            }

            // Back substitution
            solution(n_ - 1) = d_prime(n_ - 1);
            for (int i = n_ - 2; i >= 0; --i) {
                solution(i) = d_prime(i) - c_prime(i) * solution(i + 1);
            }
        });
        Kokkos::fence();
    }
};

void test_tridiagonal_residual_kokkos() {
    using timer = std::chrono::high_resolution_clock;
    const int n = 50000;  // Adjust the dimension as needed
    std::cout << "\nKokkos Dimension: " << n << std::endl;

    GridBasedTridiagonalKokkos system(n);

    // Create host mirror for values initialization
    auto h_values = Kokkos::create_mirror_view(system.values);

    // Set up test pattern (-1, 2, -1)
    // First point special case
    h_values(0 * system.point_stride + 1) = 2.0;     // b(0)
    h_values(0 * system.point_stride + 2) = -1.0;    // c(0)

    // Middle points
    for (int i = 1; i < n - 1; i++) {
        int idx = i * system.point_stride;
        h_values(idx + 0) = -1.0;      // a(i)
        h_values(idx + 1) = 2.0;       // b(i)
        h_values(idx + 2) = -1.0;      // c(i)
    }

    // Last point special case
    int last_idx = (n - 1) * system.point_stride;
    h_values(last_idx + 0) = -1.0;     // a(n-1)
    h_values(last_idx + 1) = 2.0;      // b(n-1)
    // c(n-1) is not used

    // Create random right-hand side
    std::vector<double> b(n);
    std::srand(52); // Seed for reproducibility
    for (int i = 0; i < n; i++) {
        b[i] = std::rand() / (RAND_MAX + 1.0);
        h_values[i * system.point_stride + 3] = b[i];  // Set d(i)
    }

    // Copy initialized values to device
    Kokkos::deep_copy(system.values, h_values);

    // Solve system
    Kokkos::View<double*> solution("solution", n);

    auto t_start_solve = timer::now();
    system.solve(solution);
    Kokkos::fence();
    auto t_end_solve = timer::now();

    std::cout << "\nSolve time: "
              << std::chrono::duration<double>(t_end_solve - t_start_solve).count()
              << " seconds" << std::endl;

    // Copy solution back to host
    auto h_solution = Kokkos::create_mirror_view(solution);
    Kokkos::deep_copy(h_solution, solution);

    // Verify solution by computing residual
    Kokkos::View<double*> Ax("Ax", n);

    auto t_start_multip = timer::now();
    system.multiply(Ax, solution);
    Kokkos::fence();
    auto t_end_multip = timer::now();

    std::cout << "Explicit time: "
              << std::chrono::duration<double>(t_end_multip - t_start_multip).count()
              << " seconds" << std::endl;

    // Copy Ax back to host
    auto h_Ax = Kokkos::create_mirror_view(Ax);
    Kokkos::deep_copy(h_Ax, Ax);

    // Compute residual norm ||Ax - b||
    double residual_norm = 0.0;
    for (int i = 0; i < n; i++) {
        double res = h_Ax(i) - b[i];
        residual_norm += res * res;
    }
    residual_norm = std::sqrt(residual_norm);

    std::cout << "Residual norm ||Ax - b||: " << residual_norm << std::endl;
}


void test_trid() {
    std::cout << "Entering test_trid()" << std::endl;  // Add this line
    test_tridiagonal_residual();
    test_standard_tridiagonal();

    Kokkos::initialize();
    {
        std::cout << "Starting Kokkos tridiagonal test..." << std::endl;
        test_tridiagonal_residual_kokkos();
    }
    Kokkos::finalize();

}