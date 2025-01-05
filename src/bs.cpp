#include "bs.hpp"

void initialize_A0_values(Kokkos::View<double**>& values,
                        const Grid& grid,
                        const double rho,
                        const double sigma) {
    auto values_host = Kokkos::create_mirror_view(values);
    Kokkos::deep_copy(values_host, 0.0);

    // Loop structure from Python implementation
    for(int j = 0; j < grid.Vec_v.size()-2; ++j) {  // m2-1 rows
        for(int i = 0; i < grid.Vec_s.size()-2; ++i) {  // m1-1 cols
            double c = rho * sigma * grid.Vec_s[i+1] * grid.Vec_v[j+1];

            for(int k = -1; k <= 1; ++k) {
                for(int l = -1; l <= 1; ++l) {
                    double beta_s_val = beta_s(i, k, grid.Delta_s);
                    double beta_v_val = beta_v(j, l, grid.Delta_v);
                    
                    int idx = i * 9 + (l + 1) * 3 + (k + 1);
                    values_host(j, idx) = c * beta_s_val * beta_v_val;
                }
            }
        }
    }

    Kokkos::deep_copy(values, values_host);
}

void initialize_A1_matrices(Kokkos::View<double**>& main_diags,
                          Kokkos::View<double**>& lower_diags,
                          Kokkos::View<double**>& upper_diags,
                          const Grid& grid,
                          const double rho,
                          const double sigma,
                          const double r_d,
                          const double r_f) {
    auto main_diags_host = Kokkos::create_mirror_view(main_diags);
    auto lower_diags_host = Kokkos::create_mirror_view(lower_diags);
    auto upper_diags_host = Kokkos::create_mirror_view(upper_diags);
    
    Kokkos::deep_copy(main_diags_host, 0.0);
    Kokkos::deep_copy(lower_diags_host, 0.0);
    Kokkos::deep_copy(upper_diags_host, 0.0);

    // For j in range(m2 + 1)
    for(int j = 0; j <= grid.Vec_v.size()-1; j++) {
        for(int i = 1; i < grid.Vec_s.size()-1; i++) {
            double a = 0.5 * grid.Vec_s[i] * grid.Vec_s[i] * grid.Vec_v[j];
            double b = (r_d - r_f) * grid.Vec_s[i];
            
            // Populate diagonals using central difference coefficients
            lower_diags_host(j, i-1) = a * delta_s(i-1, -1, grid.Delta_s) + 
                                    b * beta_s(i-1, -1, grid.Delta_s);
            main_diags_host(j, i) = a * delta_s(i-1, 0, grid.Delta_s) + 
                                b * beta_s(i-1, 0, grid.Delta_s) - 0.5 * r_d;
            upper_diags_host(j, i) = a * delta_s(i-1, 1, grid.Delta_s) + 
                                    b * beta_s(i-1, 1, grid.Delta_s);
        }
        // Add boundary term
        main_diags_host(j, grid.Vec_s.size()-1) = -0.5 * r_d;
    }

    Kokkos::deep_copy(main_diags, main_diags_host);
    Kokkos::deep_copy(lower_diags, lower_diags_host);
    Kokkos::deep_copy(upper_diags, upper_diags_host);
}

void initialize_A2_matrices(Kokkos::View<double*>& main_diag,
                          Kokkos::View<double*>& lower_diag,
                          Kokkos::View<double*>& upper_diag,
                          Kokkos::View<double*>& upper2_diag,
                          const Grid& grid,
                          const double rho,
                          const double sigma,
                          const double r_d,
                          const double kappa,
                          const double eta) {
    auto h_main = Kokkos::create_mirror_view(main_diag);
    auto h_lower = Kokkos::create_mirror_view(lower_diag);
    auto h_upper = Kokkos::create_mirror_view(upper_diag);
    auto h_upper2 = Kokkos::create_mirror_view(upper2_diag);
    
    Kokkos::deep_copy(h_main, 0.0);
    Kokkos::deep_copy(h_lower, 0.0);
    Kokkos::deep_copy(h_upper, 0.0);
    Kokkos::deep_copy(h_upper2, 0.0);

    const int spacing = grid.Vec_s.size();

    // Handle j=0 case first
    for(int i = 0; i < spacing; i++) {
        double temp = kappa * (eta - grid.Vec_v[0]);
        h_main(i) += temp * gamma_v(0, 0, grid.Delta_v);
        h_upper(i) += temp * gamma_v(0, 1, grid.Delta_v);
        h_upper2(i) = temp * gamma_v(0, 2, grid.Delta_v);
    }

    // Handle remaining j values
    for(int j = 1; j < grid.Vec_v.size()-2; j++) {
        for(int i = 0; i < spacing; i++) {
            double temp = kappa * (eta - grid.Vec_v[j]);
            double temp2 = 0.5 * sigma * sigma * grid.Vec_v[j];

            // Main diagonal index
            int main_idx = i + j*spacing;
            
            // Lower diagonal index - needs to start from 0 and use (j-1) blocks
            int lower_idx = i + (j-1)*spacing;
            
            // Upper diagonal index - aligns with main diagonal
            int upper_idx = main_idx;

            h_lower[lower_idx] += (temp * beta_v(j-1, -1, grid.Delta_v) + 
                                temp2 * delta_v(j-1, -1, grid.Delta_v));
            h_main[main_idx] += (temp * beta_v(j-1, 0, grid.Delta_v) + 
                                temp2 * delta_v(j-1, 0, grid.Delta_v));
            h_upper[upper_idx] += (temp * beta_v(j-1, 1, grid.Delta_v) + 
                                temp2 * delta_v(j-1, 1, grid.Delta_v));
            
            // Add reaction term to main diagonal
            h_main[main_idx] += -0.5 * r_d;
        }
    }

    Kokkos::deep_copy(main_diag, h_main);
    Kokkos::deep_copy(lower_diag, h_lower);
    Kokkos::deep_copy(upper_diag, h_upper);
    Kokkos::deep_copy(upper2_diag, h_upper2);
}

void build_A1_implicit(
    Kokkos::View<double**>& impl_main,
    Kokkos::View<double**>& impl_lower,
    Kokkos::View<double**>& impl_upper,
    const Kokkos::View<double**>& main,
    const Kokkos::View<double**>& lower,
    const Kokkos::View<double**>& upper,
    const double theta,
    const double delta_t,
    const int m1,
    const int m2
) {
    // Create host mirrors
    auto h_impl_main = Kokkos::create_mirror_view(impl_main);
    auto h_impl_lower = Kokkos::create_mirror_view(impl_lower);
    auto h_impl_upper = Kokkos::create_mirror_view(impl_upper);
    
    auto h_main = Kokkos::create_mirror_view(main);
    auto h_lower = Kokkos::create_mirror_view(lower);
    auto h_upper = Kokkos::create_mirror_view(upper);
    
    // Copy explicit matrices to host
    Kokkos::deep_copy(h_main, main);
    Kokkos::deep_copy(h_lower, lower);
    Kokkos::deep_copy(h_upper, upper);

    // Build implicit matrices on host
    for(int j = 0; j <= m2; j++) {
        for(int i = 0; i <= m1; i++) {
            h_impl_main(j,i) = 1.0 - theta * delta_t * h_main(j,i);
        }
        for(int i = 0; i < m1; i++) {
            h_impl_lower(j,i) = -theta * delta_t * h_lower(j,i);
            h_impl_upper(j,i) = -theta * delta_t * h_upper(j,i);
        }
    }

    // Copy back to device
    Kokkos::deep_copy(impl_main, h_impl_main);
    Kokkos::deep_copy(impl_lower, h_impl_lower);
    Kokkos::deep_copy(impl_upper, h_impl_upper);
}

void build_A2_implicit(
    Kokkos::View<double*>& impl_main,
    Kokkos::View<double*>& impl_lower,
    Kokkos::View<double*>& impl_upper,
    Kokkos::View<double*>& impl_upper2,
    const Kokkos::View<double*>& main,
    const Kokkos::View<double*>& lower,
    const Kokkos::View<double*>& upper,
    const Kokkos::View<double*>& upper2,
    const double theta,
    const double delta_t,
    const int m1,
    const int m2
) {
    // Create host mirrors
    auto h_impl_main = Kokkos::create_mirror_view(impl_main);
    auto h_impl_lower = Kokkos::create_mirror_view(impl_lower);
    auto h_impl_upper = Kokkos::create_mirror_view(impl_upper);
    auto h_impl_upper2 = Kokkos::create_mirror_view(impl_upper2);
    
    auto h_main = Kokkos::create_mirror_view(main);
    auto h_lower = Kokkos::create_mirror_view(lower);
    auto h_upper = Kokkos::create_mirror_view(upper);
    auto h_upper2 = Kokkos::create_mirror_view(upper2);

    // Copy explicit matrices to host
    Kokkos::deep_copy(h_main, main);
    Kokkos::deep_copy(h_lower, lower);
    Kokkos::deep_copy(h_upper, upper);
    Kokkos::deep_copy(h_upper2, upper2);

    // Initialize implicit_main_diag with identity
    for(int i = 0; i < (m2+1)*(m1+1); i++) {
        h_impl_main(i) = 1.0;
    }

    // Subtract theta*delta_t*A2 from main diagonal where A2 is defined
    for(int i = 0; i < (m2-1)*(m1+1); i++) {
        h_impl_main(i) -= theta * delta_t * h_main(i);
    }

    // Build the off-diagonal terms
    for(int i = 0; i < (m2-2)*(m1+1); i++) {
        h_impl_lower(i) = -theta * delta_t * h_lower(i);
    }

    for(int i = 0; i < (m2-1)*(m1+1); i++) {
        h_impl_upper(i) = -theta * delta_t * h_upper(i);
    }

    for(int i = 0; i < m1+1; i++) {
        h_impl_upper2(i) = -theta * delta_t * h_upper2(i);
    }

    // Copy back to device
    Kokkos::deep_copy(impl_main, h_impl_main);
    Kokkos::deep_copy(impl_lower, h_impl_lower);
    Kokkos::deep_copy(impl_upper, h_impl_upper);
    Kokkos::deep_copy(impl_upper2, h_impl_upper2);
}

void build_boundary_conditions(
    Kokkos::View<double*> b0,
    Kokkos::View<double*> b1,
    Kokkos::View<double*> b2,
    Kokkos::View<double*> b,
    const int m1,
    const int m2,
    const int m,
    const double r_d,
    const double r_f,
    const Grid& grid,
    const int N,
    const double delta_t
) {
    // Initialize to zero
    Kokkos::deep_copy(b0, 0.0);
    Kokkos::deep_copy(b1, 0.0);
    Kokkos::deep_copy(b2, 0.0);

    // Create host mirrors
    auto h_b0 = Kokkos::create_mirror_view(b0);
    auto h_b1 = Kokkos::create_mirror_view(b1);
    auto h_b2 = Kokkos::create_mirror_view(b2);
    auto h_b = Kokkos::create_mirror_view(b);

    // Copy device to host (currently zero but good practice)
    Kokkos::deep_copy(h_b0, b0);
    Kokkos::deep_copy(h_b1, b1);
    Kokkos::deep_copy(h_b2, b2);

    // Initialize b1 (S direction boundary)
    // For j in [0, m2], at s = S_max (i = m1), set boundary condition
    for(int j = 0; j <= m2; j++) {
        int idx = m1 * (j + 1);
        if (idx < m) {
            h_b1[idx] = (r_d - r_f) * grid.Vec_s[m1] * 
                         std::exp(-r_f * delta_t * (N - 1));
        }
    }

    // Initialize b2 (V direction boundary)
    // For i in [1, m1], at v = V_max (j = m2), set boundary condition
    // offset: m - m1 - 1 + i 
    // Make sure this indexing is correct relative to your PDE structure
    for(int i = 1; i <= m1; i++) {
        int idx = m - m1 - 1 + i;
        if (idx < m) {
            h_b2[idx] = -0.5 * r_d * grid.Vec_s[i] * 
                        std::exp(-r_f * delta_t * (N - 1));
        }
    }

    // Combine boundaries
    for(int i = 0; i < m; i++) {
        h_b[i] = h_b0[i] + h_b1[i] + h_b2[i];
    }

    // Copy back to device
    Kokkos::deep_copy(b0, h_b0);
    Kokkos::deep_copy(b1, h_b1);
    Kokkos::deep_copy(b2, h_b2);
    Kokkos::deep_copy(b, h_b);
}


double DO_scheme_optimized(
    const int m1, const int m2,        // Grid dimensions
    const int N,                       // Number of time steps
    const double delta_t,              // Time step size
    const double theta,                // Weight parameter
    const double r_f,                  // Foreign interest rate
    const Grid& grid,                  // Grid information
    const double rho,                  // Correlation coefficient
    const double sigma,                // Volatility of variance
    const double r_d,                  // Domestic interest rate 
    const double kappa,                // Mean reversion rate
    const double eta,                   // Long-term variance
    const double S_0,                  // Initial stock price
    const double V_0                   // Initial variance
) {
    // Total size of system
    const int m = (m1 + 1) * (m2 + 1);

    // A0 matrix storage
    Kokkos::View<double**> A0_values("A0_values", m2 - 1, (m1 - 1) * 9);

    // A1 matrix storage
    Kokkos::View<double**> A1_main_diags("A1_main_diags", m2 + 1, m1 + 1);
    Kokkos::View<double**> A1_lower_diags("A1_lower_diags", m2 + 1, m1);
    Kokkos::View<double**> A1_upper_diags("A1_upper_diags", m2 + 1, m1);

    // A1 implicit matrix storage
    Kokkos::View<double**> A1_impl_main("A1_impl_main", m2 + 1, m1 + 1);
    Kokkos::View<double**> A1_impl_lower("A1_impl_lower", m2 + 1, m1);
    Kokkos::View<double**> A1_impl_upper("A1_impl_upper", m2 + 1, m1);

    // A2 matrix storage
    Kokkos::View<double*> A2_main_diag("A2_main", (m2-1)*(m1+1));
    Kokkos::View<double*> A2_lower_diag("A2_lower", (m2-2)*(m1+1));
    Kokkos::View<double*> A2_upper_diag("A2_upper", (m2-1)*(m1+1));
    Kokkos::View<double*> A2_upper2_diag("A2_upper2", m1+1);

    // A2 implicit matrix storage
    Kokkos::View<double*> A2_impl_main("A2_impl_main", (m2+1)*(m1+1));
    Kokkos::View<double*> A2_impl_lower("A2_impl_lower", (m2-2)*(m1+1));
    Kokkos::View<double*> A2_impl_upper("A2_impl_upper", (m2-1)*(m1+1));
    Kokkos::View<double*> A2_impl_upper2("A2_impl_upper2", m1+1);

    // Boundary condition vectors 
    Kokkos::View<double*> b0("b0", m); 
    Kokkos::View<double*> b1("b1", m);
    Kokkos::View<double*> b2("b2", m);
    Kokkos::View<double*> b("b", m);

    // Solution vectors
    Kokkos::View<double*> U("U", m);                // Current solution
    Kokkos::View<double*> Y_0("Y_0", m);           // Intermediate solution Y_0
    Kokkos::View<double*> Y_1("Y_1", m);           // Intermediate solution Y_1
    Kokkos::View<double*> rhs_1("rhs_1", m);       // Right-hand side for A1 solve
    Kokkos::View<double*> rhs_2("rhs_2", m);       // Right-hand side for A2 solve

    // Initialize U with initial condition max(S - K, 0)
    {
        auto U_host = Kokkos::create_mirror_view(U);
        for(int j = 0; j <= m2; j++) {
            for(int i = 0; i <= m1; i++) {
                U_host(i + j*(m1+1)) = std::max(grid.Vec_s[i] - grid.Vec_s[m1/2], 0.0);
            }
        }
        Kokkos::deep_copy(U, U_host);
    }

    // Initialize matrices - we'll implement these next
    initialize_A0_values(A0_values, grid, rho, sigma);
    initialize_A1_matrices(A1_main_diags, A1_lower_diags, A1_upper_diags, grid, rho, sigma, r_d, r_f);
    initialize_A2_matrices(A2_main_diag, A2_lower_diag, A2_upper_diag, A2_upper2_diag, grid, rho, sigma, r_d, kappa, eta);

    // Build implicit matrices
    build_A1_implicit(A1_impl_main, A1_impl_lower, A1_impl_upper,
                     A1_main_diags, A1_lower_diags, A1_upper_diags,
                     theta, delta_t, m1, m2);
                     
    build_A2_implicit(A2_impl_main, A2_impl_lower, A2_impl_upper, A2_impl_upper2,
                     A2_main_diag, A2_lower_diag, A2_upper_diag, A2_upper2_diag,
                     theta, delta_t, m1, m2);

    build_boundary_conditions(b0, b1, b2, b, m1, m2, m, r_d, r_f, grid, N, delta_t);


    // Main time-stepping loop 
    using timer = std::chrono::high_resolution_clock;

    auto t_start = timer::now();
    for (int n = 1; n <= N; n++) {
        Kokkos::View<double*> A0_result("A0_result", m);
        Kokkos::View<double*> A1_result("A1_result", m);
        Kokkos::View<double*> A2_result("A2_result", m);

        // Y_0 computation step
        {
            // A0 multiply 
            Kokkos::parallel_for("A0_multiply", Kokkos::RangePolicy<>(0, m2-1), KOKKOS_LAMBDA(const int j) {
                const int spacing = m1 + 1;
                for(int i = 0; i < m1-1; i++) {
                    double sum = 0.0;
                    for(int l = -1; l <= 1; l++) {
                        for(int k = -1; k <= 1; k++) {
                            int val_idx = i * 9 + (l + 1) * 3 + (k + 1);
                            int col_idx = (i + 1 + k) + (j + 1 + l) * spacing;
                            sum += A0_values(j, val_idx) * U(col_idx);
                        }
                    }
                    A0_result(i + 1 + (j + 1) * spacing) = sum;
                }
            });

            // A1 multiply
            Kokkos::parallel_for("A1_multiply", Kokkos::RangePolicy<>(0, m2+1), KOKKOS_LAMBDA(const int j) {
                const int offset = j * (m1 + 1);
                for(int i = 0; i <= m1; i++) {
                    double sum = A1_main_diags(j,i) * U(offset + i);
                    if(i > 0) {
                        sum += A1_lower_diags(j,i-1) * U(offset + i-1);
                    }
                    if(i < m1) {
                        sum += A1_upper_diags(j,i) * U(offset + i+1);
                    }
                    A1_result(offset + i) = sum;
                }
            });

            // A2 multiply
            Kokkos::parallel_for("A2_multiply", 1, KOKKOS_LAMBDA(const int) {
                const int spacing = m1 + 1;
                // First block (j=0)
                for(int i = 0; i < spacing; i++) {
                    double sum = A2_main_diag(i) * U(i);
                    sum += A2_upper_diag(i) * U(i + spacing);
                    sum += A2_upper2_diag(i) * U(i + 2*spacing);
                    A2_result(i) = sum;
                }
                // Handle remaining blocks
                for(int i = spacing; i < (m2-1)*(m1+1); i++) {
                    double sum = A2_lower_diag(i-spacing) * U(i-spacing);
                    sum += A2_main_diag(i) * U(i);
                    sum += A2_upper_diag(i) * U(i + spacing);
                    A2_result(i) = sum;
                }
            });

            // Combine results into Y_0
            Kokkos::parallel_for("Y0_computation", m, KOKKOS_LAMBDA(const int i) {
                double exp_factor = std::exp(r_f * delta_t * (n-1));
                Y_0(i) = U(i) + delta_t * (A0_result(i) + A1_result(i) + A2_result(i) + b(i) * exp_factor);
            });
        }

        // A1 implicit solve step
        {
            // Compute A1*U using same parallel explicit multiply as before
            Kokkos::parallel_for("A1_multiply", Kokkos::RangePolicy<>(0, m2+1), KOKKOS_LAMBDA(const int j) {
                const int offset = j * (m1 + 1);
                for(int i = 0; i <= m1; i++) {
                    double sum = A1_main_diags(j,i) * U(offset + i);
                    if(i > 0) {
                        sum += A1_lower_diags(j,i-1) * U(offset + i-1);
                    }
                    if(i < m1) {
                        sum += A1_upper_diags(j,i) * U(offset + i+1);
                    }
                    A1_result(offset + i) = sum;
                }
            });

            // Compute RHS
            Kokkos::parallel_for("A1_rhs_computation", m, KOKKOS_LAMBDA(const int i) {
                double exp_factor = std::exp(r_f * delta_t * n);
                double rhs = Y_0(i) + theta * delta_t * (b1(i) * exp_factor - A1_result(i));
                Y_0(i) = rhs;  // Reuse Y_0 to store RHS
            });

            // Create temporary storage for tridiagonal solve
            Kokkos::View<double**> temp("A1_temp", m2+1, m1+1);

            // A1 implicit solve - parallel over variance levels
            Kokkos::parallel_for("A1_solve", Kokkos::RangePolicy<>(0, m2+1), KOKKOS_LAMBDA(const int j) {
                const int offset = j * (m1 + 1);
                
                // Forward sweep
                temp(j,0) = A1_impl_main(j,0);
                Y_1(offset) = Y_0(offset);
                
                for(int i = 1; i <= m1; i++) {
                    double m = A1_impl_lower(j,i-1) / temp(j,i-1);
                    temp(j,i) = A1_impl_main(j,i) - m * A1_impl_upper(j,i-1);
                    Y_1(offset + i) = Y_0(offset + i) - m * Y_1(offset + i-1);
                }

                // Back substitution
                Y_1(offset + m1) /= temp(j,m1);
                for(int i = m1-1; i >= 0; i--) {
                    Y_1(offset + i) = (Y_1(offset + i) - 
                        A1_impl_upper(j,i) * Y_1(offset + i+1)) / temp(j,i);
                }
            });
        }

        // A2 implicit solve step
        {
        // Compute A2*U using same multiply as before
        Kokkos::parallel_for("A2_multiply", 1, KOKKOS_LAMBDA(const int) {
            const int spacing = m1 + 1;
            
            // First block (j=0)
            for(int i = 0; i < spacing; i++) {
                double sum = A2_main_diag(i) * U(i);
                sum += A2_upper_diag(i) * U(i + spacing);
                sum += A2_upper2_diag(i) * U(i + 2*spacing);
                A2_result(i) = sum;
            }

            // Handle remaining blocks
            for(int i = spacing; i < (m2-1)*(m1+1); i++) {
                double sum = A2_lower_diag(i-spacing) * U(i-spacing);
                sum += A2_main_diag(i) * U(i);
                sum += A2_upper_diag(i) * U(i + spacing);
                A2_result(i) = sum;
            }
        });

        // Compute RHS
        Kokkos::parallel_for("A2_rhs_computation", m, KOKKOS_LAMBDA(const int i) {
            double exp_factor = std::exp(r_f * delta_t * n);
            double rhs = Y_1(i) + theta * delta_t * (b2(i) * exp_factor - A2_result(i));
            Y_1(i) = rhs;  // Reuse Y_1 to store RHS
        });

        // Create temporary storage for block solve
        const int spacing = m1 + 1;
        const int num_rows = (m2-1)*spacing;
        const int total_size = (m2+1)*spacing;

        Kokkos::View<double*> c_star("c_star", num_rows);
        Kokkos::View<double*> c2_star("c2_star", spacing);
        Kokkos::View<double*> d_star("d_star", total_size);

        // A2 implicit solve
        Kokkos::parallel_for("A2_solve", 1, KOKKOS_LAMBDA(const int) {
            // Identity block
            for (int i = num_rows; i < total_size; i++) {
                d_star(i) = Y_1(i);
            }

            // Normalize first m1+1 rows and upper2_diagonal
            for (int i = 0; i < spacing; i++) {
                c_star(i) = A2_impl_upper(i) / A2_impl_main(i);
                c2_star(i) = A2_impl_upper2(i) / A2_impl_main(i);
                d_star(i) = Y_1(i) / A2_impl_main(i);
            }

            // First block forward sweep
            for (int i = 0; i < spacing; i++) {
                double c_upper = A2_impl_upper(i+spacing) - c2_star(i)*A2_impl_lower(i);
                double m = 1.0 / (A2_impl_main(i+spacing) - c_star(i)*A2_impl_lower(i));
                c_star(i+spacing) = c_upper * m;
                d_star(i+spacing) = (Y_1(i+spacing) - A2_impl_lower(i) * d_star(i)) * m;
            }

            // Middle blocks forward sweep
            for (int i = spacing; i < num_rows - spacing; i++) {
                double m = 1.0 / (A2_impl_main(i+spacing) - c_star(i)*A2_impl_lower(i));
                c_star(i+spacing) = A2_impl_upper(i+spacing) * m;
                d_star(i+spacing) = (Y_1(i+spacing) - A2_impl_lower(i) * d_star(i)) * m;
            }

            // Pre-backward sweep
            for (int i = num_rows - spacing; i < num_rows; i++) {
                d_star(i) -= d_star(i+spacing)*c_star(i);
            }

            // Last m1+1 rows
            for (int i = num_rows - spacing; i < num_rows; i++) {
                U(i) = d_star(i);
            }

            // Backward sweep
            for (int i = num_rows - 1; i >= 3 * spacing; i--) {
                U(i - spacing) = d_star(i - spacing) - c_star(i - spacing) * U(i);
            }

            // First block back substitution with upper2_diag
            for (int i = 3 * spacing - 1; i >= 2 * spacing; i--) {
                U(i - spacing) = d_star(i - spacing) - c_star(i - spacing) * U(i);
                d_star(i - 2*spacing) -= c2_star(i - 2*spacing) * U(i);
            }

            // Last backward substitution
            for (int i = 2 * spacing - 1; i >= spacing; i--) {
                U(i - spacing) = d_star(i - spacing) - c_star(i - spacing) * U(i);
            }

            // Identity block
            for (int i = num_rows; i < total_size; i++) {
                U(i) = d_star(i);
            }
        });
        }
    }
    auto t_end = timer::now();

    std::cout << "DO optimized time: "
              << std::chrono::duration<double>(t_end - t_start).count()
              << " seconds" << std::endl;
    
    // Get final price
    auto h_U = Kokkos::create_mirror_view(U);
    Kokkos::deep_copy(h_U, U);

    int index_s = std::find(grid.Vec_s.begin(), grid.Vec_s.end(), S_0) - grid.Vec_s.begin();
    int index_v = std::find(grid.Vec_v.begin(), grid.Vec_v.end(), V_0) - grid.Vec_v.begin();
    
    return h_U[index_s + index_v*(m1+1)];
}


void test_Call_opti(){
    // Market parameters
    const double K = 100.0;
    const double S_0 = 100.0;
    const double V_0 = 0.04;
    const double T = 1.0;
    const double r_d = 0.025;
    const double r_f = 0.0;
    const double rho = -0.9;
    const double sigma = 0.3;
    const double kappa = 1.5;
    const double eta = 0.04;
    
    // Grid/solver parameters
    const int m1 = 50;
    const int m2 = 50;
    const int N = 20;
    const double delta_t = T / N;
    const double theta = 0.8;

    // Create grid
    Grid grid(m1, 8*K, S_0, K, K/5, m2, 5.0, V_0, 5.0/500);

    std::cout << "Starting DO scheme optimization test..." << std::endl;
    
    double option_price = DO_scheme_optimized(m1, m2, N, delta_t, theta, r_f, grid, 
                                            rho, sigma, r_d, kappa, eta, S_0, V_0);
    
    // Compare with reference price
    const double reference_price = 8.8948693600540167;
    std::cout << std::setprecision(16) << "Option price: " << option_price << std::endl;
    std::cout << "Relative error: " << std::abs(option_price - reference_price)/reference_price << std::endl;

    std::cout << "Test completed successfully" << std::endl;
}

void test_A1_operations() {
    // Test dimensions
    const int m1 = 300;
    const int m2 = 100;
    const int m = (m1 + 1) * (m2 + 1);
    std::cout << "Testing A1 with dimensions m1=" << m1 << ", m2=" << m2 << "\n";

    // Create grid
    Grid grid = create_test_grid(m1, m2);
    
    // Matrix parameters
    double rho = -0.9;
    double sigma = 0.3;
    double r_d = 0.025;
    double r_f = 0.0;
    double theta = 0.8;
    double delta_t = 1.0/20;

    // Create Views for A1 matrix
    Kokkos::View<double**> A1_main_diags("A1_main_diags", m2+1, m1+1);
    Kokkos::View<double**> A1_lower_diags("A1_lower_diags", m2+1, m1);
    Kokkos::View<double**> A1_upper_diags("A1_upper_diags", m2+1, m1);

    // Create Views for implicit A1 matrix
    Kokkos::View<double**> A1_impl_main("A1_impl_main", m2+1, m1+1);
    Kokkos::View<double**> A1_impl_lower("A1_impl_lower", m2+1, m1);
    Kokkos::View<double**> A1_impl_upper("A1_impl_upper", m2+1, m1);

    // Test vectors
    Kokkos::View<double*> x("x", m);
    Kokkos::View<double*> b("b", m);
    Kokkos::View<double*> result("result", m);
    
    // Initialize matrix
    initialize_A1_matrices(A1_main_diags, A1_lower_diags, A1_upper_diags, 
                            grid, rho, sigma, r_d, r_f);

    // Build implicit matrix
    build_A1_implicit(A1_impl_main, A1_impl_lower, A1_impl_upper,
                    A1_main_diags, A1_lower_diags, A1_upper_diags,
                    theta, delta_t, m1, m2);

    // Initialize test vectors with random values
    auto h_x = Kokkos::create_mirror_view(x);
    auto h_b = Kokkos::create_mirror_view(b);
    for (int i = 0; i < m; ++i) {
        h_x(i) = std::rand() / (RAND_MAX + 1.0);
        h_b(i) = std::rand() / (RAND_MAX + 1.0);
    }
    Kokkos::deep_copy(x, h_x);
    Kokkos::deep_copy(b, h_b);

    // Test explicit multiply
    auto t_start = std::chrono::high_resolution_clock::now();
    
    Kokkos::parallel_for("A1_multiply", Kokkos::RangePolicy<>(0, m2+1), 
        KOKKOS_LAMBDA(const int j) {
        const int offset = j * (m1 + 1);
        for(int i = 0; i <= m1; i++) {
            double sum = A1_main_diags(j,i) * x(offset + i);
            if(i > 0) {
                sum += A1_lower_diags(j,i-1) * x(offset + i-1);
            }
            if(i < m1) {
                sum += A1_upper_diags(j,i) * x(offset + i+1);
            }
            result(offset + i) = sum;
        }
    });
    Kokkos::fence();
    
    auto t_end = std::chrono::high_resolution_clock::now();
    std::cout << "Explicit multiply time: " 
                << std::chrono::duration<double>(t_end - t_start).count()
                << " seconds" << std::endl;

    // Test implicit solve
    // Create persistent temp array for parallel solve
    Kokkos::View<double**> temp("A1_temp", m2+1, m1+1);  // Changed to 2D array

    // Test implicit solve
    t_start = std::chrono::high_resolution_clock::now();

    Kokkos::parallel_for("A1_solve", Kokkos::RangePolicy<>(0, m2+1), 
        KOKKOS_LAMBDA(const int j) {
        const int offset = j * (m1 + 1);
        
        // Forward sweep
        temp(j,0) = A1_impl_main(j,0);  // Use 2D indexing for temp
        x(offset) = b(offset);
        
        for(int i = 1; i <= m1; i++) {
            double m = A1_impl_lower(j,i-1) / temp(j,i-1);  // Use 2D indexing
            temp(j,i) = A1_impl_main(j,i) - m * A1_impl_upper(j,i-1);
            x(offset + i) = b(offset + i) - m * x(offset + i-1);
        }

        // Back substitution
        x(offset + m1) /= temp(j,m1);
        for(int i = m1-1; i >= 0; i--) {
            x(offset + i) = (x(offset + i) - 
                A1_impl_upper(j,i) * x(offset + i+1)) / temp(j,i);
        }
    });
    Kokkos::fence();
    
    t_end = std::chrono::high_resolution_clock::now();
    std::cout << "Implicit solve time: "
                << std::chrono::duration<double>(t_end - t_start).count()
                << " seconds" << std::endl;

    // Verify solution by computing residual
    Kokkos::View<double*> verify("verify", m);
    
    // Compute A1*x
    Kokkos::parallel_for("residual", Kokkos::RangePolicy<>(0, m2+1), 
        KOKKOS_LAMBDA(const int j) {
        const int offset = j * (m1 + 1);
        for(int i = 0; i <= m1; i++) {
            double sum = A1_main_diags(j,i) * x(offset + i);
            if(i > 0) {
                sum += A1_lower_diags(j,i-1) * x(offset + i-1);
            }
            if(i < m1) {
                sum += A1_upper_diags(j,i) * x(offset + i+1);
            }
            verify(offset + i) = sum;
        }
    });
    
    // Compute residual norm
    auto h_verify = Kokkos::create_mirror_view(verify);
    auto h_x_final = Kokkos::create_mirror_view(x);
    Kokkos::deep_copy(h_verify, verify);
    Kokkos::deep_copy(h_x_final, x);

    double residual = 0.0;
    for(int i = 0; i < m; i++) {
        double res = h_x_final(i) - theta * delta_t * h_verify(i) - h_b(i);
        residual += res * res;
    }
    residual = std::sqrt(residual);
    
    std::cout << "Residual norm: " << residual << std::endl;
}


#include <Kokkos_Core.hpp>
#include <iostream>
#include <cmath>
#include <chrono>

// Assume you have:
// struct Grid { ... };
// Grid create_test_grid(int m1, int m2);

// We'll just use fixed coefficients as in your snippet (-2 for main, 1 for lower/upper).
// Replace with your actual computations if needed.

// Build the explicit matrix in flattened form
void build_A1_flat_matrix(const Grid& grid, int m1, int m2,
                          Kokkos::View<double*> main_diag,
                          Kokkos::View<double*> lower_diag,
                          Kokkos::View<double*> upper_diag) {
    auto h_main = Kokkos::create_mirror_view(main_diag);
    auto h_lower = Kokkos::create_mirror_view(lower_diag);
    auto h_upper = Kokkos::create_mirror_view(upper_diag);

    int total_size = (m1+1)*(m2+1);
    for (int i = 0; i < total_size; i++) {
        h_main(i) = 0.0; // Initialize to 0 first
    }

    // Fill block by block
    for(int j = 0; j <= m2; j++) {
        int block_offset = j*(m1+1);
        // Interior rows
        for(int i = 1; i < m1; i++) {
            int idx = block_offset + i;
            h_lower(idx-1) = 1.0;
            h_main(idx) = -2.0;
            h_upper(idx) = 1.0;
        }
        // Last row in block
        h_main(block_offset + m1) = -2.0;

        // Boundaries between blocks (if any)
        if(j < m2) {
            h_lower(block_offset + m1) = 0.0;
            h_upper(block_offset + m1) = 0.0;
        }
    }

    // Copy to device
    Kokkos::deep_copy(main_diag, h_main);
    Kokkos::deep_copy(lower_diag, h_lower);
    Kokkos::deep_copy(upper_diag, h_upper);
}

// Build the implicit matrix
void build_A1_flat_implicit(double theta, double delta_t,
                            Kokkos::View<double*> main_diag,
                            Kokkos::View<double*> lower_diag,
                            Kokkos::View<double*> upper_diag,
                            Kokkos::View<double*> implicit_main_diag,
                            Kokkos::View<double*> implicit_lower_diag,
                            Kokkos::View<double*> implicit_upper_diag) {
    int N = main_diag.extent(0);
    double alpha = theta * delta_t;

    auto h_main = Kokkos::create_mirror_view(main_diag);
    auto h_lower = Kokkos::create_mirror_view(lower_diag);
    auto h_upper = Kokkos::create_mirror_view(upper_diag);

    Kokkos::deep_copy(h_main, main_diag);
    Kokkos::deep_copy(h_lower, lower_diag);
    Kokkos::deep_copy(h_upper, upper_diag);

    auto h_impl_main = Kokkos::create_mirror_view(implicit_main_diag);
    auto h_impl_lower = Kokkos::create_mirror_view(implicit_lower_diag);
    auto h_impl_upper = Kokkos::create_mirror_view(implicit_upper_diag);

    for (int i = 0; i < N; i++) {
        h_impl_main(i) = 1.0 - alpha * h_main(i);
    }

    for (int i = 0; i < N - 1; i++) {
        h_impl_lower(i) = -alpha * h_lower(i);
        h_impl_upper(i) = -alpha * h_upper(i);
    }

    Kokkos::deep_copy(implicit_main_diag, h_impl_main);
    Kokkos::deep_copy(implicit_lower_diag, h_impl_lower);
    Kokkos::deep_copy(implicit_upper_diag, h_impl_upper);
}

// Multiply function
void A1_flat_multiply(Kokkos::View<double*> main_diag,
                      Kokkos::View<double*> lower_diag,
                      Kokkos::View<double*> upper_diag,
                      const Kokkos::View<double*>& x,
                      Kokkos::View<double*>& result) {
    int N = main_diag.extent(0);
    Kokkos::parallel_for("A1_flat_multiply", N, KOKKOS_LAMBDA(const int i) {
        double val = main_diag(i)*x(i);
        if(i > 0) {
            val += lower_diag(i-1)*x(i-1);
        }
        if(i < N-1) {
            val += upper_diag(i)*x(i+1);
        }
        result(i) = val;
    });
    Kokkos::fence();
}

// Solve implicit using Thomas algorithm (c_prime and d_prime given)
void A1_flat_solve_implicit(Kokkos::View<double*> implicit_main_diag,
                            Kokkos::View<double*> implicit_lower_diag,
                            Kokkos::View<double*> implicit_upper_diag,
                            Kokkos::View<double*> c_prime,
                            Kokkos::View<double*> d_prime,
                            Kokkos::View<double*>& x,
                            const Kokkos::View<double*>& b) {
    int N = implicit_main_diag.extent(0);

    // We'll do sequential solve on device (single thread)
    Kokkos::parallel_for("A1_thomas_solve_flat", Kokkos::RangePolicy<>(0,1), KOKKOS_LAMBDA(const int) {
        // Forward sweep
        c_prime(0) = implicit_upper_diag(0)/implicit_main_diag(0);
        d_prime(0)=b(0)/implicit_main_diag(0);

        for(int i=1;i<N;i++){
            double denom = implicit_main_diag(i)-implicit_lower_diag(i-1)*c_prime(i-1);
            if(i<N-1){
                c_prime(i)=implicit_upper_diag(i)/denom;
            }
            d_prime(i)=(b(i)-implicit_lower_diag(i-1)*d_prime(i-1))/denom;
        }

        // Back substitution
        x(N-1)=d_prime(N-1);
        for(int i=N-2;i>=0;i--){
            x(i)=d_prime(i)-c_prime(i)*x(i+1);
        }
    });
    Kokkos::fence();
}


void test_A1_flat_performance_no_class() {
    using timer = std::chrono::high_resolution_clock;

    const int m1 = 300;
    const int m2 = 100;
    std::cout << "Testing flat A1 with dimensions m1=" << m1 << ", m2=" << m2 << "\n";

    Grid grid = create_test_grid(m1, m2);

    double rho = -0.9;
    double sigma = 0.3;
    double r_d = 0.025;
    double r_f = 0.0;
    double theta = 1.0;
    double delta_t = 1.0;

    int total_size = (m1 + 1)*(m2 + 1);

    // Allocate all Views
    Kokkos::View<double*> main_diag("main_diag", total_size);
    Kokkos::View<double*> lower_diag("lower_diag", total_size-1);
    Kokkos::View<double*> upper_diag("upper_diag", total_size-1);

    Kokkos::View<double*> implicit_main_diag("implicit_main", total_size);
    Kokkos::View<double*> implicit_lower_diag("implicit_lower", total_size-1);
    Kokkos::View<double*> implicit_upper_diag("implicit_upper", total_size-1);

    Kokkos::View<double*> c_prime("c_prime", total_size);
    Kokkos::View<double*> d_prime("d_prime", total_size);

    Kokkos::View<double*> x("x", total_size);
    Kokkos::View<double*> b("b", total_size);
    Kokkos::View<double*> result("result", total_size);

    // Initialize arrays
    build_A1_flat_matrix(grid, m1, m2, main_diag, lower_diag, upper_diag);

    // Initialize b and x
    auto h_b=Kokkos::create_mirror_view(b);
    auto h_x=Kokkos::create_mirror_view(x);
    for(int i=0;i<total_size;i++){
        h_b(i)=std::rand()/(RAND_MAX+1.0);
        h_x(i)=std::rand()/(RAND_MAX+1.0);
    }
    Kokkos::deep_copy(b,h_b);
    Kokkos::deep_copy(x,h_x);

    // Build implicit matrix
    build_A1_flat_implicit(theta, delta_t,
                      main_diag, lower_diag, upper_diag,
                      implicit_main_diag, implicit_lower_diag, implicit_upper_diag);

    auto t_start = timer::now();

    int N = total_size;

    // Run Thomas solve on device (single thread)
    Kokkos::parallel_for("A1_thomas_solve_flat", Kokkos::RangePolicy<>(0,1), KOKKOS_LAMBDA(const int) {
        // Forward sweep
        c_prime(0) = implicit_upper_diag(0)/implicit_main_diag(0);
        d_prime(0) = b(0)/implicit_main_diag(0);

        for(int i=1; i<N; i++){
            double denom = implicit_main_diag(i) - implicit_lower_diag(i-1)*c_prime(i-1);
            if(i<N-1) {
                c_prime(i) = implicit_upper_diag(i)/denom;
            }
            d_prime(i) = (b(i) - implicit_lower_diag(i-1)*d_prime(i-1))/denom;
        }

        // Back substitution
        x(N-1)=d_prime(N-1);
        for(int i=N-2; i>=0; i--){
            x(i) = d_prime(i) - c_prime(i)*x(i+1);
        }
    });
    Kokkos::fence();
     
    auto t_end = timer::now();
    std::cout << "Implicit time: " 
              << std::chrono::duration<double>(t_end - t_start).count()
              << " seconds" << std::endl; 

    // Verify solution by computing residual
    // Multiply A*x
    A1_flat_multiply(main_diag, lower_diag, upper_diag, x, result);

    auto h_result = Kokkos::create_mirror_view(result);
    auto h_b_final = Kokkos::create_mirror_view(b);
    auto h_x_final = Kokkos::create_mirror_view(x);
    Kokkos::deep_copy(h_result, result);
    Kokkos::deep_copy(h_b_final, b);
    Kokkos::deep_copy(h_x_final, x);

    double residual=0.0;
    for(int i=0; i<total_size; i++){
        double res = h_x_final(i) - theta*delta_t*h_result(i) - h_b_final(i);
        residual += res*res;
    }
    residual = std::sqrt(residual);
    std::cout<<"Residual norm: "<<residual<<"\n";
}



void test_DO_scheme_optimized() {
    Kokkos::initialize();
    {
        try {
            //test_Call_opti();
            //test_A1_operations();
            test_A1_flat_performance_no_class();
        }
        catch (std::exception& e) {
            std::cout << "Error during test: " << e.what() << std::endl;
        }
    }
    Kokkos::finalize();
}