#include "hes_mat_fac.hpp"
#include <iostream>

//for std::setprec() output debugging of flatten A1
#include <iomanip>

/*

A0 class constructor and build function

*/
heston_A0Storage_gpu::heston_A0Storage_gpu(int m1_in, int m2_in) 
    : m1(m1_in), m2(m2_in) {
    values = Kokkos::View<double**>("A0_values", m2 - 1, (m1 - 1) * 9);
}

void heston_A0Storage_gpu::build_matrix(const Grid& grid, double rho, double sigma) {
    auto values_host = Kokkos::create_mirror_view(values);
    Kokkos::deep_copy(values_host, 0.0);

    for(int j = 0; j < m2-1; ++j) {
        for(int i = 0; i < m1-1; ++i) {
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


/*

A1 class constructor and build function

*/
heston_A1Storage_gpu::heston_A1Storage_gpu(int m1_in, int m2_in) : m1(m1_in), m2(m2_in) {
    main_diags = Kokkos::View<double**>("A1_main_diags", m2+1, m1+1);
    lower_diags = Kokkos::View<double**>("A1_lower_diags", m2+1, m1);
    upper_diags = Kokkos::View<double**>("A1_upper_diags", m2+1, m1);

    implicit_main_diags = Kokkos::View<double**>("A1_impl_main_diags", m2+1, m1+1);
    implicit_lower_diags = Kokkos::View<double**>("A1_impl_lower_diags", m2+1, m1);
    implicit_upper_diags = Kokkos::View<double**>("A1_impl_upper_diags", m2+1, m1);

    temp_sequ = Kokkos::View<double*>("temp_sequ", m1+1);
    temp_para = Kokkos::View<double**>("temp_para", m2 + 1, m1 + 1);
}

void heston_A1Storage_gpu::build_matrix(const Grid& grid, double rho, double sigma, double r_d, double r_f) {
    auto main_diags_host = Kokkos::create_mirror_view(main_diags);
    auto lower_diags_host = Kokkos::create_mirror_view(lower_diags);
    auto upper_diags_host = Kokkos::create_mirror_view(upper_diags);
    
    Kokkos::deep_copy(main_diags_host, 0.0);
    Kokkos::deep_copy(lower_diags_host, 0.0);
    Kokkos::deep_copy(upper_diags_host, 0.0);

    // For j in range(m2 + 1)
    for(int j = 0; j <= m2; j++) {
        for(int i = 1; i < m1; i++) {
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
        main_diags_host(j, m1) = -0.5 * r_d;
    }

    Kokkos::deep_copy(main_diags, main_diags_host);
    Kokkos::deep_copy(lower_diags, lower_diags_host);
    Kokkos::deep_copy(upper_diags, upper_diags_host);
}

void heston_A1Storage_gpu::build_implicit(const double theta, const double delta_t) {
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

A2 class constructor and build function

*/
/*
heston_A2Storage_gpu::heston_A2Storage_gpu(int m1_in, int m2_in) : m1(m1_in), m2(m2_in) {
    main_diag = Kokkos::View<double*>("A2_main_diag", (m2-1)*(m1+1));
    lower_diag = Kokkos::View<double*>("A2_lower_diag", (m2-2)*(m1+1));
    upper_diag = Kokkos::View<double*>("A2_upper_diag", (m2-1)*(m1+1));
    upper2_diag = Kokkos::View<double*>("A2_upper2_diag", m1+1);

    implicit_main_diag = Kokkos::View<double*>("A2_impl_main_diag", (m2+1)*(m1+1));
    implicit_lower_diag = Kokkos::View<double*>("A2_impl_lower_diag", (m2-2)*(m1+1));
    implicit_upper_diag = Kokkos::View<double*>("A2_impl_upper_diag", (m2-1)*(m1+1));
    implicit_upper2_diag = Kokkos::View<double*>("A2_impl_upper2_diag", m1+1);
}


void heston_A2Storage_gpu::build_matrix(const Grid& grid, double rho, double sigma, double r_d, 
                                      double kappa, double eta) {
    std::cout<< "in functionm";
    auto h_main = Kokkos::create_mirror_view(main_diag);
    auto h_lower = Kokkos::create_mirror_view(lower_diag);
    auto h_upper = Kokkos::create_mirror_view(upper_diag);
    auto h_upper2 = Kokkos::create_mirror_view(upper2_diag);
    
    Kokkos::deep_copy(h_main, 0.0);
    Kokkos::deep_copy(h_lower, 0.0);
    Kokkos::deep_copy(h_upper, 0.0);
    Kokkos::deep_copy(h_upper2, 0.0);

    int spacing = m1+1;

    // Handle j=0 case first
    for(int i = 0; i < m1+1; i++) {
        double temp = kappa * (eta - grid.Vec_v[0]);
        // Use l_9c = [0,1,2]
        h_main(i) += temp * gamma_v(0, 0, grid.Delta_v);
        h_upper(i) += temp * gamma_v(0, 1, grid.Delta_v);
        h_upper2(i) = temp * gamma_v(0, 2, grid.Delta_v);
    }

    // Handle remaining j values
    for(int j = 1; j < m2-1; j++) {
        for(int i = 0; i < m1+1; i++) {
            double temp = kappa * (eta - grid.Vec_v[j]);
            double temp2 = 0.5 * sigma * sigma * grid.Vec_v[j];

            if(grid.Vec_v[j] > 1.0) {
                int main_idx = i + (j+1)*(m1+1);
                // Using l_9a = [-2,-1,0]
                if(j > 0) h_lower[main_idx-spacing] += temp * alpha_v(j, -2, grid.Delta_v);
                h_lower[main_idx] += temp * alpha_v(j, -1, grid.Delta_v);
                h_main[main_idx] += temp * alpha_v(j, 0, grid.Delta_v);

                // Add regular central differences
                for(int k = -1; k <= 1; k++) {
                    int idx = j > 0 ? main_idx + k*spacing : i + k*spacing;
                    if(k == -1) h_lower[idx] += temp2 * delta_v(j-1, k, grid.Delta_v);
                    else if(k == 0) h_main[main_idx] += temp2 * delta_v(j-1, k, grid.Delta_v);
                    else h_upper[main_idx] += temp2 * delta_v(j-1, k, grid.Delta_v);
                }
            } else {
                int main_idx = i + j*(m1+1);
                for(int k = -1; k <= 1; k++) {
                    if(k == -1 && j > 0) {
                        h_lower[main_idx] += (temp * beta_v(j-1, k, grid.Delta_v) + 
                                            temp2 * delta_v(j-1, k, grid.Delta_v));
                    }
                    else if(k == 0) {
                        h_main[main_idx] += (temp * beta_v(j-1, k, grid.Delta_v) + 
                                           temp2 * delta_v(j-1, k, grid.Delta_v));
                    }
                    else {
                        h_upper[main_idx] += (temp * beta_v(j-1, k, grid.Delta_v) + 
                                            temp2 * delta_v(j-1, k, grid.Delta_v));
                    }
                }
            }
            h_main[i + j*(m1+1)] += -0.5 * r_d;
        }
    }

    Kokkos::deep_copy(main_diag, h_main);
    Kokkos::deep_copy(lower_diag, h_lower);
    Kokkos::deep_copy(upper_diag, h_upper);
    Kokkos::deep_copy(upper2_diag, h_upper2);
}

void heston_A2Storage_gpu::build_implicit(const double theta, const double delta_t) {
    Kokkos::parallel_for("build_implicit", 1, KOKKOS_LAMBDA(const int) {
        // Initialize implicit_main_diag with identity
        for(int i = 0; i < (m2+1)*(m1+1); i++) {
            implicit_main_diag(i) = 1.0;
        }

        // Subtract theta*delta_t*A2 from main diagonal where A2 is defined
        for(int i = 0; i < (m2-1)*(m1+1); i++) {
            implicit_main_diag(i) -= theta * delta_t * main_diag(i);
        }

        // Build the off-diagonal terms
        for(int i = 0; i < (m2-2)*(m1+1); i++) {
            implicit_lower_diag(i) = -theta * delta_t * lower_diag(i);
        }

        for(int i = 0; i < (m2-1)*(m1+1); i++) {
            implicit_upper_diag(i) = -theta * delta_t * upper_diag(i);
        }

        for(int i = 0; i < m1+1; i++) {
            implicit_upper2_diag(i) = -theta * delta_t * upper2_diag(i);
        }
    });
    Kokkos::fence();
}
*/



/*
struct heston_A0Storage_gpu {
    int m1, m2;
    Kokkos::View<double**> values; // [m2 - 1][(m1 - 1) * 9]

    heston_A0Storage_gpu(int m1_in, int m2_in) : m1(m1_in), m2(m2_in) {
        // Allocate the values View
        values = Kokkos::View<double**>("A0_values", m2 - 1, (m1 - 1) * 9);
    }

    // Initialize the matrix
    void build_matrix(const Grid& grid, double rho, double sigma) {
        auto values_host = Kokkos::create_mirror_view(values);
        
        // Set everything to zero initially
        Kokkos::deep_copy(values_host, 0.0);

        // Python's range(1, m2) and range(1, m1)
        for(int j = 0; j < m2-1; ++j) {  // j maps to j+1 in Python
            for(int i = 0; i < m1-1; ++i) { // i maps to i+1 in Python
                double c = rho * sigma * grid.Vec_s[i+1] * grid.Vec_v[j+1];

                // Loop over k and l as in Python's l_11
                for(int k = -1; k <= 1; ++k) {
                    for(int l = -1; l <= 1; ++l) {
                        // Calculate beta coefficients
                        double beta_s_val = beta_s(i, k, grid.Delta_s);
                        double beta_v_val = beta_v(j, l, grid.Delta_v);
                        
                        // Store in flattened format
                        // Convert k,l to linear index in range [0,8]
                        int idx = i * 9 + (l + 1) * 3 + (k + 1);
                        values_host(j, idx) = c * beta_s_val * beta_v_val;
                    }
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

};

void test_heston_A0() {
   int m1 = 5;
   int m2 = 5;
   
   // Create grid
   Grid grid = create_test_grid(m1, m2);
   
   // Create and build A0 matrix
   heston_A0Storage_gpu A0(m1, m2);
   double rho = -0.9;
   double sigma = 0.3;
   A0.build_matrix(grid, rho, sigma);
   
   // Get host copy
   auto values_host = Kokkos::create_mirror_view(A0.values);
   Kokkos::deep_copy(values_host, A0.values);
   
   // Print matrix structure
   std::cout << "A0 Matrix Structure:" << std::endl;
   std::cout << "--------------------" << std::endl;
   
   for(int j = 0; j < m2-1; ++j) {
       std::cout << "\nVariance level j=" << j << ":" << std::endl;
       for(int i = 0; i < m1-1; ++i) {
           std::cout << "Row " << i << ": ";
           for(int val = 0; val < 9; ++val) {
               std::cout << values_host(j, i*9 + val) << " ";
           }
           std::cout << std::endl;
       }
   }
   
   // Print dimensions
   std::cout << "\nDimensions:" << std::endl;
   std::cout << "Values shape: [" << m2-1 << "][" << (m1-1)*9 << "]" << std::endl;
}
*/
//A0 class test
void test_heston_A0() {
    {
        int m1 = 5;
        int m2 = 5;
        
        // Create grid
        Grid grid = create_test_grid(m1, m2);
        
        // Create and build A0 matrix
        heston_A0Storage_gpu A0(m1, m2);
        double rho = -0.9;
        double sigma = 0.3;
        A0.build_matrix(grid, rho, sigma);
        
        // Get host copy using the getter method
        auto values = A0.get_values();
        auto values_host = Kokkos::create_mirror_view(values);
        Kokkos::deep_copy(values_host, values);
        
        // Print matrix structure
        std::cout << "A0 Matrix Structure:" << std::endl;
        std::cout << "--------------------" << std::endl;
        
        for(int j = 0; j < m2-1; ++j) {
            std::cout << "\nVariance level j=" << j << ":" << std::endl;
            for(int i = 0; i < m1-1; ++i) {
                std::cout << "Row " << i << ": ";
                for(int val = 0; val < 9; ++val) {
                    std::cout << values_host(j, i*9 + val) << " ";
                }
                std::cout << std::endl;
            }
        }
        
        // Print dimensions
        std::cout << "\nDimensions:" << std::endl;
        std::cout << "Values shape: [" << m2-1 << "][" << (m1-1)*9 << "]" << std::endl;
        std::cout << "m1: " << A0.get_m1() << ", m2: " << A0.get_m2() << std::endl;
    }
}


/*
struct heston_A1Storage_gpu {
    int m1, m2;
    
    Kokkos::View<double**> main_diags;
    Kokkos::View<double**> lower_diags;
    Kokkos::View<double**> upper_diags;

    Kokkos::View<double**> implicit_main_diags;
    Kokkos::View<double**> implicit_lower_diags;
    Kokkos::View<double**> implicit_upper_diags;

    heston_A1Storage_gpu(int m1_, int m2_) : m1(m1_), m2(m2_) {
        main_diags = Kokkos::View<double**>("A1_main_diags", m2+1, m1+1);
        lower_diags = Kokkos::View<double**>("A1_lower_diags", m2+1, m1);
        upper_diags = Kokkos::View<double**>("A1_upper_diags", m2+1, m1);

        implicit_main_diags = Kokkos::View<double**>("A1_impl_main_diags", m2+1, m1+1);
        implicit_lower_diags = Kokkos::View<double**>("A1_impl_lower_diags", m2+1, m1);
        implicit_upper_diags = Kokkos::View<double**>("A1_impl_upper_diags", m2+1, m1);
    }

    void build_matrix(const Grid& grid, double rho, double sigma, double r_d, double r_f) {
        auto main_diags_host = Kokkos::create_mirror_view(main_diags);
        auto lower_diags_host = Kokkos::create_mirror_view(lower_diags);
        auto upper_diags_host = Kokkos::create_mirror_view(upper_diags);
        
        Kokkos::deep_copy(main_diags_host, 0.0);
        Kokkos::deep_copy(lower_diags_host, 0.0);
        Kokkos::deep_copy(upper_diags_host, 0.0);

        // For j in range(m2 + 1)
        for(int j = 0; j <= m2; j++) {
            for(int i = 1; i < m1; i++) {
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
            main_diags_host(j, m1) = -0.5 * r_d;
        }

        Kokkos::deep_copy(main_diags, main_diags_host);
        Kokkos::deep_copy(lower_diags, lower_diags_host);
        Kokkos::deep_copy(upper_diags, upper_diags_host);
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

    //explicit method single thread
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
    
    //This is a sequential implicict solve. Only one thread is handling everything
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
    
};

void test_heston_A1() {
    int m1 = 5;
    int m2 = 5;
    Grid grid = create_test_grid(m1, m2);
    
    heston_A1Storage_gpu A1(m1, m2);
    double rho = -0.9;
    double sigma = 0.3;
    double r_d = 0.025;
    double r_f = 0.0;
    
    A1.build_matrix(grid, rho, sigma, r_d, r_f);
    
    auto main_host = Kokkos::create_mirror_view(A1.main_diags);
    auto lower_host = Kokkos::create_mirror_view(A1.lower_diags);
    auto upper_host = Kokkos::create_mirror_view(A1.upper_diags);
    
    Kokkos::deep_copy(main_host, A1.main_diags);
    Kokkos::deep_copy(lower_host, A1.lower_diags);
    Kokkos::deep_copy(upper_host, A1.upper_diags);
    
    std::cout << "A1 Matrix Structure:\n";
    for(int j = 0; j <= m2; j++) {
        std::cout << "\nVariance level j=" << j << ":\n";
        std::cout << "Lower diagonal: ";
        for(int i = 0; i < m1; i++) std::cout << lower_host(j,i) << " ";
        std::cout << "\nMain diagonal:  ";
        for(int i = 0; i <= m1; i++) std::cout << main_host(j,i) << " ";
        std::cout << "\nUpper diagonal: ";
        for(int i = 0; i < m1; i++) std::cout << upper_host(j,i) << " ";
        std::cout << "\n";
    }
}
*/
//A1 class test
void test_heston_A1() {
    int m1 = 100;
    int m2 = 75;
    Grid grid = create_test_grid(m1, m2);
    
    heston_A1Storage_gpu A1(m1, m2);
    double rho = -0.9;
    double sigma = 0.3;
    double r_d = 0.025;
    double r_f = 0.0;

    const double theta = 0.8;
    const double delta_t = 1.0/40.0; 

    using timer = std::chrono::high_resolution_clock;
    
    auto t_start = timer::now();
    A1.build_matrix(grid, rho, sigma, r_d, r_f);
    A1.build_implicit(theta, delta_t);
    auto t_end = timer::now();
    
    std::cout << "Build matrix time: "
              << std::chrono::duration<double>(t_end - t_start).count()
              << " seconds" << std::endl;
    
    // Get Views using getters
    auto main = A1.get_main_diags();
    auto lower = A1.get_lower_diags();
    auto upper = A1.get_upper_diags();

    auto implicit_main = A1.get_implicit_main_diags();
    auto implicit_lower = A1.get_implicit_lower_diags();
    auto implicit_upper = A1.get_implicit_upper_diags();

    // Create mirror views (fixed typo in Kokkos::)
    auto main_host = Kokkos::create_mirror_view(main);
    auto lower_host = Kokkos::create_mirror_view(lower);
    auto upper_host = Kokkos::create_mirror_view(upper);

    auto implicit_main_host = Kokkos::create_mirror_view(implicit_main);
    auto implicit_lower_host = Kokkos::create_mirror_view(implicit_lower);
    auto implicit_upper_host = Kokkos::create_mirror_view(implicit_upper);
    
    // Copy to host
    Kokkos::deep_copy(main_host, main);
    Kokkos::deep_copy(lower_host, lower);
    Kokkos::deep_copy(upper_host, upper);

    Kokkos::deep_copy(implicit_main_host, implicit_main);
    Kokkos::deep_copy(implicit_lower_host, implicit_lower);
    Kokkos::deep_copy(implicit_upper_host, implicit_upper);
    
    // Print matrices
    /*
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "A1 Matrix Structure:\n";
    std::cout << "--------------------\n";
    
    for(int j = 0; j <= m2; j++) {
        std::cout << "\nVariance level j=" << j << ":\n";
        std::cout << "Lower diagonal: ";
        for(int i = 0; i < m1; i++) std::cout << lower_host(j,i) << " ";
        std::cout << "\nMain diagonal:  ";
        for(int i = 0; i <= m1; i++) std::cout << main_host(j,i) << " ";
        std::cout << "\nUpper diagonal: ";
        for(int i = 0; i < m1; i++) std::cout << upper_host(j,i) << " ";
        std::cout << "\n";

        std::cout << "Implicit diagonals:\n";
        std::cout << "Lower diagonal: ";
        for(int i = 0; i < m1; i++) std::cout << implicit_lower_host(j,i) << " ";
        std::cout << "\nMain diagonal:  ";
        for(int i = 0; i <= m1; i++) std::cout << implicit_main_host(j,i) << " ";
        std::cout << "\nUpper diagonal: ";
        for(int i = 0; i < m1; i++) std::cout << implicit_upper_host(j,i) << " ";
        std::cout << "\n";
        std::cout << "----------------------------------------\n";

    }
    */

    // Print dimensions
    std::cout << "\nDimensions:\n";
    std::cout << "m1: " << A1.get_m1() << ", m2: " << A1.get_m2() << "\n";
}   


//This test compare the explicit and implicit output of a simple test case vector. It is compared to the python implementation
//and checked that the outputs align. This test was written when i saw oszillatory behavior in the m1 direction when increasing
//this dimension size
//RESULT OF TEST: No discrepencies between ypthon and c++
void test_A1_structure() {
    // Test dimensions
    int m1 = 4;  // Small dimensions for readable output
    int m2 = 3;
    
    Grid grid = create_test_grid(m1, m2);
    
    // Create and build A1 matrix
    heston_A1Storage_gpu A1(m1, m2);
    double rho = -0.9;
    double sigma = 0.3;
    double r_d = 0.025;
    double r_f = 0.0;
    
    A1.build_matrix(grid, rho, sigma, r_d, r_f);
    
    const int total_size = (m1 + 1) * (m2 + 1);
    std::cout << "\nA1 Matrix Structure:";
    std::cout << "\nShape: [" << total_size << ", " << total_size << "]" << std::endl;

    std::cout << "\nA1 Matrix Values by Block:" << std::endl;
        
    auto h_main = Kokkos::create_mirror_view(A1.get_main_diags());
    auto h_lower = Kokkos::create_mirror_view(A1.get_lower_diags());
    auto h_upper = Kokkos::create_mirror_view(A1.get_upper_diags());
    
    Kokkos::deep_copy(h_main, A1.get_main_diags());
    Kokkos::deep_copy(h_lower, A1.get_lower_diags());
    Kokkos::deep_copy(h_upper, A1.get_upper_diags());
    
    for(int j = 0; j <= m2; j++) {
        std::cout << "\nBlock j=" << j << ":" << std::endl;
        
        // Print lower diagonal
        std::cout << "\n  Lower diagonal for block " << j << ":" << std::endl;
        for(int i = 0; i < m1; i++) {
            double val = h_lower(j,i);
            if(std::abs(val) > 1e-10) {
                std::cout << "    [" << i+1 << "," << i << "] = " 
                            << std::fixed << std::setprecision(6) << val << std::endl;
            }
        }
        
        // Print main diagonal
        std::cout << "\n  Main diagonal for block " << j << ":" << std::endl;
        for(int i = 0; i <= m1; i++) {
            double val = h_main(j,i);
            if(std::abs(val) > 1e-10) {
                std::cout << "    [" << i << "," << i << "] = " 
                            << std::fixed << std::setprecision(6) << val << std::endl;
            }
        }
        
        // Print upper diagonal
        std::cout << "\n  Upper diagonal for block " << j << ":" << std::endl;
        for(int i = 0; i < m1; i++) {
            double val = h_upper(j,i);
            if(std::abs(val) > 1e-10) {
                std::cout << "    [" << i << "," << i+1 << "] = " 
                            << std::fixed << std::setprecision(6) << val << std::endl;
            }
        }
    }
    
    // Create test vectors
    Kokkos::View<double*> x("x", total_size);
    Kokkos::View<double*> b("b", total_size);
    Kokkos::View<double*> result("result", total_size);
    
    auto h_x = Kokkos::create_mirror_view(x);
    auto h_b = Kokkos::create_mirror_view(b);
    
    // Simple counting vectors
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
    
    // Test explicit multiplication
    A1.multiply(x, result);
    
    auto h_result = Kokkos::create_mirror_view(result);
    Kokkos::deep_copy(h_result, result);
    
    std::cout << "\nExplicit multiplication first 10 results:" << std::endl;
    for(int i = 0; i < std::min(30, total_size); i++) {
        std::cout << "result[" << i << "] = " << std::fixed 
                    << std::setprecision(6) << h_result(i) << std::endl;
    }
    
    // Test implicit solve
    double theta = 0.8;
    double delta_t = 1.0/20;
    A1.build_implicit(theta, delta_t);
    A1.solve_implicit(x, b);
    
    auto h_implicit = Kokkos::create_mirror_view(x);
    Kokkos::deep_copy(h_implicit, x);
    
    std::cout << "\nImplicit solve first 10 results:" << std::endl;
    for(int i = 0; i < std::min(30, total_size); i++) {
        std::cout << "implicit_result[" << i << "] = " << std::fixed 
                    << std::setprecision(6) << h_implicit(i) << std::endl;
    }
    
    // Print structure of first block
    /*
    auto h_main = Kokkos::create_mirror_view(A1.get_main_diags());
    auto h_lower = Kokkos::create_mirror_view(A1.get_lower_diags());
    auto h_upper = Kokkos::create_mirror_view(A1.get_upper_diags());
    
    Kokkos::deep_copy(h_main, A1.get_main_diags());
    Kokkos::deep_copy(h_lower, A1.get_lower_diags());
    Kokkos::deep_copy(h_upper, A1.get_upper_diags());
    
    std::cout << "\nSparsity pattern for first block:" << std::endl;
    for(int i = 0; i <= m1; i++) {
        for(int j = 0; j <= m1; j++) {
            bool is_nonzero = false;
            if(j == i) is_nonzero = std::abs(h_main(0,i)) > 1e-10;
            if(j == i-1 && i > 0) is_nonzero = std::abs(h_lower(0,i-1)) > 1e-10;
            if(j == i+1 && i < m1) is_nonzero = std::abs(h_upper(0,i)) > 1e-10;
            std::cout << (is_nonzero ? 'X' : '.');
        }
        std::cout << std::endl;
    }
    */
}

/*
struct heston_A2Storage_gpu {
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

    heston_A2Storage_gpu(int m1_, int m2_) : m1(m1_), m2(m2_) {
        main_diag = Kokkos::View<double*>("A2_main_diag", (m2-1)*(m1+1));
        lower_diag = Kokkos::View<double*>("A2_lower_diag", (m2-2)*(m1+1));
        upper_diag = Kokkos::View<double*>("A2_upper_diag", (m2-1)*(m1+1));
        upper2_diag = Kokkos::View<double*>("A2_upper2_diag", m1+1);

        implicit_main_diag = Kokkos::View<double*>("A2_impl_main_diag", (m2+1)*(m1+1));
        implicit_lower_diag = Kokkos::View<double*>("A2_impl_lower_diag", (m2-2)*(m1+1));
        implicit_upper_diag = Kokkos::View<double*>("A2_impl_upper_diag", (m2-1)*(m1+1));
        implicit_upper2_diag = Kokkos::View<double*>("A2_impl_upper2_diag", m1+1);
    }

    void build_matrix(const Grid& grid, double rho, double sigma, double r_d, double kappa, double eta) {
        auto h_main = Kokkos::create_mirror_view(main_diag);
        auto h_lower = Kokkos::create_mirror_view(lower_diag);
        auto h_upper = Kokkos::create_mirror_view(upper_diag);
        auto h_upper2 = Kokkos::create_mirror_view(upper2_diag);
        
        Kokkos::deep_copy(h_main, 0.0);
        Kokkos::deep_copy(h_lower, 0.0);
        Kokkos::deep_copy(h_upper, 0.0);
        Kokkos::deep_copy(h_upper2, 0.0);

        int spacing = m1+1;

        // Handle j=0 case first
        for(int i = 0; i < m1+1; i++) {
            double temp = kappa * (eta - grid.Vec_v[0]);
            // Use l_9c = [0,1,2]
            h_main(i) += temp * gamma_v(0, 0, grid.Delta_v);
            h_upper(i) += temp * gamma_v(0, 1, grid.Delta_v);
            h_upper2(i) = temp * gamma_v(0, 2, grid.Delta_v);
        }

        // Handle remaining j values
        for(int j = 1; j < m2-1; j++) {
            for(int i = 0; i < m1+1; i++) {
                double temp = kappa * (eta - grid.Vec_v[j]);
                double temp2 = 0.5 * sigma * sigma * grid.Vec_v[j];

                if(grid.Vec_v[j] > 1.0) {
                    int main_idx = i + (j+1)*(m1+1);
                    // Using l_9a = [-2,-1,0]
                    if(j > 0) h_lower[main_idx-spacing] += temp * alpha_v(j, -2, grid.Delta_v);
                    h_lower[main_idx] += temp * alpha_v(j, -1, grid.Delta_v);
                    h_main[main_idx] += temp * alpha_v(j, 0, grid.Delta_v);

                    // Add regular central differences
                    for(int k = -1; k <= 1; k++) {
                        int idx = j > 0 ? main_idx + k*spacing : i + k*spacing;
                        if(k == -1) h_lower[idx] += temp2 * delta_v(j-1, k, grid.Delta_v);
                        else if(k == 0) h_main[main_idx] += temp2 * delta_v(j-1, k, grid.Delta_v);
                        else h_upper[main_idx] += temp2 * delta_v(j-1, k, grid.Delta_v);
                    }
                } else {
                    int main_idx = i + j*(m1+1);
                    for(int k = -1; k <= 1; k++) {
                        if(k == -1 && j > 0) h_lower[main_idx-spacing] += (temp * beta_v(j-1, k, grid.Delta_v) + 
                                                                temp2 * delta_v(j-1, k, grid.Delta_v));
                        else if(k == 0) h_main[main_idx] += (temp * beta_v(j-1, k, grid.Delta_v) + 
                                                        temp2 * delta_v(j-1, k, grid.Delta_v));
                        else h_upper[main_idx] += (temp * beta_v(j-1, k, grid.Delta_v) + 
                                                temp2 * delta_v(j-1, k, grid.Delta_v));
                    }
                }
                h_main[i + j*(m1+1)] += -0.5 * r_d;
            }
        }

        Kokkos::deep_copy(main_diag, h_main);
        Kokkos::deep_copy(lower_diag, h_lower);
        Kokkos::deep_copy(upper_diag, h_upper);
        Kokkos::deep_copy(upper2_diag, h_upper2);
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

};

void test_heston_A2() {
    {
        int m1 = 5;
        int m2 = 5;
        Grid grid = create_test_grid(m1, m2);

        //test_grid();
        
        heston_A2Storage_gpu A2(m1, m2);
        double rho = -0.9;
        double sigma = 0.3;
        double r_d = 0.025;
        double kappa = 1.5;
        double eta = 0.04;
        std::cout << "here" << std::endl;
        A2.build_matrix(grid, rho, sigma, r_d, kappa, eta);
        
        // Get host copies
        auto h_main = Kokkos::create_mirror_view(A2.main_diag);
        auto h_lower = Kokkos::create_mirror_view(A2.lower_diag);
        auto h_upper = Kokkos::create_mirror_view(A2.upper_diag);
        auto h_upper2 = Kokkos::create_mirror_view(A2.upper2_diag);
        
        Kokkos::deep_copy(h_main, A2.main_diag);
        Kokkos::deep_copy(h_lower, A2.lower_diag);
        Kokkos::deep_copy(h_upper, A2.upper_diag);
        Kokkos::deep_copy(h_upper2, A2.upper2_diag);

        std::cout << "A2 Matrix Structure:\n";
        std::cout << "First block (j=0) upper2 diagonal:\n";
        for(int i = 0; i < m1+1; i++) 
            std::cout << h_upper2[i] << " ";
        std::cout << "\n\n";

        for(int j = 0; j < m2-1; j++) {
            std::cout << "Block j=" << j << ":\n";
            if(j > 0) {
                std::cout << "Lower diagonal: ";
                for(int i = 0; i < m1+1; i++) 
                    std::cout << h_lower[i + j*(m1+1)] << " ";
                std::cout << "\n";
            }
            std::cout << "Main diagonal:  ";
            for(int i = 0; i < m1+1; i++) 
                std::cout << h_main[i + j*(m1+1)] << " ";
            std::cout << "\nUpper diagonal: ";
            for(int i = 0; i < m1+1; i++) 
                std::cout << h_upper[i + j*(m1+1)] << " ";
            std::cout << "\n\n";
        }
    }
}
*/
//A2 class test
/*
void test_heston_A2() {
    // Create scope for Kokkos objects
    int m1 = 5;
    int m2 = 5;
    Grid grid = create_test_grid(m1, m2);
    
    // Create and build A2 matrix
    heston_A2Storage_gpu A2(m1, m2);
    double rho = -0.9;
    double sigma = 0.3;
    double r_d = 0.025;
    double kappa = 1.5;
    double eta = 0.04;
    
    
    A2.build_matrix(grid, rho, sigma, r_d, kappa, eta);
    
    // Get Views using getters
    auto main = A2.get_main_diag();
    auto lower = A2.get_lower_diag();
    auto upper = A2.get_upper_diag();
    auto upper2 = A2.get_upper2_diag();

    // Create mirror views
    auto h_main = Kokkos::create_mirror_view(main);
    auto h_lower = Kokkos::create_mirror_view(lower);
    auto h_upper = Kokkos::create_mirror_view(upper);
    auto h_upper2 = Kokkos::create_mirror_view(upper2);
    
    // Copy to host
    Kokkos::deep_copy(h_main, main);
    Kokkos::deep_copy(h_lower, lower);
    Kokkos::deep_copy(h_upper, upper);
    Kokkos::deep_copy(h_upper2, upper2);
    
    // Print matrices
    std::cout << "A2 Matrix Structure:\n";
    std::cout << "--------------------\n";
    
    std::cout << "First block (j=0) upper2 diagonal:\n";
    for(int i = 0; i < m1+1; i++) 
        std::cout << h_upper2[i] << " ";
    std::cout << "\n\n";

    for(int j = 0; j < m2-1; j++) {
        std::cout << "Block j=" << j << ":\n";
        if(j > 0) {
            std::cout << "Lower diagonal: ";
            for(int i = 0; i < m1+1; i++) 
                std::cout << h_lower[i + j*(m1+1)] << " ";
            std::cout << "\n";
        }
        std::cout << "Main diagonal:  ";
        for(int i = 0; i < m1+1; i++) 
            std::cout << h_main[i + j*(m1+1)] << " ";
        std::cout << "\nUpper diagonal: ";
        for(int i = 0; i < m1+1; i++) 
            std::cout << h_upper[i + j*(m1+1)] << " ";
        std::cout << "\n\n";
    }

    // Print dimensions
    std::cout << "\nDimensions:\n";
    std::cout << "m1: " << A2.get_m1() << ", m2: " << A2.get_m2() << "\n";
    std::cout << "Values shape (main/upper): [" << (m2-1)*(m1+1) << "]\n";
    std::cout << "Values shape (lower): [" << (m2-2)*(m1+1) << "]\n";
    std::cout << "Values shape (upper2): [" << m1+1 << "]\n";
}
*/


/*

Here come the numerical tests for the A_i matrix classes. This is largely just a copy of the tests 
in the mat_fac file. We basically compare the residual between implicict and explicit steps

*/

void test_A0_multiply() {
    using timer = std::chrono::high_resolution_clock;
    
    // Test dimensions
    const int m1 = 5; 
    const int m2 = 20;
    std::cout << "Testing A0 multiply with dimensions m1=" << m1 << ", m2=" << m2 << "\n";

    // Create grid
    Grid grid = create_test_grid(m1, m2);

    // Initialize A0 matrix
    heston_A0Storage_gpu A0(m1, m2);
    double rho = -0.9;
    double sigma = 0.3;
    A0.build_matrix(grid, rho, sigma);
    
    // Total size
    const int total_size = (m1 + 1) * (m2 + 1);

    // Create test vectors
    Kokkos::View<double*> x("x", total_size);
    Kokkos::View<double*> result("result", total_size);
    
    // Initialize x with values 1,2,3,...
    Kokkos::parallel_for("init_x", total_size, KOKKOS_LAMBDA(const int idx) {
        x(idx) = static_cast<double>(idx + 1);
    });

    
    // Zero result vector
    Kokkos::deep_copy(result, 0.0);

    // Test multiply
    std::cout << "\nTesting multiply...\n";
    auto t_start = timer::now();
    A0.multiply_seq(x, result);
    auto t_end = timer::now();

    std::cout << "Multiply time: "
              << std::chrono::duration<double>(t_end - t_start).count()
              << " seconds" << std::endl;

    // Copy results back and check
    auto h_x = Kokkos::create_mirror_view(x);
    auto h_result = Kokkos::create_mirror_view(result);
    Kokkos::deep_copy(h_x, x);
    Kokkos::deep_copy(h_result, result);

    // Print first few results
    std::cout << "\nFirst 10 results:\n";
    for(int i = 0; i < std::min(60, total_size); i++) {
        std::cout << "result[" << i << "] = " << h_result(i) << " ";
    }
}

void test_A1_multiply_and_implicit() {
    using timer = std::chrono::high_resolution_clock;
    
    // Test dimensions
    const int m1 = 100;
    const int m2 = 75;
    std::cout << "Testing A1 with dimensions m1=" << m1 << ", m2=" << m2 << "\n";

    // Create grid
    Grid grid = create_test_grid(m1, m2);
    
    // Initialize A1
    heston_A1Storage_gpu A1(m1, m2);
    double rho = -0.9;
    double sigma = 0.3;
    double r_d = 0.025;
    double r_f = 0.0;

    double theta = 0.8;
    double delta_t = 1.0/40;

    A1.build_matrix(grid, rho, sigma, r_d, r_f);

    // After building A1
    auto h_main_check = Kokkos::create_mirror_view(A1.get_main_diags());
    auto h_lower_check = Kokkos::create_mirror_view(A1.get_lower_diags());
    auto h_upper_check = Kokkos::create_mirror_view(A1.get_upper_diags());

    Kokkos::deep_copy(h_main_check, A1.get_main_diags());
    Kokkos::deep_copy(h_lower_check, A1.get_lower_diags());
    Kokkos::deep_copy(h_upper_check, A1.get_upper_diags());

    const int total_size = (m1 + 1) * (m2 + 1);
    
    // Create test vectors
    Kokkos::View<double*> x("x", total_size);
    Kokkos::View<double*> b("b", total_size);
    Kokkos::View<double*> result("result", total_size);
    Kokkos::deep_copy(result, 0.0);  // Initialize result to zero
    
    // Initialize with random values
    auto h_b = Kokkos::create_mirror_view(b);
    auto h_x = Kokkos::create_mirror_view(x);
    for (int i = 0; i < total_size; ++i) {
        h_b(i) = total_size - i;//std::rand() / (RAND_MAX + 1.0);
        h_x(i) = 1.0 + i;//std::rand() / (RAND_MAX + 1.0);
    }
    Kokkos::deep_copy(b, h_b);
    Kokkos::deep_copy(x, h_x);

    // Build implicit matrix
    A1.build_implicit(theta, delta_t);

    
    // Test multiply
    auto t_start = timer::now();
    A1.multiply_parallel_s_and_v(x, result);
    auto t_end = timer::now();
    
    std::cout << "Multiply time: "
              << std::chrono::duration<double>(t_end - t_start).count()
              << " seconds" << std::endl;

    // Check multiply result
    /*
    auto h_result = Kokkos::create_mirror_view(result);
    Kokkos::deep_copy(h_result, result);
    std::cout << "After multiply, first few results: ";
    for(int i = 0; i < total_size; i++) std::cout << h_result(i) << " ";
    std::cout << "\n";
    */


    // Test implicit solve
    t_start = timer::now();
    A1.solve_implicit_parallel_v(x, b); //x values changed here
    t_end = timer::now();
    
    std::cout << "Implicit solve time: "
              << std::chrono::duration<double>(t_end - t_start).count()
              << " seconds" << std::endl;

    // Verify solution
    Kokkos::View<double*> verify("verify", total_size);
    A1.multiply(x, verify);
    
    // Compute residual
    auto h_verify = Kokkos::create_mirror_view(verify);
    Kokkos::deep_copy(h_verify, verify);
    
    double residual = 0.0;
    Kokkos::deep_copy(h_x, x);  // Make sure we have latest x values

    std::cout << "\nDebug values for first block:\n";
    for(int i = 0; i < m1+1; i++) {
        double res = h_x(i) - theta * delta_t * h_verify(i) - h_b(i);
        residual += res * res;
    }
    residual = std::sqrt(residual);
    
    std::cout << "Residual norm: " << residual << std::endl;
}

/*
void test_A2_multiply_and_implicit() {
    using timer = std::chrono::high_resolution_clock;

    // Test dimensions
    int m1 = 5;
    int m2 = 5;
    Grid grid = create_test_grid(m1, m2);

    // Create and build A2 matrix
    heston_A2Storage_gpu A2(m1, m2);
    double rho = -0.9;
    double sigma = 0.3;
    double r_d = 0.025;
    double kappa = 1.5;
    double eta = 0.04;

    std::cout << "here";
    
    A2.build_matrix(grid, rho, sigma, r_d, kappa, eta);
    std::cout<< "here";
    const int total_size = (m1 + 1) * (m2 + 1);

    // Create test vectors
    Kokkos::View<double*> x("x", total_size);
    Kokkos::View<double*> b("b", total_size);
    Kokkos::View<double*> result("result", total_size);

    // Initialize with random values
    auto h_b = Kokkos::create_mirror_view(b);
    auto h_x = Kokkos::create_mirror_view(x);
    for (int i = 0; i < total_size; ++i) {
        h_b(i) = std::rand() / (RAND_MAX + 1.0);
        h_x(i) = std::rand() / (RAND_MAX + 1.0);
    }
    Kokkos::deep_copy(b, h_b);
    Kokkos::deep_copy(x, h_x);

    // Build implicit matrix
    double theta = 0.8;
    double delta_t = 0.001;
    A2.build_implicit(theta, delta_t);

    // Test multiply
    auto t_start = timer::now();
    multiply(A2, x, result);
    auto t_end = timer::now();

    std::cout << "Multiply time: "
                << std::chrono::duration<double>(t_end - t_start).count()
                << " seconds" << std::endl;

    // Test implicit solve
    t_start = timer::now();
    solve_implicit(A2, x, b);
    t_end = timer::now();

    std::cout << "Implicit solve time: "
                << std::chrono::duration<double>(t_end - t_start).count()
                << " seconds" << std::endl;

    // Verify solution
    Kokkos::View<double*> verify("verify", total_size);
    multiply(A2, x, verify);

    // Compute residual
    auto h_verify = Kokkos::create_mirror_view(verify);
    Kokkos::deep_copy(h_verify, verify);


    double residual = 0.0;
    for(int i = 0; i < total_size; i++) {
        double res = h_x(i) - theta * delta_t * h_verify(i) - h_b(i);
        residual += res * res;
    }
    residual = std::sqrt(residual);

    std::cout << "Residual norm: " << residual << std::endl;
}
*/


/*

Here comes the test for the device A1 class. This is not workable as it stands right now, since the class is 
living on the host, eventho it has Views as memeber variables, when we call it form a device function we are still 
going through host memory. This results in a core dump error, i.e. accessing memory not allocated on the device.

One fix is to rwrite the class and making it "Easily copyable" or to write the entire class on the device (this is 
very very tricky)

*/
/*
void test_A1_device_in_one_kernel() {
    
    // Test dimensions
    const int m1 = 5;
    const int m2 = 5;
    
    // Create test grid
    Grid grid = create_test_grid(m1, m2);
    
    // Create A1 device matrix
    // 1) Create on host, as before
    heston_A1_device A1_host(m1, m2);

    

    /*
    // Get Views for verification
    auto main = A1_dev_ptr->get_main_diags();
    auto lower = A1_dev_ptr->get_lower_diags();
    auto upper = A1_dev_ptr->get_upper_diags();

    // Create host mirrors
    auto h_main = Kokkos::create_mirror_view(main);
    auto h_lower = Kokkos::create_mirror_view(lower);
    auto h_upper = Kokkos::create_mirror_view(upper);
    
    // Copy to host
    Kokkos::deep_copy(h_main, main);
    Kokkos::deep_copy(h_lower, lower);
    Kokkos::deep_copy(h_upper, upper);
    
    // Print matrices for verification
    std::cout << "A1 Device Matrix Structure:\n";
    std::cout << "-------------------------\n";
    
    for(int j = 0; j <= m2; j++) {
        std::cout << "\nVariance level j=" << j << ":\n";
        
        std::cout << "Lower diagonal: ";
        for(int i = 0; i < m1; i++) {
            std::cout << h_lower(j,i) << " ";
        }
        
        std::cout << "\nMain diagonal:  ";
        for(int i = 0; i <= m1; i++) {
            std::cout << h_main(j,i) << " ";
        }
        
        std::cout << "\nUpper diagonal: ";
        for(int i = 0; i < m1; i++) {
            std::cout << h_upper(j,i) << " ";
        }
        std::cout << "\n";
    }

    // Print dimensions
    std::cout << "\nDimensions:\n";
    std::cout << "m1: " << A1_dev_ptr->get_m1() << ", m2: " << A1_dev_ptr->get_m2() << "\n";
    
}
*/

// DeviceCallable.hpp
// Simple test to make the class device idea working. It isnt working
/*
#pragma once
#include <Kokkos_Core.hpp>

class DeviceCallable {
private:
    int size;
    Kokkos::View<double*> data;

public:
    // Default constructor - must be device callable
    KOKKOS_FUNCTION
    DeviceCallable() : size(0) {}

    // Constructor that can be called on device
    KOKKOS_FUNCTION
    DeviceCallable(int size_in, Kokkos::View<double*> data_in) 
        : size(size_in), data(data_in) {}

    // Device-callable build method
    KOKKOS_FUNCTION
    void build_matrix_device(const double factor1, const double factor2) {
        for(int i = 0; i < size; i++) {
            data(i) = factor1 * i + factor2;
        }
    }

    // Device-callable compute method
    KOKKOS_FUNCTION
    double compute_device(const int idx) const {
        if(idx < size) {
            return data(idx) * data(idx);
        }
        return 0.0;
    }

    // Getters
    KOKKOS_FUNCTION
    int get_size() const { return size; }
    
    KOKKOS_FUNCTION
    const Kokkos::View<double*>& get_data() const { return data; }
};

// Helper function to create device view of the class
inline Kokkos::View<DeviceCallable*> create_device_object(int size) {
    // Allocate data View that will be used by device object
    Kokkos::View<double*> data("object_data", size);
    
    // Create View to hold device object
    Kokkos::View<DeviceCallable*> d_object("device_object", 1);
    
    // Create host mirror
    auto h_object = Kokkos::create_mirror_view(d_object);
    
    // Initialize host object
    h_object(0) = DeviceCallable(size, data);
    
    // Copy to device
    Kokkos::deep_copy(d_object, h_object);
    
    return d_object;
}

// Test function
void test_device_callable() {
            const int size = 100;
            double factor1 = 2.0;
            double factor2 = 1.0;
            
            // Create device object
            auto d_object = create_device_object(size);
            
            // Call build_matrix_device on device
            Kokkos::parallel_for("build", 1, KOKKOS_LAMBDA(const int) {
                auto& obj = d_object(0);
                obj.build_matrix_device(factor1, factor2);
            });
            Kokkos::fence();
            
            // Create result View for compute test
            Kokkos::View<double*> results("results", size);
            
            // Call compute_device on device
            Kokkos::parallel_for("compute", size, KOKKOS_LAMBDA(const int i) {
                auto& obj = d_object(0);
                results(i) = obj.compute_device(i);
            });
            Kokkos::fence();
            
            // Verify results
            auto h_results = Kokkos::create_mirror_view(results);
            Kokkos::deep_copy(h_results, results);
            
            // Check a few values
            for(int i = 0; i < 5; i++) {
                double expected = (factor1 * i + factor2) * (factor1 * i + factor2);
                std::cout << "results[" << i << "] = " << h_results(i) 
                         << " (expected: " << expected << ")" << std::endl;
            }

}
*/


/*

The two tests below show, that when we change the data layout of A1 into three diagonals, we do 
not get any speedups compared to our initial 2D data layout. Both take the same amount of time 

*/

/*

This is the coalesc A1 test

*/
/*
// Basic structure test
void test_heston_A1_coalesc() {
    {   // Create scope for Kokkos objects
        int m1 = 5;
        int m2 = 5;
        Grid grid = create_test_grid(m1, m2);
        
        heston_A1Storage_coalesc A1(m1, m2);
        double rho = -0.9;
        double sigma = 0.3;
        double r_d = 0.025;
        double r_f = 0.0;
        
        A1.build_matrix(grid, rho, sigma, r_d, r_f);
        
        // Get Views using getters
        auto main = A1.get_main_diags();
        auto lower = A1.get_lower_diags();
        auto upper = A1.get_upper_diags();

        // Create mirror views
        auto main_host = Kokkos::create_mirror_view(main);
        auto lower_host = Kokkos::create_mirror_view(lower);
        auto upper_host = Kokkos::create_mirror_view(upper);
        
        // Copy to host
        Kokkos::deep_copy(main_host, main);
        Kokkos::deep_copy(lower_host, lower);
        Kokkos::deep_copy(upper_host, upper);
        
        // Print matrices
        std::cout << "A1 Matrix Structure:\n";
        std::cout << "--------------------\n";
        
        for(int j = 0; j <= m2; j++) {
            std::cout << "\nVariance level j=" << j << ":\n";
            
            std::cout << "Lower diagonal: ";
            for(int i = 0; i < m1; i++) {
                std::cout << lower_host(j*m1 + i) << " ";
            }
            
            std::cout << "\nMain diagonal:  ";
            for(int i = 0; i <= m1; i++) {
                std::cout << main_host(j*(m1+1) + i) << " ";
            }
            
            std::cout << "\nUpper diagonal: ";
            for(int i = 0; i < m1; i++) {
                std::cout << upper_host(j*m1 + i) << " ";
            }
            std::cout << "\n";
        }

        // Print dimensions
        std::cout << "\nDimensions:\n";
        std::cout << "m1: " << A1.get_m1() << ", m2: " << A1.get_m2() << "\n";
    }
}

// Performance test
void test_A1_multiply_and_implicit_coalesc() {
    using timer = std::chrono::high_resolution_clock;
    
    const int m1 = 300;
    const int m2 = 100;
    std::cout << "Testing A1 with dimensions m1=" << m1 << ", m2=" << m2 << "\n";

    Grid grid = create_test_grid(m1, m2);
    
    // Initialize A1
    heston_A1Storage_coalesc A1(m1, m2);
    double rho = -0.9;
    double sigma = 0.3;
    double r_d = 0.025;
    double r_f = 0.0;
    A1.build_matrix(grid, rho, sigma, r_d, r_f);

    // Check matrices after building
    auto h_main_check = Kokkos::create_mirror_view(A1.get_main_diags());
    auto h_lower_check = Kokkos::create_mirror_view(A1.get_lower_diags());
    auto h_upper_check = Kokkos::create_mirror_view(A1.get_upper_diags());

    Kokkos::deep_copy(h_main_check, A1.get_main_diags());
    Kokkos::deep_copy(h_lower_check, A1.get_lower_diags());
    Kokkos::deep_copy(h_upper_check, A1.get_upper_diags());

    const int total_size = (m1 + 1) * (m2 + 1);
    
    // Create test vectors
    Kokkos::View<double*> x("x", total_size);
    Kokkos::View<double*> b("b", total_size);
    Kokkos::View<double*> result("result", total_size);
    Kokkos::deep_copy(result, 0.0);
    
    // Initialize with random values
    auto h_b = Kokkos::create_mirror_view(b);
    auto h_x = Kokkos::create_mirror_view(x);
    for (int i = 0; i < total_size; ++i) {
        h_b(i) = std::rand() / (RAND_MAX + 1.0);
        h_x(i) = std::rand() / (RAND_MAX + 1.0);
    }
    Kokkos::deep_copy(b, h_b);
    Kokkos::deep_copy(x, h_x);

    // Build implicit matrix
    double theta = 0.8;
    double delta_t = 1.0/14;
    A1.build_implicit(theta, delta_t);

    /*
    // Test sequential multiply
    auto t_start = timer::now();
    A1.multiply_seq(x, result);
    auto t_end = timer::now();
    std::cout << "Sequential multiply time: "
              << std::chrono::duration<double>(t_end - t_start).count()
              << " seconds" << std::endl;
    
    
    // Test parallel multiply
    t_start = timer::now();
    A1.multiply_parallel(x, result);
    auto t_end_parallel = timer::now();
    std::cout << "Parallel multiply time: "
              << std::chrono::duration<double>(t_end_parallel - t_start).count()
              << " seconds" << std::endl;
    
    // Test implicit solve
    for(int i=0; i<5; i++){
        auto t_start = timer::now();
        A1.solve_implicit_parallel_v(x, b);
        auto t_end = timer::now();
        std::cout << "Implicit solve time: "
                << std::chrono::duration<double>(t_end - t_start).count()
                << " seconds" << std::endl;
    }
    
    /*
    // Verify solution
    Kokkos::View<double*> verify("verify", total_size);
    A1.multiply_parallel(x, verify);
    
    // Compute residual for first block
    auto h_verify = Kokkos::create_mirror_view(verify);
    Kokkos::deep_copy(h_verify, verify);
    Kokkos::deep_copy(h_x, x);

    double residual = 0.0;
    std::cout << "\nDebug values for first block:\n";
    for(int i = 0; i < m1+1; i++) {
        double res = h_x(i) - theta * delta_t * h_verify(i) - h_b(i);
        residual += res * res;
    }
    residual = std::sqrt(residual);
    
    std::cout << "Residual norm: " << residual << std::endl;
    
}
*/

/*

This is the basic A1 test with just three diagonasl

*/

// Basic structure test and performance test for flat storage A1 implementation
/*
void test_heston_A1_flat() {
    {   // Create scope for Kokkos objects
        int m1 = 5;
        int m2 = 5;
        Grid grid = create_test_grid(m1, m2);
        
        heston_A1_flat A1(m1, m2);
        double rho = -0.9;
        double sigma = 0.3;
        double r_d = 0.025;
        double r_f = 0.0;
        
        A1.build_matrix(grid, rho, sigma, r_d, r_f);
        
        // Get Views and create host mirrors
        auto main = A1.get_main_diag();
        auto lower = A1.get_lower_diag();
        auto upper = A1.get_upper_diag();

        auto main_host = Kokkos::create_mirror_view(main);
        auto lower_host = Kokkos::create_mirror_view(lower);
        auto upper_host = Kokkos::create_mirror_view(upper);
        
        // Copy to host
        Kokkos::deep_copy(main_host, main);
        Kokkos::deep_copy(lower_host, lower);
        Kokkos::deep_copy(upper_host, upper);
        
        // Print all diagonals
        std::cout << "A1 Matrix Diagonals:\n";
        std::cout << "-------------------\n\n";

        const int block_size = m1 + 1;
        
        std::cout << "Lower diagonal values [" << lower_host.extent(0) << " entries]:\n";
        for(int i = 0; i < lower_host.extent(0); i++) {
            if(i > 0 && i % block_size == 0) std::cout << "| "; // Block boundary marker
            std::cout << lower_host(i) << " ";
            if((i + 1) % 5 == 0) std::cout << "\n";
        }
        std::cout << "\n\n";

        std::cout << "Main diagonal values [" << main_host.extent(0) << " entries]:\n";
        for(int i = 0; i < main_host.extent(0); i++) {
            if(i > 0 && i % block_size == 0) std::cout << "| "; // Block boundary marker
            std::cout <<  main_host(i) << " ";
            if((i + 1) % 5 == 0) std::cout << "\n";
        }
        std::cout << "\n\n";

        std::cout << "Upper diagonal values [" << upper_host.extent(0) << " entries]:\n";
        for(int i = 0; i < upper_host.extent(0); i++) {
            if(i > 0 && i % block_size == 0) std::cout << "| "; // Block boundary marker
            std::cout <<  upper_host(i) << " ";
            if((i + 1) % 5 == 0) std::cout << "\n";
        }
        std::cout << "\n\n";

        // Print dimensions
        std::cout << "\nDimensions:\n";
        std::cout << "m1: " << A1.get_m1() << ", m2: " << A1.get_m2() << "\n";
        std::cout << "Block size: " << block_size << "\n";
        std::cout << "Total size: " << A1.get_total_size() << "\n";
        std::cout << "Number of blocks: " << m2 + 1 << "\n";
    }
}

void test_A1_flat_performance() {
    using timer = std::chrono::high_resolution_clock;
    
    // Test with larger dimensions
    const int m1 = 300;
    const int m2 = 100;
    std::cout << "Testing flat A1 with dimensions m1=" << m1 << ", m2=" << m2 << "\n";

    Grid grid = create_test_grid(m1, m2);
    
    // Initialize A1
    heston_A1_flat A1(m1, m2);
    double rho = -0.9;
    double sigma = 0.3;
    double r_d = 0.025;
    double r_f = 0.0;
    
    A1.build_matrix(grid, rho, sigma, r_d, r_f);
   
    const int total_size = (m1 + 1) * (m2 + 1);
    
    // Create test vectors
    Kokkos::View<double*> x("x", total_size);
    Kokkos::View<double*> b("b", total_size);
    Kokkos::View<double*> result("result", total_size);
    
    // Initialize with random values
    auto h_b = Kokkos::create_mirror_view(b);
    auto h_x = Kokkos::create_mirror_view(x);
    for (int i = 0; i < total_size; ++i) {
        h_b(i) = std::rand() / (RAND_MAX + 1.0);
        h_x(i) = std::rand() / (RAND_MAX + 1.0);
    }
    Kokkos::deep_copy(b, h_b);
    Kokkos::deep_copy(x, h_x);

    // Build implicit matrix
    double theta = 1.0;
    double delta_t = 1.0;
    
    
    A1.build_implicit(theta, delta_t);
    
    // Test multiply
    //auto t_start = timer::now();
    A1.multiply(x, result);
    //auto t_end = timer::now();
    //std::cout << "Matrix multiply time: "
              //<< std::chrono::duration<double>(t_end - t_start).count()
              //<< " seconds\n";
    
    // Test implicit solve
    for(int i = 0; i < 5; i++) {
        auto t_start = timer::now();
        A1.solve_implicit(x, b);
        auto t_end = timer::now();
        std::cout << "Implicit solve time: "
                << std::chrono::duration<double>(t_end - t_start).count()
                << " seconds" << std::endl;
    }
    
    
    // Verify solution
    Kokkos::View<double*> verify("verify", total_size);
    A1.multiply(x, verify);
    
    auto h_verify = Kokkos::create_mirror_view(verify);
    Kokkos::deep_copy(h_verify, verify);
    Kokkos::deep_copy(h_x, x);

    // Compute residual
    double residual = 0.0;
    for(int i = 0; i < total_size; i++) {
        double res = h_x(i) - theta * delta_t * h_verify(i) - h_b(i);
        residual += res * res;
    }
    residual = std::sqrt(residual);
    
    std::cout << "Total residual norm: " << residual << "\n";
}
*/

void test_hes_mat_fac() {
    // Initialize Kokkos
    Kokkos::initialize();
    {
        try {
            std::cout << "Default execution space: " << Kokkos::DefaultExecutionSpace::name() << std::endl;

            //test_heston_A0();
            //test_heston_A1();
            //test_heston_A2();

            //test_A1_structure();
            
            //test_A0_multiply();
            //test_parallel_tridiagonal();
            test_A1_multiply_and_implicit();
            //test_A2_multiply_and_implicit();

            //test_A1_device_in_one_kernel();
            //test_device_callable();

            //test_heston_A1_coalesc();
            //test_A1_multiply_and_implicit_coalesc();

            //test_heston_A1_flat();
            //test_A1_flat_performance();
        }
        catch (std::exception& e) {
            std::cout << "Error: " << e.what() << std::endl;
        }
    } // All test objects destroyed here
    Kokkos::finalize();
}