#pragma once

#include <Kokkos_Core.hpp>
#include "grid.hpp"
#include "coeff.hpp"

/*

A2 class

*/
class heston_A2Storage_gpu {
private:
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

    // Persistent temporary storage for implicit solve
    Kokkos::View<double*> c_star;   
    Kokkos::View<double*> c2_star;  
    Kokkos::View<double*> d_star;

public:
    KOKKOS_FUNCTION
    heston_A2Storage_gpu() = default;

    heston_A2Storage_gpu(int m1_in, int m2_in);

    void build_matrix(const Grid& grid, double rho, double sigma, double r_d, double kappa, double eta);

    void build_implicit(const double theta, const double delta_t);

    // Multiply function - kept inline for performance
    inline void multiply(const Kokkos::View<double*>& x, Kokkos::View<double*>& result);

    // Multiply parallel in v function - kept inline for performance
    inline void multiply_parallel_v(const Kokkos::View<double*>& x, Kokkos::View<double*>& result);

    // Multiply function parallel in stock and variance - kept inline for performance
    inline void multiply_parallel_s_and_v(const Kokkos::View<double*>& x, Kokkos::View<double*>& result);


    // Solve implicit function - kept inline for performance
    inline void solve_implicit(Kokkos::View<double*>& x, const Kokkos::View<double*>& b);

    // Getters
    KOKKOS_INLINE_FUNCTION int get_m1() const { return m1; }
    KOKKOS_INLINE_FUNCTION int get_m2() const { return m2; }
    KOKKOS_INLINE_FUNCTION const Kokkos::View<double*>& get_main_diag() const { return main_diag; }
    KOKKOS_INLINE_FUNCTION const Kokkos::View<double*>& get_lower_diag() const { return lower_diag; }
    KOKKOS_INLINE_FUNCTION const Kokkos::View<double*>& get_upper_diag() const { return upper_diag; }
    KOKKOS_INLINE_FUNCTION const Kokkos::View<double*>& get_upper2_diag() const { return upper2_diag; }

    //for debugging:
    KOKKOS_INLINE_FUNCTION const Kokkos::View<double*>& get_implicit_main_diag() const { return implicit_main_diag; }
    KOKKOS_INLINE_FUNCTION const Kokkos::View<double*>& get_implicit_lower_diag() const { return implicit_lower_diag; }
    KOKKOS_INLINE_FUNCTION const Kokkos::View<double*>& get_implicit_upper_diag() const { return implicit_upper_diag; }
    KOKKOS_INLINE_FUNCTION const Kokkos::View<double*>& get_implicit_upper2_diag() const { return implicit_upper2_diag; }
};

//sequential single threaded
inline void heston_A2Storage_gpu::multiply(const Kokkos::View<double*>& x, Kokkos::View<double*>& result) {
    const int local_m1 = m1;
    const int local_m2 = m2;

    const auto local_main = main_diag;
    const auto local_lower = lower_diag;
    const auto local_upper = upper_diag;
    const auto local_upper2 = upper2_diag;

    const int spacing = m1 + 1;
    //Kokkos::deep_copy(result, 0.0);
    
    
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

        //Handle remaining zero entries
        for(int i = (local_m2-1)*(local_m1+1); i < (local_m1+1)*(local_m2+1); i++) {
            result(i) = 0;
        }
        
    });
    Kokkos::fence();
}

//parallel in v
inline void heston_A2Storage_gpu::multiply_parallel_v(const Kokkos::View<double*>& x, Kokkos::View<double*>& result) {
    const int local_m1 = m1;
    const int local_m2 = m2;

    const auto local_main = main_diag;
    const auto local_lower = lower_diag;
    const auto local_upper = upper_diag;
    const auto local_upper2 = upper2_diag;

    const int spacing = local_m1 + 1;
    
    // First set result to zero
    Kokkos::deep_copy(result, 0.0);
    
    Kokkos::parallel_for("multiply", Kokkos::RangePolicy<>(0, local_m2 - 1), KOKKOS_LAMBDA(const int j) {
        if (j == 0) {
            // First block (j=0)
            for (int i = 0; i < spacing; i++) {
                double temp = local_main(i) * x(i);
                temp += local_upper(i) * x(i + spacing);
                temp += local_upper2(i) * x(i + 2 * spacing);
                result(i) = temp;
            }
        } 
        else {
            // Handle remaining blocks
            int block_start = j * spacing;
            for (int i = 0; i < spacing; i++) {
                int idx = block_start + i;
                double temp = local_lower(idx - spacing) * x(idx - spacing);
                temp += local_main(idx) * x(idx);
                temp += local_upper(idx) * x(idx + spacing);
                result(idx) = temp;
            }
        }
    });
    Kokkos::fence();
}

//parallel in s and v
inline void heston_A2Storage_gpu::multiply_parallel_s_and_v(const Kokkos::View<double*>& x, Kokkos::View<double*>& result) {
    const int local_m1 = m1;
    const int local_m2 = m2;
    const auto local_main = main_diag;
    const auto local_lower = lower_diag;
    const auto local_upper = upper_diag;
    const auto local_upper2 = upper2_diag;
    const int spacing = local_m1 + 1;

    // Define team policy: one team per variance level j
    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;

    //Kokkos::deep_copy(result, 0.0);

    Kokkos::parallel_for("A2_multiply",
        team_policy(local_m2 - 1, Kokkos::AUTO),  // One team per variance block
        KOKKOS_LAMBDA(const member_type& team_member) {
            const int j = team_member.league_rank();
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, spacing), [=](const int i) {
                if (j == 0) {
                    // First variance block
                    double temp = local_main(i)*x(i)
                                + local_upper(i)*x(i+spacing)
                                + local_upper2(i)*x(i+2*spacing);
                    result(i) = temp;
                } else {
                    // Subsequent variance blocks
                    int idx = j*spacing + i;
                    double temp = local_lower(idx-spacing)*x(idx-spacing)
                                + local_main(idx)*x(idx)
                                + local_upper(idx)*x(idx+spacing);
                    result(idx) = temp;
                }
            });
        }
    );

    // Now zero out the remaining elements
    int start_zero = (local_m2 - 1)*spacing;
    int end_zero = (local_m2 + 1)*spacing; // total size = (m1+1)*(m2+1)

    Kokkos::parallel_for("A2_zero_tail", Kokkos::RangePolicy<>(start_zero, end_zero), KOKKOS_LAMBDA(const int i) {
        result(i) = 0.0;
    });

    Kokkos::fence();
}

inline void heston_A2Storage_gpu::solve_implicit(Kokkos::View<double*>& x, const Kokkos::View<double*>& b) {
    const int local_m1 = m1;
    const int local_m2 = m2;
    const int spacing = local_m1 + 1;
    const int num_rows = (local_m2 - 1) * spacing;
    const int total_size = (local_m2 + 1) * spacing;


    auto local_c_star = c_star;
    auto local_c2_star = c2_star;
    auto local_d_star = d_star;

    // Get diagonal arrays
    const auto local_impl_main = implicit_main_diag;
    const auto local_impl_lower = implicit_lower_diag;
    const auto local_impl_upper = implicit_upper_diag;
    const auto local_impl_upper2 = implicit_upper2_diag;

    //Debugging the DO scheme solver. Before we compute anything we set the results to zero
    //Kokkos::deep_copy(x, 0.0);

    Kokkos::parallel_for("solve_implicit", 1, KOKKOS_LAMBDA(const int) {
        // Identity block
        for (int i = num_rows; i < total_size; i++) {
            local_d_star(i) = b(i);
        }

        
        // Normalize first m1+1 rows and upper2_diagonal
        for (int i = 0; i < spacing; i++) {
            local_c_star(i) = local_impl_upper(i) / local_impl_main(i);
            local_c2_star(i) = local_impl_upper2(i) / local_impl_main(i);
            local_d_star(i) = b(i) / local_impl_main(i);
        }

        // First block forward sweep (handle upper2_diag)
        for (int i = 0; i < spacing; i++) {
            double c_upper = local_impl_upper(i + spacing) - local_c2_star(i) * local_impl_lower(i);
            double m = 1.0 / (local_impl_main(i + spacing) - local_c_star(i) * local_impl_lower(i));
            local_c_star(i + spacing) = c_upper * m;
            local_d_star(i + spacing) = (b(i + spacing) - local_impl_lower(i) * local_d_star(i)) * m;
        }

        // Middle blocks forward sweep
        for (int i = spacing; i < num_rows - spacing; i++) {
            double m = 1.0 / (local_impl_main(i + spacing) - local_c_star(i) * local_impl_lower(i));
            local_c_star(i + spacing) = local_impl_upper(i + spacing) * m;
            local_d_star(i + spacing) = (b(i + spacing) - local_impl_lower(i) * local_d_star(i)) * m;
        }

        // Pre-backward sweep
        for (int i = num_rows - spacing; i < num_rows; i++) {
            local_d_star(i) -= local_d_star(i + spacing) * local_c_star(i);
        }

        // Last m1+1 rows
        for (int i = num_rows - spacing; i < num_rows; i++) {
            x(i) = local_d_star(i);
        }

        // Backward sweep
        for (int i = num_rows - 1; i >= 3 * spacing; i--) {
            x(i - spacing) = local_d_star(i - spacing) - local_c_star(i - spacing) * x(i);
        }

        // First block back substitution with upper2_diag
        for (int i = 3 * spacing - 1; i >= 2 * spacing; i--) {
            x(i - spacing) = local_d_star(i - spacing) - local_c_star(i - spacing) * x(i);
            local_d_star(i - 2 * spacing) -= local_c2_star(i - 2 * spacing) * x(i);
        }

        // Last backward substitution
        for (int i = 2 * spacing - 1; i >= spacing; i--) {
            x(i - spacing) = local_d_star(i - spacing) - local_c_star(i - spacing) * x(i);
        }

        // Identity block
        for (int i = num_rows; i < total_size; i++) {
            x(i) = local_d_star(i);
        }
    });
    Kokkos::fence();
}





/*

A2 shuffled class

*/

//this class comes from the FD when we shuffle the solution vector not wrt to the variance direction
//but the stokc direction. Resulting in a similar structure as the A1 class

class heston_A2_shuffled {
private:
    int m1, m2;  // Grid dimensions
    
    // Explicit system diagonals
    Kokkos::View<double**> main_diags;     // [m1+1][m2+1]
    Kokkos::View<double**> lower_diags;    // [m1+1][m2]
    Kokkos::View<double**> lower2_diags;   // [m1+1][m2-1]
    Kokkos::View<double**> upper_diags;    // [m1+1][m2]
    Kokkos::View<double**> upper2_diags;   // [m1+1][m2-1]
    
    // Implicit system diagonals
    Kokkos::View<double**> implicit_main_diags;
    Kokkos::View<double**> implicit_lower_diags;
    Kokkos::View<double**> implicit_lower2_diags;
    Kokkos::View<double**> implicit_upper_diags;
    Kokkos::View<double**> implicit_upper2_diags;
    
    // Temporary storage for implicit solve
    Kokkos::View<double**> c_prime;      // [m1+1][m2+1]
    Kokkos::View<double**> c2_prime;     // [m1+1][m2+1]
    Kokkos::View<double**> d_prime;      // [m1+1][m2+1]

public:
    // Default constructor needed for some Kokkos operations
    KOKKOS_FUNCTION
    heston_A2_shuffled() = default;
    
    // Main constructor
    heston_A2_shuffled(int m1_in, int m2_in);
    
    // Build matrix and implicit system
    void build_matrix(const Grid& grid, double rho, double sigma, double r_d, double kappa, double eta);
    void build_implicit(const double theta, const double delta_t);
    
    // Matrix operations
    inline void multiply(const Kokkos::View<double*>& x, Kokkos::View<double*>& result);
    inline void solve_implicit(Kokkos::View<double*>& x, const Kokkos::View<double*>& b);
    
    // Getters
    KOKKOS_INLINE_FUNCTION int get_m1() const { return m1; }
    KOKKOS_INLINE_FUNCTION int get_m2() const { return m2; }
    KOKKOS_INLINE_FUNCTION const Kokkos::View<double**>& get_main_diags() const { return main_diags; }
    KOKKOS_INLINE_FUNCTION const Kokkos::View<double**>& get_lower_diags() const { return lower_diags; }
    KOKKOS_INLINE_FUNCTION const Kokkos::View<double**>& get_lower2_diags() const { return lower2_diags; }
    KOKKOS_INLINE_FUNCTION const Kokkos::View<double**>& get_upper_diags() const { return upper_diags; }
    KOKKOS_INLINE_FUNCTION const Kokkos::View<double**>& get_upper2_diags() const { return upper2_diags; }

    //for debugging
    KOKKOS_INLINE_FUNCTION const Kokkos::View<double**>& get_implicit_main_diags() const { return implicit_main_diags; }
    KOKKOS_INLINE_FUNCTION const Kokkos::View<double**>& get_implicit_lower_diags() const { return implicit_lower_diags; }
    KOKKOS_INLINE_FUNCTION const Kokkos::View<double**>& get_implicit_lower2_diags() const { return implicit_lower2_diags; }
    KOKKOS_INLINE_FUNCTION const Kokkos::View<double**>& get_implicit_upper_diags() const { return implicit_upper_diags; }
    KOKKOS_INLINE_FUNCTION const Kokkos::View<double**>& get_implicit_upper2_diags() const { return implicit_upper2_diags; }
};

// Multiply implementation parallel in s
inline void heston_A2_shuffled::multiply(const Kokkos::View<double*>& x, 
                                       Kokkos::View<double*>& result) {
    const int local_m1 = m1;
    const int local_m2 = m2;
    const auto local_main = main_diags;
    const auto local_lower = lower_diags;
    const auto local_lower2 = lower2_diags;
    const auto local_upper = upper_diags;
    const auto local_upper2 = upper2_diags;
    
    // Parallel over stock price blocks
    Kokkos::parallel_for("A2_multiply", local_m1 + 1, KOKKOS_LAMBDA(const int i) {
        const int block_offset = i * (local_m2 + 1);
        
        // Handle first row
        result(block_offset) = local_main(i, 0) * x(block_offset);
        if(0 < local_m2) {
            result(block_offset) += local_upper(i, 0) * x(block_offset + 1);
        }
        if(1 < local_m2) {
            result(block_offset) += local_upper2(i, 0) * x(block_offset + 2);
        }
        
        // Handle second row
        if(0 < local_m2) {
            result(block_offset + 1) = local_lower(i, 0) * x(block_offset) +
                                     local_main(i, 1) * x(block_offset + 1);
            if(1 < local_m2) {
                result(block_offset + 1) += local_upper(i, 1) * x(block_offset + 2);
            }
            if(2 < local_m2) {
                result(block_offset + 1) += local_upper2(i, 1) * x(block_offset + 3);
            }
        }
        
        // Handle middle rows
        for(int j = 2; j < local_m2 - 1; j++) {
            result(block_offset + j) = local_lower2(i, j-2) * x(block_offset + j-2) +
                                     local_lower(i, j-1) * x(block_offset + j-1) +
                                     local_main(i, j) * x(block_offset + j) +
                                     local_upper(i, j) * x(block_offset + j+1);
            if(j < local_m2 - 2) {
                result(block_offset + j) += local_upper2(i, j) * x(block_offset + j+2);
            }
        }
        
        // Handle second-to-last row
        if(local_m2 > 2) {
            const int j = local_m2 - 1;
            result(block_offset + j) = local_lower2(i, j-2) * x(block_offset + j-2) +
                                     local_lower(i, j-1) * x(block_offset + j-1) +
                                     local_main(i, j) * x(block_offset + j);
            if(j < local_m2) {
                result(block_offset + j) += local_upper(i, j) * x(block_offset + j+1);
            }
        }
        
        // Handle last row
        if(local_m2 > 1) {
            const int j = local_m2;
            result(block_offset + j) = local_lower2(i, j-2) * x(block_offset + j-2) +
                                     local_lower(i, j-1) * x(block_offset + j-1) +
                                     local_main(i, j) * x(block_offset + j);
        }
    });
    Kokkos::fence();
}

//implicit impleemntation parallel in s
inline void heston_A2_shuffled::solve_implicit(Kokkos::View<double*>& x, 
                                             const Kokkos::View<double*>& b) {
    const int local_m1 = m1;
    const int local_m2 = m2;
    
    // Get local references to Views
    const auto local_main = implicit_main_diags;
    const auto local_lower = implicit_lower_diags;
    const auto local_lower2 = implicit_lower2_diags;
    const auto local_upper = implicit_upper_diags;
    const auto local_upper2 = implicit_upper2_diags;
    
    // Get temporary storage
    const auto local_c = c_prime;
    const auto local_c2 = c2_prime;
    const auto local_d = d_prime;

    // Parallelize over stock price blocks
    Kokkos::parallel_for("A2_implicit_solve", local_m1 + 1, KOKKOS_LAMBDA(const int i) {
        const int block_offset = i * (local_m2 + 1);
        const int block_size = local_m2 + 1;
        
        // Forward sweep
        // First row
        local_c(i, 0) = local_upper(i, 0) / local_main(i, 0);
        local_c2(i, 0) = local_upper2(i, 0) / local_main(i, 0);
        local_d(i, 0) = b(block_offset) / local_main(i, 0);
        
        // Second row
        if(block_size > 1) {
            double m1 = 1.0 / (local_main(i, 1) - local_lower(i, 0) * local_c(i, 0));
            local_c(i, 1) = (local_upper(i, 1) - local_lower(i, 0) * local_c2(i, 0)) * m1;
            local_c2(i, 1) = local_upper2(i, 1) * m1;
            local_d(i, 1) = (b(block_offset + 1) - local_lower(i, 0) * local_d(i, 0)) * m1;
        }
        
        // Main forward sweep
        for(int j = 2; j < block_size; j++) {
            double den = local_main(i, j) - 
                        (local_lower(i, j-1) - local_lower2(i, j-2) * local_c(i, j-2)) * local_c(i, j-1) - 
                        local_lower2(i, j-2) * local_c2(i, j-2);
            double m = 1.0 / den;
            
            // Update c coefficients
            local_c(i, j) = (local_upper(i, j) - 
                            (local_lower(i, j-1) - local_lower2(i, j-2) * local_c(i, j-2)) * local_c2(i, j-1)) * m;
            if(j < block_size - 2) {
                local_c2(i, j) = local_upper2(i, j) * m;
            }
            
            // Update d
            local_d(i, j) = (b(block_offset + j) - 
                            (local_lower(i, j-1) - local_lower2(i, j-2) * local_c(i, j-2)) * local_d(i, j-1) - 
                            local_lower2(i, j-2) * local_d(i, j-2)) * m;
        }
        
        // Back substitution
        x(block_offset + block_size - 1) = local_d(i, block_size - 1);
        
        if(block_size > 1) {
            x(block_offset + block_size - 2) = local_d(i, block_size - 2) - 
                                              local_c(i, block_size - 2) * x(block_offset + block_size - 1);
        }
        
        for(int j = block_size - 3; j >= 0; j--) {
            x(block_offset + j) = local_d(i, j) - 
                                 local_c(i, j) * x(block_offset + j + 1) - 
                                 local_c2(i, j) * x(block_offset + j + 2);
        }
    });
    Kokkos::fence();
}

/*
inline void shuffle_vector(const Kokkos::View<double*>& input, 
                         Kokkos::View<double*>& output,
                         const int m1, 
                         const int m2);

inline void unshuffle_vector(const Kokkos::View<double*>& input, 
                              Kokkos::View<double*>& output,
                              const int m1, 
                              const int m2);
*/

// Helper functions for shuffling
inline void shuffle_vector(const Kokkos::View<double*>& input, 
    Kokkos::View<double*>& output,
    const int m1, 
    const int m2) {
    // Shuffles from [v0(s0,s1,...), v1(s0,s1,...), ...] 
    // to [s0(v0,v1,...), s1(v0,v1,...), ...]
    Kokkos::parallel_for("shuffle", m1 + 1, KOKKOS_LAMBDA(const int i) {
        for(int j = 0; j <= m2; j++) {
        // From original idx = j*(m1+1) + i 
        // To shuffled idx = i*(m2+1) + j
        output(i*(m2+1) + j) = input(j*(m1+1) + i);
        }
    });
    Kokkos::fence();
}

inline void unshuffle_vector(const Kokkos::View<double*>& input, 
         Kokkos::View<double*>& output,
         const int m1, 
         const int m2) {
    // Shuffles from [s0(v0,v1,...), s1(v0,v1,...), ...] 
    // back to [v0(s0,s1,...), v1(s0,s1,...), ...]
    Kokkos::parallel_for("shuffle_back", m1 + 1, KOKKOS_LAMBDA(const int i) {
        for(int j = 0; j <= m2; j++) {
        // From shuffled idx = i*(m2+1) + j
        // To original idx = j*(m1+1) + i
        output(j*(m1+1) + i) = input(i*(m2+1) + j);
        }
    });
    Kokkos::fence();
}

void test_heston_A2_mat();