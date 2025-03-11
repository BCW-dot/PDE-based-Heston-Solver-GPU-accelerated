#pragma once

#include <Kokkos_Core.hpp>
#include "grid.hpp"
#include "coeff.hpp"

/*

A0 class

*/
class heston_A0Storage_gpu {
private:
    int m1, m2;
    Kokkos::View<double**> values; // [m2 - 1][(m1 - 1) * 9]

public:
    // Constructor - marked as host function since it allocates memory
    KOKKOS_FUNCTION
    heston_A0Storage_gpu() = default;  // Default constructor needed for some Kokkos operations

    // Constructor that allocates memory - must run on host
    heston_A0Storage_gpu(int m1_in, int m2_in);

    // Method to build the matrix - runs on host
    void build_matrix(const Grid& grid, double rho, double sigma);

    // Method to multiply - runs on device - is called from host
    void multiply(const Kokkos::View<double*>& x, Kokkos::View<double*>& result) const;

    // Method to multiply, parallised over variance and stock - runs on device - is called from host
    void multiply_parallel_s_and_v(const Kokkos::View<double*>& x, Kokkos::View<double*>& result) const;

    // Getters
    KOKKOS_INLINE_FUNCTION
    int get_m1() const { return m1; }
    
    KOKKOS_INLINE_FUNCTION
    int get_m2() const { return m2; }
    
    KOKKOS_INLINE_FUNCTION
    const Kokkos::View<double**>& get_values() const { return values; }
};

//parallisation over the variance direction
inline void heston_A0Storage_gpu::multiply(const Kokkos::View<double*>& x, 
                                             Kokkos::View<double*>& result) const {
    int total_size = x.size();
    int m1_ = m1;
    int m2_ = m2;
    auto values_ = values;
    auto x_ = x;

    // Zero out result vector in parallel before computation
    Kokkos::parallel_for("A0_zero_result", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, 1), KOKKOS_LAMBDA(const int i) {
        result(i) = 0.0;
    });
    Kokkos::fence();

    // Parallelize over variance direction j
    Kokkos::parallel_for("A0_multiply_parallel_v", m2_-1, KOKKOS_LAMBDA(const int j) {
        int row_offset = (j + 1) * (m1_ + 1);
        
        // Process all stock points for this variance level
        for (int i = 0; i < m1_ - 1; ++i) {
            double sum = 0.0;
            
            // Process the 3x3 stencil around point (i,j)
            for (int l = -1; l <= 1; ++l) {
                for (int k = -1; k <= 1; ++k) {
                    int val_idx = i * 9 + (l + 1) * 3 + (k + 1);
                    int col_idx = (i + 1 + k) + (j + 1 + l) * (m1_ + 1);
                    
                    if (col_idx >= 0 && col_idx < total_size) {
                        sum += values_(j, val_idx) * x_(col_idx);
                    }
                }
            }
            
            // Store result for this stock point
            result(row_offset + i + 1) = sum;
        }
    });
    Kokkos::fence();
}

// Parallel multiply method (parallelizes over variance levels and stock price levels)
inline void heston_A0Storage_gpu::multiply_parallel_s_and_v(const Kokkos::View<double*>& x, Kokkos::View<double*>& result) const {
    int total_size = x.size();
    int m1_ = m1;
    int m2_ = m2;
    auto values_ = values;
    auto x_ = x;
    
    //Kokkos::deep_copy(result, 0.0);
    Kokkos::parallel_for("A0_zero_result", total_size, KOKKOS_LAMBDA(const int i) {
        result(i) = 0.0;
    });
    Kokkos::fence();

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
        result(row_offset + i + 1) += sum;
    });
    Kokkos::fence();

}


/*

A1 class

*/
class heston_A1Storage_gpu {
private:
    int m1, m2;

    // Explicit system diagonals
    Kokkos::View<double**> main_diags;
    Kokkos::View<double**> lower_diags;
    Kokkos::View<double**> upper_diags;

    // Implicit system diagonals
    Kokkos::View<double**> implicit_main_diags;
    Kokkos::View<double**> implicit_lower_diags;
    Kokkos::View<double**> implicit_upper_diags;
    
    // Persistent temporary storage for implicit solve
    //sequential
    Kokkos::View<double*> temp_sequ; 
    //parallel
    Kokkos::View<double**> temp_para; 

public:
    // Constructor - marked as host function since it allocates memory
    KOKKOS_FUNCTION
    heston_A1Storage_gpu() = default;

    heston_A1Storage_gpu(int m1_in, int m2_in);

    // Build matrix function declaration - implementation in cpp
    void build_matrix(const Grid& grid, double rho, double sigma, double r_d, double r_f);

    // Build implicit function - kept inline for performance
    void build_implicit(const double theta, const double delta_t);

    // Multiply function - kept inline for performance
    inline void multiply(const Kokkos::View<double*>& x, Kokkos::View<double*>& result);

    // Multiply function parallel in stock and varaince - kept inline for performance
    inline void multiply_parallel_s_and_v(const Kokkos::View<double*>& x, Kokkos::View<double*>& result);

    // Solve implicit function - kept inline for performance
    inline void solve_implicit(Kokkos::View<double*>& x, const Kokkos::View<double*>& b);

    // Solve implicit function parallel in varaince - kept inline for performance
    inline void solve_implicit_parallel_v(Kokkos::View<double*>& x, const Kokkos::View<double*>& b);

    // Getters
    KOKKOS_INLINE_FUNCTION int get_m1() const { return m1; }
    KOKKOS_INLINE_FUNCTION int get_m2() const { return m2; }
    KOKKOS_INLINE_FUNCTION const Kokkos::View<double**>& get_main_diags() const { return main_diags; }
    KOKKOS_INLINE_FUNCTION const Kokkos::View<double**>& get_lower_diags() const { return lower_diags; }
    KOKKOS_INLINE_FUNCTION const Kokkos::View<double**>& get_upper_diags() const { return upper_diags; }

    KOKKOS_INLINE_FUNCTION const Kokkos::View<double**>& get_implicit_main_diags() const { return implicit_main_diags; }
    KOKKOS_INLINE_FUNCTION const Kokkos::View<double**>& get_implicit_lower_diags() const { return implicit_lower_diags; }
    KOKKOS_INLINE_FUNCTION const Kokkos::View<double**>& get_implicit_upper_diags() const { return implicit_upper_diags; }
};

// Multiply function - kept inline for performance
//parallle in v
inline void heston_A1Storage_gpu::multiply(const Kokkos::View<double*>& x, Kokkos::View<double*>& result) {
    const int local_m1 = m1;
    const int local_m2 = m2;
    const auto local_main = main_diags;
    const auto local_lower = lower_diags;
    const auto local_upper = upper_diags;


    Kokkos::parallel_for("multiply", Kokkos::RangePolicy<>(0, local_m2 + 1), KOKKOS_LAMBDA(const int j) {
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
    });
    Kokkos::fence();
}

//this did not yield any faster improvement over the parallisation in v
inline void heston_A1Storage_gpu::multiply_parallel_s_and_v(const Kokkos::View<double*>& x, Kokkos::View<double*>& result) {
    const int local_m1 = m1;
    const int local_m2 = m2;
    const auto local_main = main_diags;
    const auto local_lower = lower_diags;
    const auto local_upper = upper_diags;

    // Initialize result to zero
    //Kokkos::deep_copy(result, 0.0);

    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;

    Kokkos::parallel_for("A1_multiply", team_policy(local_m2 + 1, Kokkos::AUTO), KOKKOS_LAMBDA(const member_type& team_member) {
        const int j = team_member.league_rank();
        const int offset = j * (local_m1 + 1);

        // Parallelize over i using team-level parallelism
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

// Solve implicit function - kept inline for performance
inline void heston_A1Storage_gpu::solve_implicit(Kokkos::View<double*>& x, const Kokkos::View<double*>& b) {
        const int local_m1 = m1;
        const int local_m2 = m2;
        const auto local_impl_main = implicit_main_diags;
        const auto local_impl_lower = implicit_lower_diags;
        const auto local_impl_upper = implicit_upper_diags;
        
        auto local_temp = temp_sequ;

        //Do scheme debugging
        //Kokkos::deep_copy(x, 0.0);

        Kokkos::parallel_for("solve_implicit", Kokkos::RangePolicy<>(0, 1), KOKKOS_LAMBDA(const int) {
            for(int j = 0; j <= local_m2; j++) {
                const int offset = j * (local_m1 + 1); //index for first entry of b for each block
                
                // Forward sweep

                //first entry (0,0) of each block is 1 since A1 (0,0) is 0 (boundary condition at s=s_min=0 is 0)
                //so I - thata*delta_t*A1 is 1
                local_temp(0) = local_impl_main(j,0); //this is 1 for each j (see right above explanaiton)
                x(offset) = b(offset);
                
                for(int i = 1; i <= local_m1; i++) {
                    double m = local_impl_lower(j,i-1) / local_temp(i-1); // 
                    local_temp(i) = local_impl_main(j,i) - m * local_impl_upper(j,i-1); // (wikipedia notation) b_2-a2*c_1', this is the new value of the diagonal
                    x(offset + i) = b(offset + i) - m * x(offset + i-1); //updating solution with x_2 = x_2 - m * x_1
                }

                // Back substitution
                x(offset + local_m1) /= local_temp(local_m1);
                for(int i = local_m1-1; i >= 0; i--) {
                    x(offset + i) = (x(offset + i) - 
                        local_impl_upper(j,i) * x(offset + i+1)) / local_temp(i);
                }
            }
        });
        Kokkos::fence();
    }
  

inline void heston_A1Storage_gpu::solve_implicit_parallel_v(Kokkos::View<double*>& x, const Kokkos::View<double*>& b) {
    const int local_m1 = m1;
    const int local_m2 = m2;
    const auto local_impl_main = implicit_main_diags;
    const auto local_impl_lower = implicit_lower_diags;
    const auto local_impl_upper = implicit_upper_diags;

    // Temporary storage: one line per variance block
    // Each block (j) gets a scratch space for the Thomas algorithm temp array
    //Kokkos::View<double**> temp("A1_temp", local_m2 + 1, local_m1 + 1);
    
    auto local_temp = temp_para;

    // Parallelize over variance levels j
    // Each j is handled by one thread of execution in this parallel_for.
    // The forward and backward sweeps remain sequential over i.
    Kokkos::parallel_for("A1_solve_implicit", local_m2 + 1, KOKKOS_LAMBDA(const int j) {
        const int offset = j * (local_m1 + 1);

        // Forward sweep for block j
        local_temp(j,0) = local_impl_main(j,0);
        x(offset) = b(offset);

        for (int i = 1; i <= local_m1; i++) {
            double m = local_impl_lower(j,i-1) / local_temp(j,i-1);
            local_temp(j,i) = local_impl_main(j,i) - m * local_impl_upper(j,i-1);
            x(offset + i) = b(offset + i) - m * x(offset + i - 1);
        }

        // Back substitution for block j
        x(offset + local_m1) /= local_temp(j,local_m1);
        for (int i = local_m1 - 1; i >= 0; i--) {
            x(offset + i) = (x(offset + i) -
                local_impl_upper(j,i) * x(offset + i + 1)) / local_temp(j,i);
        }
    });
    Kokkos::fence();
}


void test_hes_mat_fac();