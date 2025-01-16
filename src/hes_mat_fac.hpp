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
    void multiply_seq(const Kokkos::View<double*>& x, Kokkos::View<double*>& result) const;

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

inline void heston_A0Storage_gpu::multiply_seq(const Kokkos::View<double*>& x, 
                                             Kokkos::View<double*>& result) const {
    int total_size = x.size();
    int m1_ = m1;
    int m2_ = m2;
    auto values_ = values;
    auto x_ = x;

    //Debugging D0 scheme
    //Kokkos::deep_copy(result, 0.0);
    //auto result_ = result;


    // Run on a single thread
    Kokkos::parallel_for("A0_multiply_seq", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, 1), 
                            KOKKOS_LAMBDA(const int&) {
        //first zero block
        
        for(int k = 0; k<1+m1_+1; k++){
            result(k) = 0.0;
        }
        

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
                result(row_offset + i + 1) += sum;
            }
        }
        
        for(int k=m1_ + (m2_-1)*(m1_+1); k< (m1_+1)*(m2_+1); k++){
            result(k) = 0;
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
    // Zero the first block
    {
        int start = 0;
        int end = 1 + m1_ + 1; 
        Kokkos::parallel_for("A0_zero_front", Kokkos::RangePolicy<>(start, end), KOKKOS_LAMBDA(const int i) {
            result(i) = 0.0;
        });
        Kokkos::fence();
    }

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

    // Zero the tail region
    {
        int start_tail = m1_ + (m2_ - 1)*(m1_ + 1);
        int end_tail = (m1_ + 1)*(m2_ + 1);
        Kokkos::parallel_for("A0_zero_tail", Kokkos::RangePolicy<>(start_tail, end_tail), 
            KOKKOS_LAMBDA(const int i) {
                result(i) = 0.0;
            }
        );
        Kokkos::fence();
    }
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
inline void heston_A1Storage_gpu::multiply(const Kokkos::View<double*>& x, Kokkos::View<double*>& result) {
    const int local_m1 = m1;
    const int local_m2 = m2;
    const auto local_main = main_diags;
    const auto local_lower = lower_diags;
    const auto local_upper = upper_diags;

    //Debugging DO scheme. This is inefficient
    //Kokkos::deep_copy(result, 0.0);

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


/*

GPU A1 class. This implements the above A1 class, but this time s.t. its methods are callable form the GPU. This is needed for the
calibration process where we fill the Jacobianmatrix J in parallel. Each entry is one option value computation and therefor need to 
call the methods like A1.explicit() in parallel

*/

class heston_A1_device {
private:
    int m1, m2;

    // Explicit system diagonals - 2D Views [variance_level][stock_index]
    Kokkos::View<double**> main_diags;    // [m2+1][m1+1]
    Kokkos::View<double**> lower_diags;   // [m2+1][m1]
    Kokkos::View<double**> upper_diags;   // [m2+1][m1]

    // Implicit system diagonals
    Kokkos::View<double**> implicit_main_diags;
    Kokkos::View<double**> implicit_lower_diags;
    Kokkos::View<double**> implicit_upper_diags;

    // Temporary storage for implicit solve
    Kokkos::View<double**> temp_para;     // [m2+1][m1+1]

public:
    // Default constructor for device compatibility
    KOKKOS_DEFAULTED_FUNCTION
    heston_A1_device() = default;

    // Main constructor - host only as it allocates memory
    heston_A1_device(int m1_in, int m2_in) : m1(m1_in), m2(m2_in) {
        // Allocate explicit system diagonals
        main_diags = Kokkos::View<double**>("A1_main_diags", m2+1, m1+1);
        lower_diags = Kokkos::View<double**>("A1_lower_diags", m2+1, m1);
        upper_diags = Kokkos::View<double**>("A1_upper_diags", m2+1, m1);

        // Allocate implicit system diagonals
        implicit_main_diags = Kokkos::View<double**>("A1_impl_main_diags", m2+1, m1+1);
        implicit_lower_diags = Kokkos::View<double**>("A1_impl_lower_diags", m2+1, m1);
        implicit_upper_diags = Kokkos::View<double**>("A1_impl_upper_diags", m2+1, m1);

        // Allocate workspace
        temp_para = Kokkos::View<double**>("A1_temp_para", m2+1, m1+1);
    }

    // Matrix building method - callable from device
    KOKKOS_FUNCTION
    void build_matrix_device(double rho, double sigma, double r_d, double r_f) {
        const int local_m1 = m1;
        const int local_m2 = m2;
        const auto local_main = main_diags;
        const auto local_lower = lower_diags;
        const auto local_upper = upper_diags;
        
        // Loop over variance levels
        /*
        for(int j = 0; j <= local_m2; j++) {
            // Interior points
            for(int i = 1; i < local_m1; i++) {
                // PDE coefficients
                double a = 0.5 * grid.device_Vec_s(i) * grid.device_Vec_s(i) * grid.device_Vec_v(j);
                double b = (r_d - r_f) * grid.device_Vec_s(i);

                // Build tridiagonal system for this level
                // Lower diagonal
                local_lower(j,i-1) = a * device_delta_s(i-1, -1, grid.device_Delta_s) + 
                                    b * device_beta_s(i-1, -1, grid.device_Delta_s);
                
                // Main diagonal
                local_main(j,i) = a * device_delta_s(i-1, 0, grid.device_Delta_s) + 
                                 b * device_beta_s(i-1, 0, grid.device_Delta_s) - 0.5 * r_d;
                
                // Upper diagonal
                local_upper(j,i) = a * device_delta_s(i-1, 1, grid.device_Delta_s) + 
                                  b * device_beta_s(i-1, 1, grid.device_Delta_s);
            }
        }
        */
        
        // Build tridiagonal system for this level
        // Lower diagonal
        local_lower(0,0) = sigma;

        // Main diagonal
        local_main(0,0) = rho;

        // Upper diagonal
        local_upper(0,0) = rho;


        // Last entry in block
        local_main(0,local_m1) = -0.5 * r_d;
    }

    // Build implicit system - callable from device
    KOKKOS_FUNCTION
    void build_implicit_device(const double theta, const double delta_t) {
        for(int j = 0; j <= m2; j++) {
            // Main diagonal entries
            for(int i = 0; i <= m1; i++) {
                implicit_main_diags(j,i) = 1.0 - theta * delta_t * main_diags(j,i);
            }
            
            // Off-diagonal entries
            for(int i = 0; i < m1; i++) {
                implicit_lower_diags(j,i) = -theta * delta_t * lower_diags(j,i);
                implicit_upper_diags(j,i) = -theta * delta_t * upper_diags(j,i);
            }
        }
    }

    // Explicit multiply for a single block - callable from device
    KOKKOS_FUNCTION
    inline void multiply_device(const Kokkos::View<double*>& x, Kokkos::View<double*>& result);

    // Implicit solve for a single block - callable from device
    KOKKOS_FUNCTION
    inline void solve_implicit_device(Kokkos::View<double*>& x, const Kokkos::View<double*>& b);

    // Getters
    KOKKOS_INLINE_FUNCTION int get_m1() const { return m1; }
    KOKKOS_INLINE_FUNCTION int get_m2() const { return m2; }
    KOKKOS_INLINE_FUNCTION const Kokkos::View<double**>& get_main_diags() const { return main_diags; }
    KOKKOS_INLINE_FUNCTION const Kokkos::View<double**>& get_lower_diags() const { return lower_diags; }
    KOKKOS_INLINE_FUNCTION const Kokkos::View<double**>& get_upper_diags() const { return upper_diags; }
};


// Multiply function - kept inline for performance
KOKKOS_FUNCTION
inline void heston_A1_device::multiply_device(const Kokkos::View<double*>& x, Kokkos::View<double*>& result){
    using policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
    policy parallel_policy({0,0}, {m2+1, m1+1});

    Kokkos::parallel_for("A1_multiply", parallel_policy,
    KOKKOS_LAMBDA(const int j, const int i) {
        const int offset = j * (m1 + 1);
        double sum = main_diags(j,i) * x(offset + i);

        if(i > 0) {
        sum += lower_diags(j,i-1) * x(offset + i-1);
        }
        if(i < m1) {
        sum += upper_diags(j,i) * x(offset + i+1);
        }
        result(offset + i) = sum;
    }
    );
    Kokkos::fence();
}

KOKKOS_FUNCTION
inline void heston_A1_device::solve_implicit_device(Kokkos::View<double*>& x, const Kokkos::View<double*>& b){
    // Parallel over variance levels
    Kokkos::parallel_for("A1_implicit_solve", m2 + 1, KOKKOS_LAMBDA(const int j) {
        const int offset = j * (m1 + 1);
        
        // Forward sweep
        // First entry
        temp_para(j,0) = implicit_main_diags(j,0);
        x(offset) = b(offset);
        
        // Interior points forward sweep
        for(int i = 1; i <= m1; i++) {
            double m = implicit_lower_diags(j,i-1) / temp_para(j,i-1);
            temp_para(j,i) = implicit_main_diags(j,i) - m * implicit_upper_diags(j,i-1);
            x(offset + i) = b(offset + i) - m * x(offset + i-1);
        }
        
        // Back substitution
        // Last point
        x(offset + m1) /= temp_para(j,m1);
        
        // Interior points back substitution
        for(int i = m1-1; i >= 0; i--) {
            x(offset + i) = (x(offset + i) - 
                implicit_upper_diags(j,i) * x(offset + i+1)) / temp_para(j,i);
        }
    });
    Kokkos::fence();
}

/*

A2 class moved to an individual .hpp (hes_A2_mat) file for debgugging. There was an indexing error

*/
/*
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

public:
    KOKKOS_FUNCTION
    heston_A2Storage_gpu() = default;

    heston_A2Storage_gpu(int m1_in, int m2_in);

    void build_matrix(const Grid& grid, double rho, double sigma, double r_d, double kappa, double eta);

    void build_implicit(const double theta, const double delta_t);

    // Multiply function - kept inline for performance
    inline void multiply(const Kokkos::View<double*>& x, Kokkos::View<double*>& result);

    // Solve implicit function - kept inline for performance
    inline void solve_implicit(Kokkos::View<double*>& x, const Kokkos::View<double*>& b);

    // Getters
    KOKKOS_INLINE_FUNCTION int get_m1() const { return m1; }
    KOKKOS_INLINE_FUNCTION int get_m2() const { return m2; }
    KOKKOS_INLINE_FUNCTION const Kokkos::View<double*>& get_main_diag() const { return main_diag; }
    KOKKOS_INLINE_FUNCTION const Kokkos::View<double*>& get_lower_diag() const { return lower_diag; }
    KOKKOS_INLINE_FUNCTION const Kokkos::View<double*>& get_upper_diag() const { return upper_diag; }
    KOKKOS_INLINE_FUNCTION const Kokkos::View<double*>& get_upper2_diag() const { return upper2_diag; }
};

inline void heston_A2Storage_gpu::multiply(const Kokkos::View<double*>& x, Kokkos::View<double*>& result) {
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
        } else {
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

inline void heston_A2Storage_gpu::solve_implicit(Kokkos::View<double*>& x, const Kokkos::View<double*>& b) {
    const int local_m1 = m1;
    const int local_m2 = m2;
    const int spacing = local_m1 + 1;
    const int num_rows = (local_m2 - 1) * spacing;
    const int total_size = (local_m2 + 1) * spacing;

    // Temporary storage on device
    Kokkos::View<double*> c_star("c_star", num_rows);
    Kokkos::View<double*> c2_star("c2_star", spacing);
    Kokkos::View<double*> d_star("d_star", total_size);

    // Get diagonal arrays
    const auto local_impl_main = implicit_main_diag;
    const auto local_impl_lower = implicit_lower_diag;
    const auto local_impl_upper = implicit_upper_diag;
    const auto local_impl_upper2 = implicit_upper2_diag;

    Kokkos::parallel_for("solve_implicit", 1, KOKKOS_LAMBDA(const int) {
        // Identity block
        for (int i = num_rows; i < total_size; i++) {
            d_star(i) = b(i);
        }

        // Normalize first m1+1 rows and upper2_diagonal
        for (int i = 0; i < spacing; i++) {
            c_star(i) = local_impl_upper(i) / local_impl_main(i);
            c2_star(i) = local_impl_upper2(i) / local_impl_main(i);
            d_star(i) = b(i) / local_impl_main(i);
        }

        // First block forward sweep (handle upper2_diag)
        for (int i = 0; i < spacing; i++) {
            double c_upper = local_impl_upper(i + spacing) - c2_star(i) * local_impl_lower(i);
            double m = 1.0 / (local_impl_main(i + spacing) - c_star(i) * local_impl_lower(i));
            c_star(i + spacing) = c_upper * m;
            d_star(i + spacing) = (b(i + spacing) - local_impl_lower(i) * d_star(i)) * m;
        }

        // Middle blocks forward sweep
        for (int i = spacing; i < num_rows - spacing; i++) {
            double m = 1.0 / (local_impl_main(i + spacing) - c_star(i) * local_impl_lower(i));
            c_star(i + spacing) = local_impl_upper(i + spacing) * m;
            d_star(i + spacing) = (b(i + spacing) - local_impl_lower(i) * d_star(i)) * m;
        }

        // Pre-backward sweep
        for (int i = num_rows - spacing; i < num_rows; i++) {
            d_star(i) -= d_star(i + spacing) * c_star(i);
        }

        // Last m1+1 rows
        for (int i = num_rows - spacing; i < num_rows; i++) {
            x(i) = d_star(i);
        }

        // Backward sweep
        for (int i = num_rows - 1; i >= 3 * spacing; i--) {
            x(i - spacing) = d_star(i - spacing) - c_star(i - spacing) * x(i);
        }

        // First block back substitution with upper2_diag
        for (int i = 3 * spacing - 1; i >= 2 * spacing; i--) {
            x(i - spacing) = d_star(i - spacing) - c_star(i - spacing) * x(i);
            d_star(i - 2 * spacing) -= c2_star(i - 2 * spacing) * x(i);
        }

        // Last backward substitution
        for (int i = 2 * spacing - 1; i >= spacing; i--) {
            x(i - spacing) = d_star(i - spacing) - c_star(i - spacing) * x(i);
        }

        // Identity block
        for (int i = num_rows; i < total_size; i++) {
            x(i) = d_star(i);
        }
    });
    Kokkos::fence();
}
*/


/*

This is a A1 class test where i implement coalesc memory access
The idea was to improve the implicit solver speed significantly, but it actually did not improve
at all. This lead me to believe we should implement the A1 matrix with just rhee  diagonals and
not try to optimize memory allocation by the block structure

*/

class heston_A1Storage_coalesc {
private:
    int m1, m2;
    // Instead of m2+1 separate tridiagonal matrices, store all diagonals contiguously
    Kokkos::View<double*> main_diags;    // [(m2+1)*(m1+1)]
    Kokkos::View<double*> lower_diags;   // [(m2+1)*m1]
    Kokkos::View<double*> upper_diags;   // [(m2+1)*m1]

    // For implicit solver
    Kokkos::View<double*> implicit_main_diags;
    Kokkos::View<double*> implicit_lower_diags;
    Kokkos::View<double*> implicit_upper_diags;

    Kokkos::View<double*> temp_storage;  // [(m2+1)*(m1+1)]
    
public:
    heston_A1Storage_coalesc(int m1_in, int m2_in) : m1(m1_in), m2(m2_in) {
        main_diags = Kokkos::View<double*>("A1_main_diags", (m2+1)*(m1+1));
        lower_diags = Kokkos::View<double*>("A1_lower_diags", (m2+1)*m1);
        upper_diags = Kokkos::View<double*>("A1_upper_diags", (m2+1)*m1);

        implicit_main_diags = Kokkos::View<double*>("A1_impl_main_diags", (m2+1)*(m1+1));
        implicit_lower_diags = Kokkos::View<double*>("A1_impl_lower_diags", (m2+1)*m1);
        implicit_upper_diags = Kokkos::View<double*>("A1_impl_upper_diags", (m2+1)*m1);

        temp_storage = Kokkos::View<double*>("A1_temp_storage", (m2+1)*(m1+1));
    }

    // Helper functions for indexing
    KOKKOS_INLINE_FUNCTION
    int main_index(int j, int i) const { return j*(m1+1) + i; }
    
    KOKKOS_INLINE_FUNCTION
    int off_index(int j, int i) const { return j*m1 + i; }

    // Build functions would need similar adjustments to use linear indexing
    void build_matrix(const Grid& grid, double rho, double sigma, double r_d, double r_f) {
        auto h_main = Kokkos::create_mirror_view(main_diags);
        auto h_lower = Kokkos::create_mirror_view(lower_diags);
        auto h_upper = Kokkos::create_mirror_view(upper_diags);
        
        Kokkos::deep_copy(h_main, 0.0);
        Kokkos::deep_copy(h_lower, 0.0);
        Kokkos::deep_copy(h_upper, 0.0);

        // Fill arrays using linear indexing
        for(int j = 0; j <= m2; j++) {
            for(int i = 1; i < m1; i++) {
                double a = 0.5 * grid.Vec_s[i] * grid.Vec_s[i] * grid.Vec_v[j];
                double b = (r_d - r_f) * grid.Vec_s[i];
                
                h_lower[off_index(j,i-1)] = a * delta_s(i-1, -1, grid.Delta_s) + 
                                        b * beta_s(i-1, -1, grid.Delta_s);
                h_main[main_index(j,i)] = a * delta_s(i-1, 0, grid.Delta_s) + 
                                    b * beta_s(i-1, 0, grid.Delta_s) - 0.5 * r_d;
                h_upper[off_index(j,i)] = a * delta_s(i-1, 1, grid.Delta_s) + 
                                        b * beta_s(i-1, 1, grid.Delta_s);
            }
            h_main[main_index(j,m1)] = -0.5 * r_d;
        }

        Kokkos::deep_copy(main_diags, h_main);
        Kokkos::deep_copy(lower_diags, h_lower);
        Kokkos::deep_copy(upper_diags, h_upper);
    }

    void build_implicit(const double theta, const double delta_t) {
        // Create host mirrors for all Views
        auto h_main_diag = Kokkos::create_mirror_view(main_diags);
        auto h_lower_diag = Kokkos::create_mirror_view(lower_diags);
        auto h_upper_diag = Kokkos::create_mirror_view(upper_diags);
        
        auto h_impl_main = Kokkos::create_mirror_view(implicit_main_diags);
        auto h_impl_lower = Kokkos::create_mirror_view(implicit_lower_diags);
        auto h_impl_upper = Kokkos::create_mirror_view(implicit_upper_diags);

        // Copy existing data from device to host
        Kokkos::deep_copy(h_main_diag, main_diags);
        Kokkos::deep_copy(h_lower_diag, lower_diags);
        Kokkos::deep_copy(h_upper_diag, upper_diags);

        // Build implicit system on host
        for(int j = 0; j <= m2; j++) {
            for(int i = 0; i <= m1; i++) {
                const int main_idx = j*(m1+1) + i;
                h_impl_main(main_idx) = 1.0 - theta * delta_t * h_main_diag(main_idx);
            }
            
            for(int i = 0; i < m1; i++) {
                const int off_idx = j*m1 + i;
                h_impl_lower(off_idx) = -theta * delta_t * h_lower_diag(off_idx);
                h_impl_upper(off_idx) = -theta * delta_t * h_upper_diag(off_idx);
            }
        }

        // Copy results back to device
        Kokkos::deep_copy(implicit_main_diags, h_impl_main);
        Kokkos::deep_copy(implicit_lower_diags, h_impl_lower);
        Kokkos::deep_copy(implicit_upper_diags, h_impl_upper);
    }

    // Optimized parallel solve
    void solve_implicit_parallel_v(Kokkos::View<double*>& x, const Kokkos::View<double*>& b) {
        const int local_m1 = m1;
        const auto local_impl_main = implicit_main_diags;
        const auto local_impl_lower = implicit_lower_diags;
        const auto local_impl_upper = implicit_upper_diags;

        const auto temp = temp_storage;

        Kokkos::parallel_for("A1_solve_implicit", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,m2+1), KOKKOS_LAMBDA(const int j) {
            const int block_offset = j * (local_m1 + 1);

            // Forward sweep
            // Use direct indexing into contiguous memory
            temp(block_offset) = local_impl_main(block_offset);
            x(block_offset) = b(block_offset);

            for (int i = 1; i <= local_m1; i++) {
                const int curr_idx = block_offset + i;
                const int prev_idx = curr_idx - 1;
                const int off_idx = j*local_m1 + (i-1);  // Index into off-diagonals
                
                double m = local_impl_lower(off_idx) / temp(prev_idx);
                temp(curr_idx) = local_impl_main(curr_idx) - m * local_impl_upper(off_idx);
                x(curr_idx) = b(curr_idx) - m * x(prev_idx);
            }

            // Back substitution
            x(block_offset + local_m1) /= temp(block_offset + local_m1);
            for (int i = local_m1 - 1; i >= 0; i--) {
                const int curr_idx = block_offset + i;
                const int next_idx = curr_idx + 1;
                const int off_idx = j*local_m1 + i;

                x(curr_idx) = (x(curr_idx) - local_impl_upper(off_idx) * x(next_idx)) 
                             / temp(curr_idx);
            }
        });
        Kokkos::fence();
    }
    
    // Sequential multiply method
    void multiply_seq(const Kokkos::View<double*>& x, Kokkos::View<double*>& result) const {
        // Run on a single thread
        Kokkos::parallel_for("A1_multiply_seq", 1, KOKKOS_LAMBDA(const int) {
            // Loop over variance levels
            for(int j = 0; j <= m2; j++) {
                const int block_offset = j * (m1 + 1);
                
                // First entry of each block
                result(block_offset) = main_diags(block_offset) * x(block_offset);
                if(block_offset < (m2*(m1+1))) {
                    result(block_offset) += upper_diags(j*m1) * x(block_offset + 1);
                }
                
                // Middle entries
                for(int i = 1; i < m1; i++) {
                    const int curr_idx = block_offset + i;
                    result(curr_idx) = main_diags(curr_idx) * x(curr_idx) +
                                    lower_diags(j*m1 + i - 1) * x(curr_idx - 1) +
                                    upper_diags(j*m1 + i) * x(curr_idx + 1);
                }
                
                // Last entry of each block
                const int last_idx = block_offset + m1;
                result(last_idx) = main_diags(last_idx) * x(last_idx) +
                                lower_diags(j*m1 + m1 - 1) * x(last_idx - 1);
            }
        });
        Kokkos::fence();
    }

    // Parallel multiply method
    void multiply_parallel(const Kokkos::View<double*>& x, Kokkos::View<double*>& result) const {
        Kokkos::parallel_for("A1_multiply_parallel", m2 + 1, KOKKOS_LAMBDA(const int j) {
            const int block_offset = j * (m1 + 1);
            
            // First entry of block
            result(block_offset) = main_diags(block_offset) * x(block_offset);
            if(block_offset < (m2*(m1+1))) {
                result(block_offset) += upper_diags(j*m1) * x(block_offset + 1);
            }
            
            // Middle entries
            for(int i = 1; i < m1; i++) {
                const int curr_idx = block_offset + i;
                result(curr_idx) = main_diags(curr_idx) * x(curr_idx) +
                                lower_diags(j*m1 + i - 1) * x(curr_idx - 1) +
                                upper_diags(j*m1 + i) * x(curr_idx + 1);
            }
            
            // Last entry of block
            const int last_idx = block_offset + m1;
            result(last_idx) = main_diags(last_idx) * x(last_idx) +
                            lower_diags(j*m1 + m1 - 1) * x(last_idx - 1);
        });
        Kokkos::fence();
    }

    // Getters
    KOKKOS_INLINE_FUNCTION int get_m1() const { return m1; }
    KOKKOS_INLINE_FUNCTION int get_m2() const { return m2; }
    KOKKOS_INLINE_FUNCTION const Kokkos::View<double*>& get_main_diags() const { return main_diags; }
    KOKKOS_INLINE_FUNCTION const Kokkos::View<double*>& get_lower_diags() const { return lower_diags; }
    KOKKOS_INLINE_FUNCTION const Kokkos::View<double*>& get_upper_diags() const { return upper_diags; }

};

/*

This is the "basic" idea of the A1 class where we just store it as three diagonasl and use one
thread to solve the impolicit system. 

*/

class heston_A1_flat {
private:
    int m1, m2;  // Grid dimensions
    int total_size;  // (m1+1)*(m2+1)

    // Explicit matrix storage - single long diagonal format
    Kokkos::View<double*> main_diag;    // Length: total_size
    Kokkos::View<double*> lower_diag;   // Length: total_size-1
    Kokkos::View<double*> upper_diag;   // Length: total_size-1

    // Implicit matrix storage (I - theta*dt*A)
    Kokkos::View<double*> implicit_main_diag;
    Kokkos::View<double*> implicit_lower_diag;
    Kokkos::View<double*> implicit_upper_diag;

    // Temporary storage for Thomas algorithm
    Kokkos::View<double*> c_prime;      // Length: total_size
    Kokkos::View<double*> d_prime;      // Length: total_size

public:
    // Constructor and default constructor remain unchanged
    KOKKOS_FUNCTION
    heston_A1_flat() = default;

    heston_A1_flat(int m1_in, int m2_in) : m1(m1_in), m2(m2_in), total_size((m1+1)*(m2+1)) {
        main_diag = Kokkos::View<double*>("A1_main_diag", total_size);
        lower_diag = Kokkos::View<double*>("A1_lower_diag", total_size-1);
        upper_diag = Kokkos::View<double*>("A1_upper_diag", total_size-1);

        implicit_main_diag = Kokkos::View<double*>("A1_impl_main_diag", total_size);
        implicit_lower_diag = Kokkos::View<double*>("A1_impl_lower_diag", total_size-1);
        implicit_upper_diag = Kokkos::View<double*>("A1_impl_upper_diag", total_size-1);

        c_prime = Kokkos::View<double*>("A1_c_prime", total_size);
        d_prime = Kokkos::View<double*>("A1_d_prime", total_size);
    }

    // Revised build_matrix function
    void build_matrix(const Grid& grid, double rho, double sigma, double r_d, double r_f) {
        auto h_main = Kokkos::create_mirror_view(main_diag);
        auto h_lower = Kokkos::create_mirror_view(lower_diag);
        auto h_upper = Kokkos::create_mirror_view(upper_diag);

        // Initialize all to zero
        Kokkos::deep_copy(h_main, 0.0);
        Kokkos::deep_copy(h_lower, 0.0);
        Kokkos::deep_copy(h_upper, 0.0);

        // Fill matrices block by block
        for(int j = 0; j <= m2; j++) {
            int block_offset = j * (m1 + 1);

            // First row in each block has main diagonal = 0 (already set by initialization)
            // h_main(block_offset) = 0.0;  // Not needed as we initialized to zero

            // Interior rows of block
            for(int i = 1; i < m1; i++) {
                double a = 0.5 * grid.Vec_s[i] * grid.Vec_s[i] * grid.Vec_v[j];
                double b = (r_d - r_f) * grid.Vec_s[i];
                int idx = block_offset + i;

                // Lower diagonal (non-zero only for interior points)
                h_lower(idx-1) = 1;//a * delta_s(i-1, -1, grid.Delta_s) + b * beta_s(i-1, -1, grid.Delta_s);

                // Main diagonal
                h_main(idx) = -2;//a * delta_s(i-1, 0, grid.Delta_s) + b * beta_s(i-1, 0, grid.Delta_s) - 0.5 * r_d;

                // Upper diagonal 
                h_upper(idx) = 1;//a * delta_s(i-1, 1, grid.Delta_s) + b * beta_s(i-1, 1, grid.Delta_s);
            }

            // Last row in block
            h_main(block_offset + m1) = -2;//-0.5 * r_d;

            // Zero out block boundaries in off-diagonals
            if(j < m2) {
                // Zero out last connection of current block to next block
                h_lower(block_offset + m1) = 0.0;
                h_upper(block_offset + m1) = 0.0;
            }
        }

        // Copy to device
        Kokkos::deep_copy(main_diag, h_main);
        Kokkos::deep_copy(lower_diag, h_lower);
        Kokkos::deep_copy(upper_diag, h_upper);
    }

    // Other functions remain unchanged
    void build_implicit(const double theta, const double delta_t) {
        // Create host mirrors for all implicit matrices
        auto h_impl_main = Kokkos::create_mirror_view(implicit_main_diag);
        auto h_impl_lower = Kokkos::create_mirror_view(implicit_lower_diag);
        auto h_impl_upper = Kokkos::create_mirror_view(implicit_upper_diag);

        // Get the explicit matrices on host to build from
        auto h_main = Kokkos::create_mirror_view(main_diag);
        auto h_lower = Kokkos::create_mirror_view(lower_diag);
        auto h_upper = Kokkos::create_mirror_view(upper_diag);

        // Copy explicit matrices to host
        Kokkos::deep_copy(h_main, main_diag);
        Kokkos::deep_copy(h_lower, lower_diag);
        Kokkos::deep_copy(h_upper, upper_diag);

        // Build implicit matrices on host
        // Main diagonal: I - theta*dt*A
        for(int i = 0; i < total_size; i++) {
            h_impl_main(i) = 1.0 - theta * delta_t * h_main(i);
        }

        // Off diagonals: -theta*dt*A
        for(int i = 0; i < total_size - 1; i++) {
            h_impl_lower(i) = -theta * delta_t * h_lower(i);
            h_impl_upper(i) = -theta * delta_t * h_upper(i);
        }

        // Copy results back to device
        Kokkos::deep_copy(implicit_main_diag, h_impl_main);
        Kokkos::deep_copy(implicit_lower_diag, h_impl_lower);
        Kokkos::deep_copy(implicit_upper_diag, h_impl_upper);
    }
    
    void multiply(const Kokkos::View<double*>& x, Kokkos::View<double*>& result) {
        const int local_total_size = total_size;
        const auto local_main = main_diag;
        const auto local_lower = lower_diag;
        const auto local_upper = upper_diag;
        const auto local_x = x;
        const auto local_result = result;

        Kokkos::parallel_for("A1_multiply", local_total_size, KOKKOS_LAMBDA(const int i) {
            local_result(i) = local_main(i) * local_x(i);
            
            if(i > 0) {
                local_result(i) += local_lower(i-1) * local_x(i-1);
            }
            
            if(i < local_total_size - 1) {
                local_result(i) += local_upper(i) * local_x(i+1);
            }
        });
        Kokkos::fence();
    }

    //first implcicit solver
    void solve_implicit(Kokkos::View<double*>& x, const Kokkos::View<double*>& b) {
        const int local_total_size = total_size;
        const auto local_impl_main = implicit_main_diag;
        const auto local_impl_lower = implicit_lower_diag;
        const auto local_impl_upper = implicit_upper_diag;
        const auto local_c_prime = c_prime;
        const auto local_d_prime = d_prime;
        const auto local_x = x;
        const auto local_b = b;

        Kokkos::parallel_for("A1_thomas_solve", Kokkos::RangePolicy<>(0, 1), 
            KOKKOS_LAMBDA(const int) {
                // Forward sweep
                local_c_prime(0) = local_impl_upper(0) / local_impl_main(0);
                local_d_prime(0) = local_b(0) / local_impl_main(0);

                for(int i = 1; i < local_total_size; i++) {
                    double denominator = local_impl_main(i) - local_impl_lower(i-1) * local_c_prime(i-1);
                    
                    if(i < local_total_size - 1) {
                        local_c_prime(i) = local_impl_upper(i) / denominator;
                    }
                    
                    local_d_prime(i) = (local_b(i) - local_impl_lower(i-1) * local_d_prime(i-1)) / denominator;
                }

                // Back substitution
                local_x(local_total_size - 1) = local_d_prime(local_total_size - 1);
                
                for(int i = local_total_size - 2; i >= 0; i--) {
                    local_x(i) = local_d_prime(i) - local_c_prime(i) * local_x(i+1);
                }
        });
        Kokkos::fence();
    }

    /*
    void solve_implicit(Kokkos::View<double*>& x, const Kokkos::View<double*>& b) {
        const int local_size = total_size;
        const auto local_main = implicit_main_diag;
        const auto local_lower = implicit_lower_diag;
        const auto local_upper = implicit_upper_diag;
        const auto local_temp = c_prime;  // Using c_prime as temp storage
        const auto local_x = x;
        const auto local_b = b;

        // Sequential Thomas algorithm on device
        Kokkos::parallel_for("thomas_solve", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,1), 
            KOKKOS_LAMBDA(const int) {
                // Forward sweep
                local_temp(0) = local_main(0);
                local_x(0) = local_b(0);
                
                for(int i = 1; i < local_size; ++i) {
                    double w = local_lower(i-1) / local_temp(i-1);
                    local_temp(i) = local_main(i) - w * local_upper(i-1);
                    local_x(i) = local_b(i) - w * local_x(i-1);
                }

                // Backward substitution
                local_x(local_size-1) = local_x(local_size-1) / local_temp(local_size-1);
                for(int i = local_size-2; i >= 0; --i) {
                    local_x(i) = (local_x(i) - local_upper(i) * local_x(i+1)) / local_temp(i);
                }
        });
        Kokkos::fence();
    }
    */

    /*
    void solve_implicit(Kokkos::View<double*>& x, const Kokkos::View<double*>& b) {
        const int N = total_size;

        // Create local copies of the implicit diagonal arrays
        Kokkos::View<double*> main_copy("main_copy", N);
        Kokkos::View<double*> lower_copy("lower_copy", N-1);
        Kokkos::View<double*> upper_copy("upper_copy", N-1);

        Kokkos::deep_copy(main_copy, implicit_main_diag);
        Kokkos::deep_copy(lower_copy, implicit_lower_diag);
        Kokkos::deep_copy(upper_copy, implicit_upper_diag);

        // Create a local copy of b
        Kokkos::View<double*> b_local("b_local", N);
        Kokkos::deep_copy(b_local, b);

        // Create a temporary array for the Thomas algorithm
        Kokkos::View<double*> temp("temp", N);

        // Initialize x to 0.0
        Kokkos::deep_copy(x, 0.0);

        // Now run the Thomas solve using the local copies
        using timer = std::chrono::high_resolution_clock;
        auto t_start = timer::now();
        Kokkos::parallel_for("tridiagonal_solve",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,1),
            KOKKOS_LAMBDA(const int) {
                // Forward sweep
                temp(0) = main_copy(0);
                x(0) = b_local(0);
                for(int i = 1; i < N; ++i) {
                    double w = lower_copy(i-1) / temp(i-1);
                    temp(i) = main_copy(i) - w * upper_copy(i-1);
                    x(i) = b_local(i) - w * x(i-1);
                }

                // Backward substitution
                x(N-1) = x(N-1) / temp(N-1);
                for(int i = N-2; i >= 0; --i) {
                    x(i) = (x(i) - upper_copy(i) * x(i+1)) / temp(i);
                }
            }
        );
        Kokkos::fence();
        auto t_end = timer::now();
        std::cout << "Run " << " implicit solve time: "
                << std::chrono::duration<double>(t_end - t_start).count()
                << " seconds\n";
    }
    */

    // Getters
    KOKKOS_INLINE_FUNCTION int get_m1() const { return m1; }
    KOKKOS_INLINE_FUNCTION int get_m2() const { return m2; }
    KOKKOS_INLINE_FUNCTION int get_total_size() const { return total_size; }
    KOKKOS_INLINE_FUNCTION const Kokkos::View<double*>& get_main_diag() const { return main_diag; }
    KOKKOS_INLINE_FUNCTION const Kokkos::View<double*>& get_lower_diag() const { return lower_diag; }
    KOKKOS_INLINE_FUNCTION const Kokkos::View<double*>& get_upper_diag() const { return upper_diag; }
};


void test_hes_mat_fac();