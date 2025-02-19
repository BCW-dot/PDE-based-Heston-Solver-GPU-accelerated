#ifndef BOUNDARY_CONDITIONS_HPP
#define BOUNDARY_CONDITIONS_HPP

#include <Kokkos_Core.hpp>
#include <cmath>

/*

This class implements the boundary conditions for a call option. It has a misleading name, since the decision to also
discuss Put options was done at a later time. 

*/
class BoundaryConditions {
private:
    // Member views to store boundary values
    Kokkos::View<double*> b0_; // Mixed derivative boundary
    Kokkos::View<double*> b1_; // S direction boundary 
    Kokkos::View<double*> b2_; // V direction boundary
    Kokkos::View<double*> b_;  // Combined boundary vector

    // Grid dimensions
    int m1_; // Number of S grid points - 1
    int m2_; // Number of V grid points - 1
    int m_;  // Total size (m1+1)*(m2+1)

    // Parameters
    double r_d_;  // Domestic interest rate
    double r_f_;  // Foreign interest rate
    int N_;       // Number of time steps
    double delta_t_; // Time step size

public:
    // Constructor 
    BoundaryConditions(int m1, int m2, double r_d, double r_f, int N, double delta_t)
        : m1_(m1), m2_(m2), m_((m1+1)*(m2+1)), 
          r_d_(r_d), r_f_(r_f), N_(N), delta_t_(delta_t) {
        
        // Allocate views with proper Kokkos constructors
        b0_ = Kokkos::View<double*>("b0", m_);
        b1_ = Kokkos::View<double*>("b1", m_);
        b2_ = Kokkos::View<double*>("b2", m_);
        b_  = Kokkos::View<double*>("b", m_);

        // Initialize to zero
        Kokkos::deep_copy(b0_, 0.0);
        Kokkos::deep_copy(b1_, 0.0);
        Kokkos::deep_copy(b2_, 0.0);
        Kokkos::deep_copy(b_, 0.0);
    }

    // Initialize boundary conditions given stock price grid
    void initialize(const Kokkos::View<double*>& Vec_s) {
        // Create host mirrors for all arrays
        auto h_b0 = Kokkos::create_mirror_view(b0_);
        auto h_b1 = Kokkos::create_mirror_view(b1_);
        auto h_b2 = Kokkos::create_mirror_view(b2_);
        auto h_b = Kokkos::create_mirror_view(b_);
        auto h_Vec_s = Kokkos::create_mirror_view(Vec_s);

        // Copy Vec_s to host
        Kokkos::deep_copy(h_Vec_s, Vec_s);

        // Initialize to zero
        for(int i = 0; i < m_; i++) {
            h_b0[i] = 0.0;
            h_b1[i] = 0.0;
            h_b2[i] = 0.0;
        }

        // Initialize b1 (S direction boundary)
        for(int j = 0; j <= m2_; j++) {
            h_b1[m1_ * (j + 1)] = (r_d_ - r_f_) * h_Vec_s[m1_] * 
                                std::exp(-r_f_ * delta_t_ * (N_ - 1));
        }

        // Initialize b2 (V direction boundary)
        for(int i = 1; i <= m1_; i++) {
            h_b2[m_ - m1_ - 1 + i] = -0.5 * r_d_ * h_Vec_s[i] * 
                                    std::exp(-r_f_ * delta_t_ * (N_ - 1));
        }

        // Combine boundaries on host
        for(int i = 0; i < m_; i++) {
            h_b[i] = h_b0[i] + h_b1[i] + h_b2[i];
        }

        // Copy all results to device at once
        Kokkos::deep_copy(b0_, h_b0);
        Kokkos::deep_copy(b1_, h_b1);
        Kokkos::deep_copy(b2_, h_b2);
        Kokkos::deep_copy(b_, h_b);
    }

    // Getters for boundary views
    Kokkos::View<double*> get_b0() const { return b0_; }
    Kokkos::View<double*> get_b1() const { return b1_; }
    Kokkos::View<double*> get_b2() const { return b2_; }
    Kokkos::View<double*> get_b() const { return b_; }

    int get_size() const { return m_; }
};


class BoundaryConditions_put {
    private:
        // Member views to store boundary values
        Kokkos::View<double*> b0_; // Mixed derivative boundary
        Kokkos::View<double*> b1_; // S direction boundary 
        Kokkos::View<double*> b2_; // V direction boundary
        Kokkos::View<double*> b_;  // Combined boundary vector
    
        // Grid dimensions
        int m1_; // Number of S grid points - 1
        int m2_; // Number of V grid points - 1
        int m_;  // Total size (m1+1)*(m2+1)
    
        // Parameters
        double r_d_;  // Domestic interest rate
        double r_f_;  // Foreign interest rate
        int N_;       // Number of time steps
        double delta_t_; // Time step size
    
    public:
        // Constructor 
        BoundaryConditions_put(int m1, int m2, double r_d, double r_f, int N, double delta_t)
            : m1_(m1), m2_(m2), m_((m1+1)*(m2+1)), 
              r_d_(r_d), r_f_(r_f), N_(N), delta_t_(delta_t) {
            
            // Allocate views with proper Kokkos constructors
            b0_ = Kokkos::View<double*>("b0", m_);
            b1_ = Kokkos::View<double*>("b1", m_);
            b2_ = Kokkos::View<double*>("b2", m_);
            b_  = Kokkos::View<double*>("b", m_);
    
            // Initialize to zero
            Kokkos::deep_copy(b0_, 0.0);
            Kokkos::deep_copy(b1_, 0.0);
            Kokkos::deep_copy(b2_, 0.0);
            Kokkos::deep_copy(b_, 0.0);
        }
    
        // Initialize boundary conditions given stock price grid
        void initialize(const double K) {
            // Create host mirrors for all arrays
            auto h_b0 = Kokkos::create_mirror_view(b0_);
            auto h_b1 = Kokkos::create_mirror_view(b1_);
            auto h_b2 = Kokkos::create_mirror_view(b2_);
            auto h_b = Kokkos::create_mirror_view(b_);
    
            // Initialize to zero
            for(int i = 0; i < m_; i++) {
                h_b0[i] = 0.0;
                h_b1[i] = 0.0;
                h_b2[i] = 0.0;
            }
    
            // Initialize b1 (S direction boundary at S=0)
            for(int j = 0; j <= m2_; j++) {
                h_b1[j * (m1_ + 1)] =  0;//K * std::exp(-r_d_); 
            }
    
            // Combine boundaries on host
            for(int i = 0; i < m_; i++) {
                h_b[i] = h_b0[i] + h_b1[i] + h_b2[i];
            }
    
            // Copy all results to device
            Kokkos::deep_copy(b0_, h_b0);
            Kokkos::deep_copy(b1_, h_b1);
            Kokkos::deep_copy(b2_, h_b2);
            Kokkos::deep_copy(b_, h_b);
        }
    
        // Getters for boundary views
        Kokkos::View<double*> get_b0() const { return b0_; }
        Kokkos::View<double*> get_b1() const { return b1_; }
        Kokkos::View<double*> get_b2() const { return b2_; }
        Kokkos::View<double*> get_b() const { return b_; }
    
        int get_size() const { return m_; }
};


void test_boundary_conditions();

#endif // BOUNDARY_CONDITIONS_HPP