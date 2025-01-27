#pragma once
#include <Kokkos_Core.hpp>
#include "grid_pod.hpp"


template<class DeviceType>
struct Device_BoundaryConditions {
    typedef DeviceType execution_space;
    typedef typename DeviceType::memory_space memory_space;
    
    Kokkos::View<double*, DeviceType> b0_;
    Kokkos::View<double*, DeviceType> b1_;
    Kokkos::View<double*, DeviceType> b2_;
    Kokkos::View<double*, DeviceType> b_;
    
    int m1_, m2_, m_;
    double r_d_, r_f_;
    int N_;
    double delta_t_;
    
    KOKKOS_FUNCTION
    Device_BoundaryConditions() = default;
    
    Device_BoundaryConditions(int m1, int m2, double r_d, double r_f, int N, double delta_t) 
        : m1_(m1), m2_(m2), m_((m1+1)*(m2+1)), 
          r_d_(r_d), r_f_(r_f), N_(N), delta_t_(delta_t) {
        
        b0_ = Kokkos::View<double*, DeviceType>("b0", m_);
        b1_ = Kokkos::View<double*, DeviceType>("b1", m_);
        b2_ = Kokkos::View<double*, DeviceType>("b2", m_);
        b_  = Kokkos::View<double*, DeviceType>("b", m_);

        Kokkos::deep_copy(b0_, 0.0);
        Kokkos::deep_copy(b1_, 0.0);
        Kokkos::deep_copy(b2_, 0.0);
        Kokkos::deep_copy(b_, 0.0);
    }
    
    template<class GridType>
    KOKKOS_FUNCTION
    void initialize(const GridType& grid,
                   const Kokkos::TeamPolicy<>::member_type& team) {
        // Zero initialization
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, m_),
            [&](const int i) {
                b0_(i) = 0.0;
                b1_(i) = 0.0;
                b2_(i) = 0.0;
                b_(i) = 0.0;
            });
        team.team_barrier();

        // Initialize b1
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, m2_ + 1),
            [&](const int j) {
                const double exp_factor = Kokkos::exp(-r_f_ * delta_t_ * (N_ - 1));
                b1_(m1_ * (j + 1)) = (r_d_ - r_f_) * grid.device_Vec_s[m1_] * exp_factor;
            });
        team.team_barrier();

        // Initialize b2
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, m1_ + 1),
            [&](const int i) {
                const double exp_factor = Kokkos::exp(-r_f_ * delta_t_ * (N_ - 1));
                b2_(m_ - m1_ - 1 + i) = -0.5 * r_d_ * grid.device_Vec_s[i] * exp_factor;
            });
        team.team_barrier();

        // Combine boundaries
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, m_),
            [&](const int i) {
                b_(i) = b0_(i) + b1_(i) + b2_(i);
            });
        team.team_barrier();
    }
};