#pragma once
#include <Kokkos_Core.hpp>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>

struct GridViews {
  Kokkos::View<double*> device_Vec_s;
  Kokkos::View<double*> device_Vec_v;
  Kokkos::View<double*> device_Delta_s;
  Kokkos::View<double*> device_Delta_v;

  // optional dimension fields for convenience
  int m1, m2;

  // Possibly default constructor etc.
  GridViews() = default;

  Kokkos::View<double*> temp_v;  // For rebuilding variance grid

  // Add new device-callable method. This is needed or the calibration step V0+eps
    template<class TeamMember>
    KOKKOS_FUNCTION
    void rebuild_variance_views(const double V_0_new, const double V, const double d,
                              const TeamMember& team) {
        
        // Build initial grid
        const double Delta_eta = (1.0 / m2) * Kokkos::asinh(V / d);
        
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, m2 + 1),
            [&](const int i) {
                const double xi = i * Delta_eta;
                temp_v(i) = d * Kokkos::sinh(xi);
            });
        team.team_barrier();
        
        // Add V_0 at end
        if(team.team_rank() == 0) {
            temp_v(m2 + 1) = V_0_new;
        }
        team.team_barrier();

        // Sort the array (can implement a parallel sort if needed)
        // For now using a simple bubble sort since it's device-callable
        for(int i = 0; i < m2 + 2; i++) {
            for(int j = 0; j < m2 + 1 - i; j++) {
                if(temp_v(j) > temp_v(j + 1)) {
                    double temp = temp_v(j);
                    temp_v(j) = temp_v(j + 1);
                    temp_v(j + 1) = temp;
                }
            }
        }
        team.team_barrier();

        // Copy sorted values (excluding last element) to device_Vec_v
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, m2 + 1),
            [&](const int i) {
                device_Vec_v(i) = temp_v(i);
            });
        team.team_barrier();

        // Rebuild Delta_v
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, m2),
            [&](const int i) {
                device_Delta_v(i) = device_Vec_v(i + 1) - device_Vec_v(i);
            });
        team.team_barrier();
    }

    KOKKOS_INLINE_FUNCTION
    int find_v0_index(const double V_0){
        // If exact match not found, find closest point
        int closest_idx = 0;
        for(int i = 0; i <= m2; i++) {
            if(Kokkos::abs(device_Vec_v(i) - V_0) < 1e-10) {
                closest_idx = i;  // Store s index
                break;
            }
        }
        
        return closest_idx;
    }



};

//this function is the "constructor" for the struct GridViews
inline void buildMultipleGridViews(std::vector<GridViews> &hostGrids, int nInstances, int m1, int m2)
{
    // Resize to hold nInstances
    hostGrids.resize(nInstances);

    for(int i = 0; i < nInstances; i++) {
        // 1) Allocate device arrays for each PDE dimension
        hostGrids[i].device_Vec_s = Kokkos::View<double*>("vec_s",    m1+1);
        hostGrids[i].device_Vec_v = Kokkos::View<double*>("vec_v",    m2+1);
        hostGrids[i].device_Delta_s = Kokkos::View<double*>("delta_s",m1);
        hostGrids[i].device_Delta_v = Kokkos::View<double*>("delta_v",m2);

        hostGrids[i].temp_v = Kokkos::View<double*>("temp_v", m2 + 2);

        hostGrids[i].m1 = m1;
        hostGrids[i].m2 = m2;
    }
}



