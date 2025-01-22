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

        hostGrids[i].m1 = m1;
        hostGrids[i].m2 = m2;
    }
}



