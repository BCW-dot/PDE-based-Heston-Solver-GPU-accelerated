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



