#pragma once
#include <iostream>
#include <vector>

//for calibration
#include <Kokkos_Core.hpp>




double delta_s(int i, int pos, const std::vector<double>& Delta_s);
double delta_v(int i, int pos, const std::vector<double>& Delta_v);

double beta_s(int i, int pos, const std::vector<double>& Delta_s);
double beta_v(int i, int pos, const std::vector<double>& Delta_v);

double alpha_s(int i, int pos, const std::vector<double>& Delta_s);
double alpha_v(int i, int pos, const std::vector<double>& Delta_v);

double gamma_s(int i, int pos, const std::vector<double>& Delta_s);
double gamma_v(int i, int pos, const std::vector<double>& Delta_v);

//for calibration
KOKKOS_FUNCTION
double device_delta_s(int i, int pos, const Kokkos::View<double*>& Delta_s);
KOKKOS_FUNCTION
double device_delta_v(int i, int pos, const Kokkos::View<double*>& Delta_v);

KOKKOS_FUNCTION
double device_beta_s(int i, int pos, const Kokkos::View<double*>& Delta_s);
KOKKOS_FUNCTION
double device_beta_v(int i, int pos, const Kokkos::View<double*>& Delta_v);

KOKKOS_FUNCTION
double device_alpha_s(int i, int pos, const Kokkos::View<double*>& Delta_s);
KOKKOS_FUNCTION
double device_alpha_v(int i, int pos, const Kokkos::View<double*>& Delta_v);

KOKKOS_FUNCTION
double device_gamma_s(int i, int pos, const Kokkos::View<double*>& Delta_s);
KOKKOS_FUNCTION
double device_gamma_v(int i, int pos, const Kokkos::View<double*>& Delta_v);