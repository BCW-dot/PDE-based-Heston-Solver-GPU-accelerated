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
// Device version of delta_s
KOKKOS_INLINE_FUNCTION
double device_delta_s(int i, int pos, const Kokkos::View<double*>& Delta_s) {
    if(pos == -1) {
        return 2 / (Delta_s(i) * (Delta_s(i) + Delta_s(i + 1)));
    } else if(pos == 0) {
        return -2 / (Delta_s(i) * Delta_s(i + 1));
    } else if(pos == 1) {
        return 2 / (Delta_s(i + 1) * (Delta_s(i) + Delta_s(i + 1)));
    }
    // Note: Can't throw exception in device code
    return 0.0;
}

// Device version of delta_v
KOKKOS_INLINE_FUNCTION
double device_delta_v(int i, int pos, const Kokkos::View<double*>& Delta_v) {
    if(pos == -1) {
        return 2 / (Delta_v(i) * (Delta_v(i) + Delta_v(i + 1)));
    } else if(pos == 0) {
        return -2 / (Delta_v(i) * Delta_v(i + 1));
    } else if(pos == 1) {
        return 2 / (Delta_v(i + 1) * (Delta_v(i) + Delta_v(i + 1)));
    }
    return 0.0;
}

// Device version of beta_s
KOKKOS_INLINE_FUNCTION
double device_beta_s(int i, int pos, const Kokkos::View<double*>& Delta_s) {
    if(pos == -1) {
        return -Delta_s(i + 1) / (Delta_s(i) * (Delta_s(i) + Delta_s(i + 1)));
    } else if(pos == 0) {
        return (Delta_s(i + 1) - Delta_s(i)) / (Delta_s(i) * Delta_s(i + 1));
    } else if(pos == 1) {
        return Delta_s(i) / (Delta_s(i + 1) * (Delta_s(i) + Delta_s(i + 1)));
    }
    return 0.0;
}

// Device version of beta_v
KOKKOS_INLINE_FUNCTION
double device_beta_v(int i, int pos, const Kokkos::View<double*>& Delta_v) {
    if(pos == -1) {
        return -Delta_v(i + 1) / (Delta_v(i) * (Delta_v(i) + Delta_v(i + 1)));
    } else if(pos == 0) {
        return (Delta_v(i + 1) - Delta_v(i)) / (Delta_v(i) * Delta_v(i + 1));
    } else if(pos == 1) {
        return Delta_v(i) / (Delta_v(i + 1) * (Delta_v(i) + Delta_v(i + 1)));
    }
    return 0.0;
}

// Device version of alpha_s
KOKKOS_INLINE_FUNCTION
double device_alpha_s(int i, int pos, const Kokkos::View<double*>& Delta_s) {
    if(pos == -2) {
        return Delta_s(i) / (Delta_s(i - 1) * (Delta_s(i - 1) + Delta_s(i)));
    } else if(pos == -1) {
        return (-Delta_s(i - 1) - Delta_s(i)) / (Delta_s(i - 1) * Delta_s(i));
    } else if(pos == 0) {
        return (Delta_s(i - 1) + 2 * Delta_s(i)) / (Delta_s(i) * (Delta_s(i - 1) + Delta_s(i)));
    }
    return 0.0;
}

// Device version of alpha_v
KOKKOS_INLINE_FUNCTION
double device_alpha_v(int i, int pos, const Kokkos::View<double*>& Delta_v) {
    if(pos == -2) {
        return Delta_v(i) / (Delta_v(i - 1) * (Delta_v(i - 1) + Delta_v(i)));
    } else if(pos == -1) {
        return (-Delta_v(i - 1) - Delta_v(i)) / (Delta_v(i - 1) * Delta_v(i));
    } else if(pos == 0) {
        return (Delta_v(i - 1) + 2 * Delta_v(i)) / (Delta_v(i) * (Delta_v(i - 1) + Delta_v(i)));
    }
    return 0.0;
}

// Device version of gamma_s
KOKKOS_INLINE_FUNCTION
double device_gamma_s(int i, int pos, const Kokkos::View<double*>& Delta_s) {
    if(pos == 0) {
        return (-2 * Delta_s(i + 1) - Delta_s(i + 2)) / (Delta_s(i + 1) * (Delta_s(i + 1) + Delta_s(i + 2)));
    } else if(pos == 1) {
        return (Delta_s(i + 1) + Delta_s(i + 2)) / (Delta_s(i + 1) * Delta_s(i + 2));
    } else if(pos == 2) {
        return -Delta_s(i + 1) / (Delta_s(i + 2) * (Delta_s(i + 1) + Delta_s(i + 2)));
    }
    return 0.0;
}

// Device version of gamma_v
KOKKOS_INLINE_FUNCTION
double device_gamma_v(int i, int pos, const Kokkos::View<double*>& Delta_v) {
    if(pos == 0) {
        return (-2 * Delta_v(i + 1) - Delta_v(i + 2)) / (Delta_v(i + 1) * (Delta_v(i + 1) + Delta_v(i + 2)));
    } else if(pos == 1) {
        return (Delta_v(i + 1) + Delta_v(i + 2)) / (Delta_v(i + 1) * Delta_v(i + 2));
    } else if(pos == 2) {
        return -Delta_v(i + 1) / (Delta_v(i + 2) * (Delta_v(i + 1) + Delta_v(i + 2)));
    }
    return 0.0;
}