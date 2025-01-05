// src/heston_adi.hpp
#ifndef HESTON_ADI_HPP
#define HESTON_ADI_HPP

#include <Kokkos_Core.hpp>
// Replace KokkosSparse headers with basic Kokkos for now
#include <Kokkos_Random.hpp>

// Declare the main solver function
void heston_adi();


#endif // HESTON_ADI_HPP