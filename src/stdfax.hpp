#pragma once

#include <Kokkos_Core.hpp>
#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_spmv.hpp>
#include <KokkosSparse_gauss_seidel.hpp>
#include <KokkosBlas1_nrm2.hpp>

#include <chrono>
#include <vector>
#include <iomanip>

//KokkosKernel headers
#include <KokkosKernels_default_types.hpp>
#include "KokkosKernels_Handle.hpp"


// Type definitions for better readability
using scalar_t = double;
using lno_t = int;
using size_type = int;
using layout_t = Kokkos::LayoutLeft;
using exec_space = Kokkos::DefaultExecutionSpace;
using memory_space = exec_space::memory_space;
using device_t = Kokkos::Device<exec_space, memory_space>;

using handle_t = KokkosKernels::Experimental::KokkosKernelsHandle
    <size_type, lno_t, scalar_t, exec_space, memory_space, memory_space>;
