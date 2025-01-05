cmake_minimum_required(VERSION 3.16)
project(ADIKokkos LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set KokkosKernels paths explicitly
set(KokkosKernels_ROOT /usr/local/kokkoskernels)
list(APPEND CMAKE_PREFIX_PATH ${KokkosKernels_ROOT})

# Find packages
find_package(Kokkos REQUIRED)
find_package(KokkosKernels REQUIRED)

# Collect all source files
file(GLOB SOURCES 
    "src/*.cpp"
    "src/*.hpp"
)

# Add source files to the executable
add_executable(heston_solver ${SOURCES})

# Include directories - add KokkosKernels include path explicitly
target_include_directories(heston_solver PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${KokkosKernels_ROOT}/include
)

# Link libraries
target_link_libraries(heston_solver 
    Kokkos::kokkos
    KokkosKernels::kokkoskernels
)

# If CUDA is enabled, set CUDA-specific properties
if(Kokkos_ENABLE_CUDA)
    message(STATUS "CUDA enabled for Kokkos")
    set_property(TARGET heston_solver PROPERTY CUDA_ARCHITECTURES 75)
    set_property(TARGET heston_solver PROPERTY CUDA_SEPARABLE_COMPILATION ON)
endif()

# Print information
message(STATUS "Executable will be built in: ${CMAKE_BINARY_DIR}")
message(STATUS "KokkosKernels include path: ${KokkosKernels_ROOT}/include")
