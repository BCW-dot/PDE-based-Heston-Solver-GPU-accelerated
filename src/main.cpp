// src/main.cpp


#include "grid.hpp"

#include "hes_mat_fac.hpp"
#include "hes_A2_mat.hpp"
#include "BoundaryConditions.hpp"

#include "solver.hpp"

#include "hes_a1_kernels.hpp"
#include "hes_a2_shuffled_kernels.hpp"
#include "hes_a0_kernels.hpp"

#include "device_solver.hpp"
#include "jacobian_computation.hpp"
#include "heston_calibration.hpp"
#include "perfomance_test.hpp"

#include "MC_hes.hpp"
#include "bs.hpp"


#include <iostream>



int main(int argc, char* argv[]) {
    // Print initial message
    std::cout << "Starting various Tests with Kokkos..." << std::endl;
    
    //test_grid();

    //test_hes_mat_fac();
    //test_heston_A2_mat();

    //test_boundary_conditions();

    test_DO_scheme();

    /*
    
    Here come the test for the device class. We need these for the calibration code

    */
    //test_a1_kernel();
    //test_a2_shuffled_kernel();
    //test_a0_kernel();

    /*
    
    Test for a device callable class

    */
    //test_device_class();
    //test_heston_calibration();
    //test_jacobian_computation();
    //test_perfomance_Tests();

    /*
    
    Test helper code
    
    */
    //test_Monte_Carlo_Heston();
    //test_black_scholes();


    std::cout << "Tests have completed successfully." << std::endl;
    

    return 0;
}