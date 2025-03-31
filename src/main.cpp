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
    
    /*
    Tests the Grid construction
    */
    //test_grid();

    //Both tests are for gpu accalerated CPU code
    /*
    Tests the explicit and implicit, as well as matrix construction of A0 and A1 matrix
    */
    //test_hes_mat_fac();
    /*
    Tests the sequentiall A2 and shuffled A2 matrix 
    */
    //test_heston_A2_mat();

    /*
    Tests the boundary condition for a european call
    */
    //test_boundary_conditions();

    /*
    Tests the Douglas scheme for various option types, on the CPU with GPU accaleration
    */
    test_DO_scheme();

    /*
    
    Here come the test for the device class. We need these for the calibration code

    */
    /*
    Each tests the explicict and implicit implementation on a completely GPU code
    */
    //test_a1_kernel();
    //test_a2_shuffled_kernel();
    //test_a0_kernel();

    /*
    
    Test for a device callable class

    */
    /*
    Tests in device_solver.cpp, which test the GPU Douglas scheme (parallising Option solvers)
    */
    //test_device_class();
    /*
    Tests the calibration to various option data of the Heston model. Uses GPU accaleration
    */
    //test_heston_calibration();
    /*
    Tests the Jacobian computation, needed in the LVBM algorithm
    */
    //test_jacobian_computation();
    /*
    A perfomance test for the device_solver.cpp file. Here we can compare architectures to one another 
    */
    //test_perfomance_Tests();

    /*
    
    Test helper code
    
    */
    /*
    Comnpute MC Heston price for comparrison, only works if Feller Condition is fullfilled
    */
    //test_Monte_Carlo_Heston();
    /*
    Closed form Black Scholes implementation, used for market data generation, implied vol computation
    */
    //test_black_scholes();


    std::cout << "Tests have completed successfully." << std::endl;
    

    return 0;
}