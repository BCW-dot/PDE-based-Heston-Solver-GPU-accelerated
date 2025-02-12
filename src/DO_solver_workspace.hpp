#pragma once
#include <Kokkos_Core.hpp>

template<class Device>
struct DO_Workspace {
    // Main solution arrays
    Kokkos::View<double**, Device> U;
    Kokkos::View<double**, Device> Y_0;
    Kokkos::View<double**, Device> Y_1;
    
    // Results arrays
    Kokkos::View<double**, Device> A0_result;
    Kokkos::View<double**, Device> A1_result;
    Kokkos::View<double**, Device> A2_result_unshuf;
    
    // Shuffled arrays
    Kokkos::View<double**, Device> U_shuffled;
    Kokkos::View<double**, Device> Y_1_shuffled;
    Kokkos::View<double**, Device> A2_result_shuffled;
    Kokkos::View<double**, Device> U_next_shuffled;

    // Add lambda for American options
    Kokkos::View<double**, Device> lambda_bar;
    // For dividend processing
    Kokkos::View<double**, Device> U_temp;

    DO_Workspace(int nInstances, int total_size) {
        U = Kokkos::View<double**, Device>("U", nInstances, total_size);
        Y_0 = Kokkos::View<double**, Device>("Y_0", nInstances, total_size);
        Y_1 = Kokkos::View<double**, Device>("Y_1", nInstances, total_size);

        A0_result = Kokkos::View<double**, Device>("A0_result", nInstances, total_size);
        A1_result = Kokkos::View<double**, Device>("A1_result", nInstances, total_size);
        A2_result_unshuf = Kokkos::View<double**, Device>("A2_result_unshuf", nInstances, total_size);

        U_shuffled = Kokkos::View<double**, Device>("U_shuffled", nInstances, total_size);
        Y_1_shuffled = Kokkos::View<double**, Device>("Y_1_shuffled", nInstances, total_size);
        A2_result_shuffled = Kokkos::View<double**, Device>("A2_result_shuffled", nInstances, total_size);
        U_next_shuffled = Kokkos::View<double**, Device>("U_next_shuffled", nInstances, total_size);

        lambda_bar = Kokkos::View<double**, Device>("lambda_bar", nInstances, total_size);
        U_temp = Kokkos::View<double**, Device>("U_temp", nInstances, total_size);
    }
};
