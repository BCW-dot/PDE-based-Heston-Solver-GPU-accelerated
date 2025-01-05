#include <vector>
#include <cmath>
#include <iostream>

#include "grid.hpp"

double Grid::Map_s(double xi, double K, double c) {
    return K + c * std::sinh(xi);
}

double Grid::Map_v(double xi, double d) {
    return d * std::sinh(xi);
}

// Main grid generation function
Grid::Grid(int m1, double S, double S_0, double K, double c, 
           int m2, double V, double V_0, double d) {
        
    // Initialize vectors with correct sizes
    Vec_s.resize(m1 + 1);
    Delta_s.resize(m1);
    Vec_v.resize(m2 + 1);
    Delta_v.resize(m2);

    // Generate stock price grid
    double Delta_xi = (1.0 / m1) * (std::asinh((S - K) / c) - std::asinh(-K / c));
    std::vector<double> Uniform_s(m1 + 1);
    
    for(int i = 0; i <= m1; ++i) {
        Uniform_s[i] = std::asinh(-K / c) + i * Delta_xi;
        Vec_s[i] = Map_s(Uniform_s[i], K, c);
    }
    
    // Insert S_0 and sort
    Vec_s.push_back(S_0);
    std::sort(Vec_s.begin(), Vec_s.end());
    Vec_s.pop_back(); // Remove last element

    // Calculate Delta_s
    for(int i = 0; i < m1; ++i) {
        Delta_s[i] = Vec_s[i + 1] - Vec_s[i];
    }

    // Generate variance grid
    double Delta_eta = (1.0 / m2) * std::asinh(V / d);
    std::vector<double> Uniform_v(m2 + 1);
    
    for(int i = 0; i <= m2; ++i) {
        Uniform_v[i] = i * Delta_eta;
        Vec_v[i] = Map_v(Uniform_v[i], d);
    }

    // Insert V_0 and sort
    Vec_v.push_back(V_0);
    std::sort(Vec_v.begin(), Vec_v.end());
    Vec_v.pop_back(); // Remove last element

    // Calculate Delta_v
    for(int i = 0; i < m2; ++i) {
        Delta_v[i] = Vec_v[i + 1] - Vec_v[i];
    }
}

// Helper function to create test grid
Grid create_test_grid(int m1, int m2) {
   double S_0 = 100.0;
   double K = 100.0;
   double S = 8 * K;
   double c = K / 5.0;
   
   double V = 5.0;
   double V_0 = 0.04;
   double d = V / 500.0;
   
   return Grid(m1, S, S_0, K, c, m2, V, V_0, d);
}

//this test waas checked against the python implementation from vizual outputs
void test_grid(){
    // Example parameters
    int m1 = 25;
    double S_0 = 100.0;
    double K = 100.0;
    double S = 8 * K;
    double c = K / 5.0;

    int m2 = 25;
    double V = 5.0;
    double V_0 = 0.04;
    double d = V / 500.0;

    // Create grid
    Grid grid(m1, S, S_0, K, c, m2, V, V_0, d);

    // Access grid points
    std::cout << "\nStock Dimesnion: " << grid.Vec_s.size() << "==" << m1+1 << std::endl;
    std::cout << "\nStock prices:\n";
    for(int i = 0; i < grid.Vec_s.size(); i++) {
        std::cout << "[" << i << "] = " << grid.Vec_s[i] << ",";
    }

    std::cout << "\nStock Delta Dim: " << grid.Delta_s.size() << "==" << m1 << std::endl;
    std::cout << "\nStock delta:\n";
    for(int i = 0; i < grid.Delta_s.size(); i++) {
        std::cout << "[" << i << "] = " << grid.Delta_s[i] << ",";
    }


    std::cout << "\nVariance Dimesnion: " << grid.Vec_v.size() << "==" << m2+1 << std::endl;
    std::cout << "\nVariance values:\n";
    for(int i = 0; i < grid.Vec_v.size(); i++) {
        std::cout << "[" << i << "] = " << grid.Vec_v[i] << ",";
    }

    std::cout << "\nVariance Delta Dim: " << grid.Delta_v.size() << "==" << m2 << std::endl;
    std::cout << "\nVariance delta:\n";
    for(int i = 0; i < grid.Delta_v.size(); i++) {
        std::cout << "[" << i << "] = " << grid.Delta_v[i] << ",";
    }

}










