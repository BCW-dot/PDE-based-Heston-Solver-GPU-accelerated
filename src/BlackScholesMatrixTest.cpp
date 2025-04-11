#include "BlackScholesMatrixTest.hpp"
#include <iostream> //std
#include <string> //print vector
#include <chrono> //time 
#include <random> //random vector for testin

void print_vector(std::vector<double>& v, const std::string& name){
    std::cout<< "\n" << name << " = ";
    std::cout<< "{" << v[0] << ", ";
    for (int i = 1; i < int(v.size()) -1 ; i++){
        std::cout << v[i] << ", ";
    }
    std::cout<< v[v.size()-1] << "}";
    std::cout<< "\n";
}

void fill_random_vector(std::vector<double>& v){
    for(int i = 0; i<v.size(); i++){
        v[i] = static_cast<double>(rand())/RAND_MAX;
    }
}


//Tridiagonal matrix
BlackScholes_standard::BlackScholes_standard(int size){
    size_of_matrix = size;
    a = std::vector<double>(size_of_matrix-1, 0.0);
    b = std::vector<double>(size_of_matrix, 0.0);
    c = std::vector<double>(size_of_matrix-1, 0.0);

    // Create new vectors, not references
    c_star = std::vector<double>(size_of_matrix-1, 0.0);
    d_star = std::vector<double>(size_of_matrix, 0.0);
}

void BlackScholes_standard::build_black_scholes_matrix() {
    // Initialize all elements of b
    for(int i = 0; i < size_of_matrix; i++) {
        b[i] = 2;
    }
    
    // Initialize all elements of a and c
    for(int i = 0; i < size_of_matrix-1; i++) {
        a[i] = -1;
        c[i] = -1;
    }
}

void BlackScholes_standard::multiply(std::vector<double>& x, std::vector<double>& result) {
    
    // First row (special case)
    result[0] = b[0] * x[0] + c[0] * x[1];
    
    // Middle rows
    for(int i = 1; i < size_of_matrix-1; i++) {
        result[i] = a[i-1] * x[i-1] + b[i] * x[i] + c[i] * x[i+1];
    }
    
    // Last row (special case)
    result[size_of_matrix-1] = a[size_of_matrix-2] * x[size_of_matrix-2] + b[size_of_matrix-1] * x[size_of_matrix-1];
    
}

void BlackScholes_standard::solve(std::vector<double>& result, std::vector<double>& d) {
    // First forward substitution step
    c_star[0] = c[0] / b[0];
    d_star[0] = d[0] / b[0];

    // Forward substitution (note the changed loop bound)
    for (int i=1; i < size_of_matrix-1; i++) {
        double m = 1.0 / (b[i] - a[i-1] * c_star[i-1]);
        c_star[i] = c[i] * m;
        d_star[i] = (d[i] - a[i-1] * d_star[i-1]) * m;
    }

    // Handle the last element separately
    d_star[size_of_matrix-1] = (d[size_of_matrix-1] - a[size_of_matrix-1-1] * d_star[size_of_matrix-1-1]) / 
                                (b[size_of_matrix-1] - a[size_of_matrix-1-1] * c_star[size_of_matrix-1-1]);

    // Backward substitution
    result[size_of_matrix-1] = d_star[size_of_matrix-1];
    
    for (int i=size_of_matrix-2; i >= 0; i--) {
        result[i] = d_star[i] - c_star[i] * result[i+1];
    }
}



void test_black_scholes_matrix() {
    std::cout << "Starting BSM matrix comparrison Tests " << std::endl;

    int problem_size = 10000;
    std::cout << "Standard BSM Matrix size: " << problem_size << std::endl;

    // Create a right-hand side vector b
    std::vector<double> b(problem_size);
    fill_random_vector(b);

    // Create a solution vector that will hold x (after solving)
    std::vector<double> x(problem_size, 0.0);

    BlackScholes_standard bsm(problem_size);
    bsm.build_black_scholes_matrix();

    // Save original b for later comparison
    std::vector<double> original_b = b;

    // IMPLICIT STEP: Solve Ax = b for x
    auto start = std::chrono::high_resolution_clock::now();
    bsm.solve(x, b);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end-start);
    std::cout << "Execution time of standard BSM matrix implicit step: " << duration.count() << "ms" << std::endl;

    // EXPLICIT STEP: Compute A*x
    std::vector<double> Ax(problem_size, 0.0);
    start = std::chrono::high_resolution_clock::now();
    bsm.multiply(x, Ax);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end-start);
    std::cout << "Execution time of standard BSM matrix explicit step: " << duration.count() << "ms" << std::endl;

    // Calculate ||Ax - b||
    double norm = 0.0;
    for (int i = 0; i < problem_size; i++) {
        double diff = Ax[i] - original_b[i];
        norm += diff * diff;
    }
    norm = std::sqrt(norm);

    std::cout << "Residual norm ||Ax - b|| = " << norm << std::endl;
    std::cout << "Ending BSM matrix comparrison Tests " << std::endl;
}








