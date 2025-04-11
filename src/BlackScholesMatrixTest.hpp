#pragma once

#include <vector>



class BlackScholes_standard{
public:
    std::vector<double> a,b,c;
    std::vector<double> c_star, d_star;
    int size_of_matrix;

    //basic constructor, inits three vectors for lower, main, upper diagonal
    BlackScholes_standard(int size);

    //builds a first implementation of the matrix for debugging and testing purposes
    void build_black_scholes_matrix();

    void multiply(std::vector<double>& x, std::vector<double>& result);

    void solve(std::vector<double>& result, std::vector<double>& b);

};

void test_black_scholes_matrix();