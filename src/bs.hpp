// bs.hpp
#pragma once
#include <cmath>
#include <iostream>

//THIS IS WRONG; THE CALL PRICES ARE COMPUTED NOT CORRECTLY I STRONGLY BELIEF
class BlackScholes {
private:
    // Constants for normal CDF approximation
    static constexpr double a1 = 0.254829592;
    static constexpr double a2 = -0.284496736;
    static constexpr double a3 = 1.421413741;
    static constexpr double a4 = -1.453152027;
    static constexpr double a5 = 1.061405429;
    static constexpr double p = 0.3275911;

    // Helper function for normal CDF approximation (Abramowitz and Stegun formula)
    static double norm_cdf(double x) {
        // Take absolute value of x
        double sign = 1.0;
        if (x < 0) {
            sign = -1.0;
            x = -x;
        }
        
        // Formula 7.1.26 from Abramowitz and Stegun
        double t = 1.0 / (1.0 + p * x);
        double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * std::exp(-x * x / 2.0);
        
        return 0.5 * (1.0 + sign * y);
    }

    // Helper function for d_j calculation
    static double d_j(int j, double S, double K, double r, double v, double T) {
        return (std::log(S / K) + (r + std::pow(-1.0, j-1) * 0.5 * v * v) * T) / (v * std::sqrt(T));
    }

public:
    // Calculate call option price
    
    static double call_price(int CP, double S, double K, double r, double v, double T) {
        const double sqrt_T = std::sqrt(T);
        const double log_SK = std::log(S/K);
        const double vol_sqrt_T = v * sqrt_T;
        
        const double d1 = (log_SK + (r + 0.5 * v * v) * T) / vol_sqrt_T;
        const double d2 = d1 - vol_sqrt_T;
        
        return S * std::erfc(-d1/std::sqrt(2.0))/2.0 
               - K * std::exp(-r * T) * std::erfc(-d2/std::sqrt(2.0))/2.0;
    }
               
    
    /*
    static double call_price(int CP, double S, double K, double r, double v, double T) {
        double d1 = d_j(1, S, K, r, v, T);
        double d2 = d_j(2, S, K, r, v, T);

        return CP * (S * norm_cdf(CP * d1) - K * std::exp(-r * T) * norm_cdf(CP * d2));
    }
    */
    
   

    // Calculate vega
    static double call_vega(int CP, double S, double K, double r, double v, double T) {
        double d = d_j(1, S, K, r, v, T);
        return CP * S * std::exp(-d * d / 2.0) * std::sqrt(T / (2.0 * M_PI));
    }
    

    // Reverse BS using dichotomy method
    static double reverse_BS_dic(int CP, double S, double K, double r, double T, 
        double C_target, double epsilon, double a, double b) {
        const int MAX_ITER = 1000;
        int iter = 0;

        double x = (b + a) / 2;
        double C = call_price(CP, S, K, r, x, T);

        while(std::abs(C - C_target) > epsilon && iter < MAX_ITER) {
            C = call_price(CP, S, K, r, x, T);

            if(C > C_target) {
                b = x;
            } 

            else {
                a = x;
            }

            x = (b + a) / 2;
            iter++;
        }

        if(iter >= MAX_ITER) {
            std::cout << "Warning: Maximum iterations reached in bisection method\n";
            std::cout << "Final error: " << std::abs(C - C_target) << "\n";
        }

        return x;
    }


        

    // Newton method for implied volatility with fallback to dichotomy
    static double reverse_BS(int CP, double S, double K, double r, double T, 
        double v_0, double C_target, double epsilon) {
        double x = v_0;
        double C = call_price(CP, S, K, r, x, T);
        bool fail = false;

        while (std::abs(C - C_target) > epsilon) {
            C = call_price(CP, S, K, r, x, T);
            double V = call_vega(CP, S, K, r, x, T);

            if (std::abs(V) < 1e-10) {
                fail = true;
                std::cout << "Newton method fails for S = " << S << "strike" << K << std::endl;
                break;
            }
            x -= (C - C_target) / V;
        }

        if (fail) {
            std::cout << "Keep reversing BS using dichotomy method." << std::endl;
            double a = 0.001;
            double b = 1.0;
            x = reverse_BS_dic(CP, S, K, r, T, C_target, epsilon, a, b);
        }

        return x;
    }

};


