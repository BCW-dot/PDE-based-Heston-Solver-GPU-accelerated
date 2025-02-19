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

    // Function to generate synthetic market prices
    static void generate_market_data(
        const double S_0,          // Spot price
        const double T,            // Time to maturity
        const double r_d,          // Risk-free rate
        const std::vector<double>& strikes,  // Array of strikes
        Kokkos::View<double*>::HostMirror& h_market_prices  // Output market prices on host
    ) {
        // Fixed market volatility for synthetic data generation
        const double market_vol = 0.2;  
        std::cout << "Volatility Black Scholes: " << market_vol << std::endl;

        //std::cout << "prices " << std::endl;
        // Generate market prices using Black-Scholes
        for(size_t i = 0; i < strikes.size(); ++i) {
            const double K = strikes[i];
    
            h_market_prices(i) = call_price(1, S_0, K, r_d, market_vol, T);
            std::cout << h_market_prices(i) << ", ";
        }
    }
    
    static void generate_market_data_with_dividends(
        const double S_0,          // Initial spot price
        const double T,            // Time to maturity
        const double r_d,          // Risk-free rate
        const std::vector<double>& strikes,  // Array of strikes
        const std::vector<double>& dividend_dates,
        const std::vector<double>& dividend_amounts,
        const std::vector<double>& dividend_percentages,
        Kokkos::View<double*>::HostMirror& h_market_prices
    ) {
        const double market_vol = 0.2;  
        std::cout << "Volatility Black Scholes: " << market_vol << std::endl;
    
        // Calculate adjusted spot price
        double S_adjusted = S_0;
        for(size_t i = 0; i < dividend_dates.size(); ++i) {
            if(dividend_dates[i] < T) {  // Only consider dividends before maturity
                //std::cout<< "div applied at " << dividend_dates[i] << std::endl;
                // Fixed amount dividend
                S_adjusted -= dividend_amounts[i] * std::exp(-r_d * dividend_dates[i]);
                //std::cout<< "stock after cash " << S_adjusted << std::endl;
                // Percentage dividend
                S_adjusted -= (S_0 * dividend_percentages[i]) * std::exp(-r_d * dividend_dates[i]);
                //std::cout<< "stock after percentage " << S_adjusted << std::endl;
            }
        }
    
        //std::cout << "prices with dividends " << std::endl;
        // Generate prices using adjusted spot
        for(size_t i = 0; i < strikes.size(); ++i) {
            const double K = strikes[i];
        
            h_market_prices(i) = call_price(1, S_adjusted, K, r_d, market_vol, T);

            std::cout << h_market_prices(i) << ", ";
        }
    }
    

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
                std::cout << "Call Price " << C << std::endl;
                std::cout << "Vega " << V << std::endl;
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


