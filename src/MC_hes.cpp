#include "MC_hes.hpp"

#include <cmath>
#include <ctime>
#include <iostream>
#include <algorithm>
#include <iomanip>

// Constructor implementation
HestonEuropeanCall::HestonEuropeanCall(double initial_price, double initial_vol, double strike, double risk_free_rate,
                 double mean_reversion, double long_term_vol, double vol_of_vol, double correlation,
                 double maturity, int num_simulations) 
    : S_0(initial_price), V_0(initial_vol), K(strike), r(risk_free_rate),
      kappa(mean_reversion), theta(long_term_vol), sigma(vol_of_vol), rho(correlation),
      T(maturity), M(num_simulations) {}

// Check Feller condition implementation
bool HestonEuropeanCall::checkFellerCondition() const {
    return (2 * kappa * theta > sigma * sigma);
}

// Main pricing method that automatically selects the appropriate implementation
std::vector<double> HestonEuropeanCall::priceOption() {
    if (checkFellerCondition()) {
        std::cout << "Using standard Heston model (Feller condition satisfied)." << std::endl;
        return priceOptionStandard();
    } else {
        std::cout << "Using Heston model with reflection (Feller condition violated)." << std::endl;
        return priceOptionWithReflection();
    }
}

// Method to force a specific pricing method
std::vector<double> HestonEuropeanCall::priceOption(const std::string& method) {
    if (method == "standard") {
        std::cout << "Forcing standard Heston model." << std::endl;
        return priceOptionStandard();
    } else if (method == "reflection") {
        std::cout << "Forcing Heston model with reflection." << std::endl;
        return priceOptionWithReflection();
    } else {
        std::cerr << "Unknown method specified. Using automatic selection." << std::endl;
        return priceOption();
    }
}

// Standard Heston model implementation (when Feller condition is satisfied)
std::vector<double> HestonEuropeanCall::priceOptionStandard() {
    std::clock_t start_time = clock();
    double sum_price = 0.0;
    double sum_2_price = 0.0;
    double discount_factor = exp(-r * T);
    int N = N_DAYS_PER_YEAR;
    double dt = T / static_cast<double>(N);

    for (int i = 0; i < M; ++i) {
        double S = S_0;
        double V = V_0;
        
        for (int j = 0; j < N; ++j) {
            // Generate random normal variables
            double X_1 = static_cast<double>(rand()) / RAND_MAX;
            double X_2 = static_cast<double>(rand()) / RAND_MAX;
            double Z_1 = sqrt(std::abs(2 * log(X_1))) * sin(2 * M_PI * X_2);
            double Z_2 = sqrt(std::abs(2 * log(X_1))) * cos(2 * M_PI * X_2);
            
            // Correlated Wiener processes
            double dW_v = sqrt(dt) * Z_1;
            double dW_s = sqrt(dt) * (rho * Z_1 + sqrt(1 - rho * rho) * Z_2);
            
            // Update volatility
            double d_V = kappa * (theta - V) * dt + sigma * sqrt(std::max(V, 0.0)) * dW_v;
            
            // Update stock price
            double d_S = r * S * dt + S * sqrt(std::max(V, 0.0)) * dW_s;
            
            V += d_V;
            S += d_S;
        }
        
        // Calculate payoff and discount
        double payoff = std::max(S - K, 0.0);
        double discounted_payoff = discount_factor * payoff;
        
        sum_price += discounted_payoff;
        sum_2_price += std::pow(discounted_payoff, 2);
    }
    
    // Calculate mean and variance
    double mean_price = sum_price / static_cast<double>(M);
    double variance = ((1.0 / static_cast<double>(M)) * sum_2_price - mean_price * mean_price) / static_cast<double>(M);
    
    // Calculate confidence interval and execution time
    double conf_interval = NORMAL_SCORE * sqrt(variance / M);
    double execution_time = (std::clock() - start_time) / static_cast<double>(CLOCKS_PER_SEC);
    
    // Return price, confidence interval, and execution time
    return {mean_price, conf_interval, execution_time};
}

// Heston model with reflection implementation (when Feller condition is violated)
// Heston model with reflection implementation (when Feller condition is violated)
std::vector<double> HestonEuropeanCall::priceOptionWithReflection() {
    std::clock_t start_time = clock();
    double sum_price = 0.0;
    double sum_2_price = 0.0;
    double discount_factor = exp(-r * T);
    int N = N_DAYS_PER_YEAR;
    double dt = T / static_cast<double>(N);

    for (int i = 0; i < M; ++i) {
        double S = S_0;
        double V = V_0;
        
        for (int j = 0; j < N; ++j) {
            // Generate random normal variables
            double X_1 = static_cast<double>(rand()) / RAND_MAX;
            double X_2 = static_cast<double>(rand()) / RAND_MAX;
            double Z_1 = sqrt(std::abs(2 * log(X_1))) * sin(2 * M_PI * X_2);
            double Z_2 = sqrt(std::abs(2 * log(X_1))) * cos(2 * M_PI * X_2);
            
            // Correlated Wiener processes
            double dW_v = sqrt(dt) * Z_1;
            double dW_s = sqrt(dt) * (rho * Z_1 + sqrt(1 - rho * rho) * Z_2);

            // Update volatility with proper treatment when Feller condition is violated
            double drift = kappa * (theta - V) * dt;
            double diffusion = sigma * sqrt(std::max(V, 0.0)) * dW_v;
            double d_V = drift + diffusion;
            
            // Apply absorption at zero instead of reflection
            V = std::max(V + d_V, 0.0);
            
            // Update stock price
            double d_S = r * S * dt + S * sqrt(V) * dW_s;
            S += d_S;
        }
        
        // Calculate payoff and discount
        double payoff = std::max(S - K, 0.0);
        double discounted_payoff = discount_factor * payoff;
        
        sum_price += discounted_payoff;
        sum_2_price += std::pow(discounted_payoff, 2);
    }
    
    // Calculate mean and variance
    double mean_price = sum_price / static_cast<double>(M);
    double variance = ((1.0 / static_cast<double>(M)) * sum_2_price - mean_price * mean_price) / static_cast<double>(M);
    
    // Calculate confidence interval and execution time
    double conf_interval = NORMAL_SCORE * sqrt(variance / M);
    double execution_time = (std::clock() - start_time) / static_cast<double>(CLOCKS_PER_SEC);
    
    // Return price, confidence interval, and execution time
    return {mean_price, conf_interval, execution_time};
}

/*

Monte Carlo

*/
void test_MC(){
    // Initialize random seed
    srand(static_cast<unsigned int>(time(nullptr)));
    
    std::cout << std::fixed << std::setprecision(6);

    double S_0 = 100.0;    // Initial stock price
    double V_0 = 0.04;     // Initial volatility (squared)

    double K = 100.0;      // Strike price

    double r = 0.025;       // Risk-free rate

    double rho = -0.9;     // Correlation
    double sigma = 0.6;    // Volatility of volatility (2*kappa*theta = 0.16 > sigma^2 = 0.09)
    double kappa = 0.5;    // Rate of mean reversion
    double eta = 0.04;   // Long-term volatility (squared)
    

    double T = 1.0;        // Time to maturity (1 year)
    int M = 100000;         // Number of simulations
    
    // Create and price the option
    HestonEuropeanCall option(S_0, V_0, K, r, kappa, eta, sigma, rho, T, M);
    
    // Auto-select method based on Feller condition
    std::cout << "\nAuto-selected method:" << std::endl;
    std::vector<double> result = option.priceOption();
    
    // Output results
    std::cout << "European Call Option Price: " << result[0] << std::endl;
    std::cout << "95% Confidence Interval: [" << result[0] - result[1] << ", " << result[0] + result[1] << "]" << std::endl;
    std::cout << "Execution Time: " << result[2] << " seconds" << std::endl;
}

void test_Monte_Carlo_Heston(){
    test_MC();
}