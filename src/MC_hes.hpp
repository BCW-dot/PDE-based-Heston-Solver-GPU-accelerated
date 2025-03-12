#ifndef MC_hes_HPP
#define MC_hes_HPP

#include <vector>
#include <string>

// Constants
const int N_DAYS_PER_YEAR = 350;
const double NORMAL_SCORE = 1.96; // For 95% confidence interval

class HestonEuropeanCall {
private:
    double S_0;      // Initial stock price
    double V_0;      // Initial volatility
    double K;        // Strike price

    double r_d;        // Domestic Risk-free rate
    double r_f;        // Foreign Risk-free rate

    double kappa;    // Rate of mean reversion
    double eta;    // Long-term volatility
    double sigma;    // Volatility of volatility
    double rho;      // Correlation between stock and volatility processes

    double T;        // Time to maturity in years
    int M;           // Number of Monte Carlo simulations

    // Private implementation methods
    std::vector<double> priceOptionStandard();
    std::vector<double> priceOptionWithReflection();

public:
    // Constructor
    HestonEuropeanCall(double initial_price, double initial_vol, double strike, double d_risk_free_rate,
                        double f_risk_free_rate,double mean_reversion, double long_term_vol, double vol_of_vol, 
                        double correlation, double maturity, int num_simulations);

    // Check Feller condition
    bool checkFellerCondition() const;

    // Price the option, automatically selecting the appropriate method
    std::vector<double> priceOption();
    
    // Force a specific pricing method
    std::vector<double> priceOption(const std::string& method);
};


double compute_MC_heston_call_price(
    double S_0,      // Initial stock price
    double V_0,      // Initial volatility (squared)
    double K,        // Strike price
    double r_d,      // Domestic risk-free rate
    double r_f,      // Foreign risk-free rate
    double rho,      // Correlation
    double sigma,    // Volatility of volatility
    double kappa,    // Rate of mean reversion
    double eta,      // Long-term volatility (squared)
    double T,        // Time to maturity
    int M = 100000           // Number of simulations
);

void test_Monte_Carlo_Heston();

#endif 