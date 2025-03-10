#include "bs.hpp"


void test_black_scholes() {
    std::cout << "\nTesting Black-Scholes Implementation\n";
    std::cout << "===================================\n\n";
    
    // Test parameters
    double S = 100.0;  // Current stock price
    double K = 100.0;  // Strike price
    double r = 0.025;  // Risk-free rate
    double T = 1.0;    // Time to maturity
    double v = 0.3;    // Volatility
    int CP = 1;        // Call option
    
    std::cout << "Parameters:\n";
    std::cout << "Stock price (S): " << S << "\n";
    std::cout << "Strike price (K): " << K << "\n";
    std::cout << "Risk-free rate (r): " << r << "\n";
    std::cout << "Time to maturity (T): " << T << "\n";
    std::cout << "Volatility (v): " << v << "\n\n";
    
    // Test 1: Call price calculation
    double call_price = BlackScholes::call_price(CP, S, K, r, v, T);
    std::cout << "Test 1 - Call Price Calculation\n";
    std::cout << "Call price: " << call_price << "\n";
    // For ATM option with these parameters, price should be roughly 11-13
    if (call_price > 11.0 && call_price < 13.0) {
        std::cout << "✓ Price is in expected range\n\n";
    } else {
        std::cout << "✗ Price is outside expected range\n\n";
    }
    
    // Test 2: Vega calculation
    double vega = BlackScholes::call_vega(CP, S, K, r, v, T);
    std::cout << "Test 2 - Vega Calculation\n";
    std::cout << "Vega: " << vega << "\n";
    // Vega should be positive and roughly 30-40 for ATM option
    if (vega > 0 && vega < 50.0) {
        std::cout << "✓ Vega is in expected range\n\n";
    } else {
        std::cout << "✗ Vega is outside expected range\n\n";
    }
    
    // Test 3: Implied volatility calculation (reverse engineering)
    double epsilon = 0.0001;
    double test_vol = BlackScholes::reverse_BS(CP, S, K, r, T, 0.5, call_price, epsilon);
    std::cout << "Test 3 - Implied Volatility Calculation\n";
    std::cout << "Original volatility: " << v << "\n";
    std::cout << "Calculated implied volatility: " << test_vol << "\n";
    if (std::abs(test_vol - v) < epsilon) {
        std::cout << "✓ Implied volatility matches original within tolerance\n\n";
    } else {
        std::cout << "✗ Implied volatility calculation failed\n\n";
    }
    
    // Test 4: Edge cases
    std::cout << "Test 4 - Edge Cases\n";
    
    // Deep ITM option
    double itm_price = BlackScholes::call_price(CP, 150.0, K, r, v, T);
    std::cout << "Deep ITM price (S=150): " << itm_price << "\n";
    if (itm_price > 50.0) {  // Should be at least intrinsic value
        std::cout << "✓ Deep ITM price is reasonable\n";
    } else {
        std::cout << "✗ Deep ITM price seems incorrect\n";
    }
    
    // Deep OTM option
    double otm_price = BlackScholes::call_price(CP, 50.0, K, r, v, T);
    std::cout << "Deep OTM price (S=50): " << otm_price << "\n";
    if (otm_price < 1.0) {  // Should be small but positive
        std::cout << "✓ Deep OTM price is reasonable\n";
    } else {
        std::cout << "✗ Deep OTM price seems incorrect\n";
    }
    
    // Zero volatility case
    double zero_vol_price = BlackScholes::call_price(CP, S, K, r, 0.0001, T);
    double intrinsic = std::max(0.0, S - K * std::exp(-r * T));
    std::cout << "Near-zero volatility price: " << zero_vol_price << "\n";
    std::cout << "Intrinsic value: " << intrinsic << "\n";
    if (std::abs(zero_vol_price - intrinsic) < 0.1) {
        std::cout << "✓ Zero volatility case converges to intrinsic value\n";
    } else {
        std::cout << "✗ Zero volatility case failed\n";
    }
    
    std::cout << "\nBlack-Scholes testing completed.\n";
}
