#include "solver.hpp"
#include <cmath>

//for the Exporter class
#include <string>
#include <fstream>
//for std::setprecision
#include <iomanip>
//implied vol
#include "bs.hpp"


class ResultsExporter {
public:
    static void exportToCSV(const std::string& baseFilename, 
                          const Grid& grid, 
                          const Kokkos::View<double*>& results) {
        // Export grid
        std::ofstream gridFile(baseFilename + "_grid.csv");
        gridFile << "s_values,v_values\n";
        // Export all s values first, then all v values
        for(size_t i = 0; i < grid.Vec_s.size(); ++i) {
            gridFile << grid.Vec_s[i];
            if(i < grid.Vec_s.size() - 1) gridFile << ",";
        }
        gridFile << "\n";
        for(size_t i = 0; i < grid.Vec_v.size(); ++i) {
            gridFile << grid.Vec_v[i];
            if(i < grid.Vec_v.size() - 1) gridFile << ",";
        }
        gridFile.close();

        // Export results
        auto h_results = Kokkos::create_mirror_view(results);
        Kokkos::deep_copy(h_results, results);
        
        std::ofstream resultsFile(baseFilename + "_results.csv");
        resultsFile << "values\n";
        for(int i = 0; i < h_results.extent(0); ++i) {
            resultsFile << h_results(i) << "\n";
        }
        resultsFile.close();
    }
};

class ConvergenceExporter {
public:
    struct ConvergenceData {
        std::vector<int> m1_sizes;     // Stock price dimensions
        std::vector<int> m2_sizes;     // Variance dimensions
        std::vector<double> errors;    // Relative errors
        std::vector<double> times;     // Computation times
        std::vector<double> prices;    // Computed prices
    };

    // New struct for time step convergence
    struct TimeStepData {
        std::vector<int> N_values;     // Number of time steps
        std::vector<double> errors;
        std::vector<double> times;
        std::vector<double> prices;
    };

    static ConvergenceData testFixedM2VaryM1(int fixed_m2, 
                                           const std::vector<int>& m1_sizes,
                                           double ref_price,
                                           const double K,
                                           const double S_0,
                                           const double V_0,
                                           const double T,
                                           const double r_d,
                                           const double r_f,
                                           const double rho,
                                           const double sigma,
                                           const double kappa,
                                           const double eta) {
        ConvergenceData data;
        data.m1_sizes = m1_sizes;
        data.m2_sizes.resize(m1_sizes.size(), fixed_m2);
        
        for (size_t i = 0; i < m1_sizes.size(); ++i) {
            int m1 = m1_sizes[i];
            int m2 = fixed_m2;
            
            // Record dimensions
            std::cout << "Testing m1 = " << m1 << ", m2 = " << m2 << std::endl;
            
            // Start timing
            auto start = std::chrono::high_resolution_clock::now();
            
            // Setup grid and matrices
            int m = (m1 + 1) * (m2 + 1);
            Grid grid(m1, 8*K, S_0, K, K/5, m2, 5.0, V_0, 5.0/500);
            
            // Initialize matrices
            heston_A0Storage_gpu A0(m1, m2);
            heston_A1Storage_gpu A1(m1, m2);
            heston_A2Storage_gpu A2(m1, m2);

            A0.build_matrix(grid, rho, sigma);
            A1.build_matrix(grid, rho, sigma, r_d, r_f);
            A2.build_matrix(grid, rho, sigma, r_d, kappa, eta);

            const int N = 20;
            const double delta_t = T / N;
            const double theta = 0.8;

            // Build implicit matrices
            A1.build_implicit(theta, delta_t);
            A2.build_implicit(theta, delta_t);

            // Initialize boundary conditions
            BoundaryConditions bounds(m1, m2, r_d, r_f, N, delta_t);
            bounds.initialize(Kokkos::View<double*>(grid.Vec_s.data(), grid.Vec_s.size()));

            // Initial condition
            Kokkos::View<double*> U_0("U_0", m);
            Kokkos::View<double*> U("U", m);
            
            auto h_U_0 = Kokkos::create_mirror_view(U_0);
            for(int j = 0; j <= m2; j++) {
                for(int i = 0; i <= m1; i++) {
                    h_U_0[i + j*(m1+1)] = std::max(grid.Vec_s[i] - K, 0.0);
                }
            }
            Kokkos::deep_copy(U_0, h_U_0);

            // Run solver
            DO_scheme<Kokkos::View<double*>>(m, N, U_0, delta_t, theta, A0, A1, A2, bounds, r_f, U);
            //CS_scheme<Kokkos::View<double*>>(m, N, U_0, delta_t, theta, A0, A1, A2, bounds, r_f, U);

            // Get price
            auto h_U = Kokkos::create_mirror_view(U);
            Kokkos::deep_copy(h_U, U);

            int index_s = std::find(grid.Vec_s.begin(), grid.Vec_s.end(), S_0) - grid.Vec_s.begin();
            int index_v = std::find(grid.Vec_v.begin(), grid.Vec_v.end(), V_0) - grid.Vec_v.begin();
            double price = h_U[index_s + index_v*(m1+1)];
            
            // Record time
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = end - start;
            
            // Store results
            data.times.push_back(duration.count());
            data.prices.push_back(price);
            data.errors.push_back(std::abs(price - ref_price) / ref_price);
        }
        
        return data;
    }

    static ConvergenceData testFixedM1VaryM2(int fixed_m1, 
                                           const std::vector<int>& m2_sizes,
                                           double ref_price,
                                           const double K,
                                           const double S_0,
                                           const double V_0,
                                           const double T,
                                           const double r_d,
                                           const double r_f,
                                           const double rho,
                                           const double sigma,
                                           const double kappa,
                                           const double eta) {
        ConvergenceData data;
        data.m2_sizes = m2_sizes;
        data.m1_sizes.resize(m2_sizes.size(), fixed_m1);
        
        for (size_t i = 0; i < m2_sizes.size(); ++i) {
            int m1 = fixed_m1;
            int m2 = m2_sizes[i];
            
            // Record dimensions
            std::cout << "Testing m1 = " << m1 << ", m2 = " << m2 << std::endl;
            
            // Setup grid and matrices
            int m = (m1 + 1) * (m2 + 1);
            Grid grid(m1, 8*K, S_0, K, K/5, m2, 5.0, V_0, 5.0/500);
            
            // Initialize matrices
            heston_A0Storage_gpu A0(m1, m2);
            heston_A1Storage_gpu A1(m1, m2);
            heston_A2Storage_gpu A2(m1, m2);

            A0.build_matrix(grid, rho, sigma);
            A1.build_matrix(grid, rho, sigma, r_d, r_f);
            A2.build_matrix(grid, rho, sigma, r_d, kappa, eta);

            const int N = 20;
            const double delta_t = T / N;
            const double theta = 0.8;

            // Build implicit matrices
            A1.build_implicit(theta, delta_t);
            A2.build_implicit(theta, delta_t);

            // Initialize boundary conditions
            BoundaryConditions bounds(m1, m2, r_d, r_f, N, delta_t);
            bounds.initialize(Kokkos::View<double*>(grid.Vec_s.data(), grid.Vec_s.size()));

            // Initial condition
            Kokkos::View<double*> U_0("U_0", m);
            Kokkos::View<double*> U("U", m);
            
            auto h_U_0 = Kokkos::create_mirror_view(U_0);
            for(int j = 0; j <= m2; j++) {
                for(int i = 0; i <= m1; i++) {
                    h_U_0[i + j*(m1+1)] = std::max(grid.Vec_s[i] - K, 0.0);
                }
            }
            Kokkos::deep_copy(U_0, h_U_0);

            // Run solver
            // Start timing
            auto start = std::chrono::high_resolution_clock::now();

            DO_scheme<Kokkos::View<double*>>(m, N, U_0, delta_t, theta, A0, A1, A2, bounds, r_f, U);
            //CS_scheme<Kokkos::View<double*>>(m, N, U_0, delta_t, theta, A0, A1, A2, bounds, r_f, U);


            // Record time
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = end - start;

            // Get price
            auto h_U = Kokkos::create_mirror_view(U);
            Kokkos::deep_copy(h_U, U);

            int index_s = std::find(grid.Vec_s.begin(), grid.Vec_s.end(), S_0) - grid.Vec_s.begin();
            int index_v = std::find(grid.Vec_v.begin(), grid.Vec_v.end(), V_0) - grid.Vec_v.begin();
            double price = h_U[index_s + index_v*(m1+1)];
            
            
            // Store results
            data.times.push_back(duration.count());
            data.prices.push_back(price);
            data.errors.push_back(std::abs(price - ref_price) / ref_price);
        }
        
        return data;
    }

    
    static TimeStepData testVaryTimeSteps(int fixed_m1,
                                          int fixed_m2,
                                          const std::vector<int>& N_steps,
                                          double ref_price,
                                          const double K,
                                          const double S_0,
                                          const double V_0,
                                          const double T,
                                          const double r_d,
                                          const double r_f,
                                          const double rho,
                                          const double sigma,
                                          const double kappa,
                                          const double eta) {
        TimeStepData data;
        data.N_values = N_steps;
        
        for (int N : N_steps) {
            std::cout << "Testing with N = " << N << " time steps" << std::endl;
            int m1 = fixed_m1;
            int m2 = fixed_m2;
            int m = (m1 + 1) * (m2 + 1);

            Grid grid(m1, 8*K, S_0, K, K/5, m2, 5.0, V_0, 5.0/500);
            
            heston_A0Storage_gpu A0(m1, m2);
            heston_A1Storage_gpu A1(m1, m2);
            heston_A2Storage_gpu A2(m1, m2);

            A0.build_matrix(grid, rho, sigma);
            A1.build_matrix(grid, rho, sigma, r_d, r_f);
            A2.build_matrix(grid, rho, sigma, r_d, kappa, eta);

            const double delta_t = T / N;
            const double theta = 0.8;

            A1.build_implicit(theta, delta_t);
            A2.build_implicit(theta, delta_t);

            BoundaryConditions bounds(m1, m2, r_d, r_f, N, delta_t);
            bounds.initialize(Kokkos::View<double*>(grid.Vec_s.data(), grid.Vec_s.size()));

            Kokkos::View<double*> U_0("U_0", m);
            Kokkos::View<double*> U("U", m);
            
            auto h_U_0 = Kokkos::create_mirror_view(U_0);
            for(int j = 0; j <= m2; j++) {
                for(int i = 0; i <= m1; i++) {
                    h_U_0[i + j*(m1+1)] = std::max(grid.Vec_s[i] - K, 0.0);
                }
            }
            Kokkos::deep_copy(U_0, h_U_0);

            auto start = std::chrono::high_resolution_clock::now();
            DO_scheme<Kokkos::View<double*>>(m, N, U_0, delta_t, theta, A0, A1, A2, bounds, r_f, U);
            //CS_scheme<Kokkos::View<double*>>(m, N, U_0, delta_t, theta, A0, A1, A2, bounds, r_f, U);

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = end - start;

            auto h_U = Kokkos::create_mirror_view(U);
            Kokkos::deep_copy(h_U, U);

            int index_s = std::find(grid.Vec_s.begin(), grid.Vec_s.end(), S_0) - grid.Vec_s.begin();
            int index_v = std::find(grid.Vec_v.begin(), grid.Vec_v.end(), V_0) - grid.Vec_v.begin();
            double price = h_U[index_s + index_v*(m1+1)];

            data.times.push_back(duration.count());
            data.prices.push_back(price);
            data.errors.push_back(std::abs(price - ref_price) / ref_price);
        }
        
        return data;
    }
    
    static ConvergenceData testFixedM2VaryM1_shuffled(int fixed_m2, 
                                                 const std::vector<int>& m1_sizes,
                                                 double ref_price,
                                                 const double K,
                                                 const double S_0,
                                                 const double V_0,
                                                 const double T,
                                                 const double r_d,
                                                 const double r_f,
                                                 const double rho,
                                                 const double sigma,
                                                 const double kappa,
                                                 const double eta) {
    ConvergenceData data;
    data.m1_sizes = m1_sizes;
    data.m2_sizes.resize(m1_sizes.size(), fixed_m2);
    
    for (size_t i = 0; i < m1_sizes.size(); ++i) {
        int m1 = m1_sizes[i];
        int m2 = fixed_m2;

        auto t_start = std::chrono::high_resolution_clock::now();
        
        std::cout << "Testing m1 = " << m1 << ", m2 = " << m2 << std::endl;
        
        //auto start = std::chrono::high_resolution_clock::now();
        
        int m = (m1 + 1) * (m2 + 1);
        Grid grid(m1, 8*K, S_0, K, K/5, m2, 5.0, V_0, 5.0/500);
        
        // Initialize all matrices including shuffled A2
        heston_A0Storage_gpu A0(m1, m2);
        heston_A1Storage_gpu A1(m1, m2);
        heston_A2Storage_gpu A2(m1, m2);
        heston_A2_shuffled A2_shuf(m1, m2);

        A0.build_matrix(grid, rho, sigma);
        A1.build_matrix(grid, rho, sigma, r_d, r_f);
        A2.build_matrix(grid, rho, sigma, r_d, kappa, eta);
        A2_shuf.build_matrix(grid, rho, sigma, r_d, kappa, eta);

        const int N = 20;
        const double delta_t = T / N;
        const double theta = 0.5;

        A1.build_implicit(theta, delta_t);
        A2.build_implicit(theta, delta_t);
        A2_shuf.build_implicit(theta, delta_t);

        BoundaryConditions bounds(m1, m2, r_d, r_f, N, delta_t);
        bounds.initialize(Kokkos::View<double*>(grid.Vec_s.data(), grid.Vec_s.size()));

        Kokkos::View<double*> U_0("U_0", m);
        Kokkos::View<double*> U("U", m);
        
        auto h_U_0 = Kokkos::create_mirror_view(U_0);
        for(int j = 0; j <= m2; j++) {
            for(int i = 0; i <= m1; i++) {
                h_U_0[i + j*(m1+1)] = std::max(grid.Vec_s[i] - K, 0.0);
            }
        }
        Kokkos::deep_copy(U_0, h_U_0);

        auto start = std::chrono::high_resolution_clock::now();

        DO_scheme_shuffle(m, m1, m2, N, U_0, delta_t, theta, A0, A1, A2, A2_shuf, bounds, r_f, U);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = (end - start);
        

        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Entire setz up everythig time: "
                << std::chrono::duration<double>(t_end - t_start).count()
                << " seconds" << std::endl;

        auto h_U = Kokkos::create_mirror_view(U);
        Kokkos::deep_copy(h_U, U);

        int index_s = std::find(grid.Vec_s.begin(), grid.Vec_s.end(), S_0) - grid.Vec_s.begin();
        int index_v = std::find(grid.Vec_v.begin(), grid.Vec_v.end(), V_0) - grid.Vec_v.begin();
        double price = h_U[index_s + index_v*(m1+1)];
        
        
        data.times.push_back(duration.count());
        data.prices.push_back(price);
        data.errors.push_back(std::abs(price - ref_price) / ref_price);
    }

    return data;
}

    static ConvergenceData testFixedM1VaryM2_shuffled(int fixed_m1, 
                                                 const std::vector<int>& m2_sizes,
                                                 double ref_price,
                                                 const double K,
                                                 const double S_0,
                                                 const double V_0,
                                                 const double T,
                                                 const double r_d,
                                                 const double r_f,
                                                 const double rho,
                                                 const double sigma,
                                                 const double kappa,
                                                 const double eta) {
        ConvergenceData data;
        data.m2_sizes = m2_sizes;
        data.m1_sizes.resize(m2_sizes.size(), fixed_m1);
        
        for (size_t i = 0; i < m2_sizes.size(); ++i) {
            int m1 = fixed_m1;
            int m2 = m2_sizes[i];
            
            std::cout << "Testing m1 = " << m1 << ", m2 = " << m2 << std::endl;
            
            int m = (m1 + 1) * (m2 + 1);
            Grid grid(m1, 8*K, S_0, K, K/5, m2, 5.0, V_0, 5.0/500);
            
            // Initialize matrices including shuffled A2
            heston_A0Storage_gpu A0(m1, m2);
            heston_A1Storage_gpu A1(m1, m2);
            heston_A2Storage_gpu A2(m1, m2);
            heston_A2_shuffled A2_shuf(m1, m2);

            A0.build_matrix(grid, rho, sigma);
            A1.build_matrix(grid, rho, sigma, r_d, r_f);
            A2.build_matrix(grid, rho, sigma, r_d, kappa, eta);
            A2_shuf.build_matrix(grid, rho, sigma, r_d, kappa, eta);

            const int N = 80;
            const double delta_t = T / N;
            const double theta = 0.5;

            A1.build_implicit(theta, delta_t);
            A2.build_implicit(theta, delta_t);
            A2_shuf.build_implicit(theta, delta_t);

            BoundaryConditions bounds(m1, m2, r_d, r_f, N, delta_t);
            bounds.initialize(Kokkos::View<double*>(grid.Vec_s.data(), grid.Vec_s.size()));

            Kokkos::View<double*> U_0("U_0", m);
            Kokkos::View<double*> U("U", m);
            
            auto h_U_0 = Kokkos::create_mirror_view(U_0);
            for(int j = 0; j <= m2; j++) {
                for(int i = 0; i <= m1; i++) {
                    h_U_0[i + j*(m1+1)] = std::max(grid.Vec_s[i] - K, 0.0);
                }
            }
            Kokkos::deep_copy(U_0, h_U_0);

            auto start = std::chrono::high_resolution_clock::now();
            
            DO_scheme_shuffle(m, m1, m2, N, U_0, delta_t, theta, A0, A1, A2, A2_shuf, bounds, r_f, U);

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = end - start;

            auto h_U = Kokkos::create_mirror_view(U);
            Kokkos::deep_copy(h_U, U);

            int index_s = std::find(grid.Vec_s.begin(), grid.Vec_s.end(), S_0) - grid.Vec_s.begin();
            int index_v = std::find(grid.Vec_v.begin(), grid.Vec_v.end(), V_0) - grid.Vec_v.begin();
            double price = h_U[index_s + index_v*(m1+1)];
            
            data.times.push_back(duration.count());
            data.prices.push_back(price);
            data.errors.push_back(std::abs(price - ref_price) / ref_price);
        }
        
        return data;
    }

    static TimeStepData testVaryTimeSteps_shuffled(int fixed_m1,
                                                int fixed_m2,
                                                const std::vector<int>& N_steps,
                                                double ref_price,
                                                const double K,
                                                const double S_0,
                                                const double V_0,
                                                const double T,
                                                const double r_d,
                                                const double r_f,
                                                const double rho,
                                                const double sigma,
                                                const double kappa,
                                                const double eta) {
        TimeStepData data;
        data.N_values = N_steps;
        
        for (int N : N_steps) {
            std::cout << "Testing with N = " << N << " time steps" << std::endl;
            int m1 = fixed_m1;
            int m2 = fixed_m2;
            int m = (m1 + 1) * (m2 + 1);

            Grid grid(m1, 8*K, S_0, K, K/5, m2, 5.0, V_0, 5.0/500);
            
            heston_A0Storage_gpu A0(m1, m2);
            heston_A1Storage_gpu A1(m1, m2);
            heston_A2Storage_gpu A2(m1, m2);
            heston_A2_shuffled A2_shuf(m1, m2);

            A0.build_matrix(grid, rho, sigma);
            A1.build_matrix(grid, rho, sigma, r_d, r_f);
            A2.build_matrix(grid, rho, sigma, r_d, kappa, eta);
            A2_shuf.build_matrix(grid, rho, sigma, r_d, kappa, eta);

            const double delta_t = T / N;
            const double theta = 0.5;

            A1.build_implicit(theta, delta_t);
            A2.build_implicit(theta, delta_t);
            A2_shuf.build_implicit(theta, delta_t);

            BoundaryConditions bounds(m1, m2, r_d, r_f, N, delta_t);
            bounds.initialize(Kokkos::View<double*>(grid.Vec_s.data(), grid.Vec_s.size()));

            Kokkos::View<double*> U_0("U_0", m);
            Kokkos::View<double*> U("U", m);
            
            auto h_U_0 = Kokkos::create_mirror_view(U_0);
            for(int j = 0; j <= m2; j++) {
                for(int i = 0; i <= m1; i++) {
                    h_U_0[i + j*(m1+1)] = std::max(grid.Vec_s[i] - K, 0.0);
                }
            }
            Kokkos::deep_copy(U_0, h_U_0);

            auto start = std::chrono::high_resolution_clock::now();
            
            DO_scheme_shuffle(m, m1, m2, N, U_0, delta_t, theta, A0, A1, A2, A2_shuf, bounds, r_f, U);
            
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = end - start;

            auto h_U = Kokkos::create_mirror_view(U);
            Kokkos::deep_copy(h_U, U);

            int index_s = std::find(grid.Vec_s.begin(), grid.Vec_s.end(), S_0) - grid.Vec_s.begin();
            int index_v = std::find(grid.Vec_v.begin(), grid.Vec_v.end(), V_0) - grid.Vec_v.begin();
            double price = h_U[index_s + index_v*(m1+1)];

            data.times.push_back(duration.count());
            data.prices.push_back(price);
            data.errors.push_back(std::abs(price - ref_price) / ref_price);
        }
        
        return data;
    }


    // Export time step convergence results
    static void exportTimeStepConvergenceToCSV(const std::string& filename, const TimeStepData& data) {
        std::ofstream file(filename + "_timestep_convergence.csv");
        
        file << "N,price,error,time\n";
        
        for (size_t i = 0; i < data.N_values.size(); ++i) {
            file << data.N_values[i] << ","
                 << std::scientific << std::setprecision(10) << data.prices[i] << ","
                 << data.errors[i] << ","
                 << data.times[i] << "\n";
        }
    }

    //Exporting space step convergence results
    static void exportToCSV(const std::string& filename, const ConvergenceData& data) {
        std::ofstream file(filename + "_convergence.csv");
        
        // Write header
        file << "m1,m2,price,error,time\n";
        
        // Write data
        for (size_t i = 0; i < data.m1_sizes.size(); ++i) {
            file << data.m1_sizes[i] << ","
                 << data.m2_sizes[i] << ","
                 << std::scientific << std::setprecision(10) << data.prices[i] << ","
                 << data.errors[i] << ","
                 << data.times[i] << "\n";
        }
    }
};

/*

DO scheme tests, working

*/
void test_heston_call(){
    using timer = std::chrono::high_resolution_clock;
    // Market parameters
    const double K = 100.0;
    const double S_0 = 100.0;
    const double V_0 = 0.04;
    const double T = 1.0;
    const double r_d = 0.025;
    const double r_f = 0.0;
    const double rho = -0.9;
    const double sigma = 0.3;
    const double kappa = 1.5;
    const double eta = 0.04;
    
    // Test parameters matching Python version
    const int m1 = 300;
    const int m2 = 100;
    std::cout << "Dimesnion StockxVariance: " << m1+1 << "x" << m2+1 << std::endl;

    const int m = (m1 + 1) * (m2 + 1);

    const int N = 20;
    const double delta_t = T / N;
    const double theta = 0.8;
    std::cout << "Time Dimension: " << N << std::endl;
    std::cout << "Theta: " << theta << std::endl;


    // Create grid and matrices
    Grid grid(m1, 8*K, S_0, K, K/5, m2, 5.0, V_0, 5.0/500);

    // Initialize matrices
    heston_A0Storage_gpu A0(m1, m2);
    heston_A1Storage_gpu A1(m1, m2);
    heston_A2Storage_gpu A2(m1, m2);

    A0.build_matrix(grid, rho, sigma);
    A1.build_matrix(grid, rho, sigma, r_d, r_f);
    A2.build_matrix(grid, rho, sigma, r_d, kappa, eta);

    // Build implicit matrices
    A1.build_implicit(theta, delta_t);
    A2.build_implicit(theta, delta_t);

    // Initialize boundary conditions
    BoundaryConditions bounds(m1, m2, r_d, r_f, N, delta_t);
    bounds.initialize(Kokkos::View<double*>(grid.Vec_s.data(), grid.Vec_s.size()));

    // Initial condition: max(S - K, 0)
    Kokkos::View<double*> U_0("U_0", m);
    Kokkos::View<double*> U("U", m);

    
    auto h_U_0 = Kokkos::create_mirror_view(U_0);
    for(int j = 0; j <= m2; j++) {
        for(int i = 0; i <= m1; i++) {
            h_U_0[i + j*(m1+1)] = std::max(grid.Vec_s[i] - K, 0.0);
        }
    }
    Kokkos::deep_copy(U_0, h_U_0);

    // Run solver
    //auto t_start = timer::now();
    DO_scheme<Kokkos::View<double*>>(m, N, U_0, delta_t, theta, A0, A1, A2, bounds, r_f, U);
    //auto t_end = timer::now();

    //std::cout << "DO time: "
            //<< std::chrono::duration<double>(t_end - t_start).count()
            //<< " seconds" << std::endl;

    // Verify solution
    auto h_U = Kokkos::create_mirror_view(U);
    Kokkos::deep_copy(h_U, U);

    // Find option price at S_0 and V_0
    int index_s = std::find(grid.Vec_s.begin(), grid.Vec_s.end(), S_0) - grid.Vec_s.begin();
    int index_v = std::find(grid.Vec_v.begin(), grid.Vec_v.end(), V_0) - grid.Vec_v.begin();
    double option_price = h_U[index_s + index_v*(m1+1)];

    // Compare with reference price (from Python/Monte Carlo)
    const double reference_price = 8.8948693600540167;
    std::cout << std::setprecision(16) << option_price << std::endl;
    std::cout << "Relative error: " << std::abs(option_price - reference_price)/reference_price << std::endl;
    //EXPECT_NEAR(option_price, reference_price, 0.1);

    //for python plotting
    ResultsExporter::exportToCSV("heston_do_scheme", grid, U);
}

void test_DO_m1_convergence() {
    // Market parameters
    const double K = 100.0;
    const double S_0 = 100.0;
    const double V_0 = 0.04;
    const double T = 1.0;
    const double r_d = 0.025;
    const double r_f = 0.0;
    const double rho = -0.9;
    const double sigma = 0.3;
    const double kappa = 1.5;
    const double eta = 0.04;
    
    const double ref_price = 8.8948693600540167;
    
    // Test parameters
    std::vector<int> m1_sizes = {20, 30, 40, 50, 60, 70, 80, 100, 120, 170, 200, 250, 300};
    int fixed_m2 = 50;
    
    std::cout << "Starting convergence test with fixed m2 = " << fixed_m2 << std::endl;
    
    auto data = ConvergenceExporter::testFixedM2VaryM1(
        fixed_m2, m1_sizes, ref_price,
        K, S_0, V_0, T, r_d, r_f, rho, sigma, kappa, eta
    );
    
    ConvergenceExporter::exportToCSV("do_scheme", data);
    
    std::cout << "\nResults:" << std::endl;
    std::cout << "m1\tError\t\tTime" << std::endl;
    for (size_t i = 0; i < data.m1_sizes.size(); ++i) {
        std::cout << data.m1_sizes[i] << "\t"
                    << std::scientific << data.errors[i] << "\t"
                    << std::fixed << data.times[i] << "s" << std::endl;
    }
}

void test_all_convergence() {
    // Market parameters
    const double K = 100.0;
    const double S_0 = 100.0;
    const double V_0 = 0.04;
    const double T = 1.0;
    const double r_d = 0.025;
    const double r_f = 0.0;
    const double rho = -0.9;
    const double sigma = 0.3;
    const double kappa = 1.5;
    const double eta = 0.04;
    
    const double ref_price = 8.8948693600540167;
    
    // Test m1 convergence
    std::vector<int> m1_sizes = {50, 75, 100, 150, 200, 250, 300};
    int fixed_m2 = 100;
    auto data_m1 = ConvergenceExporter::testFixedM2VaryM1(
        fixed_m2, m1_sizes, ref_price,
        K, S_0, V_0, T, r_d, r_f, rho, sigma, kappa, eta
    );
    ConvergenceExporter::exportToCSV("do_scheme_m1", data_m1);
    
    // Test m2 convergence
    std::vector<int> m2_sizes = {20, 50, 70, 75, 85, 90, 100};
    int fixed_m1 = 100;
    auto data_m2 = ConvergenceExporter::testFixedM1VaryM2(
        fixed_m1, m2_sizes, ref_price,
        K, S_0, V_0, T, r_d, r_f, rho, sigma, kappa, eta
    );
    ConvergenceExporter::exportToCSV("do_scheme_m2", data_m2);
    
    // Test time step convergence
    std::vector<int> N_steps = {10, 20, 30, 40, 50, 60, 70};
    auto data_N = ConvergenceExporter::testVaryTimeSteps(
        100, 50, N_steps, ref_price,
        K, S_0, V_0, T, r_d, r_f, rho, sigma, kappa, eta
    );
    ConvergenceExporter::exportTimeStepConvergenceToCSV("do_scheme_N", data_N);
}

/*

Shuffled DO scheme tests

*/

// Function to compute European call price using DO scheme with shuffled A2
void test_heston_call_shuffled() {
    // Test parameters
    double K = 100.0;
    double S_0 = K;
    double V_0 = 0.04;
    double T = 1.0;
    double r_d = 0.025;
    double r_f = 0.0;
    double rho = -0.9;
    double sigma = 0.3;
    double kappa = 1.5;
    double eta = 0.04;

    int m1 = 100;
    int m2 = 50;
    int m = (m1 + 1) * (m2 + 1);
    int N = 20;
    double theta = 0.8;

    // Create grid
    Grid grid = create_test_grid(m1, m2);

    // Initialize matrices
    heston_A0Storage_gpu A0(m1, m2);
    heston_A1Storage_gpu A1(m1, m2);
    heston_A2Storage_gpu A2(m1, m2);  // Original A2
    heston_A2_shuffled A2_shuf(m1, m2);  // Shuffled A2

    // Build matrices
    A0.build_matrix(grid, rho, sigma);
    A1.build_matrix(grid, rho, sigma, r_d, r_f);
    A2.build_matrix(grid, rho, sigma, r_d, kappa, eta);
    A2_shuf.build_matrix(grid, rho, sigma, r_d, kappa, eta);

    // Time step size
    double delta_t = T / N;

    // Build boundary conditions
    BoundaryConditions bounds(m1, m2, r_d, r_f, N, delta_t);
    bounds.initialize(Kokkos::View<double*>(grid.Vec_s.data(), m1 + 1));

    // Initial condition
    Kokkos::View<double*> U_0("U_0", m);
    Kokkos::View<double*> U("U", m);

    // Set initial condition on host
    auto h_U_0 = Kokkos::create_mirror_view(U_0);
    for (int j = 0; j <= m2; j++) {
        for (int i = 0; i <= m1; i++) {
            h_U_0(i + j * (m1 + 1)) = std::max(grid.Vec_s[i] - K, 0.0);
        }
    }
    Kokkos::deep_copy(U_0, h_U_0);

    // Build implicit matrices
    double theta_t = theta * delta_t;
    A1.build_implicit(theta, delta_t);
    A2.build_implicit(theta, delta_t);
    A2_shuf.build_implicit(theta, delta_t);

    // Solve using DO scheme
    //for( int i = 0; i<5; i++){
        DO_scheme_shuffle(m, m1, m2, N, U_0, delta_t, theta, A0, A1, A2, A2_shuf, bounds, r_f, U);
    //}

    // Get result
    auto h_U = Kokkos::create_mirror_view(U);
    Kokkos::deep_copy(h_U, U);

    // Find price at S_0, V_0
    // Find option price at S_0 and V_0
    int index_s = std::find(grid.Vec_s.begin(), grid.Vec_s.end(), S_0) - grid.Vec_s.begin();
    int index_v = std::find(grid.Vec_v.begin(), grid.Vec_v.end(), V_0) - grid.Vec_v.begin();
    double option_price = h_U[index_s + index_v*(m1+1)];

    // Compare with reference price (from Python/Monte Carlo)
    const double reference_price = 8.8948693600540167;
    std::cout << std::setprecision(16) << option_price << std::endl;
    std::cout << "Absolut error: " << std::abs(option_price - reference_price) << std::endl;///reference_price << std::endl;
    std::cout << "Relative error: " << std::abs(option_price - reference_price)/reference_price << std::endl;

    ResultsExporter::exportToCSV("shuffled_heston_do_scheme", grid, U);
}

#include <numeric>
void test_heston_call_shuffled_vary_m1() {
    // Fixed parameters
    double K = 100.0;
    double S_0 = K;
    double V_0 = 0.04;
    double T = 1.0;
    double r_d = 0.025;
    double r_f = 0.0;
    double rho = -0.9;
    double sigma = 0.3;
    double kappa = 1.5;
    double eta = 0.04;
    const double reference_price = 8.8948693600540167;

    // Fixed m2 and vary m1
    int fixed_m2 = 100;
    std::vector<int> m1_values = {50, 100, 150, 200, 250, 300, 350, 400};
    
    std::cout << "\nTesting with different m1 values (fixed m2=" << fixed_m2 << "):\n";
    std::cout << std::left  // Left align text
              << std::setw(8) << "m1"
              << std::setw(12) << "Price"
              << std::setw(12) << "Rel. Error"
              << std::setw(12) << "Min Time(s)"
              << std::setw(12) << "Avg Time(s)" << "\n";
    std::cout << std::string(56, '-') << "\n";

    for(int m1 : m1_values) {
        int m2 = fixed_m2;
        int m = (m1 + 1) * (m2 + 1);
        int N = 20;
        double theta = 0.5;

        // Create grid
        Grid grid = create_test_grid(m1, m2);

        // Initialize matrices
        heston_A0Storage_gpu A0(m1, m2);
        heston_A1Storage_gpu A1(m1, m2);
        heston_A2Storage_gpu A2(m1, m2);
        heston_A2_shuffled A2_shuf(m1, m2);

        // Build matrices
        A0.build_matrix(grid, rho, sigma);
        A1.build_matrix(grid, rho, sigma, r_d, r_f);
        A2.build_matrix(grid, rho, sigma, r_d, kappa, eta);
        A2_shuf.build_matrix(grid, rho, sigma, r_d, kappa, eta);

        double delta_t = T / N;

        // Build boundary conditions
        BoundaryConditions bounds(m1, m2, r_d, r_f, N, delta_t);
        bounds.initialize(Kokkos::View<double*>(grid.Vec_s.data(), m1 + 1));

        // Initial condition
        Kokkos::View<double*> U_0("U_0", m);
        Kokkos::View<double*> U("U", m);

        auto h_U_0 = Kokkos::create_mirror_view(U_0);
        for (int j = 0; j <= m2; j++) {
            for (int i = 0; i <= m1; i++) {
                h_U_0(i + j * (m1 + 1)) = std::max(grid.Vec_s[i] - K, 0.0);
            }
        }
        Kokkos::deep_copy(U_0, h_U_0);

        // Build implicit matrices
        A1.build_implicit(theta, delta_t);
        A2.build_implicit(theta, delta_t);
        A2_shuf.build_implicit(theta, delta_t);

        // Run solver multiple times and measure performance
        const int num_runs = 1;
        std::vector<double> run_times;
        
        for(int i = 0; i < num_runs; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            
            DO_scheme_shuffle(m, m1, m2, N, U_0, delta_t, theta, A0, A1, A2, A2_shuf, bounds, r_f, U);
            
            auto end = std::chrono::high_resolution_clock::now();
            double duration = std::chrono::duration<double>(end - start).count();
            run_times.push_back(duration);
        }

        // Get final result
        auto h_U = Kokkos::create_mirror_view(U);
        Kokkos::deep_copy(h_U, U);

        int index_s = std::find(grid.Vec_s.begin(), grid.Vec_s.end(), S_0) - grid.Vec_s.begin();
        int index_v = std::find(grid.Vec_v.begin(), grid.Vec_v.end(), V_0) - grid.Vec_v.begin();
        double option_price = h_U[index_s + index_v*(m1+1)];

        // Calculate relative error
        double rel_error = std::abs(option_price - reference_price)/reference_price;

        // Calculate min and average time
        double min_time = *std::min_element(run_times.begin(), run_times.end());
        double avg_time = std::accumulate(run_times.begin(), run_times.end(), 0.0) / num_runs;

        // Print results
        std::cout << std::left
                  << std::setw(8) << m1 
                  << std::fixed << std::setprecision(5)
                  << std::setw(12) << option_price
                  << std::setw(12) << rel_error
                  << std::setw(12) << min_time
                  << std::setw(12) << avg_time << "\n";
    }
}

void test_shuffled_convergence() {
    // Market parameters (same as before)
    const double K = 100.0;
    const double S_0 = 100.0;
    const double V_0 = 0.04;
    const double T = 1.0;
    const double r_d = 0.025;
    const double r_f = 0.0;
    const double rho = -0.9;
    const double sigma = 0.3;
    const double kappa = 1.5;
    const double eta = 0.04;
    
    const double ref_price = 8.8948693600540167;
    
    // Test m1 convergence
    std::vector<int> m1_sizes = {50, 75, 100, 150, 200, 250, 300};
    int fixed_m2 = 100;
    auto data_m1 = ConvergenceExporter::testFixedM2VaryM1_shuffled(
        fixed_m2, m1_sizes, ref_price,
        K, S_0, V_0, T, r_d, r_f, rho, sigma, kappa, eta
    );
    ConvergenceExporter::exportToCSV("do_scheme_shuffled_m1", data_m1);
    
    // Test m2 convergence
    std::vector<int> m2_sizes = {20, 50, 70, 75, 85, 90, 100};
    int fixed_m1 = 300;
    auto data_m2 = ConvergenceExporter::testFixedM1VaryM2_shuffled(
        fixed_m1, m2_sizes, ref_price,
        K, S_0, V_0, T, r_d, r_f, rho, sigma, kappa, eta
    );
    ConvergenceExporter::exportToCSV("do_scheme_shuffled_m2", data_m2);
    
    // Test time step convergence
    std::vector<int> N_steps = {10, 20, 30, 40, 50, 60, 70};
    auto data_N = ConvergenceExporter::testVaryTimeSteps_shuffled(
        100, 50, N_steps, ref_price,
        K, S_0, V_0, T, r_d, r_f, rho, sigma, kappa, eta
    );
    ConvergenceExporter::exportTimeStepConvergenceToCSV("do_scheme_shuffled_N", data_N);
}

/*

CS scheme tests

*/

void test_CS_scheme_call(){
    //using timer = std::chrono::high_resolution_clock;
    // Market parameters
    const double K = 100.0;
    const double S_0 = 100.0;
    const double V_0 = 0.04;
    const double T = 1.0;
    const double r_d = 0.025;
    const double r_f = 0.0;
    const double rho = -0.9;
    const double sigma = 0.3;
    const double kappa = 1.5;
    const double eta = 0.04;
    
    // Test parameters matching Python version
    const int m1 = 200;
    const int m2 = 100;
    std::cout << "Dimesnion StockxVariance: " << m1+1 << "x" << m2+1 << std::endl;

    const int m = (m1 + 1) * (m2 + 1);

    const int N = 40;
    const double delta_t = T / N;
    const double theta = 0.8;
    std::cout << "Time Dimension: " << N << std::endl;
    std::cout << "Theta: " << theta << std::endl;


    // Create grid and matrices
    Grid grid(m1, 8*K, S_0, K, K/5, m2, 5.0, V_0, 5.0/500);

    // Initialize matrices
    heston_A0Storage_gpu A0(m1, m2);
    heston_A1Storage_gpu A1(m1, m2);
    heston_A2Storage_gpu A2(m1, m2);

    A0.build_matrix(grid, rho, sigma);
    A1.build_matrix(grid, rho, sigma, r_d, r_f);
    A2.build_matrix(grid, rho, sigma, r_d, kappa, eta);

    // Build implicit matrices
    A1.build_implicit(theta, delta_t);
    A2.build_implicit(theta, delta_t);

    // Initialize boundary conditions
    BoundaryConditions bounds(m1, m2, r_d, r_f, N, delta_t);
    bounds.initialize(Kokkos::View<double*>(grid.Vec_s.data(), grid.Vec_s.size()));

    // Initial condition: max(S - K, 0)
    Kokkos::View<double*> U_0("U_0", m);
    Kokkos::View<double*> U("U", m);

    
    auto h_U_0 = Kokkos::create_mirror_view(U_0);
    for(int j = 0; j <= m2; j++) {
        for(int i = 0; i <= m1; i++) {
            h_U_0[i + j*(m1+1)] = std::max(grid.Vec_s[i] - K, 0.0);
        }
    }
    Kokkos::deep_copy(U_0, h_U_0);

    // Run solver
    //auto t_start = timer::now();
    CS_scheme<Kokkos::View<double*>>(m, N, U_0, delta_t, theta, A0, A1, A2, bounds, r_f, U);
    //auto t_end = timer::now();

    //std::cout << "CS time: "
            //<< std::chrono::duration<double>(t_end - t_start).count()
            //<< " seconds" << std::endl;

    // Verify solution
    auto h_U = Kokkos::create_mirror_view(U);
    Kokkos::deep_copy(h_U, U);

    // Find option price at S_0 and V_0
    int index_s = std::find(grid.Vec_s.begin(), grid.Vec_s.end(), S_0) - grid.Vec_s.begin();
    int index_v = std::find(grid.Vec_v.begin(), grid.Vec_v.end(), V_0) - grid.Vec_v.begin();
    double option_price = h_U[index_s + index_v*(m1+1)];

    // Compare with reference price (from Python/Monte Carlo)
    const double reference_price = 8.8948693600540167;
    std::cout << "CS_scheme price: " << std::setprecision(12) << option_price << std::endl;
    std::cout << "CS_ scheme Relative error: " << std::abs(option_price - reference_price)/reference_price << std::endl;
    //EXPECT_NEAR(option_price, reference_price, 0.1);
    ResultsExporter::exportToCSV("heston_cs_scheme", grid, U);
}

void test_CS_convergence() {
    // Market parameters
    const double K = 100.0;
    const double S_0 = 100.0;
    const double V_0 = 0.04;
    const double T = 1.0;
    const double r_d = 0.025;
    const double r_f = 0.0;
    const double rho = -0.9;
    const double sigma = 0.3;
    const double kappa = 1.5;
    const double eta = 0.04;
    
    const double ref_price = 8.8948693600540167;
    
    // Test m1 convergence
    std::vector<int> m1_sizes = {50, 75, 100, 150, 200, 250, 300};
    int fixed_m2 = 50;
    auto data_m1 = ConvergenceExporter::testFixedM2VaryM1(
        fixed_m2, m1_sizes, ref_price,
        K, S_0, V_0, T, r_d, r_f, rho, sigma, kappa, eta
    );
    ConvergenceExporter::exportToCSV("cs_scheme_m1", data_m1);
    
    // Test m2 convergence
    std::vector<int> m2_sizes = {20, 50, 70, 75, 85, 90, 100};
    int fixed_m1 = 100;
    auto data_m2 = ConvergenceExporter::testFixedM1VaryM2(
        fixed_m1, m2_sizes, ref_price,
        K, S_0, V_0, T, r_d, r_f, rho, sigma, kappa, eta
    );
    ConvergenceExporter::exportToCSV("cs_scheme_m2", data_m2);
    
    // Test time step convergence
    std::vector<int> N_steps = {10, 20, 30, 40, 50, 60, 70};
    auto data_N = ConvergenceExporter::testVaryTimeSteps(
        100, 50, N_steps, ref_price,
        K, S_0, V_0, T, r_d, r_f, rho, sigma, kappa, eta
    );
    ConvergenceExporter::exportTimeStepConvergenceToCSV("cs_scheme_N", data_N);
}

void test_CS_shuffled() {
    // Test parameters
    double K = 100.0;
    double S_0 = K;
    double V_0 = 0.04;
    double T = 1.0;
    double r_d = 0.025;
    double r_f = 0.0;
    double rho = -0.9;
    double sigma = 0.3;
    double kappa = 1.5;
    double eta = 0.04;

    
    const int m1 = 50;
    const int m2 = 25;
    std::cout << "Dimesnion StockxVariance: " << m1+1 << "x" << m2+1 << std::endl;

    const int m = (m1 + 1) * (m2 + 1);

    const int N = 15;
    const double delta_t = T / N;
    const double theta = 0.8;
    std::cout << "Time Dimension: " << N << std::endl;
    std::cout << "Theta: " << theta << std::endl;

    // Create grid
    Grid grid(m1, 8*K, S_0, K, K/5, m2, 5.0, V_0, 5.0/500);


    // Initialize matrices
    heston_A0Storage_gpu A0(m1, m2);
    heston_A1Storage_gpu A1(m1, m2);
    heston_A2_shuffled A2_shuf(m1, m2);
    
    // Build matrices
    A0.build_matrix(grid, rho, sigma);
    A1.build_matrix(grid, rho, sigma, r_d, r_f);
    A2_shuf.build_matrix(grid, rho, sigma, r_d, kappa, eta);
    
    // Build implicit systems
    A1.build_implicit(theta, delta_t);
    A2_shuf.build_implicit(theta, delta_t);

   
    // Create boundary conditions
    BoundaryConditions bounds(m1, m2, r_d, r_f, N, delta_t);
    bounds.initialize(Kokkos::View<double*>(grid.Vec_s.data(), grid.Vec_s.size()));

    // Create initial condition and result vectors
    Kokkos::View<double*> U_0("U_0", m);
    Kokkos::View<double*> U("U", m);

    // Initialize U_0 with payoff
    auto h_U_0 = Kokkos::create_mirror_view(U_0);
    for (int j = 0; j <= m2; j++) {
        for (int i = 0; i <= m1; i++) {
            h_U_0(i + j*(m1+1)) = std::max(grid.Vec_s[i] - K, 0.0);
        }
    }
    Kokkos::deep_copy(U_0, h_U_0);

    // Run CS scheme with shuffling
    CS_scheme_shuffled(m, m1, m2, N, U_0, delta_t, theta, A0, A1, A2_shuf, bounds, r_f, U);

    // Print results if needed
    auto h_U = Kokkos::create_mirror_view(U);
    Kokkos::deep_copy(h_U, U);
    
    // Find option price at S_0 and V_0
    int index_s = std::find(grid.Vec_s.begin(), grid.Vec_s.end(), S_0) - grid.Vec_s.begin();
    int index_v = std::find(grid.Vec_v.begin(), grid.Vec_v.end(), V_0) - grid.Vec_v.begin();
    double option_price = h_U[index_s + index_v*(m1+1)];

    // Compare with reference price (from Python/Monte Carlo)
    const double reference_price = 8.8948693600540167;
    std::cout << "CS_scheme price: " << std::setprecision(12) << option_price << std::endl;
    std::cout << "CS_scheme Relative error: " << std::abs(option_price - reference_price)/reference_price << std::endl;
    //EXPECT_NEAR(option_price, reference_price, 0.1);
    ResultsExporter::exportToCSV("shuffled_heston_cs_scheme", grid, U);
}

void test_CS_schuffled_implied_vol(){
    // Test parameters
    double K = 100.0;
    double S_0 = K;
    double V_0 = 0.04;
    double T = 1.0;
    double r_d = 0.025;
    double r_f = 0.0;
    double rho = -0.9;
    double sigma = 0.3;
    double kappa = 1.5;
    double eta = 0.04;

    
    const int m1 = 200;
    const int m2 = 100;
    std::cout << "Dimesnion StockxVariance: " << m1+1 << "x" << m2+1 << std::endl;

    const int m = (m1 + 1) * (m2 + 1);

    const int N = 40;
    const double delta_t = T / N;
    const double theta = 0.8;
    std::cout << "Time Dimension: " << N << std::endl;
    std::cout << "Theta: " << theta << std::endl;

    // Create grid
    Grid grid(m1, 8*K, S_0, K, K/5, m2, 5.0, V_0, 5.0/500);


    // Initialize matrices
    heston_A0Storage_gpu A0(m1, m2);
    heston_A1Storage_gpu A1(m1, m2);
    heston_A2_shuffled A2_shuf(m1, m2);
    
    // Build matrices
    A0.build_matrix(grid, rho, sigma);
    A1.build_matrix(grid, rho, sigma, r_d, r_f);
    A2_shuf.build_matrix(grid, rho, sigma, r_d, kappa, eta);
    
    // Build implicit systems
    A1.build_implicit(theta, delta_t);
    A2_shuf.build_implicit(theta, delta_t);

   
    // Create boundary conditions
    BoundaryConditions bounds(m1, m2, r_d, r_f, N, delta_t);
    bounds.initialize(Kokkos::View<double*>(grid.Vec_s.data(), grid.Vec_s.size()));

    // Create initial condition and result vectors
    Kokkos::View<double*> U_0("U_0", m);
    Kokkos::View<double*> U("U", m);

    // Initialize U_0 with payoff
    auto h_U_0 = Kokkos::create_mirror_view(U_0);
    for (int j = 0; j <= m2; j++) {
        for (int i = 0; i <= m1; i++) {
            h_U_0(i + j*(m1+1)) = std::max(grid.Vec_s[i] - K, 0.0);
        }
    }
    Kokkos::deep_copy(U_0, h_U_0);

    // Run CS scheme with shuffling
    CS_scheme_shuffled(m, m1, m2, N, U_0, delta_t, theta, A0, A1, A2_shuf, bounds, r_f, U);

    // Print results if needed
    auto h_U = Kokkos::create_mirror_view(U);
    Kokkos::deep_copy(h_U, U);

    // Find option price at S_0 and V_0
    int index_s = std::find(grid.Vec_s.begin(), grid.Vec_s.end(), S_0) - grid.Vec_s.begin();
    int index_v = std::find(grid.Vec_v.begin(), grid.Vec_v.end(), V_0) - grid.Vec_v.begin();

    double option_price = h_U[index_s + index_v*(m1+1)];

    // Compare with reference price (from Python/Monte Carlo)
    const double reference_price = 8.8948693600540167;
    std::cout << "CS_scheme price: " << std::setprecision(12) << option_price << std::endl;
    //std::cout << "CS_scheme Relative error: " << std::abs(option_price - reference_price)/reference_price << std::endl;
    
    // Parameters for implied volatility computation
    const double epsilon = 0.01;
    const int num_s_inf = 5;
    const int num_s_sup = 6;
    const int num_v_inf = 1;
    const int num_v_sup = 10;

    // Initialize IV with correct dimensions
    std::vector<std::vector<double>> IV(num_v_sup - num_v_inf + 1, 
                                    std::vector<double>(num_s_sup + num_s_inf + 1));

    // No need for intermediate Vec_s_IV and Vec_v_IV vectors
    // Keep price array in original ordering
    std::vector<std::vector<double>> price(m2 + 1, std::vector<double>(m1 + 1));
    for(int j = 0; j <= m2; j++) {
        for(int i = 0; i <= m1; i++) {
            price[j][i] = h_U[i + j*(m1+1)];
        }
    }

    // Compute implied volatilities
    for(int j = 0; j < num_v_sup - num_v_inf + 1; j++) {
        int v_idx = index_v - num_v_inf + j;
        for(int i = 0; i < num_s_sup + num_s_inf + 1; i++) {
            int s_idx = index_s - num_s_inf + i;

            IV[j][i] = BlackScholes::reverse_BS_new(
                1,                    // CP
                grid.Vec_s[s_idx],   // S
                K,                   // Strike
                r_f,                 // Risk-free rate
                T,                   // Time to maturity
                0.5,                 // Initial guess
                price[v_idx][s_idx], // Target price
                epsilon              // Tolerance
            );
        }
    }

    // Write results to CSV file
    std::ofstream outfile("implied_volatility_surface.csv");

    // Write header
    outfile << "Variance/Stock";
    for(int i = 0; i < num_s_sup + num_s_inf + 1; i++) {
        int s_idx = index_s - num_s_inf + i;
        outfile << "," << grid.Vec_s[s_idx];
    }
    outfile << "\n";

    // Write data rows
    for(int j = 0; j < num_v_sup - num_v_inf + 1; j++) {
        int v_idx = index_v - num_v_inf + j;
        outfile << grid.Vec_v[v_idx];
        for(int i = 0; i < num_s_sup + num_s_inf + 1; i++) {
            outfile << "," << IV[j][i];
        }
        outfile << "\n";
    }
    outfile.close();
}

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
    double test_vol = BlackScholes::reverse_BS_new(CP, S, K, r, T, 0.5, call_price, epsilon);
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




void test_parallel_tridiagonal2() {
    using timer = std::chrono::high_resolution_clock;
    
    const int m1 = 300;  // size of each system
    const int m2 = 100;  // number of systems
    
    // Create arrays for all systems
    Kokkos::View<double*> diag("diag", (m2+1)*(m1+1));
    Kokkos::View<double*> lower("lower", (m2+1)*m1);
    Kokkos::View<double*> upper("upper", (m2+1)*m1);
    Kokkos::View<double*> x("x", (m2+1)*(m1+1));
    Kokkos::View<double*> b("b", (m2+1)*(m1+1));
    Kokkos::View<double*> temp("temp", (m2+1)*(m1+1));
    
    // Initialize (similar to before)
    auto h_b = Kokkos::create_mirror_view(b);
    for(int i = 0; i < (m2+1)*(m1+1); ++i) {
        h_b(i) = std::rand() / (RAND_MAX + 1.0);
    }
    Kokkos::deep_copy(b, h_b);
    Kokkos::deep_copy(diag, 2.0);
    Kokkos::deep_copy(lower, -1.0);
    Kokkos::deep_copy(upper, -1.0);

    auto t_start = timer::now();
    
    // Solve all systems in parallel
    Kokkos::parallel_for("parallel_tridiagonal", m2+1, KOKKOS_LAMBDA(const int j) {
        const int block_start = j * (m1 + 1);
        const int off_start = j * m1;
        
        // Forward sweep for system j
        temp(block_start) = diag(block_start);
        x(block_start) = b(block_start);
        
        for(int i = 1; i <= m1; i++) {
            const int curr_idx = block_start + i;
            const int off_idx = off_start + (i-1);
            
            double w = lower(off_idx) / temp(curr_idx-1);
            temp(curr_idx) = diag(curr_idx) - w * upper(off_idx);
            x(curr_idx) = b(curr_idx) - w * x(curr_idx-1);
        }
        
        // Back substitution for system j
        x(block_start + m1) /= temp(block_start + m1);
        for(int i = m1-1; i >= 0; i--) {
            const int curr_idx = block_start + i;
            const int off_idx = off_start + i;
            x(curr_idx) = (x(curr_idx) - upper(off_idx) * x(curr_idx+1)) 
                         / temp(curr_idx);
        }
    });
    
    auto t_end = timer::now();
    std::cout << "Parallel solve time: "
              << std::chrono::duration<double>(t_end - t_start).count()
              << " seconds" << std::endl;
}


void test_DO_scheme() {
    Kokkos::initialize();
        {
        //test_parallel_tridiagonal2();
        //test_heston_call();
        //test_DO_m1_convergence();
        //test_all_convergence();

        //test_heston_call_shuffled();
        //test_heston_call_shuffled_vary_m1();
        //test_shuffled_convergence();

        //has a bug in it, I dont think it is a bug, but rather bad numerics for the A2 matrix
        //we need to account for oszillation. Will produce fourth diagonal at the lower half of 
        //the matrix
        //test_CS_scheme_call();
        //test_CS_convergence();
        //test_CS_shuffled();
        test_CS_schuffled_implied_vol();

        } // All test objects destroyed here
    Kokkos::finalize();
    //test_black_scholes();
}