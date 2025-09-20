GPU-Accelerated PDE-Based Option Pricing Framework
This codebase is part of my Master's Thesis in Mathematics. The main goal was to develop high-performance computing solutions for option pricing and stochastic model calibration in real-time to market data. The numerical approach uses PDE-based methods for option pricing and the Levenberg-Marquardt algorithm for calibration.
A good starting point to understand the theory behind this codebase is my Reddit post:
https://www.reddit.com/r/quant/comments/1kj9kle/project_interactive_gpuaccelerated_pde_solver_for/
Performance Benchmarks
This framework can price American and European options, supporting underlyings that pay dividends.



Performance Benchmarks
This framework can price American and European options, supporting underlyings that pay dividends.

<img width="972" height="694" alt="image" src="https://github.com/user-attachments/assets/c7a77b51-1c52-45e4-b804-f300c12e0069" />

Key Features

Single European Option: 0.003s (A100RTX 2080)
500 American Options with Dividends: 0.02s (A100)
GPU Speedup vs CPU: 30x


Real-time Heston Calibration: GPU-accelerated Levenberg-Marquardt algorithm

<img width="942" height="592" alt="image" src="https://github.com/user-attachments/assets/6448a156-499b-4e02-9605-5359ff94dc00" />


Supported Option Types: European/American calls and puts, with and without dividends
Stochastic Volatility Models: Heston (done) and Scott-Chesney model (easily extendable)

Technical Presentation
I presented the following slides at DK Investment Bank:
[DK_Bank.pdf](https://github.com/user-attachments/files/22441146/DK_Bank.pdf)


References

Haentjens, T. & in 't Hout, K.J. (2018). "ADI schemes for pricing American options under the Heston model"

Buehler, H. (2018). "Volatility and Dividends II: Consistent Cash Dividends", J.P. Morgan QR

in 't Hout, K.J. & Foulon, S. "ADI Finite Difference Schemes for Option Pricing in the Heston Model with Correlation"

Douglas, J. (1962). "Alternating direction methods for three space variables"



