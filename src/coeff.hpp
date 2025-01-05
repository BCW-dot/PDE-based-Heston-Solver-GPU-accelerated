#pragma once
#include <iostream>
#include <vector>




double delta_s(int i, int pos, const std::vector<double>& Delta_s);
double delta_v(int i, int pos, const std::vector<double>& Delta_v);

double beta_s(int i, int pos, const std::vector<double>& Delta_s);
double beta_v(int i, int pos, const std::vector<double>& Delta_v);

double alpha_s(int i, int pos, const std::vector<double>& Delta_s);
double alpha_v(int i, int pos, const std::vector<double>& Delta_v);

double gamma_s(int i, int pos, const std::vector<double>& Delta_s);
double gamma_v(int i, int pos, const std::vector<double>& Delta_v);