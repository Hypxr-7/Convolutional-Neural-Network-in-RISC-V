#include <iostream>
#include <cmath>

// Function to compute e^x using n terms of the Taylor series
double approximateExp(double x, int n) {
    if (x < 0) {
        return 1.0 / approximateExp(-x, n); // Use identity to avoid instability
    }

    double result = 1.0;
    double term = 1.0;

    for (int i = 1; i < n; ++i) {
        term *= x / i;
        result += term;
    }

    return result;
}

int main() {
    double x = -7.942513 - 7.824346; // ≈ -15.7669
    int n = 7;

    double approx = approximateExp(x, n);
    std::cout << "Taylor Approximation of e^" << x << " using " << n << " terms: " << approx << std::endl;
    std::cout << "std::exp(x) = " << std::exp(x) << std::endl;

    return 0;
}
