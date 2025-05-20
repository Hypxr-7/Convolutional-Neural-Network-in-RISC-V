#include <iostream>

float exp_approx(float x) {
    // Simple exponential approximation using Taylor series (5 terms)
    float term = 1.0f;
    float result = 1.0f;
    for (int i = 1; i <= 30; ++i) {
        term *= x / i;
        result += term;
    }
    return result;
}

int main() {
    std::cout << exp_approx(2) << '\n';

    return 0;
}

