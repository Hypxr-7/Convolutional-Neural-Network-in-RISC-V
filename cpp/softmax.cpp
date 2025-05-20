#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>


// double exp_approx(double x) {
//     // Simple exponential approximation using Taylor series (5 terms)
//     double term = 1.0;
//     double result = 1.0;
//     for (int i = 1; i <= 5; ++i) {
//         term *= x / i;
//         result += term;
//     }
//     return result;
// }

// std::vector<double> softmax(const std::vector<double>& input) {
//     std::vector<double> output(input.size());

//     // Find the maximum for numerical stability
//     double max_val = input[0];
//     for (double val : input) {
//         if (val > max_val) max_val = val;
//     }

//     // Compute exponentials and their sum
//     double sum = 0.0;
//     for (size_t i = 0; i < input.size(); ++i) {
//         output[i] = exp_approx(input[i] - max_val);
//         sum += output[i];
//     }

//     std::cout << "Sum: " << sum << '\n';

//     // Normalize
//     for (size_t i = 0; i < output.size(); ++i) {
//         output[i] /= sum;
//     }

//     return output;
// }

// Max: 7.82435
// Sum: 1.00958

double approximateExp(double x, int n=20) {
    double result = 1.0; // Start with the 0th term
    double term = 1.0;   // To store x^i / i!

    for (int i = 1; i < n; ++i) {
        term *= x / i;   // Efficiently compute each term
        result += term;
    }

    return result;
}


std::vector<double> softmax2(const std::vector<double>& x) {
    // Find the maximum value for numerical stability
    // double max_x = *std::max_element(x.begin(), x.end());

    double max_x = x[0];
    for (const auto element : x) max_x = std::max(max_x, element);

    std::cout << "Max: " << max_x << '\n';

    // Compute exponentials after shifting
    std::cout << "Exps: ";
    std::vector<double> exps;
    exps.reserve(x.size());
    for (double val : x) {
        exps.push_back(std::exp(val - max_x));
        std::cout << std::exp(val - max_x) << ' ';
    }

    std::cout << '\n';

    std::cout << "Aprox: ";
    std::vector<double> approx;
    for (auto val : x){
        approx.push_back(approximateExp(val - max_x));
        std::cout << approximateExp(val - max_x) << ' ';
    }
    std::cout << '\n';

    // Compute sum of exponentials
    double sum_exps = std::accumulate(exps.begin(), exps.end(), 0.0);

    std::cout << "Sum: " << sum_exps << '\n';

    // Compute softmax values
    std::vector<double> softmax_vals;
    softmax_vals.reserve(x.size());
    for (double val : exps) {
        softmax_vals.push_back(val / sum_exps);
    }

    return softmax_vals;
}


int main() {
    std::vector<double> values = {-7.942513,  -12.333942, -3.000695,  3.169582,
                                  -14.738624, 7.824346,   -11.468449, -4.637082,
                                  -2.740747,  -3.158628};

    std::vector<double> result = softmax2(values);

    std::cout << "Softmax result:\n";
    for (double val : result) {
        std::cout << val << " ";
    }
    std::cout << "\n";

    return 0;
}
