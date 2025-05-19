#include <iostream>
#include <vector>

using namespace std;

int main() {
    constexpr int N = 10;
    constexpr int M = 1152;

    vector<double> dense(M, 1.0);        // 1152 x 1
    vector<double> weights(N * M, 0.5);  // 10 x 1152
    vector<double> biases(N, 0.2);       // 10 x 1
    vector<double> result(N, 0.0);       // 10 x 1

    for (int i = 0; i < N; i++) {
        int j = i * M;  // start index of the weights the will be multiplied

        result[i] = 0.0;

        for (int k = 0; k < M; k++) {
            result[i] += weights[j + k] * dense[k];
        }
        result[i] += biases[i];
    }

    for (auto val : result) std::cout << val << ' ';

    std::cout << '\n';

    return 0;
}