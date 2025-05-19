#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    constexpr int stride = 2;
    constexpr int input_dim = 8;
    constexpr int input_entries = input_dim * input_dim;
    constexpr int output_dim = input_dim / stride;
    constexpr int output_entries = output_dim * output_dim;

    std::vector<int> in_arr({1, 2, 6, 7, 8, 8, 3, 0,
                             8, 7, 1, 6, 8, 1, 4, 5,   
                             4, 6, 2, 7, 0, 2, 5, 6, 
                             3, 6, 2, 7, 4, 7, 0, 5,
                             2, 6, 1, 7, 4, 9, 3, 6,
                             4, 6, 8, 1, 4, 0, 1, 5,
                             3, 4, 7, 1, 2, 4, 7, 8,
                             7, 1, 5, 2, 4, 6, 1, 9});

    assert(in_arr.size() == input_entries);

    std::vector<int> out_arr(output_entries, 0);

    for (int k = 0; k < output_entries; k++) {
        int i = k / output_dim;
        int j = k % output_dim;

        int t1 = 2 * i;
        int t2 = 2 * i + 1;
        int t3 = 2 * j;
        int t4 = 2 * j + 1;

        int index_1 = t1 * input_dim + t3;
        int index_2 = t1 * input_dim + t4;
        int index_3 = t2 * input_dim + t3;
        int index_4 = t2 * input_dim + t4;

        out_arr[k] = std::max(in_arr[index_1], 
                     std::max(in_arr[index_2], 
                     std::max(in_arr[index_3], in_arr[index_4])));
    }

    for (int i = 0; i < output_entries; i++) {
        if (i % output_dim == 0) std::cout << '\n';
        std:: cout << out_arr[i] << ' ';
    }
    std::cout << '\n';

    return 0;
}