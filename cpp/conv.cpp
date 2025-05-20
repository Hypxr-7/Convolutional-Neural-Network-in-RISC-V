#include <iostream>
#include <vector>

using namespace std;

int main() {
    const int IMAGE_SIZE = 28;
    const int FILTER_SIZE = 5;
    const int STRIDE = 1;
    const int OUTPUT_SIZE = (IMAGE_SIZE - FILTER_SIZE) / STRIDE + 1;

    const int IMAGE_LENGTH = IMAGE_SIZE * IMAGE_SIZE;        // 784
    const int OUTPUT_LENGTH = OUTPUT_SIZE * OUTPUT_SIZE;     // 576

    vector<double> image(IMAGE_LENGTH, 1.0);                  // 28x28 input image, all 1s
    vector<double> out(OUTPUT_LENGTH, 0.0);                   // 24x24 output
    vector<double> filter(FILTER_SIZE * FILTER_SIZE, 0.5);    // 5x5 filter, all 0.5
    double bias = 1.0;

    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        for (int j = 0; j < OUTPUT_SIZE; ++j) {
            double sum = 0.0;

            // Apply filter
            for (int m = 0; m < FILTER_SIZE; ++m) {
                for (int n = 0; n < FILTER_SIZE; ++n) {
                    int image_row = i + m;
                    int image_col = j + n;

                    int image_index = image_row * IMAGE_SIZE + image_col;
                    int filter_index = m * FILTER_SIZE + n;

                    sum += image[image_index] * filter[filter_index];
                }
            }

            // Add bias
            sum += bias;

            // Store result in output
            out[i * OUTPUT_SIZE + j] = sum;
        }
    }

    // Print first few outputs
    cout << "First 10 output values:\n";
    for (int i = 0; i < 10; ++i) {
        cout << out[i] << " ";
    }
    cout << endl;   

    return 0;
}
