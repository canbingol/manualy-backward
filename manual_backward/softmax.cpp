//
// Created by Can on 4.11.2024.
//

#include "softmax.h"

std::vector<double> softmax::softmax_value(std::vector<double> input) {
    double max_val = *std::max_element(input.begin(), input.end());
    std::vector<double> eps(input.size());

    double sum = 0;
    for (int i = 0; i < input.size(); i++) {
        eps[i] = exp(input[i] - max_val);
        sum += eps[i];
    }

    std::vector<double> output(input.size());
    for (int i = 0; i < input.size(); i++) {
        output[i] = eps[i] / sum;
    }

    return output;
}
std::vector<std::vector<double>> softmax::softmax_derivative(const std::vector<double>& input) {
    std::vector<double> softmax_output = softmax_value(input);
    size_t size = softmax_output.size();

    std::vector<std::vector<double>> jacobian(size, std::vector<double>(size, 0.0));

    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            if (i == j) {
                jacobian[i][j] = softmax_output[i] * (1 - softmax_output[i]);
            } else {
                jacobian[i][j] = -softmax_output[i] * softmax_output[j];
            }
        }
    }

    return jacobian;
}
