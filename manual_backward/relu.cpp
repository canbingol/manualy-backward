//
// Created by Can on 8.11.2024.
//

#include "relu.h"

std::vector<double> relu::relu_value(std::vector<double> input) {
    std::vector<double> output;
    output.reserve(input.size());
    for (double i: input) {
        output.push_back(std::max(i, 0.0));
    }
    return output;
}
