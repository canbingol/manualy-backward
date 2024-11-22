//
// Created by Can on 4.11.2024.
//

#include "cross_entropy_loss.h"

#include <iostream>

double cross_entropy_loss::cross_entropy(const std::vector<double> &softmax_outputs,
                                         const std::vector<int> &target_class) {
    double loss = 0.0;
    for (size_t i = 0; i < target_class.size(); i++) {
        int target = target_class[i];

        if (target >= 0 && target < softmax_outputs.size()) {
            loss -= log(softmax_outputs[target] + 1e-12);
        }
    }
    return loss / target_class.size();
}

std::vector<std::vector<double>> cross_entropy_loss::backward_cross_entropy(
    const std::vector<double> &softmax_outputs, const std::vector<int> &target_class) {
    int samples = target_class.size();
    int num_classes = softmax_outputs.size();

    std::vector<std::vector<int>> one_hot_targets = eye(num_classes, target_class);

    std::vector<std::vector<double>> dinputs(samples, std::vector<double>(num_classes, 0.0));

    for (size_t i = 0; i < samples; i++) {
        int target = target_class[i];
        dinputs[i][target] = -one_hot_targets[i][target] / softmax_outputs[target];
    }

    return dinputs;
}

std::vector<std::vector<int>> cross_entropy_loss::eye(int size, const std::vector<int> &classes) {
    std::vector<std::vector<int>> eye(size, std::vector<int>(size, 0));
    for (int i = 0; i < size; i++) {
        int colNo = classes[i];
        if (colNo >= 0 && colNo < size) {
            eye[i][colNo] = 1;
        }
    }
    return eye;
}
