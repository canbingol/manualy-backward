//
// Created by Can on 4.11.2024.
//

#ifndef CROSS_ENTROPY_LOSS_H
#define CROSS_ENTROPY_LOSS_H

#include <vector>
#include <cmath>
#include <iomanip>

class cross_entropy_loss {
public:
    static double cross_entropy(const std::vector<double> &softmax_outputs, const std::vector<int> &target_class);

    static  std::vector<std::vector<double>> backward_cross_entropy(const std::vector<double> &softmax_outputs,
                                         const std::vector<int> &target_class);

   static std::vector<std::vector<int>> eye(int size, const std::vector<int> &classes);

};


#endif //CROSS_ENTROPY_LOSS_H
