//
// Created by Can on 4.11.2024.
//

#ifndef SOFTMAX_H
#define SOFTMAX_H
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>
class softmax {
public:
    static std::vector<double> softmax_value(std::vector<double> input);
  static  std::vector<std::vector<double>> softmax_derivative(const std::vector<double>& input);

};


#endif //SOFTMAX_H
