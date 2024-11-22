//
// Created by Can on 3.11.2024.
//

#ifndef RELU_H
#define RELU_H
#include <vector>


class relu {
public:
    static std::vector<double> relu_value(std::vector<double> input);

    static double deriv_relu(double input);
};


#endif //RELU_H
