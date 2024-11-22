#ifndef LAYER_H
#define LAYER_H

#include "neuron.h"
#include <vector>
#include "cross_entropy_loss.h"
#include <cmath>

class layer {
public:
    layer(const std::vector<int> &num_neurons, const std::vector<double> &input);

    void initializeLayer();

    void printLayers() const;

    double train(const std::vector<int> &ys);

    double nll_loss(const std::vector<int> &ys);

    void backwardPass(int y, const std::string &activation, double learning_rate,
                      const std::vector<double> &softmax_outputs, const std::vector<int> &target_class);
    void backwardPass(double learning_rate,
                      const std::vector<double> &softmax_outputs, const std::vector<int> &target_class);
    int num_inputs;
    std::vector<double> inputs;
private:
    int num_layer;
    std::vector<int> num_neurons; // Her katmanda kaç nöron olduğunu tutar
    std::vector<neuron> neurons;
    std::vector<double> current_input; // Mevcut katmanın girişi
};

#endif // LAYER_H
