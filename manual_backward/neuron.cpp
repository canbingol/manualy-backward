#include "neuron.h"

std::vector<double> neuron::netOut;

neuron::neuron(int num_neuron, const std::vector<double> &input, const std::string &activation)
    : num_neuron(num_neuron), input(input) {
    num_input = input.size();
    weights.resize(num_neuron, std::vector<double>(num_input));
    bias.resize(num_neuron);
    output.resize(num_neuron);
    initializeWeights();
    initializeBiases();
    forwardPass(activation);
}

void neuron::initializeWeights() {
    for (int i = 0; i < num_neuron; i++) {
        for (int j = 0; j < num_input; j++) {
            weights[i][j] = randval() ; // Ağırlıkları başlat
        }
    }
}

void neuron::initializeBiases() {
    for (int i = 0; i < num_neuron; i++) {
        bias[i] = randval() * 0.1; // Biasları başlat
    }
}

std::vector<double> activate(const std::vector<double>& input, const std::string &activation) {
    if (activation == "relu") {
        return activation::relu_value(input);
    }
    if (activation == "softmax") {
        return activation::softmax_value(input);
    }
    return activation::relu_value(input);
}


void neuron::forwardPass(const std::string &activation) {
    for (int i = 0; i < num_neuron; i++) {
        output[i] = 0; // Çıkışları sıfırla
        for (int j = 0; j < num_input; j++) {
            output[i] += weights[i][j] * input[j];
        }
        output[i] += bias[i];

    }
    output = activate(output, activation);
}

std::vector<double> neuron::getOutput() const {
    return output; // Çıkışı döndür
}

void neuron::print() const {
    std::cout << "Weights:" << std::endl;
    for (const auto &weight_vector: weights) {
        for (const auto &weight: weight_vector) {
            std::cout << weight << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "Biases:" << std::endl;
    for (const auto &b: bias) {
        std::cout << b << " ";
    }
    std::cout << std::endl;
    std::cout << "Outputs:" << std::endl;
    for (const auto &out: output) {
        std::cout << out << " ";
    }
    std::cout << std::endl;
}
