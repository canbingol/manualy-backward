#include "layer.h"

#include <stdarg.h>
layer::layer(const std::vector<int> &num_neurons, const std::vector<double> &input)
    : num_layer(num_neurons.size()), num_neurons(num_neurons), current_input(input), num_inputs(input.size()) {
    this->inputs = input;  // 'inputs' üye değişkenini başlat
    initializeLayer();
}

void layer::initializeLayer() {
    for (int i = 0; i < num_layer - 1; i++) {
        neurons.emplace_back(num_neurons[i], current_input, "relu");
        current_input = neurons.back().getOutput(); // Son nöronun çıkışını al
    }
    neurons.emplace_back(num_neurons[num_layer - 1], current_input, "softmax");
    neuron::netOut = neurons.back().getOutput();
}

void layer::printLayers() const {
    for (int i = 0; i < neurons.size(); i++) {
        std::cout << "Layer " << (i + 1) << ", Neuron Count: " << num_neurons[i] << std::endl;
        neurons[i].print();
        std::cout << "-------------------------------------" << std::endl;
    }
}

void layer::backwardPass(double learning_rate,
                         const std::vector<double> &softmax_outputs, const std::vector<int> &target_class) {
    std::cout << "1. Cross entropy gradient calculation..." << std::endl;
    std::vector<std::vector<double> > dprobs = cross_entropy_loss::backward_cross_entropy(
        softmax_outputs, target_class);

    try {
        std::cout << "Checking dprobs size: rows=" << dprobs.size()
                  << ", cols=" << (dprobs.empty() ? 0 : dprobs.at(0).size()) << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "Error accessing dprobs: " << e.what() << std::endl;
    }

    std::cout << "2. Softmax gradient calculation..." << std::endl;
    std::vector<std::vector<double> > drivsoftmax = softmax::softmax_derivative(softmax_outputs);
    int rows = drivsoftmax.size(), cols = drivsoftmax.at(0).size();
    std::vector<std::vector<double> > dsoftmax(rows, std::vector<double>(cols, 0.0));

    try {
        std::cout << "Checking drivsoftmax size: rows=" << drivsoftmax.size()
                  << ", cols=" << (drivsoftmax.empty() ? 0 : drivsoftmax.at(0).size()) << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "Error accessing drivsoftmax: " << e.what() << std::endl;
    }

    std::cout << "3. Calculating dsoftmax..." << std::endl;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            dsoftmax.at(i).at(j) += drivsoftmax.at(i).at(j) * dprobs.at(i).at(j);
        }
    }

    try {
        for (size_t i = 0; i < dsoftmax.size(); ++i) {
            for (size_t j = 0; j < dsoftmax.at(i).size(); ++j) {
                std::cout << "dsoftmax[" << i << "][" << j << "] = " << dsoftmax.at(i).at(j) << std::endl;
            }
        }
    } catch (const std::out_of_range &e) {
        std::cerr << "Out of range error in dsoftmax: " << e.what() << std::endl;
    }

    std::cout << "4. Initializing gradsLastWeights and weights_input..." << std::endl;
    std::vector<std::vector<double> > gradsLastWeights(num_neurons.at(0), std::vector<double>(num_inputs, 0.0));
    std::vector<std::vector<double> > weights_input(num_neurons.at(0), std::vector<double>(num_inputs, 0.0));

    std::cout << "5. Populating weights_input..." << std::endl;
    for (int i = 0; i < weights_input.size(); i++) {
        for (int j = 0; j < weights_input.at(0).size(); j++) {
            weights_input.at(i).at(j) = inputs.at(j);
        }
    }

    try {
        for (size_t i = 0; i < weights_input.size(); ++i) {
            for (size_t j = 0; j < weights_input.at(i).size(); ++j) {
                std::cout << "weights_input[" << i << "][" << j << "] = " << weights_input.at(i).at(j) << std::endl;
            }
        }
    } catch (const std::out_of_range &e) {
        std::cerr << "Out of range error in weights_input: " << e.what() << std::endl;
    }

    std::cout << "6. Transposing weights_input into weights_inputT..." << std::endl;
    std::vector<std::vector<double> > weights_inputT(num_neurons.at(0), std::vector<double>(num_inputs, 0.0));
    for (int i = 0; i < weights_input.size(); i++) {
        for (int j = 0; j < weights_input.at(0).size(); j++) {
            weights_inputT.at(j).at(i) = weights_input.at(i).at(j);
        }
    }

    std::cout << "7. Calculating gradsLastWeights..." << std::endl;
    for (int i = 0; i < gradsLastWeights.size(); i++) {
        for (int j = 0; j < gradsLastWeights.at(0).size(); j++) {
            gradsLastWeights.at(i).at(j) = 0.0;
            for (int k = 0; k < gradsLastWeights[0].size(); k++) {
                gradsLastWeights.at(i).at(j) += dsoftmax.at(i).at(k) * weights_inputT.at(k).at(j);
            }
        }
    }

    try {
        for (size_t i = 0; i < gradsLastWeights.size(); ++i) {
            for (size_t j = 0; j < gradsLastWeights.at(i).size(); ++j) {
                std::cout << "gradsLastWeights[" << i << "][" << j << "] = " << gradsLastWeights.at(i).at(j) << std::endl;
            }
        }
    } catch (const std::out_of_range &e) {
        std::cerr << "Out of range error in gradsLastWeights: " << e.what() << std::endl;
    }

    std::cout << "8. Updating weights..." << std::endl;
    for (int i = 0; i < gradsLastWeights.size(); i++) {
        for (int j = 0; j < gradsLastWeights.at(i).size(); j++) {
            neurons.at(0).weights.at(i).at(j) += gradsLastWeights.at(i).at(j) * learning_rate;
        }
    }

    try {
        for (size_t i = 0; i < neurons.at(0).weights.size(); ++i) {
            for (size_t j = 0; j < neurons.at(0).weights[i].size(); ++j) {
                std::cout << "neurons[0].weights[" << i << "][" << j << "] = "
                          << neurons.at(0).weights.at(i).at(j) << std::endl;
            }
        }
    } catch (const std::out_of_range &e) {
        std::cerr << "Out of range error in neurons[0].weights: " << e.what() << std::endl;
    }

    std::cout << "10. Calculating gradsLastBiases..." << std::endl;
    std::vector<double> gradsLastBiases(neurons.at(0).bias.size(), 0.0);
    for (int j = 0; j < dsoftmax.at(0).size(); j++) {
        for (int i = 0; i < dsoftmax.size(); i++) {
            gradsLastBiases.at(j) += dsoftmax.at(i).at(j);
        }
    }

    try {
        for (size_t i = 0; i < gradsLastBiases.size(); ++i) {
            std::cout << "gradsLastBiases[" << i << "] = " << gradsLastBiases.at(i) << std::endl;
        }
    } catch (const std::out_of_range &e) {
        std::cerr << "Out of range error in gradsLastBiases: " << e.what() << std::endl;
    }

    std::cout << "11. Updating biases..." << std::endl;
    for (int i = 0; i < gradsLastBiases.size(); i++) {
        neurons.at(0).bias.at(i) += gradsLastBiases.at(i)* learning_rate;
    }
    try {
        for (size_t i = 0; i < neurons.at(0).bias.size(); ++i) {
            std::cout << "neurons[0].bias[" << i << "] = "
                      << neurons.at(0).bias.at(i) << std::endl;
        }
    } catch (const std::out_of_range &e) {
        std::cerr << "Out of range error in neurons[0].bias: " << e.what() << std::endl;
    }

}



double layer::nll_loss(const std::vector<int> &ys) {
    double loss = 0.0;
    for (int i = 0; i < ys.size(); i++) {
        int target_class = ys[i];

        double predicted_prob = neuron::netOut[target_class];
        if (predicted_prob > 0) {
            loss -= log(predicted_prob);
        } else {
            loss += 0.0;
        }
    }

    loss = loss / ys.size();

    return loss;
}

double layer::train(const std::vector<int> &ys) {
    double loss = nll_loss(ys);
    std::cout << "loss: " << loss << std::endl
            << std::endl;
    std::cout << "old weighs: " << std::endl;
    for (int i = 0; i < neurons[0].weights.size(); i++) {
        for (int j = 0; j < neurons[0].weights[0].size(); j++) {
            std::cout << neurons[0].weights[i][j] << " ";
        }
        std::cout << std::endl;
    };
    std::cout << "old biases: " << std::endl;
    for (int i = 0; i < neurons[0].weights.size(); i++) {
        std::cout << neurons[0].bias[i] << " ";
    };
    backwardPass(0.01, neuron::netOut, ys);
}
