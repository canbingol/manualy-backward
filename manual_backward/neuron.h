//
// Created by Can on 2.11.2024.
//
#ifndef NEURON_H
#define NEURON_H

#include <vector>
#include <iostream>
#include <cstdlib>
#include "activation.h"


class neuron {
public:
    static std::vector<double> netOut ;

    neuron(int num_neuron, const std::vector<double> &input, const std::string &activation);

    void initializeWeights();

    void initializeBiases();

    void forwardPass(const std::string &activation);

    std::vector<double> getOutput() const;

    void print() const;
    std::vector<double> getWeights(int idx) {
        return weights[idx];
    }
    double randval() {
        return static_cast<double>(std::rand()) / RAND_MAX;
    }
    std::vector<std::vector<double> > weights;
    std::vector<double> output;

//private:
    int num_neuron;
    int num_input; // Input boyutu
    std::vector<double> input;
    std::vector<double> bias;

   /* double randval() {
        return static_cast<double>(std::rand()) / RAND_MAX * 2.0 - 1.0;
    }*/


};

#endif // NEURON_H
