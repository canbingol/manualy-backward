cmake_minimum_required(VERSION 3.29)
project(pytorch_)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(pytorch_
    activation.cpp
    activation.h
        cross_entropy_loss.cpp
        cross_entropy_loss.h
        layer.cpp
        main.h
        neuron.cpp
        relu.cpp
        main.cpp
        softmax.cpp)
