#include <iostream>
#include <vector>

#include "cross_entropy_loss.h"
#include "layer.h"


int main() {
    std::vector<int> target_class = {1,3};
    std::vector<double> input = {0,4};
    layer l({2}, input);
    l.printLayers();
    const double loss = l.train(target_class);
    printf("loss: %f\n", loss);
    return 0;
}
