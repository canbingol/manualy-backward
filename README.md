# Single Neuron Forward and Backward Propagation
This project implements forward and backward propagation for a single neuron, including weight and bias updates. The model utilizes the ReLU and Softmax activation functions and optimizes parameters via gradient descent.

## Features
- Forward Propagation: Computes outputs across layers.
- Backward Propagation: Calculates gradients and updates weights and biases using cross-entropy loss and softmax derivatives.
- Loss Function: Implements Negative Log Likelihood (NLL) loss.
## Results
### Updated Parameters 


| Parameter      | Implementation                           | PyTorch                           |
|----------------|------------------------------------------|------------------------------------|
| **Biases**     | `[0.0585009, 0.0512659]`                | `[0.0581, 0.0484]`                |
| **Weights**    | `[[0.0143656, 0.5767], [0.193304, 0.808741]]` | `[[0.0013, 0.5636], [0.1915, 0.8105]]` |


## Known Issue !!!!
- Error: Process finished with exit code -1073740940 (0xC0000374)
- Cause: Likely due to memory corruption or invalid vector access during backward propagation.

### To debug:

- Place a breakpoint at line 161 in layer.cpp.
- Run in debug mode to inspect weights, biases, and intermediate gradients.
- Contributions to resolve this issue are welcome.

## How to Run
- Compile the code using a C++ compiler (e.g., GCC, Clang).
- Execute the binary.
- Debugging outputs are printed to the console.


For further questions or contributions, please contact me.
