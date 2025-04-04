### Mini tutorial for nn::Linear class

## **default constructor of nn::Linear**

nn::Linear takes 3 total parameters: in_features: int, out_features: int and bias: bool.
```cpp
#include "machine_learning/nn/nn.h"

nn::Linear layer(5 /* in_features */, 1 /* out_features */, true /* bias */);
```

## **forward method**

There's a forward method, just like in PyTorch, that allows data to pass through the model(more 
useful for multi-layer perceptrons). nn::Linear::forward takes just one parameter as a vector of double,
the input data(you can think it as an input tensor).
```cpp
std::vector<double> data = { ... };
layer.forward(data);
```

## **update_weights**

Updates the weights and the bias of the layer. nn::Linear::update_weights takes 3 parameters, the input data(tensor),
the error of the current iteration and the learning rate. The weights are updated using gradient descent:
$W^{(l)}_{ij} = W^{(l)}_{ij} - \alpha \frac{\partial}{\partial W^{(l)}_{ij}} J(W, b)$ for the weights and
$b^{(l)}_i = b^{(l)}_i - \alpha \frac{\partial}{\partial b^{(l)}_i} J(W, b)$ for bias.

```cpp
// training process...
double error = y_pred - label[i];
if (error != 0) { // if we do a mistake
    weights.update_weights(data[i], error, learning_rate); // update the error
}
```
