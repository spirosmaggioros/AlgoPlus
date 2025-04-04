### **Mini tutorial on the MLP class**

## **Initialize a multi-layer perceptron model**
MLP::MLP takes 4 parameters: The input data: vector<vector<double> >, arch: vector<pair<int, int> >, epochs: int, learning_rate: double
```cpp
#include "machine_learning/nn/perceptron.h"

MLP::MLP model(data, { {3, 5}, {5, 5}, {5, 1} } /* arch */, 100 /* epochs */, 0.001 /* learning rate */);
```

## **fit the model on the input data**
To fit the model, you can just call the .fit() function. The training process utilizes the gradient descent of the nn::Linear
layer. You can see more in the Linear.md.
```cpp
model.fit()
```

## **perform inference**
To perform inference, you can call the perceptron::predict function that takes one parameter, the input validation data.
Returns the predicted label(you can static cast it to int if you wish).
```cpp

std::vector<double> val_data = { ... };
double y_pred = model.predict(val_data);
```
