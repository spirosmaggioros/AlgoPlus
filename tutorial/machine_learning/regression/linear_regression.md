### Mini tutorial for linear regression class
    -- linear_regression : creates a linear regression object

### **Get results from linear regression(i.e. get a and b of y):**

```cpp
#include <machine_learning/regression/linear_regression/linear_regression.h>

int main(){
    std::vector<std::vector<double> > data = {};
    linear_regression a(data);

    std::pair<double, double> a_b = a.get_results();
    ...
}
```

### **Optionally you can use python's matplotlib for visualization purposes:**

```cpp
#include <iostream>
#include "lin_reg.h"
#include "../../../../third_party/json.hpp"
#include <string>
#include <fstream>

using nlohmann::json;

int main(){
  srand((unsigned)time(NULL));
  std::ifstream ifs("data.json");
  json jf = json::parse(ifs);
  std::vector<std::vector<double> > data = jf["data"];
  linear_regression lin_reg(data);
  std::pair<double, double> a_b = lin_reg.get_results();
  std::cout << a_b.first << " " << a_b.second << '\n';
  json j;
  j["a"] = a_b.first;
  j["b"] = a_b.second;
  std::ofstream file("info.json");
  file << j;
}
```

```python
import json
import matplotlib.pyplot as plt
import os
import numpy as np

def fill_data():
    data = []
    for i in range(400):
        data.append([i, i + 10])
    return data

def plot(data,a,b):
    for x,y in data:
        plt.scatter(x,y,color='blue')
    x = np.linspace(0, 400, num = 400)
    fx = []
    for i in range(len(x)):
        fx.append(a + b*i)
    print(fx)
    plt.plot(x, fx)
    plt.grid()
    plt.show()

       
# This is just a template, if you want more clusters please fill more colors to the COLORS list
if __name__ == "__main__":
    # here you should fill the data
    data = fill_data()
    data_dict = {
            "data":data
            }
    json_object = json.dumps(data_dict)
    
    with open("data.json", "w") as outfile:
        outfile.write(json_object)
    
    # run the kmeans clustering with CLUSTERS clusters
    os.system(f"c++ -std=c++17 lin_reg.cc")
    os.system("./a.out")
    
    input_name = open("info.json", 'r') 
    file = json.load(input_name)
    a = file['a']
    b = file['b']
    
    plot(data, a, b)
```
    
