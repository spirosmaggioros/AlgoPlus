### Mini tutorial for kmeans class
    
    kmeans -- creates a kmeans object

### **Create an instance of kmeans:**:
```cpp
#include <machine_learning/clustering/kmeans/kmeans.h>

int main(){
    std::vector<std::vector<double> > data;
    int CLUSTERS;
    kmeans a(data, CLUSTERS);
    ...
}
```

### **Get assignments and cluster centers:**:

```cpp
#include <machine_learning/clustering/kmeans/kmeans.h>

int main(){
    ...
    // returns the cluster centers and assignemnts of the kmeans clustering
    std::pair<std::vector<std::vector<double> >, std::map<std::vector<double>, int64_t> > ans = a.fit();
}
```

### **optionally for visualization purposes you can write to a json file and then visualize it with python's matplotlib:**:

```cpp
#include <iostream>
#include <machine_learning/clustering/kmeans/kmeans.h>
#include "third_pary/json.hpp"
#include <string>

int main(int argc, char *argv[]){
  int CLUSTERS = atoi(argv[1]);
  srand((unsigned)time(NULL));
  std::ifstream ifs("data.json");
  json jf = json::parse(ifs);
  std::vector<std::vector<double> > data = jf["data"];

  json j;
  j["data"] = data;
  kmeans a(data, CLUSTERS);
  std::pair<std::vector<std::vector<double> >, std::map<std::vector<double> ,int64_t> > ans = a.fit();
  j["centroids"] = ans.first;
  j["assignments"] = ans.second;
  std::ofstream file("info.json");
  file << j;
}
```

```python
import json
import matplotlib.pyplot as plt
import os
import random

def fill_data():
    data = []
    for i in range(100):
        x = random.randint(0, 5)
        y = random.randint(0, 5)
        data.append([x, y])

    for i in range(100):
        x = random.randint(90, 100)
        y = random.randint(90, 100)
        data.append([x, y])

    for i in range(100):
        x = random.randint(-10, -5)
        y = random.randint(-10, -5)
        data.append([x, y])

    return data

# This is just a template, if you want more clusters please fill more colors to the COLORS list
if __name__ == "__main__":
    # here you should fill the data
    data = fill_data()
    CLUSTERS = 3
    data_dict = {
            "data":data
            }
    json_object = json.dumps(data_dict)
    
    with open("data.json", "w") as outfile:
        outfile.write(json_object)
    
    # run the kmeans clustering with CLUSTERS clusters
    os.system(f"c++ -std=c++17 kmeans.cc")
    os.system(f"./a.out {CLUSTERS}")
    
    input_name = open("info.json", 'r') 
    file = json.load(input_name)
    centroids = file['centroids']
    assignments = file['assignments']
    COLORS = ["red", "green", "orange", "blue", "black", "brown", "purple"]
    plt.figure()
    x, y = [], []
    x_noise, y_noise = [], []
    for ww in data:
        x.append(ww[0])
        y.append(ww[1])

    i = 0
    x_centroids, y_centroids, colors = [], [], []
    for ww in assignments:
        x_centroids.append(ww[0][0])
        y_centroids.append(ww[0][1])
        colors.append(COLORS[ww[1]])

    plt.scatter(x_centroids, y_centroids, color=colors)
    plt.show()

```


