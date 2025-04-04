### Mini tutorial for DBSCAN class

    DBSCAN -- creates a dbscan object


### **Create an instance of DBSCAN:**:

```cpp
#include <machine_learning/clustering/DBSCAN/dbscan.h>

int main(){
    std::vector<std::pair<double, double> > data;
    double Eps;
    int64_t MinPts;

    DBSCAN d(data, Eps, MinPts);
    ...
}
```

### **Get assignments with DBSCAN clustering:**:

```cpp
#include <machine_learning/clustering/DBSCAN/dbscan.h>

int main(){
    ...
    // get assignments for each point with DBSCAN algorithm
    std::map<std::pair<double, double>, int64_t> assignments = d.get_clusters();
}
```

### **Get noise with DBSCAN clustering:**:

```cpp
#include <machine_learning/clustering/DBSCAN/dbscan.h>

int main(){
    ...
    // get noise with DBSCAN clustering
    std::vector<std::pair<double, double> > noise = d.get_noise();
}
```

### **optionally for visualization purposes you can write to a json file and then use the python's matplotlib function:**:
```cpp
#include "../../../../src/machine_learning/clustering/DBSCAN/dbscan.h"
#include <fstream>
#include <random>
#include "../../../../third_party/json.hpp"

using nlohmann::json;

int main(int argc, char * argv[]){
  srand((unsigned)time(NULL));
  std::ifstream ifs("data.json");
  json jf = json::parse(ifs);
  std::vector<std::pair<double, double> > data = jf["data"];

  DBSCAN a(data, 4, 3);
  std::map<std::pair<double, double>, int64_t>  clusters = a.get_clusters(); 
  std::vector<std::pair<double, double> > noise = a.get_noise();
  json j;
  j["noise"] = noise;
  for(auto & x : clusters){
    std::cout << "[" << x.first.first << " " << x.first.second << "]" << ": " << x.second << '\n';
  }
  j["centroids"] = clusters;
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

    data_dict = {
            "data":data
            }
    json_object = json.dumps(data_dict)
    
    with open("data.json", "w") as outfile:
        outfile.write(json_object)

    os.system("c++ -std=c++17 run.cc ../../../../src/machine_learning/clustering/DBSCAN/dbscan.cc")
    os.system("./a.out")
    
    input_name = open("info.json", 'r') 
    file = json.load(input_name)
    centroids = file['centroids']
    noise = file['noise']

    COLORS = ["red", "green", "orange", "blue", "black", "brown", "purple"]
    plt.figure()
    x, y = [], []
    x_noise, y_noise = [], []
    for ww in data:
        x.append(ww[0])
        y.append(ww[1])

    for ww in noise:
        x_noise.append(ww[0])
        y_noise.append(ww[1])
    i = 0
    x_centroids, y_centroids, colors = [], [], []
    for ww in centroids:
        x_centroids.append(ww[0][0])
        y_centroids.append(ww[0][1])
        colors.append(COLORS[ww[1]])

    plt.scatter(x_centroids, y_centroids, color=colors)
    plt.scatter(x_noise, y_noise, color="black")
    plt.show()
```

