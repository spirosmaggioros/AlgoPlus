# Algoplus{1.0.0}

AlgoPlus is a C++ library that includes ready-to-use complex **data structures** and **algorithms** for various topics, including **Machine Learning** and **Image Processing**.

<div align="center">
  <img src="https://github.com/CSRT-NTUA/AlgoPlus/blob/main/assets/logo.png" alt="Algoplus">
</div>

[![Gitpod Ready-to-Code](https://img.shields.io/badge/Gitpod-Ready--to--Code-blue?logo=gitpod)](https://gitpod.io/#https://github.com/spirosmaggioros/AlgoPlus)
![tests](https://github.com/spirosmaggioros/AlgoPlus/actions/workflows/macos_test_cases.yml/badge.svg)
[![codecov](https://codecov.io/gh/spirosmaggioros/AlgoPlus/graph/badge.svg?token=3SBDRHUQR5)](https://codecov.io/gh/spirosmaggioros/AlgoPlus)
![GitHub repo size](https://img.shields.io/github/repo-size/spirosmaggioros/AlgoPlus)

### **See the full documentation [here](https://spirosmaggioros.github.io/AlgoPlus/)**

### **[Join](https://discord.gg/M9nYv4MHz6) our Discord**

### Example:

```cpp
#include "machine_learning/clustering/DBSCAN/dbscan.h"

// AlgoPlus now has Machine Learning classes!
int main(){
    std::vector<std::vector<double> > data;
    ...
    // Eps = 4, MinPts = 3
    DBSCAN a(data, 4, 3);

    // returns the clusters and noise of the DBSCAN clustering
    std::map<std::pair<double, double>, int64_t>  clusters = a.get_clusters();
    std::vector<std::pair<double, double> > noise = a.get_noise();
    ...
}

#include "machine_learning/image/edge_detection/sobel_operator.h"

// As well as image processing algorithms!
int main(){
  ...
  std::vector<std::vector<int32_t> > img(data);

  // Performs edge detection on image img
  std::vector<std::vector<int32_t> > resulted = Sobel(img);
  ...
}


#define ENABLE_GRAPH_VISUALIZATION
#include "graph/graph.h"
// And of course, every data structure that you need!
int main(){
  weighted_graph<int> g("undirected");
  g.add_edge(1, 4, 2);
  g.add_edge(4, 5, 6);
  g.add_edge(5, 2, 9);
  g.add_edge(2, 8, 10);

  // returns the shortest path from 1 to 2.
  std::cout << g.shortest_path(1, 2) << '\n';
  g.visualize() // You can visualize almost any of our implemented data structures!
}
```
You can see more [examples](/examples) or follow the [Tutorials](/tutorial).
> [!Note]
> This repository is a set of implementations and not a complete library meant for production or research. So whenever you see a bug or something not working as it should, please report it to us and we will try our best to fix it.

### Classes

**Machine Learning(NEW!)**
- [X] [Clustering Algorithms](https://en.wikipedia.org/wiki/Cluster_analysis)
- [X] [Encoders](https://en.wikipedia.org/wiki/Autoencoder)
- [X] [Regression Algorithms](https://en.wikipedia.org/wiki/Regression_analysis)
- [X] [Classification Algorithms](https://en.wikipedia.org/wiki/Classification)
- [X] [Shortest Path Algorithms](https://en.wikipedia.org/wiki/Shortest_path_problem)
- [X] [Image Processing Algorithms](https://en.wikipedia.org/wiki/Digital_image_processing)
- [X] [Activation Functions](https://en.wikipedia.org/wiki/Activation_function)
- [X] [Loss functions](https://en.wikipedia.org/wiki/Loss_function)

**Graphs**
- [X] [Di-Graph](https://en.wikipedia.org/wiki/Directed_graph)
- [X] [Graph](https://en.wikipedia.org/wiki/Graph_(discrete_mathematics))

**Trees**
- [X] [Red-Black Tree](https://en.wikipedia.org/wiki/Red%E2%80%93black_tree)
- [X] [AVL Tree](https://en.wikipedia.org/wiki/AVL_tree)
- [X] [Binary Search Tree](https://en.wikipedia.org/wiki/Binary_search_tree)
- [X] [Splay Tree](https://en.wikipedia.org/wiki/Splay_tree)
- [X] [Trie](https://en.wikipedia.org/wiki/Trie)
- [X] [Segment Tree](https://en.wikipedia.org/wiki/Segment_tree)
- [X] [Fenwick Tree](https://en.wikipedia.org/wiki/Fenwick_tree)
- [X] [Interval Tree](https://en.wikipedia.org/wiki/Interval_tree)
- [X] [234 Tree](https://en.wikipedia.org/wiki/2%E2%80%933%E2%80%934_tree)
- [X] [Min/Max Heap](https://en.wikipedia.org/wiki/Min-max_heap)

**Lists**
- [X] [Single Linked List](https://en.wikipedia.org/wiki/Linked_list)
- [X] [Doubly Linked List](https://en.wikipedia.org/wiki/Doubly_linked_list)
- [X] [Circular Linked List](https://www.geeksforgeeks.org/circular-linked-list)
- [X] [Skip List](https://en.wikipedia.org/wiki/Skip_list)

**Algorithms**
- [X] [Dynamic Programming](https://en.wikipedia.org/wiki/Dynamic_programming)
- [X] [Geometry](https://en.wikipedia.org/wiki/Computational_geometry)
- [X] [Number Theory](https://en.wikipedia.org/wiki/Number_theory)
- [X] [Searching](https://en.wikipedia.org/wiki/Search_algorithm)
- [X] [Sorting](https://en.wikipedia.org/wiki/Sorting_algorithm)
- [X] [String Manipulation](https://en.wikipedia.org/wiki/String_(computer_science))

**Other**
- [X] [Disjoint set](https://en.wikipedia.org/wiki/Disjoint-set_data_structure)
- [X] [Stack](https://en.wikipedia.org/wiki/Stack_(abstract_data_type))
- [X] [Queue](https://en.wikipedia.org/wiki/Queue_(abstract_data_type))
- [X] [Hash Table](https://en.wikipedia.org/wiki/Hash_table)
- [X] [Range Minimum Query](https://en.wikipedia.org/wiki/Range_minimum_query#:~:text=In%20computer%20science%2C%20a%20range,common%20prefix%20problem%20(LCP).)

> [!Tip]
> We are currently looking for contributions on machine learning classes and algorithms!

### **How to run test cases**
We have unit tests for every function of our implemented algorithms & data structures. It is very important to make sure that your code works before making any pull requests!

**Linux/MacOS**
```bash
mkdir build && cd build
cmake ..
make
cd tests
./runUnitTests
```
**Windows**
```bash
mkdir build
cmake -S . -B build -G Ninja
cmake --build build
cd build/tests
./runUnitTests
```

> [!Note]
> The splay tree's unit tests are failing in some OS's. We are working on a solution

### **Our contributors**
<a href="https://github.com/spirosmaggioros/AlgoPlus/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=spirosmaggioros/AlgoPlus" />
</a>

### **How to contribute**
1. Povide **new implementations** on our already implemented data structures & algorithms.

3. Implement **new classes / algorithms**.

2. Contribute to **unit testing** by writting unit tests for our classes and algorithms.

3. Create and contribute to **APIs for other languages**(Check AlgoPy branch).

4. **Promote** the repository on your local workshop/seminar and **get a shout-out**.

**Please have in mind that this work is open source and free to use under the Apache 2.0 licence. Please feel free
to cite the repo or our contributors for this work**.
See more [here](.github/CONTRIBUTE/CONTRIBUTE.md).

For any information or questions, please contact Spiros at **spirosmag@ieee.org**

### **Star History**

[![Star History Chart](https://api.star-history.com/svg?repos=spirosmaggioros/AlgoPlus&type=Date)](https://star-history.com/#spirosmaggioros/AlgoPlus&Date)
