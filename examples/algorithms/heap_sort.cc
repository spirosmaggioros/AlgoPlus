#ifdef __cplusplus
#include "../../src/algorithms/sorting/heap_sort.h"
#include <vector>
#endif

int main() {
    int arr[] = {59, 84, 62, 35, 32, 89, 16, 44};
    int N = sizeof(arr) / sizeof(arr[0]);
    std::vector<int> vect(arr, arr + N);
    heap_sort(vect, N);
    std::vector<int>::iterator it;
    for (it = vect.begin(); it != vect.end(); it++) {
        std::cout << *it << " ";
    }
}