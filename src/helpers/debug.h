#ifndef DEBUG_H
#define DEBUG_H

#ifdef __cplusplus
#include <iostream>
#include <type_traits>
#include <vector>
#endif

namespace DEBUG {
template <typename T> void print_arguments(const T& t) { // single argument
    std::cout << t << ' ';
}

template <typename T> void print_arguments(const std::vector<T>& t) { // vectors as arguments
    for (const T& x : t) {
        std::cout << x << ' ';
    }
}

template <class T, typename... Args>
void print_arguments(const std::vector<T>& t, Args&&... args) { // multiple args
    for (const T& x : t) {
        std::cout << x << ' ';
    }
    print_arguments(std::forward<Args>(args)...);
}

template <typename T, typename... Args>
void print_arguments(const T& t, Args&&... args) { // single args
    std::cout << t << ' ';
    print_arguments(std::forward<Args>(args)...);
}

template <typename Func, typename... Args> void get_args(Func&& callback, Args&&... args) {
    std::cout << "\033[33m" << "Arguments: ";
    print_arguments(args...);
    std::cout << '\n';
}
} // namespace DEBUG

#endif
