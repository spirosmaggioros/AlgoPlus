#ifndef TIMER_H
#define TIMER_H

#include <filesystem>
#include <utility>
#ifdef __cplusplus
#include "../../third_party/json.hpp"
#include <array>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <functional>
#include <iostream>
#include <tuple>
#endif

using nlohmann::json;

namespace TIMER {
// Helper function to invoke a function with arguments from a tuple
template <typename Func, typename Tuple, std::size_t... I>
void invoke_tuple(Func&& func, Tuple&& tup, std::index_sequence<I...>) {
    std::invoke(std::forward<Func>(func), std::get<I>(std::forward<Tuple>(tup))...);
}

// Function to measure execution time
template <typename Func, typename... Args> double exec_time(Func&& callback, Args&&... args) {
    auto start = clock();
    std::invoke(std::forward<Func>(callback), std::forward<Args>(args)...);
    auto end = clock();
    double total = (end - start) / double(CLOCKS_PER_SEC) * 1000;
    if (total > 1.0) {
        std::cout << "\033[33m" << "Total execution time: " << total << " ms\n";
    } else {
        std::cout << "\033[32m" << "Total execution time: " << total << " ms\n";
    }
    return total;
}

// Function to compare execution times of two functions
template <typename Func1, typename Tuple1, typename Func2, typename Tuple2>
int check_faster(Func1&& callback1, const Tuple1& args1, Func2&& callback2, const Tuple2& args2,
                 int verbose = 0) {
    // Run first function
    auto start = clock();
    invoke_tuple(std::forward<Func1>(callback1), args1,
                 std::make_index_sequence<std::tuple_size_v<Tuple1>>{});
    auto end = clock();
    auto total1 = (end - start) / double(CLOCKS_PER_SEC) * 1000;

    // Run second function
    start = clock();
    invoke_tuple(std::forward<Func2>(callback2), args2,
                 std::make_index_sequence<std::tuple_size_v<Tuple2>>{});
    end = clock();
    auto total2 = (end - start) / double(CLOCKS_PER_SEC) * 1000;

    if (verbose == 1) {
        std::cout << "Function 1 execution time: " << total1 << " ms\n";
        std::cout << "Function 2 execution time: " << total2 << " ms\n";
    }

    return (total1 > total2) ? 1 : 0;
}

template <typename Func, typename Tuple>
double complexity_analyzer_tuples(Func&& callback, const Tuple& args) {
    auto start = clock();
    invoke_tuple(std::forward<Func>(callback), args,
                 std::make_index_sequence<std::tuple_size_v<Tuple>>{});
    auto end = clock();
    double total = (end - start) / double(CLOCKS_PER_SEC) * 1000;
    return total;
}

template <typename Tuple> int sizeof_tuples(const Tuple args) {
    if (std::tuple_size<decltype(args)>::value == 1) {
        return std::get<0>(args);
    }

    return std::tuple_size<decltype(args)>::value;
}

template <typename Func, typename... Tuples>
void time_complexity(Func&& callback, const Tuples&... tuples) {
    if constexpr (sizeof...(tuples) < 5) {
        std::cout << "\033[34m" << "Data with size: " << (sizeof...(tuples))
                  << " are not enough to generate time complexity analysis" << '\n';
        return;
    }

    std::array<int, sizeof...(tuples)> input_size = {sizeof_tuples(tuples)...};
    std::array<double, sizeof...(tuples)> exec_time = {
        complexity_analyzer_tuples(callback, tuples)...};

    // write to file
    json j;
    j["input_size"] = input_size;
    j["execution_time"] = exec_time;
    std::ofstream file("info.json");
    file << j.dump(4);
    file.close();

    std::filesystem::path header_path = __FILE__;
    std::filesystem::path script_path = header_path.parent_path() / "analyzer.py";
    std::string command = "python3 " + script_path.string();
    int rr = system(command.c_str());
}
} // namespace TIMER

#endif
