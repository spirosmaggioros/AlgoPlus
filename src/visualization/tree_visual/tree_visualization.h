#ifndef TREE_VISUALIZATION_H
#define TREE_VISUALIZATION_H

#ifdef __cplusplus
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#endif

using namespace std;

// COLORS FOR TEXT

namespace tree_visualization {

#define OPEN_COMMAND "open"

inline void visualize(std::string& _generate, std::string newFileName = "unnamed.dot") {
    auto start_time = std::chrono::high_resolution_clock::now();
    try {
        if (newFileName.size() < 5 || newFileName.substr(newFileName.length() - 4) != ".dot")
            newFileName += ".dot";
        // newFileName = "examples/" + newFileName;
        //  Open the file for writing
        std::ofstream outFile(newFileName);

        // Check if the file is successfully opened
        if (outFile.is_open()) {
            // Write the DOT format header
            outFile << "digraph Tree {" << std::endl;

            // Generate DOT code recursively
            outFile << _generate;
            // Write the DOT format footer
            outFile << "}" << std::endl;

            // Close the file
            outFile.close();

            std::cout << "Visualization file '" << newFileName << "' created successfully."
                      << std::endl;
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration =
                std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            double runtime_sec = static_cast<double>(duration.count()) / 1e6;

            std::cout << "Visualization runtime: " << runtime_sec << " sec" << std::endl;
            std::string openCommand = OPEN_COMMAND + std::string(" ") + newFileName;
            system(openCommand.c_str());
        } else {
            std::cerr << "Error: Unable to open file '" << newFileName << "' for writing."
                      << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
};

}; // namespace tree_visualization

#endif // TREE_VISUALIZATION_H
