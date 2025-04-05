#ifndef GRAPH_VISUALIZATION_H
#define GRAPH_VISUALIZATION_H

#ifdef __cplusplus
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#endif

using namespace std;

/*
 *namespace for graph visualization function and .dot files creation.
 */
namespace graph_visualization {

#define OPEN_COMMAND "open"

/*
 *visualize function.
 *@param __generate: string that contains the needed data to generate the .dot
 *file.
 *@param newFileName: if needed, the filename can be changed(default
 *"unnamed.dot")
 */
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
            outFile << "graph G{" << std::endl;

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

}; // namespace graph_visualization

/*
 *namespace for di-graph visualization function and .dot files creation.
 */
namespace digraph_visualization {

#define OPEN_COMMAND "open"

/*
 *visualize function.
 *@param __generate: string that contains the needed data to generate the .dot
 *file.
 *@param newFileName: if needed, the filename can be changed(default
 *"unnamed.dot")
 */
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
            outFile << "digraph G{" << std::endl;

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

}; // namespace digraph_visualization

#endif // GRAPH_VISUALIZATION_H
