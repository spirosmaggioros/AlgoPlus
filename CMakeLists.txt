cmake_minimum_required(VERSION 3.12)
project(algoplus VERSION 1.0.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(EXCLUDE_DIR "third_party/")
set(unit_name "algoplus")
set(package_name "${unit_name}")
set(package_version "1.0.0")

message(STATUS "Project '${PROJECT_NAME}', package '${package_name}' version: '${package_version}'")

add_definitions(-DENABLE_LINKED_LIST_VISUALIZATION)
add_definitions(-DENABLE_TREE_VISUALIZATION)
add_definitions(-DENABLE_GRAPH_VISUALIZATION)
add_definitions(-DENABLE_FREQUENCY_LIST_VISUALIZATION)
add_definitions(-DENABLE_DOUBLY_LINKED_LIST_VISUALIZATION)
add_definitions(-DENABLE_CIRCULAR_LINKED_LIST_VISUALIZATION)
add_definitions(-DENABLE_WEIGHTED_GRAPH_VISUALIZATION)

if(ENABLE_COVERAGE)
    set(CMAKE_CXX_FLAGS "-O0 -coverage")

    find_program(LCOV lcov REQUIRED)
    find_program(GENHTML genhtml REQUIRED)

    add_custom_target(coverage
        COMMAND ${LCOV} --directory . --capture --output-file coverage.info
        # generate report
        COMMAND ${GENHTML} --demangle-cpp -o coverage coverage.info
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
endif()

# Find clang-tidy and set it as a target property
find_program(CLANG_TIDY_EXECUTABLE NAMES clang-tidy)

if (CLANG_TIDY_EXECUTABLE)
    message(STATUS "Found clang-tidy: ${CLANG_TIDY_EXECUTABLE}")
else()
    message(WARNING "clang-tidy not found!")
endif()

# Collect all source files and headers
file(GLOB_RECURSE ALL_CXX_SOURCE_FILES "${CMAKE_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE ALL_HEADER_FILES "${CMAKE_SOURCE_DIR}/src/*.h")

# Combine all source and header files into a single list
list(APPEND ALL_FILES ${ALL_CXX_SOURCE_FILES})
list(APPEND ALL_FILES ${ALL_HEADER_FILES})

# Add a custom target to run clang-tidy
if (CLANG_TIDY_EXECUTABLE)
    add_custom_target(
        run-clang-tidy
        COMMAND ${CLANG_TIDY_EXECUTABLE}
        -p ${CMAKE_BINARY_DIR}
        -header-filter=.*
        -system-headers
        ${ALL_FILES}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Running clang-tidy..."
    )
endif()

enable_testing()

add_subdirectory(tests)
