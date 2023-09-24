cmake_minimum_required(VERSION 3.10)

# Set some basic project attributes
project(OpenCVBeroepsProductKethanPonnuthurai
    VERSION 0.1
    DESCRIPTION "A Hello World Project")

include_directories(${OpenCV_INCLUDE_DIRS})

add_compile_options(-Wall -Wextra -Wconversion)

find_package(OpenCV REQUIRED)

set(SOURCES
    src/videoShapesBeroepsProduct.cpp
    src/ShapeFunctions.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})

# Link OpenCV libraries to your executable
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

# Include the configuration header in the build
target_include_directories(${PROJECT_NAME} PUBLIC "${PROJECT_BINARY_DIR}")

# Include the directory containing ShapeFunctions.hpp for header files
target_include_directories(${PROJECT_NAME} PUBLIC "${CMAKE_SOURCE_DIR}/src")