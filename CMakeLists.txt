cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 14)

project(test)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP)

add_executable(test test.cc)
target_link_libraries(test ${OpenCV_LIBS} OpenMP::OpenMP_CXX)