cmake_minimum_required(VERSION 3.21)
project(asl_2022)

set(CMAKE_CXX_STANDARD 14)
include_directories(/opt/homebrew/opt/openblas/include)
link_directories(/opt/homebrew/opt/openblas/lib)

add_executable(asl_2022
        cmake-build-debug/build.ninja
        cmake-build-debug/cmake_install.cmake
        cmake-build-debug/CMakeCache.txt
        baseline2.c)
