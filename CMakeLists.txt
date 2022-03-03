cmake_minimum_required(VERSION 3.21)
project(asl_2022)

set(CMAKE_CXX_STANDARD 14)
include_directories(/opt/OpenBLAS/include)
link_directories(/opt/OpenBLAS/lib)

add_executable(asl_2022
        src/baseline2/cmake-build-debug/build.ninja
        src/baseline2/cmake-build-debug/cmake_install.cmake
        src/baseline2/cmake-build-debug/CMakeCache.txt
        src/baseline2/baseline2.c)
