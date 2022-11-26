cmake_minimum_required(VERSION 3.23)
project(sedona_benchmarks CXX)

set(CMAKE_CXX_STANDARD 20)
set( CMAKE_CXX_FLAGS "-std=c++2a" )
include_directories(include)

add_executable(sedona_2r1p_benchmarks 2Rot1Pris/benchmark_2rot1pris.cpp)





