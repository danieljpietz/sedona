cmake_minimum_required(VERSION 3.23)
project(sedona_tests CXX)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

add_executable(sedona_tests main.cpp 2Rot1Pris/test_2rot1pris.cpp 2Rot1Pris/codegen/doublePendulumDerivative_terminate.cpp
        2Rot1Pris/codegen/doublePendulumDerivative.cpp
        2Rot1Pris/codegen/doublePendulumDerivative_data.cpp
        2Rot1Pris/codegen/doublePendulumDerivative_initialize.cpp
        2Rot1Pris/codegen/rt_nonfinite.cpp
        2Rot1Pris/codegen/rtGetInf.cpp
        2Rot1Pris/codegen/rtGetNaN.cpp)