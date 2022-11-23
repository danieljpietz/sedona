cmake_minimum_required(VERSION 3.23)
project(sedona_tests CXX)

set(CMAKE_CXX_STANDARD 20)
set( CMAKE_CXX_FLAGS "-std=c++2a" )
include_directories(include)

enable_testing()

add_executable(sedona_2r1p_tests    2Rot1Pris/test_2rot1pris.cpp
                                    2Rot1Pris/codegen/doublePendulumDerivative_terminate.cpp
                                    2Rot1Pris/codegen/doublePendulumDerivative.cpp
                                    2Rot1Pris/codegen/doublePendulumDerivative_data.cpp
                                    2Rot1Pris/codegen/doublePendulumDerivative_initialize.cpp
                                    2Rot1Pris/codegen/rt_nonfinite.cpp
                                    2Rot1Pris/codegen/rtGetInf.cpp
                                    2Rot1Pris/codegen/rtGetNaN.cpp)


add_test(
        Test1 sedona_2r1p_tests
)
