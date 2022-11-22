cmake_minimum_required(VERSION 3.23)
project(sedona CXX)

set(CMAKE_CXX_STANDARD 11)

set(SEDONA_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Print the current version from the CONFIG file
# ==============================================

file(STRINGS "${SEDONA_INCLUDE_DIR}/sedona_config.h" sedona_version_defines
        REGEX "#define SEDONA_VERSION_(MAJOR|MINOR|PATCH)")
foreach(ver ${sedona_version_defines})
    if(ver MATCHES "#define SEDONA_VERSION_(MAJOR|MINOR|PATCH) +([^ ]+)$")
        set(XSIMD_VERSION_${CMAKE_MATCH_1} "${CMAKE_MATCH_2}" CACHE INTERNAL "")
    endif()
endforeach()
set(${PROJECT_NAME}_VERSION
        ${XSIMD_VERSION_MAJOR}.${XSIMD_VERSION_MINOR}.${XSIMD_VERSION_PATCH})
message(STATUS "sedona v${${PROJECT_NAME}_VERSION}")


set(SEDONA_HEADERS
        include/sedona.h
        include/okc.h
        include/mt_okc.h
        include/okc_differential_types.h
        include/okc_link.h
        include/okc_math.h
        include/okc_system.h
        include/okc_types.h include/sedona_config.h)

add_library(sedona INTERFACE)

target_include_directories(sedona INTERFACE
        $<BUILD_INTERFACE:${SEDONA_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:include>)
