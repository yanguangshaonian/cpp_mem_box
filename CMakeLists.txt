cmake_minimum_required(VERSION 3.22.1)
project(hash)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(include)
include_directories(lib)

add_compile_options(-O3)
add_compile_options(-DNDEBUG)

add_compile_options(-march=native)
add_compile_options(-mtune=native)
add_compile_options(-mwaitpkg)

add_compile_options(-fomit-frame-pointer)
add_compile_options(-funroll-loops)
add_compile_options(-falign-functions=32)

add_executable(main src/main.cpp)

target_link_libraries(main)

include(CheckIPOSupported)
check_ipo_supported(RESULT result OUTPUT output)

if(result)
    set_property(TARGET main PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    message(STATUS "IPO/LTO enabled for main and l3_test")
else()
    message(WARNING "IPO is not supported: ${output}")
endif()