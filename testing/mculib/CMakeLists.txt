# Copyright (c) 2022 ARM Limited. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.19.0)

set(MBED_PATH ${CMAKE_CURRENT_SOURCE_DIR}/mbed-os CACHE INTERNAL "")
set(MBED_CONFIG_PATH ${CMAKE_CURRENT_BINARY_DIR} CACHE INTERNAL "")
set(APP_TARGET mculib)

# set(MAIN_SOURCE main.cpp)
# set(MAIN_SOURCE ./test/cpp/test_matmul.cpp)
set(MAIN_SOURCE ./test/python/tmp.cpp)

include(${MBED_PATH}/tools/cmake/app.cmake)

project(${APP_TARGET})

add_subdirectory(${MBED_PATH})

add_executable(${APP_TARGET}
    ${MAIN_SOURCE}
)

target_link_libraries(${APP_TARGET} mbed-baremetal)

target_include_directories(${APP_TARGET}
    PUBLIC
    # CMSIS_5/CMSIS/DSP/Include
    # CMSIS_5/CMSIS/NN/Include
    # tinylib/include
    ./include
)

mbed_set_post_build(${APP_TARGET})

option(VERBOSE_BUILD "Have a verbose build process")
if(VERBOSE_BUILD)
    set(CMAKE_VERBOSE_MAKEFILE ON)
endif()