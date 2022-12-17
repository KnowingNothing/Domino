cmake_minimum_required(VERSION 3.12)
project(domino C CXX)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")



add_subdirectory(${CMAKE_SOURCE_DIR}/3rdparty/fmt)
add_subdirectory(${CMAKE_SOURCE_DIR}/3rdparty/googletest)
add_subdirectory(${CMAKE_SOURCE_DIR}/3rdparty/pybind11)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
message(STATUS "Include Python dirs ${PYTHON_INCLUDE_DIRS}")
include_directories(${PYTHON_INCLUDE_DIRS})

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/3rdparty/fmt/include)
include_directories(${CMAKE_SOURCE_DIR}/3rdparty/pybind11/include)
include_directories(${CMAKE_SOURCE_DIR}/3rdparty/range-v3/include)

file(GLOB_RECURSE ALL_CXX_SRC_FILES ${CMAKE_SOURCE_DIR}/src/*.cc)
message(STATUS "Build with source files ${ALL_CXX_SRC_FILES}")
file(GLOB_RECURSE ALL_CXX_TEST_FILES ${CMAKE_SOURCE_DIR}/test/cpp/*.cc)

pybind11_add_module(domino MODULE ${ALL_CXX_SRC_FILES})

# link_libraries(${PYTHON_LINK_DIRS})
# link_libraries(fmt)
# add_library(domino SHARED ${ALL_CXX_SRC_FILES})
target_link_libraries(domino PUBLIC fmt)
message(STATUS "Link Python libs ${PYTHON_LIBRARIES}")
target_link_libraries(domino PUBLIC ${PYTHON_LIBRARIES})


# enable_testing()
# add_executable(test_all ${ALL_CXX_SRC_FILES} ${ALL_CXX_TEST_FILES})

# target_link_libraries(test_all fmt GTest::gtest_main)

# include(GoogleTest)
# gtest_discover_tests(test_all)
