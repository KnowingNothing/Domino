cmake_minimum_required(VERSION 3.0)
project(DOMINO C CXX)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(${CMAKE_SOURCE_DIR}/3rdparty/fmt)
add_subdirectory(${CMAKE_SOURCE_DIR}/3rdparty/googletest)

include_directories(${CMAKE_SOURCE_DIR}/include)

file(GLOB_RECURSE ALL_CXX_SRC_FILES ${CMAKE_SOURCE_DIR}/src/*.cc)
file(GLOB_RECURSE ALL_CXX_TEST_FILES ${CMAKE_SOURCE_DIR}/test/cpp/*.cc)

enable_testing()
add_executable(test_all ${ALL_CXX_SRC_FILES} ${ALL_CXX_TEST_FILES})

target_link_libraries(test_all fmt GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(test_all)
