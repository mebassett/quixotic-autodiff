cmake_minimum_required(VERSION 3.27)

project(silly-autodiff LANGUAGES CUDA CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CLANG_BUILD_NEW_DRIVER on)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set default build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# C++ compiler flags for different build types
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")

# CUDA compiler flags for different build types
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0 -DDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG --use_fast_math")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-O2 -g -lineinfo -DNDEBUG --use_fast_math")

find_package(CUDAToolkit REQUIRED)

add_subdirectory(silly_autodiff)
add_subdirectory(daft_autodiff)
add_subdirectory(mnistdata)

add_executable(ffnn examples/ffnn/ffnn.cu)
add_executable(daft-ffnn examples/ffnn/daft-ffnn.cu)
target_include_directories(ffnn PUBLIC ${PROJECT_SOURCE_DIR}/daft_autodiff)
target_include_directories(ffnn PUBLIC ${PROJECT_SOURCE_DIR}/silly_autodiff)
target_include_directories(ffnn PUBLIC ${PROJECT_SOURCE_DIR}/mnist)
target_link_libraries(ffnn PUBLIC cublas silly_autodiff mnistdata)
target_link_libraries(daft-ffnn PUBLIC cublas daft_autodiff mnistdata)


add_executable(lenet examples/lenet/lenet.cu)
add_executable(daft-lenet examples/lenet/daft-lenet.cu)
target_include_directories(lenet PUBLIC ${PROJECT_SOURCE_DIR}/silly_autodiff)
target_include_directories(lenet PUBLIC ${PROJECT_SOURCE_DIR}/daft_autodiff)
target_include_directories(lenet PUBLIC ${PROJECT_SOURCE_DIR}/mnist)
target_link_libraries(lenet PUBLIC cublas silly_autodiff mnistdata)
target_link_libraries(daft-lenet PUBLIC cublas daft_autodiff mnistdata)

#include(FetchContent)
#FetchContent_Declare(
#    googletest
#    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
#)
#
#FetchContent_MakeAvailable(googletest)
find_package(GTest REQUIRED)

enable_testing()

add_executable(
    tests
    tests/silly_autodiff.test.cu
    tests/daft_innerproduct.test.cu
    tests/daft_matrix_product.test.cu
    tests/daft_scalar_multiply.test.cu
    tests/daft_add.test.cu
    tests/daft_leaky_relu.test.cu
    tests/daft_convolution.test.cu
    tests/daft_max_pool.test.cu
    tests/daft_concat.test.cu
    tests/daft_batch_compute.test.cu
    tests/daft_batch_innerproduct.test.cu
    tests/daft_negative_identity.test.cu
)
target_link_libraries(
    tests
    PRIVATE
    GTest::gtest_main
    silly_autodiff
    daft_autodiff
)
target_include_directories(
    tests
    PRIVATE
    "${CMAKE_BINARY_DIR}/_deps/googletest-src/googletest/include"
    ${PROJECT_SOURCE_DIR}/silly_autodiff
    ${PROJECT_SOURCE_DIR}/daft_autodiff
)
set_target_properties(tests PROPERTIES
       CUDA_SEPARABLE_COMPILATION ON
       CUDA_ARCHITECTURES "native")
set_target_properties(tests PROPERTIES
       CUDA_SEPARABLE_COMPILATION ON
       CUDA_ARCHITECTURES "native"
       CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER}
   )

target_compile_options(tests PRIVATE
       $<$<COMPILE_LANGUAGE:CUDA>:-I${CMAKE_BINARY_DIR}/_deps/googletest-src/googletest/include>
)
target_compile_options(tests PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-g >
)
add_test(NAME MemCheckTest COMMAND compute-sanitizer --tool memcheck --leak-check=full $<TARGET_FILE:tests>)
set_tests_properties(MemCheckTest PROPERTIES
    FAIL_REGULAR_EXPRESSION "ERROR SUMMARY: [^0]"  # Check for any non-zero errors summary.
)
include(GoogleTest)

gtest_discover_tests(tests)

