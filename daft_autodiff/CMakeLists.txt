cmake_minimum_required(VERSION 3.27)

project(daft_autodiff LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_library(daft_autodiff daft_autodiff.cu daft_kernels.cu daft_innerproduct.cu daft_inputs.cu daft_matrix_product.cu daft_convolution.cu daft_leaky_relu.cu daft_add.cu daft_scalar.cu daft_maxpool.cu daft_concat.cu)
target_link_libraries(daft_autodiff PUBLIC cublas )
