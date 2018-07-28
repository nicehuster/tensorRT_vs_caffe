cmake_minimum_required(VERSION 3.2)
project(caffe_mint)

set(CMAKE_CXX_STANDARD 11)


INCLUDE_DIRECTORIES(/home/niceliu/data/caffe/include)
LINK_DIRECTORIES(/home/niceliu/data/caffe/build/lib)

INCLUDE_DIRECTORIES(/usr/local/cuda/include)
LINK_DIRECTORIES(/usr/local/cuda/lib64)

FILE(GLOB CPP_LIST "${CMAKE_SOURCE_DIR}/*.cpp")
ADD_EXECUTABLE(caffe_MNIST_test ${CPP_LIST})

find_package(OpenCV 3)
target_link_libraries(caffe_MNIST_test ${OpenCV_LIBS} )
target_link_libraries(caffe_MNIST_test caffe cublas boost_system glog)
