cmake_minimum_required(VERSION 2.8.4)
project(hf_cpp)

set(CMAKE_CXX_FLAGS "-pg ${CMAKE_CXX_FLAGS} -std=c++11 -O3 -march=native")
set(CMAKE_BUILD_TYPE Release)

# build supported on OSX and Linux
if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  include_directories(${CMAKE_CURRENT_BINARY_DIR}  /usr/local/include)
  link_directories(/usr/local/lib)
elseif (${CMAKE_SYSTEM_NAME} MATCHES "Linux")

  # OpenMP only for linux at this point
  find_package(OpenMP REQUIRED)
#  if (OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#  endif ()
include_directories(${CMAKE_CURRENT_BINARY_DIR} ~/local/include)
# TODO: change me for 
  link_directories(~/local/lib)
endif ()

find_package(OpenCV REQUIRED)
find_package(Armadillo REQUIRED)

add_executable(mnist_example mnist_example.cpp)
target_link_libraries(mnist_example ${OpenCV_LIBS} armadillo)


