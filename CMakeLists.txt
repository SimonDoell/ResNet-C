cmake_minimum_required(VERSION 4.2.3)
project(ResNet C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-O0 -g -Werror -Wall -Wno-unused)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3 -Werror -Wall -Wno-unused)
endif()

add_executable(ResNet
  src/matrix.c
  src/layer.c
  src/network.c

  src/optimizers/sgd.c
  src/layers/dense.c
  
  src/main.c
)

target_link_libraries(ResNet m)