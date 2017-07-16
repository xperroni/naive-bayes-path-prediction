project(MPC)

cmake_minimum_required (VERSION 3.5)

# -g allows for gdb debugging
# turn on -03 for best performance
add_definitions(-std=c++11)

set(CMAKE_CXX_FLAGS "-Wall")

add_executable(nb-predict
    "src/classifier.cpp"
    "src/main.cpp"
)
