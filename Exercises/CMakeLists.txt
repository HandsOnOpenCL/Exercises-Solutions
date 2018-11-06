cmake_minimum_required(VERSION 3.6)

# Copyright (C) 2018 Paul Keir, University of the West of Scotland

project(HOCL-Exercises)

find_package(OpenCL REQUIRED)
find_package(OpenMP REQUIRED)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "_CMakePredefinedTargets")

include_directories(${OpenCL_INCLUDE_DIRS} C_common Cpp_common)
link_libraries(${OpenCL_LIBRARY})
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")

set(name Exercise01)
set(tname ${name}-Cpp)
add_executable(${tname} ${name}/Cpp/DeviceInfo.cpp)
set_target_properties(${tname} PROPERTIES FOLDER Cpp)

set(name Exercise03)
set(tname ${name}-Cpp)
add_executable(${tname} ${name}/Cpp/vadd.cpp ${name}/Cpp/vadd.cl)
target_compile_definitions(${tname} PRIVATE EX3_VADD_CL_PATH="../${name}/Cpp/vadd.cl")
set_target_properties(${tname} PROPERTIES FOLDER Cpp)

set(name Exercise04)
set(tname ${name}-Cpp)
add_executable(${tname} ${name}/Cpp/vadd.cpp ${name}/Cpp/vadd.cl)
target_compile_definitions(${tname} PRIVATE EX4_VADD_CL_PATH="../${name}/Cpp/vadd.cl")
set_target_properties(${tname} PROPERTIES FOLDER Cpp)

set(name Exercise05)
set(tname ${name}-Cpp)
add_executable(${tname} ${name}/Cpp/vadd.cpp ${name}/Cpp/vadd.cl)
target_compile_definitions(${tname} PRIVATE EX5_VADD_CL_PATH="../${name}/Cpp/vadd.cl")
set_target_properties(${tname} PROPERTIES FOLDER Cpp)

set(name Exercise06)
set(tname ${name}-Cpp)
add_executable(${tname} ${name}/Cpp/matmul.cpp ${name}/Cpp/matmul.hpp ${name}/Cpp/matrix_lib.cpp ${name}/Cpp/matrix_lib.hpp)
set_target_properties(${tname} PROPERTIES FOLDER Cpp)

set(name Exercise09)
set(tname ${name}-Cpp)
add_executable(${tname} ${name}/Cpp/pi.cpp)
set_target_properties(${tname} PROPERTIES FOLDER Cpp)

############################

set(name Exercise01)
set(tname ${name}-C)
add_executable(${tname} ${name}/C/DeviceInfo.c)
set_target_properties(${tname} PROPERTIES FOLDER C)

set(name Exercise02)
set(tname ${name}-C)
add_executable(${tname} ${name}/C/vadd_c.c C_common/wtime.c C_common/device_info.c)
set_target_properties(${tname} PROPERTIES FOLDER C)

set(name Exercise04)
set(tname ${name}-C)
add_executable(${tname} ${name}/C/vadd_c.c C_common/wtime.c C_common/device_info.c)
set_target_properties(${tname} PROPERTIES FOLDER C)

set(name Exercise05)
set(tname ${name}-C)
add_executable(${tname} ${name}/C/vadd_c.c C_common/wtime.c C_common/device_info.c)
set_target_properties(${tname} PROPERTIES FOLDER C)

set(name Exercise06)
set(tname ${name}-C)
add_executable(${tname} ${name}/C/matmul.c ${name}/C/matmul.h ${name}/C/matrix_lib.c ${name}/C/matrix_lib.h C_common/wtime.c)
set_target_properties(${tname} PROPERTIES FOLDER C)

set(name Exercise09)
set(tname ${name}-C)
add_executable(${tname} ${name}/C/pi.c C_common/wtime.c)
set_target_properties(${tname} PROPERTIES FOLDER C)

set(name Exercise13)
set(tname ${name}-C)
add_executable(${tname} ${name}/C/gameoflife.c)
set_target_properties(${tname} PROPERTIES FOLDER C)

#set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT Exercise01-Cpp)
