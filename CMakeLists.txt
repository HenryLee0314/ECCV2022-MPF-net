CMAKE_MINIMUM_REQUIRED (VERSION 3.4)

PROJECT(EquirectProject)

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build/bin)

SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

SET(CMAKE_BUILD_TYPE "Release")

SET(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-Wall -O2 -g -std=c++11 -pthread")

ADD_DEFINITIONS(-DLOCAL_LOG_DIR=\"${PROJECT_SOURCE_DIR}/Log/\")

#########################################################
# Find OpenCV
#########################################################

FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})


MESSAGE(STATUS "OpenCV library status:")
MESSAGE(STATUS "---- version: ${OpenCV_VERSION}")
MESSAGE(STATUS "---- libraries: ${OpenCV_LIBS}")
MESSAGE(STATUS "---- include path: ${OpenCV_INCLUDE_DIRS}")

#########################################################
# Find OpenCL
#########################################################
FIND_PACKAGE(OpenCL REQUIRED)
INCLUDE_DIRECTORIES(${OpenCL_INCLUDE_DIRS})

MESSAGE(STATUS "OpenCL library status:")
MESSAGE(STATUS "---- libraries: ${OpenCL_LIBRARY}")
MESSAGE(STATUS "---- include path: ${OpenCL_INCLUDE_DIRS}")

#########################################################
# Fine Python
#########################################################
FIND_PACKAGE(Python COMPONENTS Interpreter Development REQUIRED)
INCLUDE_DIRECTORIES(${Python_INCLUDE_DIRS})

MESSAGE(STATUS "Python library status:")
MESSAGE(STATUS "---- version: ${Python_VERSION}")
MESSAGE(STATUS "---- libraries: ${Python_LIBRARY_DIRS}")
MESSAGE(STATUS "---- include path: ${Python_INCLUDE_DIRS}")

#########################################################
# Fine Pybind11
#########################################################
FIND_PACKAGE(pybind11 CONFIG REQUIRED)

#########################################################
INCLUDE_DIRECTORIES(include)
INCLUDE_DIRECTORIES(source)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/build)

set(CMAKE_OPENCL_PROGRAM_PATH "${PROJECT_SOURCE_DIR}/source/equirect2cylinder.cl")
configure_file(source/config.h.in cmake_auto_generate_config.h)

add_library(example SHARED 
    source/libexample.cpp

    source/cubemap_top.cpp
    source/log.cpp
    source/opencl_task.cpp
    source/opencl_manager.cpp
)

target_link_libraries(example PUBLIC pybind11::module)
TARGET_LINK_LIBRARIES(example PUBLIC ${OpenCV_LIBS})
target_link_libraries(example PUBLIC ${OpenCL_LIBRARY})