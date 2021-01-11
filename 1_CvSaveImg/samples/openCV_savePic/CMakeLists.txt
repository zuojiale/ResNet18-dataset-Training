cmake_minimum_required	(VERSION 3.5)
get_filename_component 	(PROJECT_NAME ${CMAKE_CURRENT_LIST_DIR} NAME) 
project	(${PROJECT_NAME})
######################################################################
set(CMAKE_SYSTEM_NAME Linux)
set(TOOLCHAIN_DIR "/usr")
set(CMAKE_FIND_ROOT_PATH ${TOOLCHAIN_DIR})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_C_COMPILER ${TOOLCHAIN_DIR}/bin/x86_64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_DIR}/bin/x86_64-linux-gnu-g++)

set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/bin)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake")


find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
include_directories(${OpenCV_INCLUDE_DIRS})

#find_package(PCL 1.7 REQUIRED)
#include_directories(${PCL_INCLUDE_DIRS})
#add_definitions(${PCL_DEFINITIONS})

include_directories (./../../include)

aux_source_directory (./src SRC_LIST)

add_executable (${PROJECT_NAME} ${SRC_LIST})

target_link_libraries(${PROJECT_NAME} ${CMAKE_CURRENT_LIST_DIR}/../../lib/libtofsdk.a ${OpenCV_LIBS} ${PCL_LIBRARIES} pthread dl rt)
#target_link_libraries(${PROJECT_NAME} ${CMAKE_CURRENT_LIST_DIR}/../../lib/libtofsdk.a pthread dl rt)


message(STATUS)
message(STATUS "General configuration for ${PROJECT_NAME}")
message(STATUS "======================================================================")
message(STATUS "Platform:")
message(STATUS "  Arch / OS:				${CMAKE_HOST_SYSTEM_PROCESSOR} / ${CMAKE_HOST_SYSTEM_NAME} ${CMAKE_HOST_SYSTEM_VERSION}")
message(STATUS "  C/C++ Compiler:			${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "======================================================================")
message(STATUS)

