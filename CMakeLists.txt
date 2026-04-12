cmake_minimum_required(VERSION 3.6)

project(rk_test)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# RKNN API path - optional local SDK layout
set(RKNN_API_PATH ${CMAKE_SOURCE_DIR}/3rd/rknn)

# Detect target arch (prefer system processor over compiler path matching)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(LIB_ARCH aarch64)
else()
    set(LIB_ARCH armhf)
endif()

# Try to find RKNN runtime library
find_library(RKNN_RT_LIB 
    NAMES rknnrt librknnrt.so
    PATHS 
        ${RKNN_API_PATH}/runtime/Linux/${LIB_ARCH}
        ${RKNN_API_PATH}/librknn_api/${LIB_ARCH}
        /usr/lib
        /usr/lib/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu
        /usr/local/lib
)

if(NOT RKNN_RT_LIB)
    message(FATAL_ERROR "RKNN runtime library not found. Please install librknnrt.so and ensure it is in /usr/lib or set RKNN_API_PATH correctly.")
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/includes)
if(EXISTS ${RKNN_API_PATH}/include)
    include_directories(${RKNN_API_PATH}/include)
endif()

find_package(Threads REQUIRED)

# C++ inference executable
add_executable(ai_ids_cpp main.cc flow_detection.cc)

target_link_libraries(ai_ids_cpp
    ${RKNN_RT_LIB}
    Threads::Threads
)

# Shared-memory consumer executable: read feature tensors from /dev/shm and run RKNN inference
add_executable(ai_ids_shm_consumer shm_rknn_consumer.cc flow_detection.cc)

target_link_libraries(ai_ids_shm_consumer
    ${RKNN_RT_LIB}
    Threads::Threads
)

# Set RPATH for runtime library loading
set_target_properties(ai_ids_cpp PROPERTIES
    INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib"
    BUILD_WITH_INSTALL_RPATH TRUE
)

set_target_properties(ai_ids_shm_consumer PROPERTIES
    INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# Print configuration info
message(STATUS "RKNN API Path: ${RKNN_API_PATH}")
message(STATUS "RKNN Runtime Library: ${RKNN_RT_LIB}")
message(STATUS "Library Architecture: ${LIB_ARCH}")


