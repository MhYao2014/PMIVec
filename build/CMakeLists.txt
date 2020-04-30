cmake_minimum_required(VERSION 3.5.1)
project(SecVec)

set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb -pthread")
set(CMAKE_CXX_STANDARD 11)
set(SOURCE_FILES
        ../src/args.cpp
        ../src/matrix.cpp
        ../src/vector.cpp
        ../src/dictionary.cpp
        ../src/Loss.cpp
        ../src/gradmanager.cpp
        ../src/LossFactory.cpp
        ../src/SecVec.cpp
        ../src/utility.cpp
        ../src/main.cpp)

set(HEADER_FILES
        ../src/args.h
        ../src/dictionary.h
        ../src/matrix.h
        ../src/vector.h
        ../src/gradmanager.h
        ../src/LossFactory.h
        ../src/SecVec.h
        ../src/utility.h
        ../src/Loss.h)

add_library(SecVec-shared SHARED ${SOURCE_FILES} ${HEADER_FILES} ../src/SecVec.cpp ../src/SecVec.h)
add_library(SecVec-static STATIC ${SOURCE_FILES} ${HEADER_FILES})
add_executable(SecVec ../src/main.cpp)
target_link_libraries(SecVec SecVec-static SecVec-shared)