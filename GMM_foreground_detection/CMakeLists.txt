cmake_minimum_required(VERSION 3.8)
project(GMM_foreground_detection)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp GMM.cpp GMM.h)
#find_library(OpenCV)
find_package(OpenCV)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(GMM_foreground_detection ${SOURCE_FILES} GMM.cpp GMM.h)
target_link_libraries(GMM_foreground_detection ${OpenCV_LIBS})