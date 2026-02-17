cmake_minimum_required(VERSION 3.21)

project(ImageProcessing LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --------------------------------------------------
# Dependencies
# --------------------------------------------------

find_package(OpenCV REQUIRED)

# --------------------------------------------------
# Engine Library
# --------------------------------------------------

# Collect all cpp files under src/
file(GLOB_RECURSE ENGINE_SOURCES
    CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

add_library(cv_engine ${ENGINE_SOURCES})

target_include_directories(cv_engine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/external
)

target_link_libraries(cv_engine PUBLIC
    opencv_core
    opencv_highgui
    opencv_imgcodecs
    opencv_imgproc
)

# --------------------------------------------------
# Main App
# --------------------------------------------------

add_executable(image_app src/main.cpp)

target_link_libraries(image_app PRIVATE
    cv_engine
)

# --------------------------------------------------
# Demo App
# --------------------------------------------------

add_executable(demo examples/demo.cpp)

target_link_libraries(demo PRIVATE
    cv_engine
)
