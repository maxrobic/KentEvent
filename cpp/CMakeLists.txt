cmake_minimum_required(VERSION 3.5)
project(KentEvent CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io features visualization)
find_package(MetavisionSDK 4.6.2 COMPONENTS base core driver cv ui REQUIRED)

include_directories(${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(KentTracker KentTracker.cpp)
target_link_libraries(KentTracker
  MetavisionSDK::base MetavisionSDK::core MetavisionSDK::driver MetavisionSDK::cv MetavisionSDK::ui
  ${OpenCV_LIBS} ${PCL_LIBRARIES})
