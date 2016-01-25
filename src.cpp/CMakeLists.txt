cmake_minimum_required(VERSION 2.8)

project (vtk)
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_COMPILER g++-4.8)

find_package(VTK 5.10 REQUIRED)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  /opt/include/vtk-5.10/
  )

link_directories(
  /opt/lib/vtk-5.10
  )

add_library(vtk SHARED vtklib_util.cpp savevtk.cpp readvtk.cpp)
TARGET_LINK_LIBRARIES(vtk ${VTK_LIBRARIES})
