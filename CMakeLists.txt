cmake_minimum_required(VERSION 2.8.5 FATAL_ERROR)
#find_package(CUDA)

PROJECT (IntersectVTK)

if(NOT VTK_BINARY_DIR)
  find_package(VTK COMPONENTS
    vtkCommonCore
    vtkCommonDataModel
    vtkIOLegacy
  )
  include(${VTK_USE_FILE})
endif()

#cuda_compile(TRIANGLE_CLIP_CU TriangleClip.cu)

add_executable(IntersectVTK IntersectVTK.cxx )
target_link_libraries(IntersectVTK ${VTK_LIBRARIES})
