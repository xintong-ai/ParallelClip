cmake_minimum_required(VERSION 2.8)
PROJECT (IntersectVTK)

find_package(CUDA)

if(NOT VTK_BINARY_DIR)
  find_package(VTK COMPONENTS
    vtkCommonCore
    vtkCommonDataModel
    vtkIOLegacy
  )
  include(${VTK_USE_FILE})
endif()

include_directories(
		${CUDA_TOOLKIT_INCLUDE}
	)
	
#cuda_compile(TRIANGLE_CLIP_CU TriangleClip.cu)
#compile(INTERSECT_VTK_O IntersectVTK.cxx)

cuda_add_executable(${PROJECT_NAME} TriangleClip.cu IntersectVTK.cxx clip.h)
#add_executable(IntersectVTK IntersectVTK.cxx)
target_link_libraries(${PROJECT_NAME} ${VTK_LIBRARIES} ${CUDA_CUDART_LIBRARY})
