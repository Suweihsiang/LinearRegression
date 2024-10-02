# CMakeList.txt: LinearRegression 的 CMake 專案，在此包含來源及定義
# 專案專屬邏輯。
#
cmake_minimum_required (VERSION 3.8)

project ("LinearRegression" LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)	
set(CMAKE_CUDA_STANDARD 17)

# 將來源新增到此專案的可執行檔。
set(EIGEN_SEARCH_HEADERS "D:/library/eigen3")
find_path(EIGEN_INCLUDE NAMES Eigen/Core PATHS ${EIGEN_SEARCH_HEADERS})
IF(NOT EXISTS ${EIGEN_INCLUDE})
	MESSAGE(FATAL_ERROR "Can't find Eigen")
ENDIF (NOT EXISTS ${EIGEN_INCLUDE})
MESSAGE("Found Eigen 3.0 : ${EIGEN_INCLUDE}")

set(OpenCV_DIR "D:/library/opencv/build")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(CUDA REQUIRED)
enable_language(CUDA)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(BEFORE ${PROJECT_SOURCE_DIR}/include${EIGEN_INCLUDE})
include_directories("D:/library")
include_directories("C:/Users/User/AppData/Local/Programs/Python/Python311/include")
include_directories("C:/Users/User/AppData/Local/Programs/Python/Python311/Lib/site-packages/numpy/core/include")
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
aux_source_directory(. SRC_LIST)
add_executable (${PROJECT_NAME} ${SRC_LIST})
target_include_directories(${PROJECT_NAME} PRIVATE "C:/Users/User/AppData/Local/Programs/Python/Python311/include")
target_link_libraries(${PROJECT_NAME} "C:/Users/User/AppData/Local/Programs/Python/Python311/libs/python311.lib")
target_link_libraries(${PROJECT_NAME} "C:/Users/User/AppData/Local/Programs/Python/Python311/libs/python311_d.lib")
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})


if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET LinearRegression PROPERTY CXX_STANDARD 20)
endif()

# TODO: 新增測試，並視需要安裝目標。
