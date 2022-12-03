# cmake 版本
cmake_minimum_required(VERSION 3.1)
# 项目名
project(trt_pro)
MESSAGE("===This is Windows,CMAKE_SYSTEM_NAME:${CMAKE_SYSTEM_NAME}")

# 编译增加C++11 支持
add_definitions(-std=c++11)
# 设置C++标准为 C++ 11
set(CMAKE_CXX_STANDARD 11)
#指定生成的版本
set(CMAKE_BUILD_TYPE Debug)
#设置可执行文件输出目录
#SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/bin)
#设置库输出目录
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/bin)

set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
#option 选项开关
#启用后，将在中使用CUDA运行时库的静态版本CUDA_LIBRARIES。
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
# 判断编译器类型
if(MSVC)     
    # Use the static C library for all build types
    foreach(var 
        CMAKE_C_FLAGS CMAKE_C_FLAGS_DEBUG CMAKE_C_FLAGS_RELEASE
        CMAKE_C_FLAGS_MINSIZEREL CMAKE_C_FLAGS_RELWITHDEBINFO
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO CUDA_NVCC_FLAGS CUDAFE_FLAGS CUDAFE_FLAGS_DEBUG
      )
      if(${var} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${var} "${${var}}")
      endif()
    endforeach()    
endif(MSVC)

# -D_MWAITXINTRIN_H_INCLUDED for solving error: identifier "__builtin_ia32_mwaitx" is undefined
##-Wall  #-Wfatal-errors 
#指定编译选项
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -W0 -MTd")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Ofast -D_MWAITXINTRIN_H_INCLUDED")

include_directories(${PROJECT_SOURCE_DIR}/include/)
include_directories(${PROJECT_SOURCE_DIR}/)

# setup CUDA
set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-std=c++11; -g; -G; -w; -gencode; arch=compute_75;code=sm_75)
#set(CUDAFE_FLAGS_DEBUG ${CUDAFE_FLAGS_DEBUG};/MTd)
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
enable_language(CUDA)  # add this line, then no need to setup cuda path in vs

# tensorrt
set(TRT_DIR "D:/tools/TensorRT-8.0.1.6-cu10.2/")  #3
include_directories(${TRT_DIR}\\include)
link_directories(${TRT_DIR}\\lib)

# setup opencv
set(OpenCV_DIR "D:/tools/opencv/build/x64/vc15/lib/")  #2
    find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# Threads
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
message(STATUS "Threads library status: ")
message(STATUS "    Threads_FOUND: ${Threads_FOUND}")
message(STATUS "    include path: ${Threads_FIND_QUIETLY}")

#打印信息
message("===message=========================================")
message(STATUS "CUDA library status:")
message(STATUS "    CUDA_FOUND: ${CUDA_FOUND}")
message(STATUS "    libraries: ${CUDA_LIBRARIES}")
message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")
message(STATUS "OpenCV library status: ")
message(STATUS "    OpenCV_FOUND: ${OpenCV_FOUND}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    OpenCV_DIR: ${OpenCV_DIR}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
message("===message=========================================")
set(LIB_FILES "")
set(SRC_FILES "")
FILE(GLOB SRC_FILES1 "src/*.*")
set(SRC_FILES ${SRC_FILES} ${SRC_FILES1})

# app_yolo
FILE(GLOB app_yolo_FILES "src/app_yolo/*.*")
include_directories(${PROJECT_SOURCE_DIR}/src/)
set(SRC_FILES ${SRC_FILES} ${app_yolo_FILES})

# app_http
FILE(GLOB app_yolo_FILES "src/app_http/*.*")
include_directories(${PROJECT_SOURCE_DIR}/src/)
set(SRC_FILES ${SRC_FILES} ${app_yolo_FILES})

include_directories(${PROJECT_SOURCE_DIR}/src/tensorRT/)

# 加载TensorRT_Pro
# builder
FILE(GLOB builder_FILES "src/tensorRT/builder/*.*")
source_group("src/tensorRT/builde" FILES ${builder_FILES})
set(SRC_FILES ${SRC_FILES} ${builder_FILES})

# common
FILE(GLOB common_FILES "src/tensorRT/common/*.*")
source_group("src/tensorRT/common" FILES ${common_FILES})
set(SRC_FILES ${SRC_FILES} ${common_FILES})


# infer
FILE(GLOB infer_FILES "src/tensorRT/infer/*.*")
source_group("src/tensorRT/infer" FILES ${infer_FILES})
set(SRC_FILES ${SRC_FILES} ${infer_FILES})

# onnxplugin
FILE(GLOB onnxplugin_FILES "src/tensorRT/onnxplugin/*.*")
source_group("src/tensorRT/onnxplugin" FILES ${onnxplugin_FILES})
set(SRC_FILES ${SRC_FILES} ${onnxplugin_FILES})

# onnxplugin-plugins
FILE(GLOB onnxplugin_plugins_FILES "src/tensorRT/onnxplugin/plugins/*.*")
source_group("src/tensorRT/onnxplugin/plugins" FILES ${onnxplugin_plugins_FILES})
set(SRC_FILES ${SRC_FILES} ${onnxplugin_plugins_FILES})

# 
include_directories(${PROJECT_SOURCE_DIR}/src/tensorRT/onnx_parser)

# onnx
include_directories(${PROJECT_SOURCE_DIR}/src/tensorRT/onnx)
FILE(GLOB onnx_FILES "src/tensorRT/onnx/*.*")
source_group("src/tensorRT/onnx" FILES ${onnx_FILES})
set(SRC_FILES ${SRC_FILES} ${onnx_FILES})

# onnx_parser
include_directories(${PROJECT_SOURCE_DIR}/src/tensorRT/onnx_parser)
FILE(GLOB onnx_parser_FILES "src/tensorRT/onnx_parser/*.*")
source_group("src/tensorRT/onnx_parser" FILES ${onnx_parser_FILES})
set(SRC_FILES ${SRC_FILES} ${onnx_parser_FILES})

##protobuf3.11.4
include_directories(${PROJECT_SOURCE_DIR}/third/protobuf3.11.4/include/)
link_directories(${PROJECT_SOURCE_DIR}/third/protobuf3.11.4/lib/)
set(LIB_FILES ${LIB_FILES} "libprotobufd")

#生成目标文件
add_executable(${PROJECT_NAME}  ${SRC_FILES})

target_link_libraries(${PROJECT_NAME} nvinfer cudart nvinfer_plugin nvonnxparser nvparsers ${CUDA_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} Threads::Threads)
target_link_libraries(${PROJECT_NAME} ${LIB_FILES})



