# 块 1：基础声明
cmake_minimum_required(VERSION 3.16)   # 声明最低 CMake 版本；低于 3.16 会报错
project(proj1_vlm CXX)                  # 项目名 proj1_vlm，语言 CXX（C++）

# 块 2：C++ 标准
set(CMAKE_CXX_STANDARD 11)          # 用 C++11 标准
set(CMAKE_CXX_STANDARD_REQUIRED ON) # 编译器不支持 C++11 就直接报错

# 块 3：找 OpenCV
find_package(OpenCV REQUIRED)   # 在系统里找 OpenCV，找不到就报错（REQUIRED）

# 块 4：定义路径变量
set(RKNN_INC   ${CMAKE_SOURCE_DIR}/include/rknn_inc)
set(RKLLM_INC  ${CMAKE_SOURCE_DIR}/include/rkllm_inc)
set(PFFT_INC   ${CMAKE_SOURCE_DIR}/include/pocketfft_inc)
set(MINIAUDIO  ${CMAKE_SOURCE_DIR}/include)
set(LIB_DIR    ${CMAKE_SOURCE_DIR}/lib)
# ${CMAKE_SOURCE_DIR}：CMake 内置变量 = CMakeLists.txt 所在目录的绝对路径


# 块 5：头文件搜索路径
include_directories(
  ${CMAKE_SOURCE_DIR}/src
  ${RKNN_INC} ${RKLLM_INC} ${PFFT_INC} ${MINIAUDIO}
  /usr/include/rga
)


# 块 6：生成可执行文件 + 链接
add_executable(demo
  src/main.cpp
  src/image_enc.cc
  src/audio_enc.cc
  src/audio_utils.cc
  src/v4l2_capture.cpp
  src/rga_converter.cpp
  src/camera_stream.cpp
)
# add_executable(目标名 源文件...)：把列出的 .cpp 编译并链接成一个叫 demo 的可执行文件；

target_link_libraries(demo PRIVATE
  ${OpenCV_LIBS}
  ${LIB_DIR}/librknnrt.so
  ${LIB_DIR}/librkllmrt.so
  rga
  pthread dl m
)
# target_link_libraries(目标 库...)：给 demo 链接这些库，对应 g++ 的 -l / 给 .so 路径；
