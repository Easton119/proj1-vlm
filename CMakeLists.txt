cmake_minimum_required(VERSION 3.16)
project(proj1_vlm CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 新增：默认 Release。
# 原来没设 CMAKE_BUILD_TYPE，cmake 会按空类型编译（等于 -O0），
# 图像预处理 / OpenCV / DSP 循环都会明显变慢。
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)        # ★ 新增：std::thread
# 阶段5：FFmpeg（只用来封装 FLV + RTMP，不做编解码）
find_package(PkgConfig REQUIRED)
pkg_check_modules(AVFORMAT REQUIRED libavformat libavcodec libavutil)
include_directories(${AVFORMAT_INCLUDE_DIRS})

set(RKNN_INC   ${CMAKE_SOURCE_DIR}/include/rknn_inc)
set(RKLLM_INC  ${CMAKE_SOURCE_DIR}/include/rkllm_inc)
set(PFFT_INC   ${CMAKE_SOURCE_DIR}/include/pocketfft_inc)
set(MINIAUDIO  ${CMAKE_SOURCE_DIR}/include)
set(LIB_DIR    ${CMAKE_SOURCE_DIR}/lib)

# ★ 新增：sherpa-onnx 不在项目里，用缓存变量方便换路径
set(SHERPA_ONNX_ROOT "/home/topeet/Qyj/sherpa-onnx" CACHE PATH "sherpa-onnx install root")

include_directories(
  ${CMAKE_SOURCE_DIR}/src
  ${RKNN_INC} ${RKLLM_INC} ${PFFT_INC} ${MINIAUDIO}
  /usr/include/rga
  ${SHERPA_ONNX_ROOT}/include          # ★ 新增
)

# ★ 新增：link_directories 必须写在 add_executable 之前才生效
link_directories(${SHERPA_ONNX_ROOT}/lib)

add_executable(demo
  src/main.cpp
  src/vlm_engine.cpp        
  src/voice_engine.cpp      
  src/tts_engine.cpp
  src/image_enc.cc
  src/audio_enc.cc
  src/audio_utils.cc
  src/v4l2_capture.cpp
  src/rga_converter.cpp
  src/camera_stream.cpp
  src/rga_lock.cpp          # 新增：RGA 全局锁（与编码器共用）
  src/mpp_encoder.cpp       # ★ 新增：MPP H.264 编码
  src/rtmp_publisher.cpp    # ★ 新增：FLV/RTMP 封装
)

target_link_libraries(demo PRIVATE
  ${OpenCV_LIBS}
  ${LIB_DIR}/librknnrt.so
  ${LIB_DIR}/librkllmrt.so
  sherpa-onnx-c-api         # ★ 新增
  asound                    # ★ 新增
  rga
  rockchip_mpp              # ★ 新增：MPP
  avformat avcodec avutil   # ★ 新增：FFmpeg（只做 FLV 封装 + RTMP 协议）
  Threads::Threads          # ★ 新增（替代裸 pthread）
  dl m
)

# ============ 阶段5：MPP 硬编码（诊断工具，先验证编码链路）============
add_executable(enc_dump
  tools/enc_dump.cpp
  src/camera_stream.cpp
  src/v4l2_capture.cpp
  src/mpp_encoder.cpp
  src/rga_lock.cpp
)

target_link_libraries(enc_dump PRIVATE
  ${OpenCV_LIBS}
  rockchip_mpp
  rga
  Threads::Threads
  dl m
)