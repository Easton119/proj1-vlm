// mpp_encoder.h —— RK3588 MPP 硬件 H.264 编码器封装
//
// 数据流：cv::Mat(BGR, CPU) --RGA--> NV12 直写 MPP 的 DMA-BUF --> MPP 编码 --> H.264 Annex-B
//
// 四个必须注意的点（都是别人的坑总结）：
//   1. buffer group 必须用 MPP_BUFFER_TYPE_DRM，mpp_buffer_get_fd() 的 fd 才能被
//      RGA 的 importbuffer_fd() 接受；用 ION 在部分内核上拿不到可用 fd。
//   2. RGA 单硬件，必须和 rga_converter 共用 GlobalRgaMutex()。
//   3. header mode 用 EACH_IDR，SPS/PPS 跟着每个 IDR 走，推流中途重连也能解。
//   4. 每帧 put_frame 后只取一次 packet（与参考实现一致）；首帧可能没有输出。

#ifndef MPP_ENCODER_H_
#define MPP_ENCODER_H_

#include <cstdint>
#include <vector>

#include <opencv2/core.hpp>
#include <rockchip/mpp_buffer.h>
#include <rockchip/mpp_frame.h>
#include <rockchip/rk_mpi.h>

class MppEncoder {
 public:
  MppEncoder() = default;
  ~MppEncoder();

  MppEncoder(const MppEncoder&) = delete;
  MppEncoder& operator=(const MppEncoder&) = delete;

  // width/height 为编码分辨率（内部按 16 对齐 stride）；bitrate 单位 bps
  bool Init(int width, int height, int fps, int bitrate_bps);
  void Release();

  // 编码一帧 BGR。返回 true 表示这一帧被编码器接受（packet 可能为空，首帧常见）。
  // keyframe 输出该包是否含 IDR。
  bool Encode(const cv::Mat& bgr, std::vector<uint8_t>* packet, bool* keyframe);

  // Annex-B 格式的 SPS/PPS（推流封装要用它构造 AVCDecoderConfigurationRecord）
  const std::vector<uint8_t>& header() const { return header_; }

  int width() const { return width_; }
  int height() const { return height_; }

 private:
  bool bgr_to_nv12(const cv::Mat& bgr, int dst_fd);

  int width_ = 0;
  int height_ = 0;
  int hor_stride_ = 0;
  int ver_stride_ = 0;
  int fps_ = 0;
  int bitrate_ = 0;
  int yuv_size_ = 0;
  int frame_count_ = 0;
  bool first_frame_ = true;

  MppCtx ctx_ = nullptr;
  MppApi* mpi_ = nullptr;
  MppBufferGroup buf_group_ = nullptr;

  std::vector<uint8_t> header_;
};

#endif  // MPP_ENCODER_H_