#include "mpp_encoder.h"

#include <im2d.hpp>      // RGA im2d API（CMake 已加 /usr/include/rga）
#include <RgaUtils.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iostream>

#include "rga_lock.h"

namespace {

inline int AlignUp(int v, int a) { return (v + a - 1) / a * a; }

// 扫 Annex-B 起始码，判断有没有 IDR(nal_unit_type == 5)
bool HasIdr(const uint8_t* d, size_t n) {
  size_t i = 0;
  while (i + 3 < n) {
    if (d[i] == 0 && d[i + 1] == 0 && d[i + 2] == 0 && d[i + 3] == 1) {
      const size_t h = i + 4;
      if (h < n && (d[h] & 0x1F) == 5) return true;
      i = h;
      continue;
    }
    if (d[i] == 0 && d[i + 1] == 0 && d[i + 2] == 1) {
      const size_t h = i + 3;
      if (h < n && (d[h] & 0x1F) == 5) return true;
      i = h;
      continue;
    }
    ++i;
  }
  return false;
}

}  // namespace

MppEncoder::~MppEncoder() { Release(); }

bool MppEncoder::Init(int width, int height, int fps, int bitrate_bps) {
  if (width <= 0 || height <= 0) return false;
  width_ = width;
  height_ = height;
  hor_stride_ = AlignUp(width, 16);
  ver_stride_ = AlignUp(height, 16);
  fps_ = fps > 0 ? fps : 15;
  bitrate_ = bitrate_bps > 0 ? bitrate_bps : 1500000;
  yuv_size_ = hor_stride_ * ver_stride_ * 3 / 2;
  frame_count_ = 0;
  first_frame_ = true;
  header_.clear();

  MPP_RET ret = mpp_create(&ctx_, &mpi_);
  if (ret != MPP_OK || ctx_ == nullptr || mpi_ == nullptr) {
    std::fprintf(stderr, "[enc] mpp_create failed: %d\n", ret);
    return false;
  }

  ret = mpp_init(ctx_, MPP_CTX_ENC, MPP_VIDEO_CodingAVC);
  if (ret != MPP_OK) {
    std::fprintf(stderr, "[enc] mpp_init(ENC, AVC) failed: %d\n", ret);
    return false;
  }

  MppEncCfg cfg = nullptr;
  if (mpp_enc_cfg_init(&cfg) != MPP_OK || cfg == nullptr) {
    std::fprintf(stderr, "[enc] mpp_enc_cfg_init failed\n");
    return false;
  }
  if (mpi_->control(ctx_, MPP_ENC_GET_CFG, cfg) != MPP_OK) {
    std::fprintf(stderr, "[enc] MPP_ENC_GET_CFG failed\n");
    mpp_enc_cfg_deinit(cfg);
    return false;
  }

  // 输入：NV12，stride 按 16 对齐
  mpp_enc_cfg_set_s32(cfg, "prep:width", width_);
  mpp_enc_cfg_set_s32(cfg, "prep:height", height_);
  mpp_enc_cfg_set_s32(cfg, "prep:hor_stride", hor_stride_);
  mpp_enc_cfg_set_s32(cfg, "prep:ver_stride", ver_stride_);
  mpp_enc_cfg_set_s32(cfg, "prep:format", MPP_FMT_YUV420SP);

  // 码控：CBR
  mpp_enc_cfg_set_s32(cfg, "rc:mode", MPP_ENC_RC_MODE_CBR);
  mpp_enc_cfg_set_s32(cfg, "rc:bps_target", bitrate_);
  mpp_enc_cfg_set_s32(cfg, "rc:bps_max", bitrate_ * 17 / 16);
  mpp_enc_cfg_set_s32(cfg, "rc:bps_min", bitrate_ * 15 / 16);
  mpp_enc_cfg_set_s32(cfg, "rc:fps_in_flex", 0);
  mpp_enc_cfg_set_s32(cfg, "rc:fps_in_num", fps_);
  mpp_enc_cfg_set_s32(cfg, "rc:fps_in_denorm", 1);
  mpp_enc_cfg_set_s32(cfg, "rc:fps_out_flex", 0);
  mpp_enc_cfg_set_s32(cfg, "rc:fps_out_num", fps_);
  mpp_enc_cfg_set_s32(cfg, "rc:fps_out_denorm", 1);
  mpp_enc_cfg_set_s32(cfg, "rc:gop", fps_);  // 1 秒一个 IDR

  mpp_enc_cfg_set_s32(cfg, "codec:type", MPP_VIDEO_CodingAVC);
  mpp_enc_cfg_set_s32(cfg, "h264:profile", 66);  // Baseline：兼容性最好
  mpp_enc_cfg_set_s32(cfg, "h264:level", 40);
  mpp_enc_cfg_set_s32(cfg, "h264:qp_init", 18);
  mpp_enc_cfg_set_s32(cfg, "h264:qp_min", 10);
  mpp_enc_cfg_set_s32(cfg, "h264:qp_max", 40);

  ret = mpi_->control(ctx_, MPP_ENC_SET_CFG, cfg);
  mpp_enc_cfg_deinit(cfg);
  if (ret != MPP_OK) {
    std::fprintf(stderr, "[enc] MPP_ENC_SET_CFG failed: %d\n", ret);
    return false;
  }

  MppEncHeaderMode header_mode = MPP_ENC_HEADER_MODE_EACH_IDR;
  if (mpi_->control(ctx_, MPP_ENC_SET_HEADER_MODE, &header_mode) != MPP_OK) {
    std::fprintf(stderr, "[enc] MPP_ENC_SET_HEADER_MODE failed\n");
    return false;
  }

  // DMA-BUF 池：DRM 类型，fd 可直接交给 RGA
  ret = mpp_buffer_group_get_internal(&buf_group_, MPP_BUFFER_TYPE_DRM);
  if (ret != MPP_OK) {
    std::fprintf(stderr, "[enc] mpp_buffer_group_get_internal failed: %d\n", ret);
    return false;
  }

  // 取 SPS/PPS
  std::vector<uint8_t> hdr_buf(4096);
  MppPacket hp = nullptr;
  if (mpp_packet_init(&hp, hdr_buf.data(), hdr_buf.size()) == MPP_OK && hp != nullptr) {
    mpp_packet_set_length(hp, 0);
    if (mpi_->control(ctx_, MPP_ENC_GET_HDR_SYNC, hp) == MPP_OK) {
      void* d = mpp_packet_get_pos(hp);
      const size_t n = mpp_packet_get_length(hp);
      if (d != nullptr && n > 0) {
        header_.assign(static_cast<uint8_t*>(d), static_cast<uint8_t*>(d) + n);
      }
    }
    mpp_packet_deinit(&hp);
  }

  std::printf("[enc] %dx%d fps=%d bitrate=%d bps stride=%dx%d header=%zu bytes\n",
              width_, height_, fps_, bitrate_, hor_stride_, ver_stride_,
              header_.size());
  return true;
}

bool MppEncoder::bgr_to_nv12(const cv::Mat& bgr, int dst_fd) {
  if (bgr.empty() || dst_fd < 0) return false;

  cv::Mat continuous;
  const cv::Mat* src = &bgr;
  if (!src->isContinuous()) {
    continuous = src->clone();  // RGA 要求连续内存
    src = &continuous;
  }

  std::lock_guard<std::mutex> lock(GlobalRgaMutex());

  const int src_wstride = static_cast<int>(src->step[0] / src->elemSize());
  const int src_size = static_cast<int>(src->step[0] * src->rows);

  rga_buffer_handle_t src_handle =
      importbuffer_virtualaddr(const_cast<void*>(static_cast<const void*>(src->data)), src_size);
  rga_buffer_handle_t dst_handle = importbuffer_fd(dst_fd, yuv_size_);
  if (src_handle == 0 || dst_handle == 0) {
    std::fprintf(stderr, "[enc] rga import failed src=%d dst=%d\n",
                 static_cast<int>(src_handle), static_cast<int>(dst_handle));
    if (src_handle) releasebuffer_handle(src_handle);
    if (dst_handle) releasebuffer_handle(dst_handle);
    return false;
  }

  rga_buffer_t src_img = wrapbuffer_handle(src_handle, src->cols, src->rows,
                                           RK_FORMAT_BGR_888, src_wstride, src->rows);
  rga_buffer_t dst_img = wrapbuffer_handle(dst_handle, width_, height_,
                                           RK_FORMAT_YCbCr_420_SP, hor_stride_,
                                           ver_stride_);

  // 一条指令完成 BGR888 -> NV12，并直接写进 MPP 的 DMA-BUF（零拷贝）
  const IM_STATUS st = imresize(src_img, dst_img);
  releasebuffer_handle(src_handle);
  releasebuffer_handle(dst_handle);

  if (st != IM_STATUS_SUCCESS) {
    std::fprintf(stderr, "[enc] rga imresize(BGR888->NV12) failed: %s\n", imStrError(st));
    return false;
  }
  return true;
}

bool MppEncoder::Encode(const cv::Mat& bgr, std::vector<uint8_t>* packet,
                        bool* keyframe) {
  if (packet != nullptr) packet->clear();
  if (keyframe != nullptr) *keyframe = false;
  if (ctx_ == nullptr || mpi_ == nullptr || buf_group_ == nullptr) return false;
  if (bgr.empty() || bgr.type() != CV_8UC3) return false;

  MppFrame frame = nullptr;
  MppBuffer buffer = nullptr;
  bool ok = false;

  do {
    if (mpp_frame_init(&frame) != MPP_OK) {
      frame = nullptr;
      std::fprintf(stderr, "[enc] mpp_frame_init failed\n");
      break;
    }
    mpp_frame_set_width(frame, width_);
    mpp_frame_set_height(frame, height_);
    mpp_frame_set_hor_stride(frame, hor_stride_);
    mpp_frame_set_ver_stride(frame, ver_stride_);
    mpp_frame_set_fmt(frame, MPP_FMT_YUV420SP);

    if (mpp_buffer_get(buf_group_, &buffer, yuv_size_) != MPP_OK) {
      std::fprintf(stderr, "[enc] mpp_buffer_get failed\n");
      break;
    }
    const int fd = mpp_buffer_get_fd(buffer);
    if (fd < 0) {
      std::fprintf(stderr, "[enc] mpp_buffer_get_fd failed\n");
      break;
    }
    if (!bgr_to_nv12(bgr, fd)) break;
    if (mpp_buffer_sync_end(buffer) != MPP_OK) {
      std::fprintf(stderr, "[enc] mpp_buffer_sync_end failed\n");
      break;
    }
    mpp_frame_set_buffer(frame, buffer);

    const int idr_interval = fps_ > 1 ? fps_ / 2 : 1;  // 每秒 2 个 IDR
    if (first_frame_ || (frame_count_ % idr_interval == 0)) {
      mpi_->control(ctx_, MPP_ENC_SET_IDR_FRAME, nullptr);
      first_frame_ = false;
    }

    if (mpi_->encode_put_frame(ctx_, frame) != MPP_OK) {
      std::fprintf(stderr, "[enc] encode_put_frame failed\n");
      break;
    }

    MppPacket pkt = nullptr;
    if (mpi_->encode_get_packet(ctx_, &pkt) == MPP_OK && pkt != nullptr) {
      uint8_t* d = static_cast<uint8_t*>(mpp_packet_get_pos(pkt));
      const size_t n = mpp_packet_get_length(pkt);
      if (d != nullptr && n > 0) {
        if (packet != nullptr) packet->assign(d, d + n);
        if (keyframe != nullptr) *keyframe = HasIdr(d, n);
      }
      mpp_packet_deinit(&pkt);
    }
    ok = true;
  } while (false);

  if (frame != nullptr) mpp_frame_deinit(&frame);
  if (buffer != nullptr) mpp_buffer_put(buffer);
  ++frame_count_;
  return ok;
}

void MppEncoder::Release() {
  if (buf_group_ != nullptr) {
    mpp_buffer_group_put(buf_group_);
    buf_group_ = nullptr;
  }
  if (ctx_ != nullptr) {
    mpp_destroy(ctx_);
    ctx_ = nullptr;
    mpi_ = nullptr;
  }
  header_.clear();
}