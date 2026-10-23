#pragma once
#include <cstdint>
#include <cstddef>
#include "rk_mpi.h"
#include "mpp_frame.h"
#include "mpp_buffer.h"
#include "mpp_meta.h"

struct DecodedFrame {
    MppFrame  frame  = nullptr;
    MppBuffer buffer = nullptr;
    int fd = -1;
    int width = 0, height = 0;
    int hor_stride = 0, ver_stride = 0;
    MppFrameFormat format = MPP_FMT_BUTT;
};

class MppJpegDecoder {
public:
    bool init(int width, int height);
    bool decode(const uint8_t* data, size_t len, DecodedFrame& out);
    void release_frame(DecodedFrame& f);
    void close();
private:
    MppCtx  mpp_ctx_ = nullptr;
    MppApi* mpi_     = nullptr;

    // 输出帧（advanced 模式预分配）
    MppBufferGroup frm_grp_ = nullptr;
    MppBuffer frm_buf_ = nullptr;
    MppFrame  out_frame_ = nullptr;

    // ★ 输入 buffer（把 MJPG 数据拷进去，packet 绑定它）
    MppBufferGroup in_grp_ = nullptr;
    MppBuffer in_buf_ = nullptr;
    size_t in_size_ = 0;

    int width_ = 0, height_ = 0;
    int hor_stride_ = 0, ver_stride_ = 0;
};