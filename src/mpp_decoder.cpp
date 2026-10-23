#include "mpp_decoder.h"
#include <unistd.h>
#include <iostream>
#include <cstring>

// ============================================================
// init(): MJPEG 用 advanced 模式（预分配输出 frame buffer）
// ============================================================
bool MppJpegDecoder::init(int width, int height) {
    width_ = width; height_ = height;
    // MPP_ALIGN(x, 16) 的等价位运算：向上对齐到 16
    hor_stride_ = (width  + 15) & ~15;
    ver_stride_ = (height + 15) & ~15;
    // ① 创建 + 初始化为 MJPEG 解码器
    if (mpp_create(&mpp_ctx_, &mpi_) != MPP_OK) { std::cerr << "mpp_create failed\n"; return false; }
    if (mpp_init(mpp_ctx_, MPP_CTX_DEC, MPP_VIDEO_CodingMJPEG) != MPP_OK) {
        std::cerr << "mpp_init MJPEG failed\n"; return false;
    }

    // ② split_parse = 1（MJPEG 也要让 MPP 自己按帧分割）
    MppDecCfg cfg = nullptr;
    mpp_dec_cfg_init(&cfg);
    mpi_->control(mpp_ctx_, MPP_DEC_GET_CFG, cfg);
    mpp_dec_cfg_set_u32(cfg, "base:split_parse", 1);
    mpi_->control(mpp_ctx_, MPP_DEC_SET_CFG, cfg);
    mpp_dec_cfg_deinit(cfg);

    // ③ ★ MJPEG 必须设置输出格式（在解码前）
    MppFrameFormat out_fmt = MPP_FMT_YUV420SP;   // NV12
    mpi_->control(mpp_ctx_, MPP_DEC_SET_OUTPUT_FORMAT, &out_fmt);

    // ④ ★ advanced 模式：创建输出帧 buffer 组并预分配一块 frame
    //    大小 *4 是为了兼容 YUV422/其它格式（官方 mpi_dec_test 也是 *4）
    if (mpp_buffer_group_get_internal(&frm_grp_, MPP_BUFFER_TYPE_ION) != MPP_OK) {
        std::cerr << "get buffer group failed\n"; return false;
    }
    size_t buf_size = (size_t)hor_stride_ * ver_stride_ * 4;
    if (mpp_buffer_get(frm_grp_, &frm_buf_, buf_size) != MPP_OK) {
        std::cerr << "get frame buffer failed\n"; return false;
    }
    if (mpp_frame_init(&out_frame_) != MPP_OK) {
        std::cerr << "frame init failed\n"; return false;
    }
    mpp_frame_set_width(out_frame_, width_);
    mpp_frame_set_height(out_frame_, height_);
    mpp_frame_set_hor_stride(out_frame_, hor_stride_);
    mpp_frame_set_ver_stride(out_frame_, ver_stride_);
    mpp_frame_set_fmt(out_frame_, MPP_FMT_YUV420SP);
    mpp_frame_set_buffer(out_frame_, frm_buf_);

    // ★ 预分配输入 buffer：够放一帧 MJPG（1280x720 JPEG 约 200~300KB，给 4MB 足够）
    in_size_ = 4 * 1024 * 1024;
    if (mpp_buffer_group_get_internal(&in_grp_, MPP_BUFFER_TYPE_ION) != MPP_OK) {
        std::cerr << "get input group failed\n"; return false;
    }
    if (mpp_buffer_get(in_grp_, &in_buf_, in_size_) != MPP_OK) {
        std::cerr << "get input buffer failed\n"; return false;
    }

    std::cout << "MPP MJPEG decoder (advanced) initialized " << width_ << "x" << height_ << "\n";

    MPP_RET fmt_ret = mpi_->control(mpp_ctx_, MPP_DEC_SET_OUTPUT_FORMAT, &out_fmt);
    std::cout << "set output format ret=" << fmt_ret << "\n";

    return true;
}

// ============================================================
// decode(): 送一帧 MJPG → 解到预分配的 frame
// ============================================================
bool MppJpegDecoder::decode(const uint8_t* data, size_t len, DecodedFrame& out) {
    if (!data || len == 0) return false;

    // // ① 封装 MJPG 成 packet
    // MppPacket packet = nullptr;
    // mpp_packet_init(&packet, (void*)data, len);

    // // ② ★ 关键：告诉 MPP "输出到这个预分配的 frame"
    // MppMeta meta = mpp_packet_get_meta(packet);
    // if (meta) {
    //     mpp_meta_set_frame(meta, KEY_OUTPUT_FRAME, out_frame_);
    // }

    // // ③ 送 packet（advanced 模式下一般不会 BUFFER_FULL）
    // MPP_RET ret = mpi_->decode_put_packet(mpp_ctx_, packet);
    // mpp_packet_deinit(&packet);

    if (len > in_size_) { std::cerr << "frame too large\n"; return false; }

    // ① 把 MJPG 数据拷进持久输入 buffer
    memcpy(mpp_buffer_get_ptr(in_buf_), data, len);

    // ② packet 绑定这块 buffer
    MppPacket packet = nullptr;
    mpp_packet_init_with_buffer(&packet, in_buf_);
    mpp_packet_set_length(packet, len);          // ★ 必须设实际长度

    // ③ 指定输出 frame
    MppMeta meta = mpp_packet_get_meta(packet);
    if (meta) {
        mpp_meta_set_frame(meta, KEY_OUTPUT_FRAME, out_frame_);
    } else {
        std::cerr << "packet meta is NULL!\n";   // 如果打印这个，说明 meta 有问题
    }

    // ④ 送包
    MPP_RET ret = mpi_->decode_put_packet(mpp_ctx_, packet);
    mpp_packet_deinit(&packet);
    if (ret != MPP_OK) {
        std::cerr << "decode_put_packet failed, ret=" << ret << " len=" << len << "\n";
        return false;
    }


    if (ret != MPP_OK) {
        std::cerr << "decode_put_packet failed, ret=" << ret << "\n";
        return false;
    }

    // ④ 取帧
    for (int t = 0; t < 100; ++t) {
        MppFrame frame = nullptr;
        if (mpi_->decode_get_frame(mpp_ctx_, &frame) != MPP_OK || !frame) {
            usleep(2000);
            continue;
        }

        if (mpp_frame_get_info_change(frame)) {
            mpi_->control(mpp_ctx_, MPP_DEC_SET_INFO_CHANGE_READY, nullptr);
            mpp_frame_deinit(&frame);
            continue;
        }

        MppBuffer buffer = mpp_frame_get_buffer(frame);
        if (!buffer) { mpp_frame_deinit(&frame); return false; }

        out.frame      = frame;
        out.buffer     = buffer;
        out.fd         = mpp_buffer_get_fd(buffer);
        out.width      = mpp_frame_get_width(frame);
        out.height     = mpp_frame_get_height(frame);
        out.hor_stride = mpp_frame_get_hor_stride(frame);
        out.ver_stride = mpp_frame_get_ver_stride(frame);
        out.format     = (MppFrameFormat)mpp_frame_get_fmt(frame);

        std::cout << "MPP decoded: " << out.width << "x" << out.height
                  << " stride=" << out.hor_stride << "x" << out.ver_stride
                  << " fd=" << out.fd << "\n";
        return true;
    }
    std::cerr << "decode_get_frame timeout\n";
    std::cout << "put packet: len=" << len << " meta=" << meta << "\n";
    return false;
}

void MppJpegDecoder::release_frame(DecodedFrame& f) {
    if (f.frame) mpp_frame_deinit(&f.frame);
    f = DecodedFrame();
}

void MppJpegDecoder::close() {
    if (out_frame_) { mpp_frame_deinit(&out_frame_); out_frame_ = nullptr; }
    if (frm_buf_)   { mpp_buffer_put(frm_buf_); frm_buf_ = nullptr; }
    if (frm_grp_)   { mpp_buffer_group_put(frm_grp_); frm_grp_ = nullptr; }
    if (in_buf_)    { mpp_buffer_put(in_buf_); in_buf_ = nullptr; }           // ★
    if (in_grp_)    { mpp_buffer_group_put(in_grp_); in_grp_ = nullptr; }      // ★
    if (mpp_ctx_)   { mpp_destroy(mpp_ctx_); mpp_ctx_ = nullptr; mpi_ = nullptr; }
}