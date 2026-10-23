#include "v4l2_capture.h"
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <iostream>

static int xioctl(int fd, unsigned long req, void* arg) {
    int r;
    do { r = ioctl(fd, req, arg); } while (r == -1 && errno == EINTR);
    return r;
}

bool V4l2MjpgCapture::open(const char* dev, int w, int h) {
    // ① 打开设备
    fd_ = ::open(dev, O_RDWR | O_NONBLOCK);
    if (fd_ < 0) { std::cerr << "open " << dev << " failed: " << strerror(errno) << "\n"; return false; }

    // ② 设置格式为 MJPG 1280x720
    v4l2_format fmt{};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width       = w;
    fmt.fmt.pix.height      = h;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
    fmt.fmt.pix.field       = V4L2_FIELD_NONE;
    if (xioctl(fd_, VIDIOC_S_FMT, &fmt) < 0) {
        std::cerr << "VIDIOC_S_FMT failed\n"; return false;
    }
    width_  = fmt.fmt.pix.width;
    height_ = fmt.fmt.pix.height;
    std::cout << "V4L2 opened " << dev << " " << width_ << "x" << height_
              << " MJPG\n";

    // ③ 申请 4 个 mmap 缓冲
    v4l2_requestbuffers req{};
    req.count  = 4;
    req.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (xioctl(fd_, VIDIOC_REQBUFS, &req) < 0) { std::cerr << "REQBUFS failed\n"; return false; }

    buffers_.resize(req.count);
    for (unsigned i = 0; i < req.count; ++i) {
        v4l2_buffer buf{};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        if (xioctl(fd_, VIDIOC_QUERYBUF, &buf) < 0) return false;

        buffers_[i].length = buf.length;
        buffers_[i].start  = mmap(nullptr, buf.length, PROT_READ | PROT_WRITE,
                                  MAP_SHARED, fd_, buf.m.offset);
        if (buffers_[i].start == MAP_FAILED) { std::cerr << "mmap failed\n"; return false; }
        xioctl(fd_, VIDIOC_QBUF, &buf);   // 入队，准备接收
    }

    // ④ 开始采集
    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(fd_, VIDIOC_STREAMON, &type) < 0) { std::cerr << "STREAMON failed\n"; return false; }
    return true;
}

bool V4l2MjpgCapture::read_frame(std::vector<uint8_t>& mjpg) {
    // ⑤ 取一帧
    v4l2_buffer buf{};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    if (xioctl(fd_, VIDIOC_DQBUF, &buf) < 0) return false;   // EAGAIN: 暂无数据

    // bytesused = 这一帧 MJPG 的实际字节数
    mjpg.assign((uint8_t*)buffers_[buf.index].start,
                (uint8_t*)buffers_[buf.index].start + buf.bytesused);

    // ⑥ 立刻入队复用
    xioctl(fd_, VIDIOC_QBUF, &buf);
    return true;
}

void V4l2MjpgCapture::close() {
    if (fd_ < 0) return;
    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    xioctl(fd_, VIDIOC_STREAMOFF, &type);
    for (auto& b : buffers_) if (b.start) munmap(b.start, b.length);
    ::close(fd_);
    fd_ = -1;
}