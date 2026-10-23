#pragma once
#include <cstdint>
#include <vector>

// V4L2 显式采集 USB 摄像头（MJPG 输出），取出一帧 MJPG 压缩数据
class V4l2MjpgCapture {
public:
    bool open(const char* dev, int w, int h);      // 打开 + 配置 MJPG
    bool read_frame(std::vector<uint8_t>& mjpg);   // 取一帧（非阻塞）
    void close();
    int width()  const { return width_; }
    int height() const { return height_; }

private:
    int fd_ = -1;
    int width_ = 0, height_ = 0;
    struct Buffer { void* start; size_t length; };
    std::vector<Buffer> buffers_;
};