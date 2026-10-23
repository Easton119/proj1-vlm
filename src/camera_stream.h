#pragma once
#include <opencv2/opencv.hpp>
#include <atomic>
#include <mutex>
#include <thread>
#include "v4l2_capture.h"

// 摄像头采集流：独立线程持续抓帧 + 解码，维护"最新一帧"（mutex 保护的单槽缓冲）
class CameraStream {
public:
    // 打开摄像头并启动采集线程
    bool start(const char* dev, int w, int h);
    // 停止采集线程、释放摄像头
    void stop();
    // 获取最新一帧（BGR），非阻塞；还没抓到帧返回 false
    bool get_latest(cv::Mat& out);

private:
    void capture_loop();   // 采集线程主循环

    V4l2MjpgCapture capture_;     // V4L2 采集（复用你已写好的类）
    std::thread      thread_;     // 采集线程句柄
    std::atomic<bool> running_{false};   // 线程运行标志

    // 最新帧（单槽缓冲，mutex 保护）
    std::mutex frame_mutex_;
    cv::Mat    latest_bgr_;   // 存解码后的 BGR（最新一帧）
    bool       has_frame_ = false;   // 是否已抓到第一帧
};