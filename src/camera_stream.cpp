#include "camera_stream.h"
#include <chrono>
#include <iostream>
#include <vector>

// ============ 启动：打开摄像头 + 起采集线程 ============
bool CameraStream::start(const char* dev, int w, int h) {
    if (!capture_.open(dev, w, h)) {
        std::cerr << "[CameraStream] open failed\n";
        return false;
    }
    running_ = true;
    thread_ = std::thread(&CameraStream::capture_loop, this);   // 起线程
    std::cout << "[CameraStream] started\n";
    return true;
}

// ============ 停止：结束线程 + 关摄像头 ============
void CameraStream::stop() {
    running_ = false;                 // 让线程退出
    if (thread_.joinable()) thread_.join();   // 等线程结束
    capture_.close();
}

// ============ 取最新帧（提问时调用）============
bool CameraStream::get_latest(cv::Mat& out) {
    std::lock_guard<std::mutex> lock(frame_mutex_);   // 加锁，防止采集线程同时写
    if (!has_frame_) return false;                     // 还没抓到第一帧
    out = latest_bgr_.clone();                         // ★ clone 深拷贝：
    //   这样提问线程拿到独立的一份，采集线程随后覆盖 latest_bgr_ 不影响 out
    return true;
}

// ============ 采集线程主循环 ============
void CameraStream::capture_loop() {
    std::vector<uint8_t> mjpg;
    while (running_) {
        // ① 抓一帧 MJPG（非阻塞：没帧就返回 false）
        if (!capture_.read_frame(mjpg)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));  // 让出 CPU
            continue;
        }

        // ② 软解 MJPG → BGR
        cv::Mat bgr = cv::imdecode(mjpg, cv::IMREAD_COLOR);
        if (bgr.empty()) continue;

        // ③ 覆盖"最新帧"（加锁，单槽，旧帧被覆盖）
        {
            std::lock_guard<std::mutex> lock(frame_mutex_);
            latest_bgr_ = bgr.clone();   // 深拷贝，避免局部 bgr 析构后数据失效
            has_frame_  = true;
        }
    }
}