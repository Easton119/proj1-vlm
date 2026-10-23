#pragma once
#include <opencv2/opencv.hpp>

// 用 RGA 硬件做 BGR→RGB + 缩放（替代 OpenCV 的 cvtColor + resize）
class RgaConverter {
public:
    // 输入：BGR 图（任意尺寸），输出：dst_w x dst_h 的 RGB 图
    bool to_rgb(const cv::Mat& bgr, int dst_w, int dst_h, cv::Mat* rgb);
};