#include "rga_converter.h"
#include <im2d.hpp>        // RGA im2d API
#include <RgaUtils.h>
#include <mutex>
#include <iostream>

// RGA 只有一个硬件，多线程同时用会崩，全局锁串行化
static std::mutex g_rga_mutex;

bool RgaConverter::to_rgb(const cv::Mat& bgr, int dst_w, int dst_h, cv::Mat* rgb) {
    if (bgr.empty() || !rgb) return false;

    rgb->create(dst_h, dst_w, CV_8UC3);

    std::lock_guard<std::mutex> lock(g_rga_mutex);

    int src_size = bgr.step[0] * bgr.rows;
    int dst_size = rgb->step[0] * rgb->rows;
    rga_buffer_handle_t src_handle = importbuffer_virtualaddr((void*)bgr.data, src_size);
    rga_buffer_handle_t dst_handle = importbuffer_virtualaddr((void*)rgb->data, dst_size);
    if (!src_handle || !dst_handle) {
        std::cerr << "rga import failed\n";
        if (src_handle) releasebuffer_handle(src_handle);
        if (dst_handle) releasebuffer_handle(dst_handle);
        return false;
    }

    rga_buffer_t src_img = wrapbuffer_handle(
        src_handle, bgr.cols, bgr.rows, RK_FORMAT_BGR_888,
        bgr.step[0] / bgr.elemSize(), bgr.rows);
    rga_buffer_t dst_img = wrapbuffer_handle(
        dst_handle, dst_w, dst_h, RK_FORMAT_RGB_888,
        rgb->step[0] / rgb->elemSize(), rgb->rows);

    auto t0 = std::chrono::high_resolution_clock::now();
    IM_STATUS ret = imresize(src_img, dst_img);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "[RGA] " << bgr.cols << "x" << bgr.rows << " -> "
              << dst_w << "x" << dst_h << " in "
              << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()
              << " us\n";

    releasebuffer_handle(src_handle);
    releasebuffer_handle(dst_handle);

    if (ret != IM_STATUS_SUCCESS) {
        std::cerr << "rga imresize failed: " << imStrError(ret) << "\n";
        return false;
    }
    return true;
}