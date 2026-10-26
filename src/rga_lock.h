#pragma once

#include <mutex>

// RGA 是单硬件单元，所有使用者（VLM 预处理 / MPP 编码前处理）必须共用同一把锁。
// 用法：std::lock_guard<std::mutex> lock(GlobalRgaMutex());
std::mutex& GlobalRgaMutex();