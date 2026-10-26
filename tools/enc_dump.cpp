// tools/enc_dump.cpp —— MPP 硬编码诊断工具
//
// 用法（板子 SSH）:
//   ./enc_dump <dev> <w> <h> <fps> <bitrate_bps> <seconds> <out.h264>
// 例:
//   ./enc_dump /dev/video21 640 480 15 1500000 10 /tmp/enc_640.h264
//
// 只做一件事：摄像头 -> MJPG 软解 -> RGA(BGR->NV12, 直写 DMA-BUF) -> MPP H.264 -> 文件
// 目的：把"MPP + RGA + DMA-BUF"这一段从推流里剥出来单独验证，不碰网络、不碰 VLM。

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>

#include "camera_stream.h"
#include "mpp_encoder.h"

int main(int argc, char** argv) {
  if (argc < 8) {
    std::fprintf(stderr,
                 "usage: %s <dev> <w> <h> <fps> <bitrate_bps> <seconds> <out.h264>\n",
                 argv[0]);
    return 1;
  }
  const char* dev = argv[1];
  const int w = std::atoi(argv[2]);
  const int h = std::atoi(argv[3]);
  const int fps = std::atoi(argv[4]);
  const int bitrate = std::atoi(argv[5]);
  const int seconds = std::atoi(argv[6]);
  const char* out_path = argv[7];

  CameraStream cam;
  if (!cam.start(dev, w, h)) {
    std::fprintf(stderr, "[dump] camera start failed\n");
    return 1;
  }

  // 等第一帧，否则编码器没数据
  cv::Mat frame;
  for (int i = 0; i < 250 && !cam.get_latest(frame); ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  if (frame.empty()) {
    std::fprintf(stderr, "[dump] no frame from camera\n");
    cam.stop();
    return 1;
  }
  std::printf("[dump] first frame %dx%d\n", frame.cols, frame.rows);

  MppEncoder enc;
  if (!enc.Init(w, h, fps, bitrate)) {
    std::fprintf(stderr, "[dump] encoder init failed\n");
    cam.stop();
    return 1;
  }
  if (enc.header().empty()) {
    std::fprintf(stderr, "[dump] WARN: empty SPS/PPS header\n");
  }

  FILE* fp = std::fopen(out_path, "wb");
  if (fp == nullptr) {
    std::fprintf(stderr, "[dump] open %s failed\n", out_path);
    enc.Release();
    cam.stop();
    return 1;
  }
  if (!enc.header().empty()) {
    std::fwrite(enc.header().data(), 1, enc.header().size(), fp);
  }

  const auto period = std::chrono::microseconds(1000000 / (fps > 0 ? fps : 15));
  auto next = std::chrono::steady_clock::now();
  const auto t_start = next;

  long frames = 0;
  long packets = 0;
  long keyframes = 0;
  long long bytes = 0;
  double enc_ms_sum = 0.0;
  long enc_n = 0;
  long last_report = -1;
  std::vector<uint8_t> pkt;

  while (true) {
    const auto now = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(now - t_start).count();
    if (elapsed >= static_cast<double>(seconds)) break;

    if (!cam.get_latest(frame)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
      continue;
    }
    if (frame.cols != w || frame.rows != h) {
      std::fprintf(stderr, "[dump] unexpected frame %dx%d\n", frame.cols, frame.rows);
      break;
    }

    bool key = false;
    const auto e0 = std::chrono::steady_clock::now();
    const bool ok = enc.Encode(frame, &pkt, &key);
    const auto e1 = std::chrono::steady_clock::now();
    enc_ms_sum += std::chrono::duration<double, std::milli>(e1 - e0).count();
    ++enc_n;

    if (!ok) {
      std::fprintf(stderr, "[dump] encode failed at frame %ld\n", frames);
      break;
    }
    ++frames;
    if (!pkt.empty()) {
      std::fwrite(pkt.data(), 1, pkt.size(), fp);
      ++packets;
      bytes += static_cast<long long>(pkt.size());
      if (key) ++keyframes;
    }

    const long sec = static_cast<long>(elapsed);
    if (sec != last_report) {
      last_report = sec;
      const double kbps = elapsed > 0 ? bytes * 8.0 / elapsed / 1000.0 : 0.0;
      std::printf("[dump] %2lds frames=%ld packets=%ld key=%ld kbps=%.0f avg_enc=%.1f ms\n",
                  sec, frames, packets, keyframes, kbps,
                  enc_n > 0 ? enc_ms_sum / enc_n : 0.0);
      std::fflush(stdout);
    }

    next += period;
    const auto after = std::chrono::steady_clock::now();
    if (after > next + period) next = after;  // 落后就重置，不堆积
    std::this_thread::sleep_until(next);
  }

  const double total = std::chrono::duration<double>(
                           std::chrono::steady_clock::now() - t_start).count();
  std::fclose(fp);
  enc.Release();
  cam.stop();

  std::printf("[dump] saved %s: %.1f s, frames=%ld packets=%ld key=%ld bytes=%lld\n",
              out_path, total, frames, packets, keyframes, bytes);
  std::printf("[dump] avg_enc=%.1f ms/frame, avg_bitrate=%.0f kbps\n",
              enc_n > 0 ? enc_ms_sum / enc_n : 0.0,
              total > 0 ? bytes * 8.0 / total / 1000.0 : 0.0);
  return 0;
}