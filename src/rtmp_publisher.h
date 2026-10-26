// rtmp_publisher.h —— H.264(Annex-B) -> FLV -> RTMP，基于 libavformat
//
// 为什么用 libavformat 而不是手写 FLV/RTMP：
//   板子上 libavformat-dev 已装（4.2.7 / libavformat 58.29.100），
//   FLV 封装、AVC sequence header、RTMP 握手全部现成，代码量从 ~400 行降到 ~150 行。
//
// 为什么要把 Annex-B 转成 AVCC：
//   MPP 输出的是 Annex-B（00 00 00 01 起始码），FLV 里存的是 AVCC（4 字节长度前缀）。
//   同时 FLV 要求先发一个 AVC sequence header（= AVCDecoderConfigurationRecord），
//   内容就是 SPS/PPS，由 MppEncoder::header() 提供。

#ifndef RTMP_PUBLISHER_H_
#define RTMP_PUBLISHER_H_

#include <cstdint>
#include <string>
#include <vector>

struct AVFormatContext;
struct AVStream;

class RtmpPublisher {
 public:
  RtmpPublisher() = default;
  ~RtmpPublisher();

  RtmpPublisher(const RtmpPublisher&) = delete;
  RtmpPublisher& operator=(const RtmpPublisher&) = delete;

  // url 支持 rtmp://host:port/app/stream，也支持本地文件（如 /tmp/test.flv）。
  // 后者用于"先验证封装、再验证网络"，两个失败面分开排查。
  bool Open(const std::string& url, int width, int height, int fps,
            const std::vector<uint8_t>& sps_pps_annexb);
  void Close();
  bool is_open() const { return fmt_ != nullptr; }

  // data 是 Annex-B 码流；pts_us 单调递增（微秒）
  bool Write(const uint8_t* data, size_t size, int64_t pts_us, bool keyframe);

  int64_t bytes_written() const { return bytes_written_; }

 private:
  AVFormatContext* fmt_ = nullptr;
  AVStream* stream_ = nullptr;
  bool wrote_header_ = false;
  int64_t last_pts_ = -1;
  int64_t bytes_written_ = 0;
};

#endif  // RTMP_PUBLISHER_H_