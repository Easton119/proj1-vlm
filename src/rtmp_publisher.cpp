#include "rtmp_publisher.h"

#include <cstdio>
#include <cstring>
#include <utility>

// ★ 必须用 extern "C" 包住：FFmpeg 是 C 库，
//   用 C++ 链接名去链 libavformat.so 会一个符号都对不上，
//   再叠加 Ubuntu 默认的 -Wl,--as-needed，链接器会把这三个库整个丢弃。
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
}

namespace {

struct NalRef {
  const uint8_t* data;
  size_t size;
};

struct Mark {
  size_t sc_start;  // 起始码起点
  size_t payload;   // NAL 数据起点
};

// 按 Annex-B 起始码（00 00 01 / 00 00 00 01）切 NAL。
// 注意：不能简单"切到下一个起始码"，否则会把下一个起始码的 0x01 当成 NAL 尾字节，
// 所以同时记录起始码起点 sc_start，用它作为上一个 NAL 的结束位置。
void SplitAnnexB(const uint8_t* d, size_t n, std::vector<NalRef>* out) {
  out->clear();
  if (d == nullptr || n < 4) return;

  std::vector<Mark> marks;
  size_t i = 0;
  while (i + 3 <= n) {
    if (d[i] == 0 && d[i + 1] == 0 && d[i + 2] == 1) {
      Mark m;
      m.sc_start = i;
      m.payload = i + 3;
      marks.push_back(m);
      i += 3;
    } else if (i + 4 <= n && d[i] == 0 && d[i + 1] == 0 && d[i + 2] == 0 &&
               d[i + 3] == 1) {
      Mark m;
      m.sc_start = i;
      m.payload = i + 4;
      marks.push_back(m);
      i += 4;
    } else {
      ++i;
    }
  }

  for (size_t k = 0; k < marks.size(); ++k) {
    size_t end = (k + 1 < marks.size()) ? marks[k + 1].sc_start : n;
    while (end > marks[k].payload && d[end - 1] == 0) --end;  // 去掉尾随 0
    if (end > marks[k].payload) {
      NalRef r;
      r.data = d + marks[k].payload;
      r.size = end - marks[k].payload;
      out->push_back(r);
    }
  }
}

// 从 SPS/PPS 构造 AVCDecoderConfigurationRecord（ISO/IEC 14496-15 5.2.4.1）
bool BuildAvcConfig(const std::vector<NalRef>& nals, std::vector<uint8_t>* out) {
  const uint8_t* sps = nullptr;
  size_t sps_n = 0;
  const uint8_t* pps = nullptr;
  size_t pps_n = 0;

  for (size_t i = 0; i < nals.size(); ++i) {
    const NalRef& r = nals[i];
    if (r.size < 1) continue;
    const int type = r.data[0] & 0x1F;
    if (type == 7 && sps == nullptr) {
      sps = r.data;
      sps_n = r.size;
    } else if (type == 8 && pps == nullptr) {
      pps = r.data;
      pps_n = r.size;
    }
  }
  if (sps == nullptr || pps == nullptr || sps_n < 4 || pps_n < 1) return false;

  out->clear();
  out->reserve(11 + sps_n + pps_n);
  out->push_back(1);       // configurationVersion
  out->push_back(sps[1]);  // AVCProfileIndication
  out->push_back(sps[2]);  // profile_compatibility
  out->push_back(sps[3]);  // AVCLevelIndication
  out->push_back(0xFF);    // 6 bits reserved + lengthSizeMinusOne = 3
  out->push_back(0xE1);    // 3 bits reserved + numOfSequenceParameterSets = 1
  out->push_back(static_cast<uint8_t>((sps_n >> 8) & 0xFF));
  out->push_back(static_cast<uint8_t>(sps_n & 0xFF));
  out->insert(out->end(), sps, sps + sps_n);
  out->push_back(1);  // numOfPictureParameterSets
  out->push_back(static_cast<uint8_t>((pps_n >> 8) & 0xFF));
  out->push_back(static_cast<uint8_t>(pps_n & 0xFF));
  out->insert(out->end(), pps, pps + pps_n);
  return true;
}

}  // namespace

RtmpPublisher::~RtmpPublisher() { Close(); }

bool RtmpPublisher::Open(const std::string& url, int width, int height, int fps,
                         const std::vector<uint8_t>& sps_pps_annexb) {
  Close();
  if (width <= 0 || height <= 0) return false;

  std::vector<NalRef> nals;
  SplitAnnexB(sps_pps_annexb.data(), sps_pps_annexb.size(), &nals);
  std::vector<uint8_t> avcc;
  if (!BuildAvcConfig(nals, &avcc)) {
    std::fprintf(stderr,
                 "[push] build AVCDecoderConfigurationRecord failed "
                 "(header=%zu bytes, nals=%zu): 上游 SPS/PPS 不正常\n",
                 sps_pps_annexb.size(), nals.size());
    return false;
  }

  avformat_network_init();

  // 显式指定 flv 封装；URL 是文件路径时同样可用
  int ret = avformat_alloc_output_context2(&fmt_, nullptr, "flv", url.c_str());
  if (ret < 0 || fmt_ == nullptr) {
    std::fprintf(stderr, "[push] alloc output context failed\n");
    fmt_ = nullptr;
    return false;
  }

  stream_ = avformat_new_stream(fmt_, nullptr);
  if (stream_ == nullptr) {
    std::fprintf(stderr, "[push] avformat_new_stream failed\n");
    Close();
    return false;
  }

  AVCodecParameters* par = stream_->codecpar;
  par->codec_type = AVMEDIA_TYPE_VIDEO;
  par->codec_id = AV_CODEC_ID_H264;
  par->width = width;
  par->height = height;
  par->format = AV_PIX_FMT_YUV420P;
  par->extradata = static_cast<uint8_t*>(
      av_mallocz(avcc.size() + AV_INPUT_BUFFER_PADDING_SIZE));
  if (par->extradata == nullptr) {
    std::fprintf(stderr, "[push] av_mallocz extradata failed\n");
    Close();
    return false;
  }
  std::memcpy(par->extradata, avcc.data(), avcc.size());
  par->extradata_size = static_cast<int>(avcc.size());

  stream_->avg_frame_rate.num = fps > 0 ? fps : 15;
  stream_->avg_frame_rate.den = 1;
  stream_->r_frame_rate = stream_->avg_frame_rate;

  if ((fmt_->oformat->flags & AVFMT_NOFILE) == 0) {
    ret = avio_open(&fmt_->pb, url.c_str(), AVIO_FLAG_WRITE);
    if (ret < 0) {
      char err[128] = {0};
      av_strerror(ret, err, sizeof(err));
      std::fprintf(stderr, "[push] avio_open(%s) failed: %s\n", url.c_str(), err);
      Close();
      return false;
    }
  }

  ret = avformat_write_header(fmt_, nullptr);
  if (ret < 0) {
    char err[128] = {0};
    av_strerror(ret, err, sizeof(err));
    std::fprintf(stderr, "[push] avformat_write_header failed: %s\n", err);
    Close();
    return false;
  }
  wrote_header_ = true;
  last_pts_ = -1;
  bytes_written_ = 0;

  std::printf("[push] opened %s | %dx%d fps=%d | extradata=%zu bytes (sps=%zu pps=%zu)\n",
              url.c_str(), width, height, fps, avcc.size(),
              sps_pps_annexb.size(), nals.size());
  std::fflush(stdout);
  return true;
}

bool RtmpPublisher::Write(const uint8_t* data, size_t size, int64_t pts_us,
                          bool keyframe) {
  if (fmt_ == nullptr || stream_ == nullptr || data == nullptr || size == 0) {
    return false;
  }

  std::vector<NalRef> nals;
  SplitAnnexB(data, size, &nals);
  if (nals.empty()) return false;

  size_t total = 0;
  for (size_t i = 0; i < nals.size(); ++i) total += 4 + nals[i].size;

  AVPacket pkt;
  av_init_packet(&pkt);
  if (av_new_packet(&pkt, static_cast<int>(total)) < 0) return false;

  uint8_t* p = pkt.data;
  for (size_t i = 0; i < nals.size(); ++i) {
    const uint32_t len = static_cast<uint32_t>(nals[i].size);
    p[0] = static_cast<uint8_t>((len >> 24) & 0xFF);
    p[1] = static_cast<uint8_t>((len >> 16) & 0xFF);
    p[2] = static_cast<uint8_t>((len >> 8) & 0xFF);
    p[3] = static_cast<uint8_t>(len & 0xFF);
    p += 4;
    std::memcpy(p, nals[i].data, nals[i].size);
    p += nals[i].size;
  }

  const AVRational us_tb = {1, 1000000};
  int64_t pts = av_rescale_q(pts_us, us_tb, stream_->time_base);
  if (pts <= last_pts_) pts = last_pts_ + 1;  // 必须严格递增
  last_pts_ = pts;

  pkt.pts = pts;
  pkt.dts = pts;
  pkt.duration = 0;
  pkt.stream_index = stream_->index;
  if (keyframe) pkt.flags |= AV_PKT_FLAG_KEY;

  // av_interleaved_write_frame 会接管 packet 所有权，成功后不要再 unref
  const int ret = av_interleaved_write_frame(fmt_, &pkt);
  if (ret < 0) {
    char err[128] = {0};
    av_strerror(ret, err, sizeof(err));
    std::fprintf(stderr, "[push] write_frame failed: %s\n", err);
    return false;
  }
  bytes_written_ += static_cast<int64_t>(total);
  return true;
}

void RtmpPublisher::Close() {
  if (fmt_ == nullptr) return;
  if (wrote_header_) {
    av_write_trailer(fmt_);
    wrote_header_ = false;
  }
  if (fmt_->pb != nullptr) {
    avio_closep(&fmt_->pb);
  }
  avformat_free_context(fmt_);
  fmt_ = nullptr;
  stream_ = nullptr;
  last_pts_ = -1;
}