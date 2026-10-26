// tts_engine.cpp —— sherpa-onnx 离线 TTS + ALSA 播放（双线程：合成 / 播放）
//
// 为什么是两个线程：
//   sherpa-onnx 的生成回调里如果直接 snd_pcm_writei，会因为缓冲满而阻塞，
//   导致"写第 N 块"和"合成第 N+1 块"串行，实测把 0.63x 的合成拖成 1.4x。
//   拆成两个线程 + FIFO 之后，合成全速跑，播放按 1x 匀速取，中间靠 FIFO 吸收。

#include "tts_engine.h"

#include <alsa/asoundlib.h>
#include <sherpa-onnx/c-api/c-api.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;

struct TtsEngine::Impl {
  TtsConfig cfg;

  const SherpaOnnxOfflineTts* tts = nullptr;
  snd_pcm_t* pcm = nullptr;
  int sample_rate = 0;
  snd_pcm_uframes_t buffer_frames = 0;

  std::thread synth_thread;
  std::thread writer_thread;
  std::atomic<bool> running{false};

  // 待合成的文本
  std::mutex mu;
  std::condition_variable cv;
  std::deque<std::string> texts;

  // PCM FIFO（int16 单声道）
  std::mutex fifo_mu;
  std::condition_variable fifo_cv;
  std::deque<int16_t> fifo;
  size_t fifo_cap = 0;   // 上限（样本数），防止合成跑太远吃内存

  // 统计
  int utt_index = 0;
  int chunk_count = 0;
  double first_chunk_ms = 0.0;
  double last_chunk_ms = 0.0;
  double max_gap_ms = 0.0;
  Clock::time_point gen_start;
  Clock::time_point last_chunk_at;

  int write_errors = 0;

  // ---------- ALSA 写入（只在播放线程里调用） ----------
  bool WriteSamples(const int16_t* data, int32_t n) {
    if (pcm == nullptr || data == nullptr || n <= 0) return false;

    int32_t offset = 0;
    while (offset < n) {
      const snd_pcm_sframes_t rc =
          snd_pcm_writei(pcm, data + offset, n - offset);
      if (rc < 0) {
        const snd_pcm_sframes_t recovered =
            snd_pcm_recover(pcm, static_cast<int>(rc), 1);
        if (recovered < 0) {
          if (write_errors < 5) {
            std::fprintf(stderr, "[tts] write failed: %s (state=%s)\n",
                         snd_strerror(static_cast<int>(rc)),
                         snd_pcm_state_name(snd_pcm_state(pcm)));
            ++write_errors;
          }
          return false;
        }
        continue;
      }
      offset += static_cast<int32_t>(rc);
    }
    return true;
  }

  // ---------- 播放线程：从 FIFO 取，按 1x 写声卡 ----------
  void WriterLoop() {
    std::vector<int16_t> buf;
    while (true) {
      {
        std::unique_lock<std::mutex> lock(fifo_mu);
        fifo_cv.wait(lock, [this]() { return !fifo.empty() || !running.load(); });
        if (fifo.empty() && !running.load()) return;
        const size_t take =
            std::min(fifo.size(), static_cast<size_t>(sample_rate / 2));
        buf.assign(fifo.begin(), fifo.begin() + take);
        fifo.erase(fifo.begin(), fifo.begin() + take);
      }
      fifo_cv.notify_all();  // 告诉合成线程：有空间了
      WriteSamples(buf.data(), static_cast<int32_t>(buf.size()));
    }
  }

  // ---------- 从回调来的 float 音频，转 int16 塞进 FIFO ----------
  void PushSamples(const float* samples, int32_t n) {
    if (samples == nullptr || n <= 0) return;

    std::vector<int16_t> tmp(static_cast<size_t>(n));
    for (int32_t i = 0; i < n; ++i) {
      float v = samples[i];
      if (v > 1.0f) v = 1.0f;
      if (v < -1.0f) v = -1.0f;
      tmp[static_cast<size_t>(i)] = static_cast<int16_t>(v * 32767.0f);
    }

    {
      std::unique_lock<std::mutex> lock(fifo_mu);
      // 背压：FIFO 超过上限就等播放消费（正常不会触发）
      fifo_cv.wait(lock, [this]() {
        return fifo.size() < fifo_cap || !running.load();
      });
      if (!running.load()) return;
      fifo.insert(fifo.end(), tmp.begin(), tmp.end());
    }
    fifo_cv.notify_all();
  }

  // 预热专用：什么都不写，只让模型把图跑一遍
  static int32_t OnWarmup(const float* /*samples*/, int32_t /*n*/,
                          float /*progress*/, void* /*arg*/) {
    return 1;
  }

  // ---------- 生成回调：只做统计 + 入 FIFO，绝不阻塞 ----------
  static int32_t OnChunk(const float* samples, int32_t n, float /*progress*/,
                         void* arg) {
    auto* self = static_cast<Impl*>(arg);
    if (self == nullptr || !self->running.load()) return 0;

    const auto now = Clock::now();
    const double since_start =
        std::chrono::duration<double, std::milli>(now - self->gen_start).count();
    if (self->chunk_count == 0) {
      self->first_chunk_ms = since_start;
    } else {
      const double gap = std::chrono::duration<double, std::milli>(
                             now - self->last_chunk_at)
                             .count();
      if (gap > self->max_gap_ms) self->max_gap_ms = gap;
    }
    self->last_chunk_at = now;
    self->last_chunk_ms = since_start;
    ++self->chunk_count;

    self->PushSamples(samples, n);
    return 1;
  }

  // ---------- ALSA 初始化 ----------
  bool OpenDevice(int rate) {
    int rc = snd_pcm_open(&pcm, cfg.device.c_str(), SND_PCM_STREAM_PLAYBACK, 0);
    if (rc < 0) {
      std::fprintf(stderr, "[tts] open %s failed: %s\n", cfg.device.c_str(),
                   snd_strerror(rc));
      return false;
    }
    rc = snd_pcm_set_params(pcm, SND_PCM_FORMAT_S16_LE,
                            SND_PCM_ACCESS_RW_INTERLEAVED, 1, rate, 1, 3000000);
    if (rc < 0) {
      std::fprintf(stderr, "[tts] set_params failed (rate=%d): %s\n", rate,
                   snd_strerror(rc));
      snd_pcm_close(pcm);
      pcm = nullptr;
      return false;
    }

    snd_pcm_uframes_t period = 0;
    if (snd_pcm_get_params(pcm, &buffer_frames, &period) == 0) {
      snd_pcm_sw_params_t* sw = nullptr;
      snd_pcm_sw_params_alloca(&sw);
      if (snd_pcm_sw_params_current(pcm, sw) == 0) {
        // 200 ms 就开口，别等积满一个 period
        snd_pcm_sw_params_set_start_threshold(pcm, sw, rate / 5);
        snd_pcm_sw_params_set_avail_min(pcm, sw, buffer_frames / 8);
        snd_pcm_sw_params(pcm, sw);
      }
      std::printf("[tts] pcm buffer=%.1f s, period=%.0f ms\n",
                  buffer_frames * 1.0 / rate, period * 1000.0 / rate);
    }

    fifo_cap = static_cast<size_t>(rate) * 30;  // 最多缓存 30 s
    return true;
  }

  // ---------- 合成线程 ----------
  void SynthLoop() {
    std::printf("[tts] synth thread started\n");
    std::fflush(stdout);

    while (true) {
      std::string text;
      {
        std::unique_lock<std::mutex> lock(mu);
        cv.wait(lock, [this]() { return !texts.empty() || !running.load(); });
        if (texts.empty() && !running.load()) {
          std::printf("[tts] synth thread exit\n");
          std::fflush(stdout);
          return;
        }
        text = texts.front();
        texts.pop_front();
      }

      ++utt_index;
      chunk_count = 0;
      first_chunk_ms = 0.0;
      last_chunk_ms = 0.0;
      max_gap_ms = 0.0;
      gen_start = Clock::now();

      std::printf("[tts] iter=%d speak(%zu bytes): %s\n", utt_index,
                  text.size(), text.c_str());
      std::fflush(stdout);

      SherpaOnnxGenerationConfig gcfg;
      std::memset(&gcfg, 0, sizeof(gcfg));
      gcfg.sid = cfg.sid;
      gcfg.speed = cfg.speed;
      gcfg.silence_scale = cfg.silence_scale;

      const auto t0 = Clock::now();
      const SherpaOnnxGeneratedAudio* audio =
          SherpaOnnxOfflineTtsGenerateWithConfig(tts, text.c_str(), &gcfg,
                                                 &Impl::OnChunk, this);
      const auto t1 = Clock::now();
      const double cost = std::chrono::duration<double>(t1 - t0).count();

      double audio_sec = 0.0;
      if (audio != nullptr) {
        audio_sec = static_cast<double>(audio->n) / audio->sample_rate;

        // ★ 落盘用于对比"模型产出的"与"喇叭播出的"
        if (!cfg.dump_dir.empty()) {
          char path[512];
          std::snprintf(path, sizeof(path), "%s/utt_%03d.wav",
                        cfg.dump_dir.c_str(), utt_index);
          SherpaOnnxWriteWave(audio->samples, audio->n, audio->sample_rate,
                              path);
        }
        SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio);
      } else {
        std::fprintf(stderr, "[tts] generate failed\n");
      }

      if (audio_sec > 0.0) {
        std::printf("[tts] iter=%d audio %.2f s | 合成 %.2f s (纯RTF %.2f) | "
                    "chunks=%d first=%.0fms last=%.0fms gap=%.0fms\n",
                    utt_index, audio_sec, cost, cost / audio_sec, chunk_count,
                    first_chunk_ms, last_chunk_ms, max_gap_ms);
      } else {
        std::printf("[tts] iter=%d synth %.2f s\n", utt_index, cost);
      }
      std::fflush(stdout);
    }
  }
};

TtsEngine::~TtsEngine() {
  Stop();
  if (impl_ != nullptr) {
    if (impl_->pcm != nullptr) snd_pcm_close(impl_->pcm);
    if (impl_->tts != nullptr) SherpaOnnxDestroyOfflineTts(impl_->tts);
    delete impl_;
    impl_ = nullptr;
  }
}

bool TtsEngine::Init(const TtsConfig& cfg) {
  impl_ = new Impl();
  impl_->cfg = cfg;

  SherpaOnnxOfflineTtsConfig config;
  std::memset(&config, 0, sizeof(config));
  config.model.vits.model = cfg.model.c_str();
  config.model.vits.lexicon = cfg.lexicon.c_str();
  config.model.vits.tokens = cfg.tokens.c_str();
  config.model.num_threads = cfg.num_threads;
  config.model.provider = "cpu";
  config.model.debug = 0;
  config.silence_scale = cfg.silence_scale;

  impl_->tts = SherpaOnnxCreateOfflineTts(&config);
  if (impl_->tts == nullptr) {
    std::fprintf(stderr, "[tts] create failed: check model/lexicon/tokens\n");
    return false;
  }

  impl_->sample_rate = SherpaOnnxOfflineTtsSampleRate(impl_->tts);
  const int speakers = SherpaOnnxOfflineTtsNumSpeakers(impl_->tts);
  std::printf("[tts] model loaded, sample_rate=%d, speakers=%d\n",
              impl_->sample_rate, speakers);
  if (cfg.sid >= speakers) {
    std::fprintf(stderr, "[tts] sid=%d out of range, fallback to 0\n", cfg.sid);
    impl_->cfg.sid = 0;
  }

  if (!impl_->OpenDevice(impl_->sample_rate)) return false;

  // ★ 预热：首次推理要做图分配/内存池初始化（实测首句 RTF 1.06 vs 后续 0.67）。
  // 在这里先合成一句短的，把这份固定开销从"第一次真实提问"里挪走。
  {
    const auto w0 = Clock::now();
    SherpaOnnxGenerationConfig wcfg;
    std::memset(&wcfg, 0, sizeof(wcfg));
    wcfg.sid = impl_->cfg.sid;
    wcfg.speed = 1.0f;
    const SherpaOnnxGeneratedAudio* warm = SherpaOnnxOfflineTtsGenerateWithConfig(
        impl_->tts, "你好", &wcfg, &Impl::OnWarmup, nullptr);
    if (warm != nullptr) SherpaOnnxDestroyOfflineTtsGeneratedAudio(warm);
    const double wms =
        std::chrono::duration<double, std::milli>(Clock::now() - w0).count();
    std::printf("[tts] warmup %.0f ms\n", wms);
  }

  impl_->running = true;
  impl_->writer_thread = std::thread([this]() { impl_->WriterLoop(); });
  impl_->synth_thread = std::thread([this]() { impl_->SynthLoop(); });
  return true;
}

void TtsEngine::Say(const std::string& text) {
  if (impl_ == nullptr || text.empty()) return;
  {
    std::lock_guard<std::mutex> lock(impl_->mu);
    if (!impl_->running.load()) return;
    // 队列里已有内容就并到队尾，避免长回答被丢段
    if (!impl_->texts.empty()) {
      impl_->texts.back() += text;
    } else {
      impl_->texts.push_back(text);
    }
  }
  impl_->cv.notify_one();
}

void TtsEngine::Stop() {
  if (impl_ == nullptr) return;
  if (!impl_->running.exchange(false)) return;

  impl_->cv.notify_all();
  impl_->fifo_cv.notify_all();

  if (impl_->synth_thread.joinable()) impl_->synth_thread.join();
  if (impl_->writer_thread.joinable()) impl_->writer_thread.join();
}