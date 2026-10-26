#include "voice_engine.h"

#include <alsa/asoundlib.h>
#include <sherpa-onnx/c-api/c-api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <mutex>
#include <vector>

namespace voice {
namespace {

constexpr int kSampleRate = 16000;
constexpr int kChannels = 1;
constexpr int kChunkSamples = 2000;
constexpr int kWindowSize = 512;
constexpr float kDcAlpha = 0.995f;   // 一阶高通，截止 ~12.7 Hz
constexpr size_t kMaxQueuedChunks = 8;

using Clock = std::chrono::steady_clock;

void Emit(const char* name, const char* from, const char* to,
          const std::string& extra) {
  std::printf("event=%s from=%s to=%s%s\n", name, from, to, extra.c_str());
  std::fflush(stdout);
}

class DcBlocker {
 public:
  void Process(std::vector<int16_t>& s) {
    for (size_t i = 0; i < s.size(); ++i) {
      const float x = static_cast<float>(s[i]);
      y_ = x - x_ + kDcAlpha * y_;
      x_ = x;
      float out = y_;
      if (out > 32767.0f) out = 32767.0f;
      if (out < -32768.0f) out = -32768.0f;
      s[i] = static_cast<int16_t>(out);
    }
  }
 private:
  float x_{0.0f};
  float y_{0.0f};
};

class PcmCapture {
 public:
  ~PcmCapture() { if (pcm_ != nullptr) snd_pcm_close(pcm_); }

  bool Open(const std::string& device) {
    device_ = device;
    return OpenDevice();
  }

  int Read(std::vector<int16_t>& buf) {
    const snd_pcm_sframes_t frames = snd_pcm_readi(pcm_, buf.data(), buf.size());
    if (frames >= 0) { errors_ = 0; return static_cast<int>(frames); }
    if (snd_pcm_recover(pcm_, static_cast<int>(frames), 1) >= 0) {
      errors_ = 0;
      return 0;
    }
    int prepared = snd_pcm_prepare(pcm_);
    if (prepared < 0) { snd_pcm_drop(pcm_); prepared = snd_pcm_prepare(pcm_); }
    if (prepared >= 0) {
      if (++errors_ >= 3) { if (!Reopen()) return -1; return 0; }
      return 0;
    }
    std::fprintf(stderr, "[mic] read failed: %s\n",
                 snd_strerror(static_cast<int>(frames)));
    return -1;
  }

 private:
  bool OpenDevice() {
    int rc = snd_pcm_open(&pcm_, device_.c_str(), SND_PCM_STREAM_CAPTURE, 0);
    if (rc < 0) {
      std::fprintf(stderr, "[mic] open %s failed: %s\n", device_.c_str(),
                   snd_strerror(rc));
      return false;
    }
    rc = snd_pcm_set_params(pcm_, SND_PCM_FORMAT_S16_LE,
                            SND_PCM_ACCESS_RW_INTERLEAVED, kChannels,
                            kSampleRate, 1, 500000);
    if (rc < 0) {
      std::fprintf(stderr, "[mic] set_params failed: %s\n", snd_strerror(rc));
      snd_pcm_close(pcm_);
      pcm_ = nullptr;
      return false;
    }
    snd_pcm_uframes_t buffer_frames = 0;
    snd_pcm_uframes_t period_frames = 0;
    if (snd_pcm_get_params(pcm_, &buffer_frames, &period_frames) == 0) {
      snd_pcm_sw_params_t* sw = nullptr;
      snd_pcm_sw_params_alloca(&sw);
      if (snd_pcm_sw_params_current(pcm_, sw) == 0) {
        snd_pcm_sw_params_set_start_threshold(pcm_, sw, 1);  // ★ 必须
        snd_pcm_sw_params_set_avail_min(pcm_, sw, period_frames);
        snd_pcm_sw_params(pcm_, sw);
      }
      std::printf("[mic] buffer=%lu frames, period=%lu frames\n",
                  static_cast<unsigned long>(buffer_frames),
                  static_cast<unsigned long>(period_frames));
    }
    errors_ = 0;
    return true;
  }

  bool Reopen() {
    if (pcm_ != nullptr) { snd_pcm_close(pcm_); pcm_ = nullptr; }
    if (!OpenDevice()) return false;
    std::fprintf(stderr, "[mic] reopened %s\n", device_.c_str());
    return true;
  }

  snd_pcm_t* pcm_{nullptr};
  std::string device_;
  int errors_{0};
};

float Rms(const int16_t* s, size_t n) {
  if (s == nullptr || n == 0) return 0.0f;
  long double sum = 0.0L;
  for (size_t i = 0; i < n; ++i) {
    const long double v = static_cast<long double>(s[i]) / 32768.0L;
    sum += v * v;
  }
  return static_cast<float>(std::sqrt(sum / static_cast<long double>(n)));
}

}  // namespace

struct Engine::Impl {
  Config cfg;

  const SherpaOnnxKeywordSpotter* spotter = nullptr;
  const SherpaOnnxOnlineStream* kws_stream = nullptr;
  const SherpaOnnxOfflineRecognizer* asr = nullptr;
  const SherpaOnnxVoiceActivityDetector* vad = nullptr;
  PcmCapture capture;
  bool capture_opened = false;

  std::deque<std::vector<int16_t>> queue;
  std::mutex queue_mutex;
  std::condition_variable queue_ready;
  bool stop = false;
  bool capture_failed = false;

  std::vector<float> vad_pending;
  size_t chunks_since_report = 0;
  float last_level = 0.0f;

  // 状态机
  enum class State { Idle, AwaitingCommand, Processing };
  State state = State::Idle;
  Clock::time_point wake_time = Clock::now();
  Clock::time_point cooldown_deadline = Clock::now();

  static const char* StateName(State s) {
    switch (s) {
      case State::Idle: return "idle";
      case State::AwaitingCommand: return "awaiting_command";
      case State::Processing: return "processing";
    }
    return "?";
  }
};

Engine::~Engine() {
  Stop();
  if (impl_ != nullptr) {
    if (impl_->vad != nullptr) SherpaOnnxDestroyVoiceActivityDetector(impl_->vad);
    if (impl_->asr != nullptr) SherpaOnnxDestroyOfflineRecognizer(impl_->asr);
    if (impl_->kws_stream != nullptr) SherpaOnnxDestroyOnlineStream(impl_->kws_stream);
    if (impl_->spotter != nullptr) SherpaOnnxDestroyKeywordSpotter(impl_->spotter);
    delete impl_;
    impl_ = nullptr;
  }
}

bool Engine::Init(const Config& cfg) {
  impl_ = new Impl();
  impl_->cfg = cfg;

  // ---- KWS ----
  SherpaOnnxKeywordSpotterConfig kws_cfg;
  std::memset(&kws_cfg, 0, sizeof(kws_cfg));
  kws_cfg.feat_config.sample_rate = kSampleRate;
  kws_cfg.feat_config.feature_dim = 80;
  kws_cfg.model_config.transducer.encoder = cfg.kws_encoder.c_str();
  kws_cfg.model_config.transducer.decoder = cfg.kws_decoder.c_str();
  kws_cfg.model_config.transducer.joiner = cfg.kws_joiner.c_str();
  kws_cfg.model_config.tokens = cfg.kws_tokens.c_str();
  kws_cfg.model_config.provider = "cpu";
  kws_cfg.model_config.num_threads = cfg.threads;
  kws_cfg.model_config.model_type = "";
  kws_cfg.model_config.modeling_unit = "cjkchar";
  kws_cfg.max_active_paths = 4;
  kws_cfg.num_trailing_blanks = 1;
  kws_cfg.keywords_score = 1.0f;
  kws_cfg.keywords_threshold = cfg.kws_threshold;
  kws_cfg.keywords_file = cfg.keywords_file.c_str();

  impl_->spotter = SherpaOnnxCreateKeywordSpotter(&kws_cfg);
  if (impl_->spotter == nullptr) {
    std::fprintf(stderr, "[kws] create failed\n");
    return false;
  }
  impl_->kws_stream = SherpaOnnxCreateKeywordStream(impl_->spotter);
  if (impl_->kws_stream == nullptr) {
    std::fprintf(stderr, "[kws] create stream failed\n");
    return false;
  }

  // ---- ASR ----
  SherpaOnnxOfflineRecognizerConfig asr_cfg;
  std::memset(&asr_cfg, 0, sizeof(asr_cfg));
  asr_cfg.feat_config.sample_rate = kSampleRate;
  asr_cfg.feat_config.feature_dim = 80;
  asr_cfg.model_config.sense_voice.model = cfg.asr_model.c_str();
  asr_cfg.model_config.sense_voice.language = cfg.language.c_str();
  asr_cfg.model_config.sense_voice.use_itn = 1;
  asr_cfg.model_config.tokens = cfg.asr_tokens.c_str();
  asr_cfg.model_config.provider = "cpu";
  asr_cfg.model_config.num_threads = cfg.threads;
  asr_cfg.decoding_method = "greedy_search";
  asr_cfg.max_active_paths = 4;

  impl_->asr = SherpaOnnxCreateOfflineRecognizer(&asr_cfg);
  if (impl_->asr == nullptr) {
    std::fprintf(stderr, "[asr] create failed\n");
    return false;
  }

  // ---- VAD ----
  SherpaOnnxVadModelConfig vad_cfg;
  std::memset(&vad_cfg, 0, sizeof(vad_cfg));
  vad_cfg.silero_vad.model = cfg.vad_model.c_str();
  vad_cfg.silero_vad.threshold = cfg.vad_threshold;
  vad_cfg.silero_vad.min_silence_duration = 0.5f;
  vad_cfg.silero_vad.min_speech_duration = 0.25f;
  vad_cfg.silero_vad.max_speech_duration = 10.0f;
  vad_cfg.silero_vad.window_size = kWindowSize;
  vad_cfg.sample_rate = kSampleRate;
  vad_cfg.num_threads = 1;
  vad_cfg.provider = "cpu";

  impl_->vad = SherpaOnnxCreateVoiceActivityDetector(&vad_cfg, 30.0f);
  if (impl_->vad == nullptr) {
    std::fprintf(stderr, "[vad] create failed\n");
    return false;
  }

  // ---- 麦克风 ----
  if (!impl_->capture.Open(cfg.device)) return false;
  impl_->capture_opened = true;
  impl_->vad_pending.reserve(kChunkSamples + kWindowSize);
  return true;
}

bool Engine::Start(QuestionHandler handler) {
  if (impl_ == nullptr || running_) return false;
  handler_ = std::move(handler);
  running_ = true;
  thread_ = std::thread([this]() { Loop(); });
  return true;
}

void Engine::Stop() {
  if (impl_ == nullptr) return;
  if (!running_.exchange(false)) return;
  {
    std::lock_guard<std::mutex> lock(impl_->queue_mutex);
    impl_->stop = true;
  }
  impl_->queue_ready.notify_all();
  if (thread_.joinable()) thread_.join();
}

void Engine::Loop() {
  Impl& s = *impl_;

  // 采集线程
  std::thread capture_thread([&s]() {
    DcBlocker dc;  // 状态必须跨块连续
    while (!s.stop) {
      std::vector<int16_t> chunk(kChunkSamples);
      const int frames = s.capture.Read(chunk);
      if (frames < 0) {
        std::lock_guard<std::mutex> lock(s.queue_mutex);
        s.capture_failed = true;
        s.stop = true;
        s.queue_ready.notify_all();
        return;
      }
      if (frames == 0) continue;
      chunk.resize(static_cast<size_t>(frames));
      dc.Process(chunk);
      {
        std::lock_guard<std::mutex> lock(s.queue_mutex);
        if (s.queue.size() >= kMaxQueuedChunks) {
          s.queue.pop_front();
          std::fprintf(stderr, "[mic] queue full, drop oldest\n");
        }
        s.queue.push_back(std::move(chunk));
      }
      s.queue_ready.notify_all();
    }
  });

  Emit("started", "idle", "idle", "");

  while (!s.stop) {
    std::vector<int16_t> chunk;
    {
      std::unique_lock<std::mutex> lock(s.queue_mutex);
      s.queue_ready.wait_for(lock, std::chrono::milliseconds(100),
                             [&s]() { return !s.queue.empty() || s.stop; });
      if (s.queue.empty()) continue;
      chunk = std::move(s.queue.front());
      s.queue.pop_front();
    }

    s.last_level = Rms(chunk.data(), chunk.size());
    const Clock::time_point now = Clock::now();

    // 1) 指令窗超时
    if (s.state == Impl::State::AwaitingCommand &&
        now - s.wake_time >=
            std::chrono::milliseconds(s.cfg.command_timeout_ms)) {
      Emit("activation_timeout", "awaiting_command", "idle", "");
      s.state = Impl::State::Idle;
    }

    // 2) KWS
    std::vector<float> f32(chunk.size());
    for (size_t i = 0; i < chunk.size(); ++i) {
      f32[i] = static_cast<float>(chunk[i]) / 32768.0f;
    }
    SherpaOnnxOnlineStreamAcceptWaveform(s.kws_stream, kSampleRate, f32.data(),
                                         static_cast<int32_t>(f32.size()));
    while (SherpaOnnxIsKeywordStreamReady(s.spotter, s.kws_stream) != 0) {
      SherpaOnnxDecodeKeywordStream(s.spotter, s.kws_stream);
      const SherpaOnnxKeywordResult* r =
          SherpaOnnxGetKeywordResult(s.spotter, s.kws_stream);
      if (r != nullptr) {
        if (r->keyword != nullptr && r->keyword[0] != '\0') {
          const std::string keyword = r->keyword;
          const Clock::time_point t = Clock::now();
          const char* from = Impl::StateName(s.state);

          if (s.state != Impl::State::Idle) {
            Emit("wake_rejected", from, from,
                 " keyword=\"" + keyword + "\" reason=busy");
          } else if (t < s.cooldown_deadline) {
            Emit("wake_rejected", from, from,
                 " keyword=\"" + keyword + "\" reason=cooldown");
          } else {
            Emit("wake_detected", from, "awaiting_command",
                 " keyword=\"" + keyword + "\" score=1.000");
            SherpaOnnxVoiceActivityDetectorClear(s.vad);
            SherpaOnnxVoiceActivityDetectorReset(s.vad);
            s.vad_pending.clear();
            s.state = Impl::State::AwaitingCommand;
            s.wake_time = t;
          }
          SherpaOnnxResetKeywordStream(s.spotter, s.kws_stream);
        }
        SherpaOnnxDestroyKeywordResult(r);
      }
    }

    // 3) VAD
    s.vad_pending.insert(s.vad_pending.end(), f32.begin(), f32.end());
    size_t offset = 0;
    while (s.vad_pending.size() - offset >= kWindowSize) {
      SherpaOnnxVoiceActivityDetectorAcceptWaveform(
          s.vad, s.vad_pending.data() + offset, kWindowSize);
      offset += kWindowSize;
    }
    s.vad_pending.erase(s.vad_pending.begin(), s.vad_pending.begin() + offset);

    // 4) 取语音段
    while (SherpaOnnxVoiceActivityDetectorEmpty(s.vad) == 0) {
      const SherpaOnnxSpeechSegment* seg =
          SherpaOnnxVoiceActivityDetectorFront(s.vad);

      if (s.state == Impl::State::AwaitingCommand && seg != nullptr &&
          seg->samples != nullptr && seg->n > 0) {
        s.state = Impl::State::Processing;
        Emit("asr_started", "awaiting_command", "processing", "");

        const SherpaOnnxOfflineStream* stream =
            SherpaOnnxCreateOfflineStream(s.asr);
        SherpaOnnxAcceptWaveformOffline(stream, kSampleRate, seg->samples,
                                        seg->n);
        const auto d0 = Clock::now();
        SherpaOnnxDecodeOfflineStream(s.asr, stream);
        const auto d1 = Clock::now();
        const SherpaOnnxOfflineRecognizerResult* res =
            SherpaOnnxGetOfflineStreamResult(stream);

        std::string text;
        if (res != nullptr) {
          if (res->text != nullptr) text = res->text;
          SherpaOnnxDestroyOfflineRecognizerResult(res);
        }
        SherpaOnnxDestroyOfflineStream(stream);

        const double audio_sec = static_cast<double>(seg->n) / kSampleRate;
        const double cost =
            std::chrono::duration<double>(d1 - d0).count();
        const long latency_ms = static_cast<long>(
            std::chrono::duration_cast<std::chrono::milliseconds>(d1 -
                                                                  s.wake_time)
                .count());

        std::printf("[asr] %s\n", text.c_str());
        char extra[256];
        std::snprintf(extra, sizeof(extra),
                      " text=\"%s\" audio=%.2fs decode=%.2fs latency=%ldms",
                      text.c_str(), audio_sec, cost, latency_ms);
        Emit("asr_final", "processing", "idle", extra);

        s.state = Impl::State::Idle;
        s.cooldown_deadline =
            Clock::now() + std::chrono::milliseconds(s.cfg.cooldown_ms);

        // ★ 把问题交给上层（这里只做入队，必须快速返回）
        if (handler_ && !text.empty()) handler_(text);
      }

      SherpaOnnxDestroySpeechSegment(seg);
      SherpaOnnxVoiceActivityDetectorPop(s.vad);
    }

    if (++s.chunks_since_report >= 8) {
      s.chunks_since_report = 0;
      std::printf("[hb] rms=%.4f state=%s\n", s.last_level,
                  Impl::StateName(s.state));
      std::fflush(stdout);
    }
  }

  {
    std::lock_guard<std::mutex> lock(s.queue_mutex);
    s.stop = true;
  }
  s.queue_ready.notify_all();
  if (capture_thread.joinable()) capture_thread.join();
  Emit("stopped", "idle", "idle", "");
}

}  // namespace voice