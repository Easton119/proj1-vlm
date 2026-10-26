// voice_engine.h —— 把 /home/topeet/Qyj/voice/app/voice_main.cpp 封装成线程化的引擎
// 用 pimpl 把 alsa / sherpa-onnx 的头文件挡在 .cpp 里，main.cpp 不用管它们

#ifndef VOICE_ENGINE_H_
#define VOICE_ENGINE_H_

#include <atomic>
#include <functional>
#include <string>
#include <thread>

namespace voice {

struct Config {
  std::string device = "plughw:2,0";
  std::string keywords_file;
  std::string kws_tokens;
  std::string kws_encoder;
  std::string kws_decoder;
  std::string kws_joiner;
  std::string asr_model;
  std::string asr_tokens;
  std::string vad_model;
  std::string language = "zh";
  int threads = 2;
  float kws_threshold = 0.25f;
  float vad_threshold = 0.25f;
  int command_timeout_ms = 8000;
  int cooldown_ms = 1500;
};

// 识别出一句完整问题后被调用（在语音线程里执行，必须快速返回、不要阻塞）
using QuestionHandler = std::function<void(const std::string& question)>;

class Engine {
 public:
  Engine() = default;
  ~Engine();

  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  bool Init(const Config& cfg);
  bool Start(QuestionHandler handler);
  void Stop();

 private:
  void Loop();

  struct Impl;
  Impl* impl_ = nullptr;
  std::thread thread_;
  std::atomic<bool> running_{false};
  QuestionHandler handler_;
};

}  // namespace voice

#endif  // VOICE_ENGINE_H_