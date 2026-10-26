// tts_engine.h —— sherpa-onnx 离线 TTS + ALSA 播放，独立线程
//
// 设计：
//   Say(text) 只入队，立即返回（VLM 线程不被 4 秒的合成阻塞）
//   内部线程：取文本 → GenerateWithConfig(回调流式) → 回调里直接写 ALSA → drain
//
// 实测基准（melo-tts-zh_en，RK3588，4 线程）：
//   5.28 s 音频，合成 4.03 s，RTF 0.76 → 生成比播放快，可以边生成边播

#ifndef TTS_ENGINE_H_
#define TTS_ENGINE_H_

#include <string>

struct TtsConfig {
  std::string model;                          // model.onnx
  std::string lexicon;                        // lexicon.txt
  std::string tokens;                         // tokens.txt
  std::string device = "plughw:1,0";          // 板载 ES8388

  std::string dump_dir;                       // ★ 新增：非空则把每段合成结果存成 wav

  int num_threads = 4;                        // RK3588 四个 A76
  int sid = 0;                                // melo 是单说话人，只能 0
  float speed = 1.0f;
  float silence_scale = 0.2f;
};

class TtsEngine {
 public:
  TtsEngine() = default;
  ~TtsEngine();

  TtsEngine(const TtsEngine&) = delete;
  TtsEngine& operator=(const TtsEngine&) = delete;

  bool Init(const TtsConfig& cfg);
  void Say(const std::string& text);   // 非阻塞：入队后立刻返回
  void Stop();

 private:
  struct Impl;
  Impl* impl_ = nullptr;
};

#endif  // TTS_ENGINE_H_