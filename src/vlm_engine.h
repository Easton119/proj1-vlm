// vlm_engine.h —— 把原 main.cpp 里的 VLM 逻辑封装成一个可反复调用的对象
//
// 和原 main.cpp 的对应关系：
//   rkllm_init / init_imgenc / camera.start  → Init()
//   抓帧 → expand2square → RGA → run_imgenc → rkllm_run → Ask()
//   release_imgenc / camera.stop / rkllm_destroy → Stop() / 析构
//
// 唯一的行为改动：callback 通过 userdata 把生成的文本收集成 std::string，
// 这样上层（将来接 TTS）能拿到完整回答，而不只是屏幕上闪过的一串字。

#ifndef VLM_ENGINE_H_
#define VLM_ENGINE_H_

#include <string>
#include <functional>
#include <opencv2/core.hpp>

struct VlmConfig {
  std::string llm_model;                        // master_w8a8_rk3588.rkllm
  std::string vision_model;                     // qwen3-vl_vision_rk3588.rknn
  std::string camera_device = "/dev/video21";
  int camera_width = 1280;
  int camera_height = 720;
  int max_new_tokens = 256;                     // 语音场景不需要 2048，短答更快
  int max_context_len = 4096;
  int npu_core_num = 3;
  std::string platform = "rk3588";
};

class VlmEngine {
 public:
  VlmEngine();
  ~VlmEngine();

  VlmEngine(const VlmEngine&) = delete;
  VlmEngine& operator=(const VlmEngine&) = delete;

  bool Init(const VlmConfig& cfg);
  void Stop();

  // 同步调用：读最新帧 → 视觉编码 → 生成回答。
  // question 不需要自带 "<image>" 前缀，函数内部会补。
  
  // on_sentence 不为空时，每凑够一句就回调一次（用于边生成边播报）
  using SentenceHandler = std::function<void(const std::string&)>;
  bool Ask(const std::string& question, std::string* answer,
           SentenceHandler on_sentence = nullptr);

  void ClearKvCache();
  // ★ 新增：给推流线程取"当前最新一帧"（只读，不触发推理、不影响 VLM 路径）
  bool LatestFrame(cv::Mat* out);

 private:
  struct Impl;
  Impl* impl_ = nullptr;
};

#endif  // VLM_ENGINE_H_