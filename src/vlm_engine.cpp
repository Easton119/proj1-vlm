#include "vlm_engine.h"

#include <chrono>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <opencv2/opencv.hpp>

#include "camera_stream.h"
#include "image_enc.h"
#include "rga_converter.h"
#include "rkllm.h"

namespace {

// 与训练时保持一致：扩方填充灰（127.5）
cv::Mat Expand2Square(const cv::Mat& img, const cv::Scalar& bg) {
  if (img.cols == img.rows) return img.clone();
  const int size = std::max(img.cols, img.rows);
  cv::Mat out(size, size, img.type(), bg);
  cv::Rect roi((size - img.cols) / 2, (size - img.rows) / 2, img.cols, img.rows);
  img.copyTo(out(roi));
  return out;
}

std::string Trim(const std::string& s) {
  const size_t b = s.find_first_not_of(" \t\r\n");
  if (b == std::string::npos) return "";
  const size_t e = s.find_last_not_of(" \t\r\n");
  return s.substr(b, e - b + 1);
}

// 找出句末位置（返回"句末之后"的下标）；找不到返回 npos。
// 中文标点是多字节，必须按子串找，不能按单字节判断。
// 为 TTS 切块：
//   1) 句末标点（。！？\n）—— 无条件切，这是最自然的边界
//   2) 逗号/顿号/分号     —— 只有切出来的片段够长（≥18 字节≈6 个汉字）才切
// 目的：让第一块音频尽快交付，缩短"首字延迟"；同时避免切出"好的，"这种碎片。
size_t FindChunkEnd(const std::string& s) {
  static const char* kHard[] = {"。", "！", "？", "!", "?", "\n"};
  static const char* kSoft[] = {"，", "；", "、"};

  size_t hard = std::string::npos;
  for (const char* e : kHard) {
    const size_t p = s.find(e);
    if (p != std::string::npos) {
      const size_t after = p + std::strlen(e);
      if (hard == std::string::npos || after < hard) hard = after;
    }
  }

  size_t soft = std::string::npos;
  for (const char* e : kSoft) {
    const size_t p = s.find(e);
    if (p != std::string::npos) {
      const size_t after = p + std::strlen(e);
      if (soft == std::string::npos || after < soft) soft = after;
    }
  }

  size_t best = hard;
  if (soft != std::string::npos && soft >= 18) {
    if (best == std::string::npos || soft < best) best = soft;
  }
  return best;
}

// 送给 TTS 之前把 markdown 和 lexicon 里没有的标点去掉。
// 不去的话，日志里那堆 "Ignore OOV '*'" 会变成莫名其妙的停顿。
std::string CleanForTts(const std::string& in) {
  std::string out;
  out.reserve(in.size());
  bool at_line_start = true;   // ★ 新增：跟踪是否在"行首"

  for (size_t i = 0; i < in.size(); ++i) {
    const char c = in[i];

    if (c == '\n' || c == '\r' || c == '\t') {
      at_line_start = true;
      continue;
    }
    if (c == ' ') {
      if (at_line_start) continue;   // 行首空格直接丢掉
      out.push_back(c);
      continue;
    }
    if (c == '*' || c == '#' || c == '`' || c == '_' || c == '>') continue;
    if (c == '-' && at_line_start) continue;  // 行首的列表符号（含前面有空格的情况）

    out.push_back(c);
    at_line_start = false;
  }

  static const char* kDrop[] = {"（", "）", "“", "”", "：", "「", "」", "…"};
  for (const char* d : kDrop) {
    const std::string bad(d);
    size_t pos = 0;
    while ((pos = out.find(bad, pos)) != std::string::npos) {
      out.erase(pos, bad.size());
    }
  }

  std::string collapsed;
  collapsed.reserve(out.size());
  bool prev_space = false;
  for (char ch : out) {
    const bool sp = (ch == ' ');
    if (sp && prev_space) continue;
    collapsed.push_back(ch);
    prev_space = sp;
  }
  return collapsed;
}


// callback 的上下文：既把 token 实时打印出来（保留原来的流式观感），
// 又把它们拼成完整字符串，并统计 token 数用于算 tok/s
struct CallbackCtx {
  std::string text;              // 完整回答（用于 [answer] 打印）
  std::string pending;           // 还没凑成整句的尾巴
  VlmEngine::SentenceHandler on_sentence;   // 可为空
  int callbacks = 0;
  bool has_perf = false;
  RKLLMPerfStat perf{};
};

int LlmCallback(RKLLMResult* result, void* userdata, LLMCallState state) {
  auto* ctx = static_cast<CallbackCtx*>(userdata);

  if (state == RKLLM_RUN_NORMAL) {
    if (result != nullptr && result->text != nullptr) {
      std::printf("%s", result->text);
      std::fflush(stdout);
      if (ctx != nullptr) {
        ctx->text += result->text;

        ctx->pending += result->text;
        if (ctx->on_sentence) {
          size_t pos;
          while ((pos = FindChunkEnd(ctx->pending)) != std::string::npos) {
            const std::string sentence = ctx->pending.substr(0, pos);
            ctx->pending.erase(0, pos);
            const std::string clean = CleanForTts(sentence);
            if (!clean.empty()) ctx->on_sentence(clean);
          }
        }

        ++ctx->callbacks;   // ★ 原来是 ctx->tokens += result->num;（1.3.1 没这个字段）
      }
    }
  } else if (state == RKLLM_RUN_FINISH) {
    std::printf("\n");
    std::fflush(stdout);
  } else if (state == RKLLM_RUN_ERROR) {
    std::fprintf(stderr, "\\run error\n");
  }

  // perf 由固件在生成过程中/结束时填充；每帧都看一眼，谁填了用谁
  if (result != nullptr && result->perf.generate_tokens > 0 && ctx != nullptr) {
    ctx->perf = result->perf;
    ctx->has_perf = true;
  }
  return 0;
}

double MsBetween(const std::chrono::high_resolution_clock::time_point& a,
                 const std::chrono::high_resolution_clock::time_point& b) {
  return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count() /
         1000.0;
}

}  // namespace

struct VlmEngine::Impl {
  VlmConfig cfg;

  LLMHandle llm = nullptr;
  rknn_app_context_t rknn_ctx;
  bool imgenc_inited = false;

  CameraStream camera;
  RgaConverter rga;
  bool camera_started = false;

  float* img_vec = nullptr;
  size_t n_image_tokens = 0;
  size_t image_width = 0;
  size_t image_height = 0;

  // Qwen3-VL 的占位 token（与原 main.cpp 一致）
  const char* img_start = "<|vision_start|>";
  const char* img_end = "<|vision_end|>";
  const char* img_content = "<|image_pad|>";
};

VlmEngine::VlmEngine() : impl_(new Impl()) {}

VlmEngine::~VlmEngine() {
  Stop();
  delete impl_;
  impl_ = nullptr;
}

bool VlmEngine::Init(const VlmConfig& cfg) {
  impl_->cfg = cfg;

  // ---- 1) 语言主干 ----
  RKLLMParam param = rkllm_createDefaultParam();
  param.model_path = impl_->cfg.llm_model.c_str();
  param.top_k = 1;
  param.max_new_tokens = impl_->cfg.max_new_tokens;
  param.max_context_len = impl_->cfg.max_context_len;
  param.skip_special_token = true;
  param.extend_param.base_domain_id = 0;

  if (impl_->cfg.platform == "rk3588" || impl_->cfg.platform == "rk3576") {
    param.extend_param.base_domain_id = 1;
  } else if (impl_->cfg.platform == "rv1126b" ||
             impl_->cfg.platform == "rk3562") {
    param.extend_param.base_domain_id = 0;
  } else {
    std::fprintf(stderr, "[vlm] unknown platform: %s\n",
                 impl_->cfg.platform.c_str());
    return false;
  }

  RKLLMCallback callback = {};
  callback.result_callback = LlmCallback;

  const auto t0 = std::chrono::high_resolution_clock::now();
  if (rkllm_init(&impl_->llm, &param, &callback) != 0) {
    std::fprintf(stderr, "[vlm] rkllm_init failed\n");
    impl_->llm = nullptr;
    return false;
  }
  const auto t1 = std::chrono::high_resolution_clock::now();
  std::printf("[vlm] LLM loaded in %.0f ms\n", MsBetween(t0, t1));

  // ---- 2) 视觉编码器 ----
  std::memset(&impl_->rknn_ctx, 0, sizeof(impl_->rknn_ctx));
  const auto t2 = std::chrono::high_resolution_clock::now();
  if (init_imgenc(impl_->cfg.vision_model.c_str(), &impl_->rknn_ctx,
                  impl_->cfg.npu_core_num) != 0) {
    std::fprintf(stderr, "[vlm] init_imgenc failed: %s\n",
                 impl_->cfg.vision_model.c_str());
    return false;
  }
  const auto t3 = std::chrono::high_resolution_clock::now();
  impl_->imgenc_inited = true;
  std::printf("[vlm] ImgEnc loaded in %.0f ms\n", MsBetween(t2, t3));

  impl_->image_width = impl_->rknn_ctx.model_width;
  impl_->image_height = impl_->rknn_ctx.model_height;
  impl_->n_image_tokens = impl_->rknn_ctx.model_image_token;

  const size_t embed_len = impl_->rknn_ctx.model_embed_size;
  const size_t n_output = impl_->rknn_ctx.io_num.n_output;
  const size_t total = impl_->n_image_tokens * embed_len * n_output;
  impl_->img_vec = static_cast<float*>(std::malloc(total * sizeof(float)));
  if (impl_->img_vec == nullptr) {
    std::fprintf(stderr, "[vlm] malloc img_vec failed (%zu floats)\n", total);
    return false;
  }
  std::memset(impl_->img_vec, 0, total * sizeof(float));

  std::printf("[vlm] vision input %zux%zu, image tokens=%zu, embed=%zu, "
              "outputs=%zu\n",
              impl_->image_width, impl_->image_height, impl_->n_image_tokens,
              embed_len, n_output);

  // ---- 3) 摄像头 ----
  if (!impl_->camera.start(impl_->cfg.camera_device.c_str(),
                           impl_->cfg.camera_width, impl_->cfg.camera_height)) {
    std::fprintf(stderr, "[vlm] camera.start(%s) failed\n",
                 impl_->cfg.camera_device.c_str());
    return false;
  }
  impl_->camera_started = true;
  std::printf("[vlm] camera %s opened (%dx%d)\n",
              impl_->cfg.camera_device.c_str(), impl_->cfg.camera_width,
              impl_->cfg.camera_height);
  return true;
}

bool VlmEngine::Ask(const std::string& question, std::string* answer,
                    SentenceHandler on_sentence) {
  if (impl_ == nullptr || answer == nullptr) return false;
  answer->clear();

  // ---- 1) 取最新帧 ----
  cv::Mat frame;
  if (!impl_->camera.get_latest(frame)) {
    std::fprintf(stderr, "[vlm] no frame yet\n");
    return false;
  }

  // ---- 2) 预处理：扩方(CPU) → RGA: BGR→RGB + resize 448 ----
  const cv::Mat square =
      Expand2Square(frame, cv::Scalar(127.5, 127.5, 127.5));
  cv::Mat rgb;
  if (!impl_->rga.to_rgb(square, static_cast<int>(impl_->image_width),
                         static_cast<int>(impl_->image_height), &rgb)) {
    std::fprintf(stderr, "[vlm] rga failed\n");
    return false;
  }

  // ---- 3) 视觉编码 ----
  const auto t0 = std::chrono::high_resolution_clock::now();
  if (run_imgenc(&impl_->rknn_ctx, rgb.data, impl_->img_vec) != 0) {
    std::fprintf(stderr, "[vlm] run_imgenc failed\n");
    return false;
  }
  const auto t1 = std::chrono::high_resolution_clock::now();

  // ---- 4) 拼 prompt（★ <image> 前缀不能少，否则模型不看图） ----
  std::string prompt = "<image>" + question +
    "\n请直接回答，不要使用分点";

  RKLLMInput input;
  std::memset(&input, 0, sizeof(input));
  input.input_type = RKLLM_INPUT_MULTIMODAL;
  input.role = "user";
  input.multimodal_input.prompt = const_cast<char*>(prompt.c_str());
  input.multimodal_input.image.image_embed = impl_->img_vec;
  input.multimodal_input.image.n_image_tokens = impl_->n_image_tokens;
  input.multimodal_input.image.n_image = 1;
  input.multimodal_input.image.image_start = impl_->img_start;
  input.multimodal_input.image.image_end = impl_->img_end;
  input.multimodal_input.image.image_content = impl_->img_content;
  input.multimodal_input.image.image_height = impl_->image_height;
  input.multimodal_input.image.image_width = impl_->image_width;

  RKLLMInferParam infer;
  std::memset(&infer, 0, sizeof(infer));
  infer.mode = RKLLM_INFER_GENERATE;
  infer.keep_history = 0;  // 每次问答独立

  // ---- 5) 生成，同时收集文本与 token 数 ----
  CallbackCtx ctx;
  ctx.on_sentence = on_sentence;
  const auto t2 = std::chrono::high_resolution_clock::now();
  const int ret = rkllm_run(impl_->llm, &input, &infer, &ctx);
  const auto t3 = std::chrono::high_resolution_clock::now();
  if (ret != 0) {
    std::fprintf(stderr, "[vlm] rkllm_run failed: %d\n", ret);
    return false;
  }

  const double img_ms = MsBetween(t0, t1);
  const double llm_ms = MsBetween(t2, t3);

  if (ctx.has_perf) {
    const double gen_ms = ctx.perf.generate_time_ms;
    const double tps =
        gen_ms > 0.0 ? ctx.perf.generate_tokens * 1000.0 / gen_ms : 0.0;
    std::printf(
        "[vlm] imgenc=%.0f ms | prefill %d tok / %.0f ms | "
        "generate %d tok / %.0f ms = %.2f tok/s | peak mem %.0f MB\n",
        img_ms, ctx.perf.prefill_tokens, ctx.perf.prefill_time_ms,
        ctx.perf.generate_tokens, ctx.perf.generate_time_ms, tps,
        ctx.perf.memory_usage_mb);
  } else {
    const double tps = llm_ms > 0.0 ? ctx.callbacks * 1000.0 / llm_ms : 0.0;
    std::printf("[vlm] imgenc=%.0f ms, llm=%.0f ms, callbacks=%d "
                "(perf 未填充), 约 %.2f tok/s\n",
                img_ms, llm_ms, ctx.callbacks, tps);
  }

  // 最后一段没有句号，也要送出去
  if (on_sentence && !ctx.pending.empty()) {
    const std::string clean = CleanForTts(ctx.pending);
    if (!clean.empty()) on_sentence(clean);
  }
  *answer = Trim(ctx.text);
  return !answer->empty();
}

void VlmEngine::ClearKvCache() {
  if (impl_ != nullptr && impl_->llm != nullptr) {
    rkllm_clear_kv_cache(impl_->llm, 1, nullptr, nullptr);
  }
}
bool VlmEngine::LatestFrame(cv::Mat* out) {
  if (impl_ == nullptr || out == nullptr || !impl_->camera_started) return false;
  return impl_->camera.get_latest(*out);
}

void VlmEngine::Stop() {
  if (impl_ == nullptr) return;

  if (impl_->camera_started) {
    impl_->camera.stop();
    impl_->camera_started = false;
  }
  if (impl_->imgenc_inited) {
    release_imgenc(&impl_->rknn_ctx);
    impl_->imgenc_inited = false;
  }
  if (impl_->img_vec != nullptr) {
    std::free(impl_->img_vec);
    impl_->img_vec = nullptr;
  }
  if (impl_->llm != nullptr) {
    LLMHandle tmp = impl_->llm;
    impl_->llm = nullptr;
    rkllm_destroy(tmp);
  }
}


