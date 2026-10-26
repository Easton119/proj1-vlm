// main.cpp —— 阶段 D：语音提问 + 摄像头实时帧 + VLM 回答
//
// 三个线程：
//   1) 语音线程（voice::Engine 内部还有采集线程）
//   2) VLM 线程：从"单槽队列"取问题 → 抓最新帧 → 视觉编码 → 生成
//   3) stdin 线程：保留打字提问的调试方式（不打字也不影响语音）
//
// 关键：语音识别出的文本在 VlmEngine::Ask 里会被加上 "<image>" 前缀，
// 否则 RKLLM 不会把它当多模态输入，模型就不会看图。

#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iostream>
#include <mutex>
#include <chrono>
#include <string>
#include <thread>

#include "vlm_engine.h"
#include "voice_engine.h"
#include "tts_engine.h"

namespace {

volatile std::sig_atomic_t g_running = 1;
void OnSignal(int) { g_running = 0; }

struct Args {
  VlmConfig vlm;
  bool voice_enabled = false;
  voice::Config voice;
  bool tts_enabled = false;     // 新增
  TtsConfig tts;                // 新增
};

void PrintUsage(const char* p) {
  std::fprintf(
      stderr,
      "usage: %s --llm=<rkllm> --vision=<rknn> [options]\n"
      "  VLM:\n"
      "    --camera=/dev/video21  --width=1280  --height=720\n"
      "    --max-new-tokens=256   --max-context-len=4096\n"
      "    --npu-core=3           --platform=rk3588\n"
      "  Voice (全部给出才启用):\n"
      "    --voice --keywords-file= --kws-tokens= --kws-encoder=\n"
      "    --kws-decoder= --kws-joiner= --asr-model= --asr-tokens=\n"
      "    --vad-model= [--device=plughw:2,0] [--language=zh]\n"
      "    [--kws-threshold=0.25] [--vad-threshold=0.25]\n"
      "    [--command-timeout-ms=8000] [--cooldown-ms=1500]\n",
      p);
}

// 返回 1 继续，0 参数错误
int ParseArgs(int argc, char** argv, Args* a) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--voice") { a->voice_enabled = true; continue; }
    if (arg == "--tts") { a->tts_enabled = true; continue; }
    if (arg.rfind("--", 0) != 0) {
      std::fprintf(stderr, "unexpected argument: %s\n", arg.c_str());
      return 0;
    }
    std::string key, value;
    const size_t eq = arg.find('=');
    if (eq != std::string::npos) {
      key = arg.substr(2, eq - 2);
      value = arg.substr(eq + 1);
    } else {
      key = arg.substr(2);
      if (i + 1 >= argc) { std::fprintf(stderr, "missing value: --%s\n", key.c_str()); return 0; }
      value = argv[++i];
    }

    if (key == "llm") a->vlm.llm_model = value;
    else if (key == "vision") a->vlm.vision_model = value;
    else if (key == "camera") a->vlm.camera_device = value;
    else if (key == "width") a->vlm.camera_width = std::atoi(value.c_str());
    else if (key == "height") a->vlm.camera_height = std::atoi(value.c_str());
    else if (key == "max-new-tokens") a->vlm.max_new_tokens = std::atoi(value.c_str());
    else if (key == "max-context-len") a->vlm.max_context_len = std::atoi(value.c_str());
    else if (key == "npu-core") a->vlm.npu_core_num = std::atoi(value.c_str());
    else if (key == "platform") a->vlm.platform = value;
    else if (key == "device") a->voice.device = value;
    else if (key == "keywords-file") a->voice.keywords_file = value;
    else if (key == "kws-tokens") a->voice.kws_tokens = value;
    else if (key == "kws-encoder") a->voice.kws_encoder = value;
    else if (key == "kws-decoder") a->voice.kws_decoder = value;
    else if (key == "kws-joiner") a->voice.kws_joiner = value;
    else if (key == "asr-model") a->voice.asr_model = value;
    else if (key == "asr-tokens") a->voice.asr_tokens = value;
    else if (key == "vad-model") a->voice.vad_model = value;
    else if (key == "language") a->voice.language = value;
    else if (key == "threads") a->voice.threads = std::atoi(value.c_str());
    else if (key == "kws-threshold") a->voice.kws_threshold = std::atof(value.c_str());
    else if (key == "vad-threshold") a->voice.vad_threshold = std::atof(value.c_str());
    else if (key == "command-timeout-ms") a->voice.command_timeout_ms = std::atoi(value.c_str());
    else if (key == "cooldown-ms") a->voice.cooldown_ms = std::atoi(value.c_str());
    else if (key == "tts-model") a->tts.model = value;
    else if (key == "tts-lexicon") a->tts.lexicon = value;
    else if (key == "tts-tokens") a->tts.tokens = value;
    else if (key == "tts-device") a->tts.device = value;
    else if (key == "tts-threads") a->tts.num_threads = std::atoi(value.c_str());
    else if (key == "tts-sid") a->tts.sid = std::atoi(value.c_str());
    else if (key == "tts-speed") a->tts.speed = static_cast<float>(std::atof(value.c_str()));
    else if (key == "tts-dump") a->tts.dump_dir = value;
    else { std::fprintf(stderr, "unknown option: --%s\n", key.c_str()); return 0; }
  }
  if (a->vlm.llm_model.empty() || a->vlm.vision_model.empty()) return 0;

  if (a->voice_enabled) {
    const std::string* req[] = {&a->voice.keywords_file, &a->voice.kws_tokens,
                                &a->voice.kws_encoder, &a->voice.kws_decoder,
                                &a->voice.kws_joiner, &a->voice.asr_model,
                                &a->voice.asr_tokens, &a->voice.vad_model};
    for (const auto* s : req) {
      if (s->empty()) {
        std::fprintf(stderr, "[voice] missing model paths\n");
        return 0;
      }
    }
  }
  if (a->tts_enabled) {
    if (a->tts.model.empty() || a->tts.lexicon.empty() ||
        a->tts.tokens.empty()) {
      std::fprintf(stderr, "[tts] need --tts-model/--tts-lexicon/--tts-tokens\n");
      return 0;
    }
  }
  return 1;
}

std::string Trim(std::string s) {
  const size_t b = s.find_first_not_of(" \t\r\n");
  if (b == std::string::npos) return "";
  const size_t e = s.find_last_not_of(" \t\r\n");
  return s.substr(b, e - b + 1);
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  if (!ParseArgs(argc, argv, &args)) {
    PrintUsage(argv[0]);
    return 2;
  }

  std::signal(SIGINT, OnSignal);
  std::signal(SIGTERM, OnSignal);

  VlmEngine vlm;
  if (!vlm.Init(args.vlm)) {
    std::fprintf(stderr, "[main] vlm init failed\n");
    return 1;
  }

  TtsEngine tts;
  bool tts_started = false;
  if (args.tts_enabled) {
    if (!tts.Init(args.tts)) {
      std::fprintf(stderr, "[main] tts init failed, continue without TTS\n");
    } else {
      tts_started = true;
    }
  }

  // ---- 单槽问题队列 ----
  std::mutex mu;
  std::condition_variable cv;
  std::deque<std::string> pending;  // 容量语义：最多 1 条
  bool running = true;

  auto submit = [&](const std::string& q) {
    const std::string text = Trim(q);
    if (text.empty()) return;
    {
      std::lock_guard<std::mutex> lock(mu);
      if (!pending.empty()) {
        std::printf("[main] vlm busy, replace pending with latest\n");
        pending.pop_back();
      }
      pending.push_back(text);
    }
    cv.notify_one();
  };

  // ---- VLM 线程 ----
  std::thread vlm_thread([&]() {
    while (true) {
      std::string question;
      {
        std::unique_lock<std::mutex> lock(mu);
        cv.wait(lock, [&]() { return !pending.empty() || !running; });
        if (!running && pending.empty()) return;
        question = pending.front();
        pending.pop_front();
      }

      std::printf("\n[ask] %s\n", question.c_str());
      std::fflush(stdout);

      // 每凑够一句就立刻交给 TTS：边生成边播报
      const auto on_sentence = [&](const std::string& sentence) {
        if (tts_started) tts.Say(sentence);
      };

      std::string answer;
      const bool ok = vlm.Ask(question, &answer, on_sentence);

      if (ok) {
        std::printf("[answer] %s\n", answer.c_str());
        std::fflush(stdout);
        // 注意：这里不要再 tts.Say(answer)，否则整段会重念一遍
      } else {
        std::fprintf(stderr, "[main] ask failed\n");
      }
    }
  });

  // ---- 语音线程 ----
  voice::Engine engine;
  bool voice_started = false;
  if (args.voice_enabled) {
    if (!engine.Init(args.voice)) {
      std::fprintf(stderr, "[main] voice init failed\n");
    } else if (!engine.Start(submit)) {
      std::fprintf(stderr, "[main] voice start failed\n");
    } else {
      voice_started = true;
      std::printf("[main] voice enabled\n");
    }
  } else {
    std::printf("[main] voice disabled (typing only)\n");
  }

  // ---- stdin 线程：保留打字提问 ----
  std::thread stdin_thread([&]() {
    std::string line;
    while (running) {
      if (!std::getline(std::cin, line)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        continue;
      }
      line = Trim(line);
      if (line == "exit") {
        running = false;
        cv.notify_all();
        break;
      }
      if (line == "clear") { vlm.ClearKvCache(); continue; }
      if (!line.empty()) submit(line);
    }
  });

  std::printf("[main] ready. 说\"你好小智\"提问，或直接打字。exit 退出。\n");

  while (g_running != 0 && running) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }

  std::printf("\n[main] shutting down...\n");
  if (tts_started) tts.Stop();
  if (voice_started) engine.Stop();
  {
    std::lock_guard<std::mutex> lock(mu);
    running = false;
  }
  cv.notify_all();
  if (vlm_thread.joinable()) vlm_thread.join();
  if (stdin_thread.joinable()) stdin_thread.join();
  vlm.Stop();
  std::printf("[main] bye\n");
  return 0;
}