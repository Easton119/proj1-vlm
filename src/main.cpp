// main.cpp —— 阶段 D：语音提问 + 摄像头实时帧 + VLM 回答
//
// 三个线程：
//   1) 语音线程（voice::Engine 内部还有采集线程）
//   2) VLM 线程：从"单槽队列"取问题 → 抓最新帧 → 视觉编码 → 生成
//   3) stdin 线程：保留打字提问的调试方式（不打字也不影响语音）
//
// 关键：语音识别出的文本在 VlmEngine::Ask 里会被加上 "<image>" 前缀，
// 否则 RKLLM 不会把它当多模态输入，模型就不会看图。

#include <atomic>
#include <condition_variable>
#include <csignal>
#include <ctime>
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
#include "mpp_encoder.h"
#include "rtmp_publisher.h"
#include <opencv2/opencv.hpp>

namespace {

volatile std::sig_atomic_t g_running = 1;
void OnSignal(int) { g_running = 0; }

struct Args {
  VlmConfig vlm;
  bool voice_enabled = false;
  voice::Config voice;
  bool tts_enabled = false;     // 新增
  TtsConfig tts;                // 新增
  // ---- 阶段⑤：MPP 硬编码推流 ----
  bool push_enabled = false;
  std::string push_url;         // rtmp://host:1935/app/stream 或 /tmp/test.flv
  int push_fps = 15;
  int push_bitrate = 1500000;   // bps
  bool push_timestamp = false;  // 把本机时间烧进画面（量端到端延迟用）
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
      "  Push (MPP H.264 -> FLV/RTMP):\n"
      "    --push=<url|/path/x.flv> [--push-fps=15]\n"
      "    [--push-bitrate=1500000] [--push-timestamp]\n",
      p);
}

// 返回 1 继续，0 参数错误
int ParseArgs(int argc, char** argv, Args* a) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--voice") { a->voice_enabled = true; continue; }
    if (arg == "--tts") { a->tts_enabled = true; continue; }
    if (arg == "--push-timestamp") { a->push_timestamp = true; continue; }
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
    else if (key == "push") { a->push_enabled = true; a->push_url = value; }
    else if (key == "push-fps") a->push_fps = std::atoi(value.c_str());
    else if (key == "push-bitrate") a->push_bitrate = std::atoi(value.c_str());
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
  if (a->push_enabled && a->push_url.empty()) {
    std::fprintf(stderr, "[push] --push needs <url|file>\n");
    return 0;
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

  // ---- 推流（阶段⑤：MPP H.264 + FLV/RTMP）----
  // 设计要点：推流线程按固定节奏向 VLM 引擎取"当前最新帧"，取不到就等 5 ms。
  //   - 不经过采集线程 → 推流不会拖慢采集
  //   - 不共享队列    → 网络阻塞时不会把帧积压（天然丢帧，始终直播最新画面）
  //   - 单独线程      → VLM 推理期间推流照常进行，互不阻塞
  MppEncoder push_enc;
  RtmpPublisher push_pub;
  std::atomic<bool> push_running{false};
  std::thread push_thread;
  bool push_started = false;

  if (args.push_enabled) {
    const int pw = args.vlm.camera_width;
    const int ph = args.vlm.camera_height;
    if (!push_enc.Init(pw, ph, args.push_fps, args.push_bitrate)) {
      std::fprintf(stderr, "[push] encoder init failed, push disabled\n");
    } else if (!push_pub.Open(args.push_url, pw, ph, args.push_fps,
                              push_enc.header())) {
      std::fprintf(stderr, "[push] open %s failed, push disabled\n",
                   args.push_url.c_str());
    } else {
      push_started = true;
      push_running = true;
      push_thread = std::thread([&]() {
        const int fps = args.push_fps > 0 ? args.push_fps : 15;
        const auto period = std::chrono::microseconds(1000000 / fps);
        auto next = std::chrono::steady_clock::now();
        auto last_report = next;
        int64_t idx = 0;
        int64_t sent = 0;
        int64_t sent_at_report = 0;
        int64_t bytes_at_report = 0;
        int64_t enc_fail = 0;
        int64_t lag_skip = 0;
        int64_t reconnects = 0;
        std::vector<uint8_t> pkt;
        cv::Mat frame;

        while (push_running.load()) {
          if (!vlm.LatestFrame(&frame)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
          }
          if (frame.cols != args.vlm.camera_width ||
              frame.rows != args.vlm.camera_height) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
          }

          if (args.push_timestamp) {
            const auto now = std::chrono::system_clock::now();
            const std::time_t t = std::chrono::system_clock::to_time_t(now);
            const int ms = static_cast<int>(
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    now.time_since_epoch()).count() % 1000);
            std::tm tmv;
            localtime_r(&t, &tmv);
            char stamp[48];
            std::snprintf(stamp, sizeof(stamp), "%02d:%02d:%02d.%03d",
                          tmv.tm_hour, tmv.tm_min, tmv.tm_sec, ms);
            cv::putText(frame, stamp, cv::Point(10, 32),
                        cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 0, 255), 2);
          }

          bool key = false;
          if (push_enc.Encode(frame, &pkt, &key)) {
            if (!pkt.empty()) {
              const int64_t pts_us = idx * 1000000LL / fps;
              if (push_pub.Write(pkt.data(), pkt.size(), pts_us, key)) {
                ++sent;
              } else {
                std::fprintf(stderr, "[push] write failed, reconnecting...\n");
                push_pub.Close();
                std::this_thread::sleep_for(std::chrono::seconds(1));
                if (push_pub.Open(args.push_url, args.vlm.camera_width,
                                  args.vlm.camera_height, fps,
                                  push_enc.header())) {
                  ++reconnects;
                } else {
                  std::fprintf(stderr, "[push] reopen failed, retry later\n");
                }
              }
            }
          } else {
            ++enc_fail;
          }
          ++idx;

          const auto now2 = std::chrono::steady_clock::now();
          const double win =
              std::chrono::duration<double>(now2 - last_report).count();
          if (win >= 1.0) {
            const double kbps =
                (push_pub.bytes_written() - bytes_at_report) * 8.0 / win / 1000.0;
            std::printf(
                "[push] fps=%.1f kbps=%.0f sent=%lld lagskip=%lld enc_fail=%lld "
                "reconn=%lld\n",
                (sent - sent_at_report) / win, kbps, (long long)sent,
                (long long)lag_skip, (long long)enc_fail,
                (long long)reconnects);
            std::fflush(stdout);
            last_report = now2;
            sent_at_report = sent;
            bytes_at_report = push_pub.bytes_written();
          }

          next += period;
          const auto after = std::chrono::steady_clock::now();
          if (after > next + period) {
            next = after;   // 落后太多就重置节奏，绝不追帧积压
            ++lag_skip;
          }
          std::this_thread::sleep_until(next);
        }
        std::printf("[push] thread exit\n");
      });
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
  
  if (push_started) {                 // ★ 先停推流：它依赖 VLM 的摄像头
    push_running = false;
    if (push_thread.joinable()) push_thread.join();
    push_pub.Close();
    push_enc.Release();
  }
  
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