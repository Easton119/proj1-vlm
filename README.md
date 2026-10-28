# vlm — RK3588 端侧多模态视觉问答

在 RK3588（6 TOPS NPU / 16 GB）上运行的**离线**多模态问答终端：摄像头实时画面 + 语音提问 → Qwen3-VL-2B 看图回答 → 语音播报；同时把摄像头画面以 H.264/RTMP 实时推送到远端。

全链路不依赖云端，模型推理、语音识别、语音合成、视频编码全部在板端完成。

```
"你好小智" → Silero VAD 断句 → SenseVoice 识别 → 抓最新帧
          → rknn 视觉编码（196 image tokens）→ rkllm 生成 → 分句
          → melo TTS → 耳机

摄像头（MJPG 640x480@30）─ 同一份解码帧 ─┬→ RGA 448 → 视觉编码器 → 语言主干
                                          └→ RGA NV12（直写 DMA-BUF）→ MPP H.264
                                             → FLV/RTMP → MediaMTX → WebRTC 浏览器
```

## 1. 能力一览

| 能力 | 状态 | 说明 |
|---|---|---|
| 单图/实时帧视觉问答 | ✅ | Qwen3-VL-2B，两段式部署（`vision.rknn` + `instruct.rkllm`） |
| 语音唤醒 | ✅ | 唤醒词「你好小智」，KWS（中英混合模型） |
| 语音断句 + 识别 | ✅ | Silero VAD + SenseVoice（中文，带标点） |
| 语音播报 | ✅ | melo TTS（44.1 kHz，中英混合），逐句流式合成 |
| 视频实时推流 | ✅ | MPP 硬编 H.264 → FLV/RTMP，PC 端 MediaMTX 转 WebRTC |
| 打字提问（调试） | ✅ | stdin 输入，不依赖麦克风 |
| 打断正在播放的语音 | ⬜ | 未实现 |
| 数字/日期口语化（`number.fst`/`date.fst`） | ⬜ | 未接入 |
| 视觉编码缓存（画面不变时跳过 2.3 s） | ⬜ | 未实现 |

---

## 2. 硬件与环境

### 2.1 硬件（实测环境）

| 项 | 值 |
|---|---|
| 开发板 | 迅为 iTOP RK3588（RK3588，6 TOPS NPU） |
| 内存 | 15949.6 MiB（≈15.6 GiB），无 Swap |
| 摄像头 | XWF-1080P USB UVC，节点 `/dev/video21`（`video22` 为 metadata） |
| 麦克风 | 同一台 UVC 摄像头内置，ALSA `plughw:2,0`（16 kHz 单声道） |
| 扬声器 | 板载 ES8388，ALSA `plughw:1,0`（外接 3.5 mm） |

> ⚠️ 摄像头与麦克风是**同一台 USB 设备**，且 `/dev/video*`、`/dev/snd` 都是**独占**的：同一时刻只能有一个进程在采集。启动前务必确认上一进程已退出。

### 2.2 软件栈（实测版本）

| 组件 | 版本 / 路径 |
|---|---|
| OS | Ubuntu 20.04.6 LTS，glibc 2.31 |
| 编译器 | gcc/g++ 9.4.0，C++11，CMake ≥ 3.16 |
| RKLLM runtime | 1.3.1（`lib/librkllmrt.so`），rknpu driver 0.9.8 |
| RKNN runtime | `lib/librknnrt.so`（具体版本未打印） |
| sherpa-onnx | v2.0.0（`/home/topeet/Qyj/sherpa-onnx`，**项目外依赖**） |
| MPP | 头文件 `/usr/include/rockchip`，库 `librockchip_mpp.so` |
| RGA | `librga.so`（运行时打印 `rga_api version 1.10.1_[0]`） |
| FFmpeg | 4.2.7（`libavformat 58.29.100`），需 `libavformat-dev` |
| OpenCV | 系统 apt 包（4.2.x） |
| 推流服务端 | MediaMTX v1.21.3（PC 侧，RTMP `:1935` / HLS `:8888` / WebRTC `:8889`） |

> **必须板端本机编译**。交叉工具链 glibc 2.33 高于板子 2.31，交叉编译产物会报 `GLIBC_2.33 not found`。

---

## 3. 模型清单

| 模型 | 体积 | 放置位置 |
|---|---|---|
| `master_w8a8_rk3588.rkllm`（语言主干，W8A8） | 2.37 GB | `models/` |
| `qwen3-vl_vision_rk3588.rknn`（视觉编码器，fp16，448×448） | 852 MB | `models/` |
| KWS：`sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20` | 约 40 MB | `/home/topeet/Qyj/kws_model_v2/…` |
| ASR：`sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17` | 228 MB | `/home/topeet/Qyj/voice/models/…` |
| VAD：`silero_vad.onnx` | 629 KB | `/home/topeet/Qyj/voice/models/` |
| TTS：`vits-melo-tts-zh_en` | 163 MB | `models/` |
| 唤醒词表 | — | `/home/topeet/Qyj/kws_model/keywords.txt` |

`models/` 与 `lib/` 已在 `.gitignore` 中，**不进仓库**。

模型转换（视觉塔用 RKNN-Toolkit2 转 fp16、语言主干用 RKLLM-Toolkit 转 w8a8，均在 WSL 中完成）的详细步骤见配套交接文档；若要随仓库分发，建议把文档放进 `docs/` 并更新本节链接。

---

## 4. 构建

```bash
cd /home/topeet/Qyj/proj1-vlm
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
```

产物：

| 可执行 | 用途 |
|---|---|
| `build/demo` | 主程序（VLM + 语音 + TTS + 推流） |
| `build/enc_dump` | 编码器诊断工具（不依赖网络和 VLM） |

> 若 `libavformat` 报 `undefined reference to avformat_*`：见 §8.3。

---

## 5. 运行

所有命令都在**板子 SSH** 内执行。

### 5.1 一键启动

```bash
bash scripts/run_demo.sh                      # 只用 VLM，键盘提问
bash scripts/run_demo.sh --tts                # VLM + 播报（打字问，语音答）
bash scripts/run_demo.sh --voice              # VLM + 语音提问（只有文字回答）
bash scripts/run_demo.sh --voice --tts        # 语音问 + 语音答
bash scripts/run_demo.sh --voice --tts --push # 再加 RTMP 推流（完整形态）
```

### 5.2 只看推流（含延迟观测）

```bash
bash scripts/run_demo.sh --push --push-timestamp
```

PC 上（Windows）先启动 `mediamtx.exe`，然后用浏览器打开：

```
http://127.0.0.1:8889/live/cam      # WebRTC，延迟最低
http://127.0.0.1:8888/live/cam      # HLS，作为备用
```

### 5.3 先验证封装、再验证网络

```bash
RTMP_URL=/tmp/test.flv bash scripts/run_demo.sh --push
ffprobe -v error -show_entries stream=codec_name,width,height,nb_read_frames -count_frames /tmp/test.flv
```

### 5.4 只验证编码器

```bash
./build/enc_dump /dev/video21 640 480 15 1500000 10 /tmp/enc_640.h264
```

### 5.5 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `RTMP_URL` | `rtmp://192.168.137.1:1935/live/cam` | 也可填本地文件路径 |
| `PUSH_FPS` | `15` | 推流帧率 |
| `PUSH_BITRATE` | `1500000` | 目标码率（bps） |
| `CAM_W` / `CAM_H` | `640` / `480` | 采集与编码分辨率 |
| `MIC_DEVICE` | `plughw:2,0` | 麦克风 |
| `TTS_DEVICE` | `plughw:1,0` | 播放 |
| `SHERPA_ROOT` | `/home/topeet/Qyj/sherpa-onnx` | sherpa-onnx 根目录 |

### 5.6 demo 参数

```
VLM:
  --llm=<rkllm>  --vision=<rknn>
  --camera=/dev/video21  --width=640  --height=480
  --max-new-tokens=256   --max-context-len=4096
  --npu-core=3           --platform=rk3588
语音输入（全部给出才启用）:
  --voice --keywords-file= --kws-tokens= --kws-encoder=
  --kws-decoder= --kws-joiner= --asr-model= --asr-tokens=
  --vad-model= [--device=] [--language=zh]
  [--kws-threshold=0.25] [--vad-threshold=0.25]
  [--command-timeout-ms=8000] [--cooldown-ms=1500]
语音输出:
  --tts --tts-model= --tts-lexicon= --tts-tokens=
  [--tts-device=] [--tts-threads=4] [--tts-sid=0]
  [--tts-speed=1.0] [--tts-dump=<dir>]
推流:
  --push=<url|/path/x.flv> [--push-fps=15] [--push-bitrate=1500000]
  [--push-timestamp]
```

交互：说「你好小智」后提问，或直接键盘输入问题；输入 `exit` 退出，`clear` 清 KV cache。

---

## 6. 实测性能

> 以下均为本机实测值，环境见 §2。测试日期 2026-10-26。

### 6.1 视觉问答（语音链路）

| 指标 | 实测 |
|---|---|
| 视觉编码 `imgenc` | **2.31 ~ 2.37 s** |
| prefill | **214 ~ 218 token / 1.0 ~ 1.3 s**（其中 196 个是图像 token） |
| 生成速度 | **6.75 ~ 10.53 tok/s** |
| 进程常驻内存（RES） | **3.9 ~ 4.1 GB**（模型 2.37 + 0.85 GB） |
| ASR 解码 | 0.13 ~ 0.24 s（audio 1.5 ~ 4.1 s） |
| TTS 边际 RTF | 0.70 ~ 0.74（与推理重叠时劣化到 1.33） |
| 首字延迟 | 约 **7.9 s**（含 imgenc + prefill 的硬地板） |
| 一次完整问答 | 约 35 s（回答 20 s 语音时） |

### 6.2 推流

| 指标 | 实测 |
|---|---|
| 单帧编码耗时 | **3.3 ms/帧**（含 RGA 转换，640×480） |
| 码率 | 1078 ~ 1524 kbps（目标 1500，CBR 收敛） |
| 稳定帧率 | **15.0 fps**，96 秒 1435 帧无掉帧 |
| VLM 推理期间 | `fps=15.0 / lagskip=0 / reconn=0`（**互不阻塞**） |
| 加速比 | 编码 ≈ 300 fps 能力，实际只用 5% 帧预算 |

### 6.3 稳定性（9.5 分钟满载、4 次问答）

| 指标 | 实测 |
|---|---|
| 温度 | 空闲 43~46 °C；问答瞬间峰值 **82 °C**，30 s 内回落；稳态 45~55 °C |
| A76 频率 | 全程 **2352000 kHz**，未触发降频 |
| 线程数 | 恒定 **20**（含 rkllm 内部线程），无线程泄漏 |
| 进程 RSS | 4255 → 4315 MB（**每次问答约 +15 MB**，是否泄漏待排查，见 §9） |

---

## 7. 项目结构

```
proj1-vlm/
├── CMakeLists.txt
├── README.md
├── src/
│   ├── main.cpp              # 参数解析 + 线程编排（VLM / 语音 / 推流 / stdin）
│   ├── vlm_engine.h/.cpp     # VLM 封装：rkllm + imgenc + 摄像头，逐句回调
│   ├── voice_engine.h/.cpp   # 语音：ALSA + DC blocker + KWS + Silero VAD + SenseVoice
│   ├── tts_engine.h/.cpp     # TTS：melo 合成线程 + ALSA 播放线程 + PCM FIFO
│   ├── camera_stream.h/.cpp  # 采集线程 + 「最新帧」单槽
│   ├── v4l2_capture.h/.cpp   # V4L2 MJPG 采集（mmap 4 buffer）
│   ├── rga_converter.h/.cpp  # RGA：BGR→RGB + 缩放（VLM 预处理）
│   ├── rga_lock.h/.cpp       # RGA 全局锁（与编码器共用）
│   ├── mpp_encoder.h/.cpp    # MPP H.264 编码封装（RGA→DMA-BUF 零拷贝）
│   ├── rtmp_publisher.h/.cpp # H.264(Annex-B) → AVCC → FLV → RTMP
│   ├── image_enc.cc/.h       # 视觉编码器（rknn）封装：init/run/release_imgenc
│   ├── audio_enc.cc/.h       # 官方 demo 遗留（Gemma4 音频），未使用
│   ├── audio_utils.cc/.h     # 官方 demo 遗留，未使用
│   ├── audio_encoder.cpp     # 官方 demo 遗留工具，未使用
│   └── img_encoder.cpp       # 官方 demo 遗留工具，未使用
├── tools/
│   └── enc_dump.cpp          # 编码器诊断工具
├── include/                  # 第三方头文件（rknn_inc / rkllm_inc / miniaudio / pocketfft）
├── lib/                      # librknnrt.so / librkllmrt.so（不进 git）
├── models/                   # 模型（不进 git）
└── scripts/
    ├── run_demo.sh
    └── soak.sh               # 长稳记录（温度 / 内存 / 线程数）
```

---

## 8. 设计要点与常见问题

### 8.1 四个关键设计

1. **两段式 VLM**：`vision.rknn`（fp16）出 196 个 image token，`instruct.rkllm`（W8A8）出文字。`prefill` 两百多 token 是「图真的进去了」的硬判据。
2. **单槽最新帧**：采集线程只保留最新一帧；VLM 与推流各按自己的节奏来取。旧帧没有价值，队列只会增加延迟。
3. **三种背压策略**：视频帧=单槽拉取（可再生）；音频块=有界队列满丢最旧（要连贯）；回答文本=合并到队尾（**一个字符都不能丢**）。
4. **零拷贝只在能省的地方省**：USB 摄像头传 MJPG 必须解码（这段做不到）；RGA 把 NV12 **直接写进 MPP 的 DMA-BUF**，省掉「解码后 BGR → 编码器输入」这次 CPU 拷贝。

### 8.2 设备被占用

```
# 检查是否有进程占着摄像头/麦克风
ps aux | grep -E "demo|enc_dump|kws_test|asr_vad|voice_app"
```

### 8.3 链接报 `undefined reference to avformat_*`

FFmpeg 是 C 库，C++ 里必须包 `extern "C"`：

```cpp
extern "C" {
#include <libavformat/avformat.h>
}
```

否则引用的是 C++ mangled 名；再叠加 Ubuntu 默认的 `-Wl,--as-needed`，链接器会把 `-lavformat` 整个丢掉。

### 8.4 退出时出现 FLV 警告

```
[flv] Failed to update header with correct duration / filesize.
```

FLV 封装器要在 trailer 里回写时长和文件大小，而 RTMP socket 不可 seek。**无害**，不影响播放。想消掉可在 `RtmpPublisher::Close()` 里判断 `fmt_->pb->seekable` 再调 `av_write_trailer`。

### 8.5 推流断了不恢复

`[push] reconn` 持续增长说明网络/服务端有问题；推流线程会自动重连（1 s 退避），不会影响采集与 VLM。先查 PC 侧 MediaMTX 是否在跑、1935 是否可达。

---

## 9. 已知限制

| 项 | 说明 |
|---|---|
| 首字延迟 ~7.9 s | 主因 `imgenc 2.3 s + prefill 1.1 s`；优化方向见交接文档（视觉编码缓存 / 降分辨率需重转模型） |
| TTS 句内偶有块间停顿 | melo 分块交付，靠 3 s ALSA 缓冲掩盖 |
| 不支持打断播报 | 新回答排队播完 |
| 数字/日期口语化未接 | `number.fst` / `date.fst` 未加载 |
| 每次问答 RSS +15 MB | 已记录待排查（疑似 `image_enc.cc` 中 `rknn_outputs_release` 只释放 1/4 个输出） |
| 温度峰值 82 °C | 接近常见 85 °C 限频点，长时间演示建议加散热 |
| 仅验证局域网推流 | 公网 / B 站等未测 |
| 断线重连未做注入测试 | 代码有，`reconn` 一直为 0 |

---
