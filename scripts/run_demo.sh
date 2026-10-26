#!/bin/bash
# run_demo.sh —— proj1-vlm 启动脚本
#
# 用法:
#   bash scripts/run_demo.sh                  # 只用 VLM，打字提问
#   bash scripts/run_demo.sh --tts            # VLM + 播报（打字问，语音答）
#   bash scripts/run_demo.sh --voice          # VLM + 语音提问（只有文字回答）
#   bash scripts/run_demo.sh --voice --tts    # 完整闭环
#   bash scripts/run_demo.sh --push           # 加 RTMP 推流（默认 640x480@15fps）
#
# 环境变量覆盖：
#   SHERPA_ROOT KWS_DIR KEYWORDS_FILE ASR_DIR VAD_MODEL MIC_DEVICE
#   LLM_MODEL VISION_MODEL TTS_DIR TTS_DEVICE TTS_THREADS

set -euo pipefail

# ---- 路径 ----
PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$PROJ_DIR/build"
APP="$BUILD_DIR/demo"

SHERPA_ROOT="${SHERPA_ROOT:-/home/topeet/Qyj/sherpa-onnx}"

# 模型
LLM_MODEL="${LLM_MODEL:-$PROJ_DIR/models/master_w8a8_rk3588.rkllm}"
VISION_MODEL="${VISION_MODEL:-$PROJ_DIR/models/qwen3-vl_vision_rk3588.rknn}"

# 语音输入（只在 --voice 时才检查/使用）
KWS_DIR="${KWS_DIR:-/home/topeet/Qyj/kws_model_v2/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20}"
KEYWORDS_FILE="${KEYWORDS_FILE:-/home/topeet/Qyj/kws_model/keywords.txt}"
ASR_DIR="${ASR_DIR:-/home/topeet/Qyj/voice/models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17}"
VAD_MODEL="${VAD_MODEL:-/home/topeet/Qyj/voice/models/silero_vad.onnx}"
MIC_DEVICE="${MIC_DEVICE:-plughw:2,0}"

# 语音输出（只在 --tts 时才检查/使用）
TTS_DIR="${TTS_DIR:-$PROJ_DIR/models/vits-melo-tts-zh_en}"
TTS_DEVICE="${TTS_DEVICE:-plughw:1,0}"
TTS_THREADS="${TTS_THREADS:-4}"

# 推流（只在 --push 时才使用）
RTMP_URL="${RTMP_URL:-rtmp://192.168.137.1:1935/live/cam}"
PUSH_FPS="${PUSH_FPS:-15}"
PUSH_BITRATE="${PUSH_BITRATE:-1500000}"
CAM_W="${CAM_W:-640}"
CAM_H="${CAM_H:-480}"

# ---- 参数：--voice / --tts 由脚本消费，其余原样透传给 demo ----
WITH_VOICE=0
WITH_TTS=0
WITH_PUSH=0
EXTRA=()
for arg in "$@"; do
  case "$arg" in
    -v|--voice) WITH_VOICE=1 ;;
    -t|--tts)   WITH_TTS=1 ;;
    -p|--push)  WITH_PUSH=1 ;;
    -h|--help)  head -11 "$0"; exit 0 ;;
    *) EXTRA+=("$arg") ;;
  esac
done

require_file() {
  if [ ! -e "$1" ]; then
    echo "[run_demo] 缺少文件: $1" >&2
    exit 1
  fi
}

require_file "$APP"
require_file "$LLM_MODEL"
require_file "$VISION_MODEL"

# rknn/rkllm 的 .so 在项目 lib/，sherpa 的 .so 在外部
export LD_LIBRARY_PATH="$PROJ_DIR/lib:$SHERPA_ROOT/lib:${LD_LIBRARY_PATH:-}"

ARGS=(
  "--llm=$LLM_MODEL"
  "--vision=$VISION_MODEL"
)

# ---- 语音输入 ----
if [ "$WITH_VOICE" -eq 1 ]; then
  require_file "$KEYWORDS_FILE"
  require_file "$KWS_DIR/tokens.txt"
  require_file "$KWS_DIR/encoder-epoch-13-avg-2-chunk-16-left-64.onnx"
  require_file "$KWS_DIR/decoder-epoch-13-avg-2-chunk-16-left-64.onnx"
  require_file "$KWS_DIR/joiner-epoch-13-avg-2-chunk-16-left-64.onnx"
  require_file "$ASR_DIR/model.int8.onnx"
  require_file "$ASR_DIR/tokens.txt"
  require_file "$VAD_MODEL"

  ARGS+=(
    "--voice"
    "--device=$MIC_DEVICE"
    "--keywords-file=$KEYWORDS_FILE"
    "--kws-tokens=$KWS_DIR/tokens.txt"
    "--kws-encoder=$KWS_DIR/encoder-epoch-13-avg-2-chunk-16-left-64.onnx"
    "--kws-decoder=$KWS_DIR/decoder-epoch-13-avg-2-chunk-16-left-64.onnx"
    "--kws-joiner=$KWS_DIR/joiner-epoch-13-avg-2-chunk-16-left-64.onnx"
    "--asr-model=$ASR_DIR/model.int8.onnx"
    "--asr-tokens=$ASR_DIR/tokens.txt"
    "--vad-model=$VAD_MODEL"
  )
  echo "[run_demo] 语音输入: 开（麦克风 $MIC_DEVICE）"
else
  echo "[run_demo] 语音输入: 关（用键盘提问）"
fi

# ---- 语音输出 ----
if [ "$WITH_TTS" -eq 1 ]; then
  require_file "$TTS_DIR/model.onnx"
  require_file "$TTS_DIR/lexicon.txt"
  require_file "$TTS_DIR/tokens.txt"

  ARGS+=(
    "--tts"
    "--tts-model=$TTS_DIR/model.onnx"
    "--tts-lexicon=$TTS_DIR/lexicon.txt"
    "--tts-tokens=$TTS_DIR/tokens.txt"
    "--tts-device=$TTS_DEVICE"
    "--tts-threads=$TTS_THREADS"
  )
  echo "[run_demo] 语音播报: 开（$TTS_DIR -> $TTS_DEVICE）"
else
  echo "[run_demo] 语音播报: 关"
fi

# ---- 推流 ----
if [ "$WITH_PUSH" -eq 1 ]; then
  ARGS+=(
    "--push=$RTMP_URL"
    "--push-fps=$PUSH_FPS"
    "--push-bitrate=$PUSH_BITRATE"
    "--width=$CAM_W"
    "--height=$CAM_H"
  )
  echo "[run_demo] RTMP 推流: 开（$RTMP_URL, ${CAM_W}x${CAM_H}@${PUSH_FPS}fps）"
else
  echo "[run_demo] RTMP 推流: 关"
fi

cd "$BUILD_DIR"
exec "$APP" "${ARGS[@]}" "${EXTRA[@]}"