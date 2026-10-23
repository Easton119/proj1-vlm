# proj1-vlm — RK3588 端侧多模态视觉问答

在 RK3588（6 TOPS NPU）上部署 Qwen3-VL-2B，实现"摄像头实时画面 + 自然语言提问 → 文本回答"。

## 环境
- 硬件：RK3588 (aarch64)，USB UVC 摄像头
- 系统：Ubuntu 20.04 (glibc 2.31)
- 依赖：OpenCV、librknnrt、librkllmrt、gcc/g++、cmake

## 模型（不进 git，需自行放置）
放到 `models/`：
- `qwen3-vl_vision_rk3588.rknn`     视觉编码器（fp16）
- `master_w8a8_rk3588.rkllm`        语言主干（W8A8）
来源：自行用 rkllm-toolkit 1.3.1 + rknn-toolkit2 从 Qwen3-VL-2B 转换。（模型名字可以自定）

## 依赖库（不进 git）
放到 `lib/`：
- `librknnrt.so`、`librkllmrt.so`

## 编译
mkdir -p build && cd build
cmake ..
make -j4