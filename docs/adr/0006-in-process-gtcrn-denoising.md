# 进程内 GTCRN 流式降噪：置于端点检测之前，失败自动旁路

## Status

accepted

## Context

噪声环境直接抬高 VAD 误触发、拉低 ASR 准确率、污染 CAM++ 声纹。采集后的第一
个管道位置（enhancer hook，`main._collect`）能同时惠及 VAD、live/批量 ASR 与
说话人识别，是本模块唯一合理的降噪插入点。

候选路线（均在线核实，2026-11-03）：

- **audio.cpp 服务端**：本仓库依赖的 audio.cpp 服务已内置 GTCRN
  （`builtin_audio_utils` 家族，task `s2s`，含 `gtcrn_streaming`），但任务路
  由是整段请求、每次运行模型状态清零（有状态流式降噪会逐请求引入边界伪影），
  `/v1/tasks/stream` 的音频输入只接受服务器本地路径而非 PCM 流（其
  `api.md`），加上逐帧 50 req/s 的 base64 JSON 与模型 busy 锁——逐帧降噪经
  HTTP 不成立。Silero VAD 那种 300ms 轮询整段重放（ADR-0003/0004）不适用
  于有状态降噪器。
- **sherpa-onnx 依赖**：其 `OnlineSpeechDenoiser` 对 GTCRN 有逐帧流式实现，
  但为单个 523KB 模型引入整套 wheel 与第二份 onnxruntime，与本仓库最小依赖
  风格不符；其 C++ 源码（`online-speech-denoiser-gtcrn-impl.h` +
  `online-speech-denoiser-stft-impl.h`）作为语义参考保留。
- **进程内 onnxruntime 直跑**：`onnxruntime`/`numpy` 已是正式依赖，进程内
  ONNX 模型与随包分发已有 CAM++ 先例（ADR-0002）。

模型事实（实测 + 官方来源核实）：GTCRN 为 ICASSP 2024 超轻量语音增强模型
（48.2K 参数、33.0 MMACs/s，MIT License，官方仓库 Xiaobin-Rong/gtcrn）。随包
模型 `models/gtcrn_simple.onnx`（535,638 字节，sha256
`e77603ac0c23dac3227dd2d7135b3a585cbee2679048aecfa886657d3ae1b534`）是官方
`stream/onnx_models/` 流式导出经 sherpa-onnx 注入元数据的版本（release
`speech-enhancement-models`）。图签名（本机 onnxruntime 实测）：输入
`mix [1,257,1,2]` 复数谱帧 + `conv/tra/inter` 三个状态缓存，输出对应
`enh` 与 `*_out` 缓存——纯前馈逐帧推理，STFT/ISTFT 在宿主侧；元数据固定
16 kHz / n_fft 512 / hop 256 / sqrt-Hann。流式语义按 sherpa 源码复刻：首个
hop 只预热分析缓冲不出样本，此后每 hop 出 256 样本（WOLA）。

## Decision

1. **进程内 GTCRN**（`src/voice_input/denoise.py`）：onnxruntime CPU 直跑
   随包模型；`GtcrnEnhancer` 实现 enhancer hook 协议
   `push_async(block) -> tuple[bytes, ...]`，宿主侧实现流式 STFT/ISTFT +
   overlap-add 与缓存回接，前端用 numpy 精确复刻 sherpa 语义（`rfft`/`irfft`
   与其手写 DFT 等价，同一 sqrt-Hann 窗）。推理经 `asyncio.to_thread` 离
   开事件循环，慢核不阻塞采集泵。
2. **帧契约**：输入输出均为 20ms/320 样本帧；hop 256 与帧 320 的公倍数
   1280 使稳态每 80ms「4 块进 4 块出」，内部缓冲有界；进程退出时残尾
   （< 40ms）丢弃。输出 int16 量化先四舍五入再夹紧（1.0 → 32767，不回绕）。
3. **延迟预算**：模型保持（hold-back）512 样本（32ms）+ hop/帧对齐缓冲，
   端到端约 40–70ms。出文本时刻仍由端点检测决定（ADR-0005），且 Silero 端点
   本身由 ≥300ms 轮询 + 尾静音窗主导，该延迟被既有端点延迟掩盖。
4. **降级与旁路**（与 live 路径 ADR-0005、VAD 永久降级 ADR-0003 同构）：
   模型加载失败 → 不降噪（`build_enhancer` 返回 None，行为与无此功能版本完全
   一致，warning 日志）；单 hop 推理失败 → 该 hop 原始样本直通、不计中断；
   连续 `MAX_ENHANCE_FAILURES`（3）次 → 本连接**永久旁路**（warning，重连或
   重启恢复）。`VOICE_IN_DENOISE=0` 整体关闭，默认开。
5. **消费面**：VAD、live/批量 ASR、说话人识别一律消费降噪后音频；
   `metadata.stable_prefix` 等帧语义不变。status 能力新增 `denoise_enabled`
   与 `denoise_model_revision` 两键。
6. 加载时严格校验图签名与元数据（照 CamppEmbedder 先例），不符即视为加载
   失败降级。

## Consequences

- 实测（12 核 x86 开发机，60s 合成噪声）：纯计算 RTF 0.178（3.6ms/帧），含
  线程切换 0.218（4.4ms/帧），远低于 20ms 帧预算；低功耗设备（树莓派级）
  按官方 RTF 0.07 的 3–8 倍外推仍在预算内，上线后仍需复测（`VOICE_IN_DENOISE=0`
  是即时兜底）。
- GTCRN 压制「不像语音」的声音（实测纯 440Hz 正弦被压至 1.2% RMS，语音谐波
  串保留 91%）：提示音、哼唱等非语音音频将难以再触发输入——对以语音为输入
  的本模块属预期收益，但意味着能量 VAD 兜底路径的能量更多来自真人声。
- 降噪器状态（hop 缓冲 + 三个模型缓存）随每条 hub 连接重建（与 speaker
  识别同生命周期），跨连接只丢一帧边界。
- 对称 sqrt-Hann WOLA 有 ≤18 LSB 的固有幅度纹波（官方导出与 sherpa 推理用
  同一窗，属模型侧固有），恒等会话测试按 ≤64 LSB 容差锁定。
- 测试三层（ADR-0002 同风格）：T1 恒等 ONNX 会话（tests/ 内 test-only 占位）
  锁定帧契约、样本守恒、WOLA 重构与旁路语义；T2 真模型（缺模型自动跳过）断言
  平稳噪声衰减至 ≤15%（实测 2%）与语音类信号保留 ≥60%（实测 91%）；官方带噪
  wav 对照经 `VOICE_IN_GTCRN_WAV` opt-in。
