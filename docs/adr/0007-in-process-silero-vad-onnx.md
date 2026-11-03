# 端点检测改为进程内 Silero VAD ONNX：删除 audio.cpp 轮询路径

## Status

accepted

supersedes: ADR-0003 与 ADR-0004 全部（Silero 端点检测的传输与服务适配决策）。
两条旧 ADR 保留为历史记录：其中对 audio.cpp streaming 会话语义的源码核实结论
（逐请求 reset、options 无效、参数烘焙）仍然成立，只是不再被本模块消费。

## Context

ADR-0003/0004 的端点检测是轮询式全缓冲重分段：每 300 ms 把整个语音缓冲 base64 成
WAV 送 audio.cpp `/v1/tasks/run`，靠盲区余量 4160 样本掩盖服务端「逐请求 reset +
固定 250ms 最短语音门限」造成的新语音可见性延迟。该实现的问题（用户决策动机，
2026-11-03）：

- audio.cpp 对该模型的封装质量差：请求侧重放整窗、状态逐请求清零（ADR-0004 源码
  核实），句尾判定延迟被推高到约 0.96 s，且请求体随语音长度线性增长；
- VAD 驻留 audio.cpp 的 GPU 推理面，与该服务真正需要 GPU 的 ASR/TTS 模型抢资源，
  而 Silero VAD 本身是 ~2.3 MB 的极轻模型，进程内 CPU 推理实测 RTF 0.011（含线程
  切换；12 核 x86 单线程，官方 60 s 测试语音），GPU 毫无必要。

事实（在线核实 + 本机实测，2026-11-03）：

- 官方仓库 snakers4/silero-vad（LICENSE=MIT，MIT (c) 2020-present Silero Team），
  README 明确 ONNX 路径只需 `onnxruntime>=1.16.1 + numpy`，torch 只是官方加载器
  的依赖——宿主自行实现 I/O 即完全不需要 torch。本仓库两依赖均已是正式依赖
  （ADR-0002/0006）。
- 随包模型 `models/silero_vad.onnx`：官方 `src/silero_vad/data/silero_vad.onnx`
  （v5/opset16 多采样率导出，`load_silero_vad(onnx=True)` 的默认件），2,327,524
  字节，sha256 `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3`；
  git blob `80c5592ef1f4c9ede3e357bbd02eb863358a6a9d` 与 master 树对象一致。
- 图签名（本机 onnxruntime 实测）：输入 `input [1,576]` f32（64 样本 context + 512
  样本 chunk，context 由宿主维护）、`state [2,1,128]` f32、`sr` int64 标量；输出
  `[1,1]` 概率 + 同形状态回接。图内维度为动态维，静态形状（[2,batch,128]）在
  `run_chunks` 运行时守卫。
- 后处理语义（官方 `utils_vad.get_speech_timestamps_from_probs` 源码逐行核实）：
  滞回阈值 0.5 / −0.15，`min_silence_duration_ms=100`，`min_speech_duration_ms=250`
  （严格大于），`speech_pad_ms=30`；prob ∈ (neg, threshold) 既不清除也不推进静音
  计时。用该官方纯 Python 函数做 oracle 校验复刻前端的概率流，在其 `tests/data/
  test.wav` 上切出 19 段——复刻正确性成立（合成谐波音不被识别：Silero 只认真实
  语音，测试按 GTCRN 先例对真实语音素材 opt-in）。

## Decision

1. **进程内 Silero VAD**（`src/voice_input/silero_vad.py` 原地重写）：
   `SileroVadModel` 按官方 `OnnxWrapper` 语义直跑随包模型（context + 状态逐 chunk
   回接、单线程 SessionOptions、加载时严格图校验）；`SileroEndpointer` 实现
   `Endpointer` 协议，官方滞回状态机流式化（丢弃仅服务于 max_speech_duration 的
   分支，单句上限沿用 `VOICE_IN_VAD_MAX_FRAMES`），推理走 `asyncio.to_thread`。
2. **默认开**：`VOICE_IN_VAD`（默认 `silero`，可设 `energy` 强制能量 VAD）。
   模型加载失败 → 构建期落能量 VAD；推理连续 `MAX_VAD_FAILURES`（3）次失败 →
   本连接永久降级能量 VAD（与 ADR-0003 的永久降级决策同构：来回切换会在切换点
   产生重复/截断端点）。
3. **判定参数**：句尾静音取官方 `min_silence_duration_ms=100ms`（经用户确认，弃
   用旧路径的 0.4 s 尾静音 + 盲区组合）；threshold 唯一开放为
   `VOICE_IN_VAD_THRESHOLD`（默认 0.5，负阈值自动 −0.15），min_silence/pad 留常量
   （官方默认）；`VOICE_IN_VAD_MIN_SPEECH_MS` 与能量路径参数不变。端点音频 = 语音
   起止各外扩 30 ms；`duration_ms` 为语音跨度（start..end，不含垫），作为说话人
   识别的 speech_ms 口径不变。
4. **live ASR 对齐**（ADR-0005 协议不变）：`utterance_open` 在概率触发时即为真，
   `utterance_audio()` 返回 `[start−pad, fed)`，与端点将发出的音频严格前缀一致；
   候选被 min_speech 判为噪声时 open 回落 False，live 跟踪器照旧中止会话。
5. **删除 HTTP 路径**：`SileroVadClient`、`VOICE_IN_VAD_URL` / `VOICE_IN_VAD_MODEL`
   / `VOICE_IN_VAD_POLL_MS` 与盲区常量全部移除；本模块不再向 audio.cpp 发任何
   VAD 请求，audio.cpp 侧无需改动（该模型可从其部署中移除以释放 GPU）。

## Consequences

- 句尾判定延迟 ≈ 100 ms 静音 + chunk 量化（32 ms），比旧路径快约 0.9 s；代价是
  min_silence 从「尾静音 0.4 s + 盲区」缩到 100 ms，句中 ≥100 ms 的真停顿会拆成两
  个端点——由核心按 ADR-0011 处理多帧输入，不视为缺陷（用户确认的取舍）。
- 实测（官方 60 s test.wav，12 核 x86 单线程）：端到端 RTF 0.011（模型本体
  0.005），19 个端点、语音合计 47.5 s；进程内开销可忽略，树莓派级设备按余量推算
  同样无压力，上线后照常观察日志即可。
- VAD 与降噪（ADR-0006）、说话人（ADR-0002）同生命周期：随每条 hub 连接重建，
  跨连接丢一段边界语音。
- 测试三层（ADR-0002 同风格）：T1 脚本化概率模型（tests/ 内 test-only 占位）锁定
  滞回语义、音频切片/垫、live 对齐、封顶、降级与图校验；T2 真模型对确定性噪声
  不出端点；官方语音 wav 经 `VOICE_IN_SILERO_WAV` opt-in（实测通过）。
- 概率驱动的能量兜底参数（`VOICE_IN_VAD_SPEECH_THRESHOLD` 等）语义不变；两条路径
  仍共用 `Endpointer` 协议，`_collect` 无感知。
