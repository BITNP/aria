# 树莓娘语音输入模块 · 开发者文档

本文面向**开发本模块的人**：与 waycast 的协议实现、各管道环节的内部机制（模型
来源与校验和、判定语义、延迟实测、降级/回退策略）、测试分层与开发命令。安装、
配置与运行请看用户文档 [README.md](README.md)；术语见 [GLOSSARY.md](GLOSSARY.md)，
决策见 [docs/adr/](docs/adr/)，逐次开发记录见 [docs/notes.zh-CN.md](docs/notes.zh-CN.md)。

## 与 waycast 的关系

- 模块连接 waycast hub 的 /ws/modules，首帧发送 capability 广告（input-only，另声明 status 查询通道），随后通过同一条连接发送 input 帧、接收并应答 status_query。
- 帧格式、capability 结构、连接令牌均来自 waycast wire 协议（waycast/src/waycast/wire.py）；本仓库在 src/aria/wire.py 本地实现该客户端侧契约（纯 JSON over WebSocket，无包依赖），不修改协议本身，互操作以 waycast 的实现为权威。
- 本模块属于 ADR-0010 之外的独立输入模块；语音输入与弹幕/IM 输入是两个独立模块，共用一个核心。

## 降噪（GTCRN，ADR-0006）

采集与端点检测之间是**进程内流式降噪**（`ARIA_DENOISE`，默认开）：每个
20ms 麦克风块经 GTCRN 流式模型增强后仍按 20ms 帧交付，VAD、live/批量 ASR 与
说话人识别消费的都是降噪后音频。

- **模型**：GTCRN（ICASSP 2024，48.2K 参数 / 33.0 MMACs/s，MIT）的官方流式
  ONNX 导出，随包分发 `src/aria/models/gtcrn_simple.onnx`（535,638
  字节，sha256 `e77603ac0c23dac3227dd2d7135b3a585cbee2679048aecfa886657d3ae1b534`；
  来源 [Xiaobin-Rong/gtcrn](https://github.com/Xiaobin-Rong/gtcrn) 的
  `stream/onnx_models/`，经
  [sherpa-onnx speech-enhancement-models](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models)
  注入元数据）。16 kHz / 512 窗 / 256 hop / sqrt-Hann，宿主侧流式
  STFT/ISTFT 按 sherpa-onnx 语义复刻。
- **延迟**：模型保持 512 样本（32ms）加帧对齐缓冲，端到端约 40–70ms；出文本
  时刻仍由端点检测决定（ADR-0005）。实测 RTF 0.18（12 核 x86，单线程）。
- **降级与旁路**：模型缺失/加载失败 → 不降噪照常运行；推理连续 3 次失败 →
  本连接永久旁路原始音频（日志 warning，重连或重启恢复）；两者的管道行为与
  `ARIA_DENOISE=0` 完全一致。
- **音频语义**：GTCRN 会压制「不像语音」的声音（提示音、哼唱等），非语音音频
  更难触发输入，属预期行为。
- 对照实验：官方带噪 wav 可通过 `ARIA_GTCRN_WAV=<16kHz 单声道 wav>` 运行
  opt-in 测试（默认跳过）。

## 端点检测（VAD，ADR-0007）

端点检测默认是**进程内流式 Silero VAD**（`ARIA_VAD`，默认开）：官方 ONNX 流式
导出（[snakers4/silero-vad](https://github.com/snakers4/silero-vad)，MIT）直接跑在
onnxruntime 上，不再经 audio.cpp 的 HTTP 任务路由（取代 ADR-0003/0004 的轮询式设计）。

- **模型**：随包分发 `src/aria/models/silero_vad.onnx`（v5/opset16 流式导出，
  2,327,524 字节，sha256 `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3`，
  git blob `80c5592e…` 与官方仓库 master 树对象核对一致）。每 512 样本（32 ms）一个
  chunk 推理：宿主维护 64 样本 context 与 `[2,1,128]` 状态回接，语义与官方
  `OnnxWrapper` 一致；推理走 `asyncio.to_thread`，护住采集泵。
- **判定语义**（官方 `utils_vad` 后处理的流式化）：概率 ≥ `ARIA_VAD_THRESHOLD`
  （默认 0.5）触发语音起点；只有低于负阈值（threshold−0.15）的概率才启动静音计时，
  连续静音达 100 ms（官方 `min_silence_duration_ms`）即出端点；两阈值之间的概率保持
  句段开放——句中换气不会被切。语音段短于 `ARIA_VAD_MIN_SPEECH_MS`（默认 250）
  视为噪声丢弃；单句仍受 `ARIA_VAD_MAX_FRAMES` 封顶。端点音频 = 150 ms 前导静音
  底（防 ASR 前端吞零静音起头的首音节，ADR-0011）+ 语音起止各外扩 150 ms（前垫覆盖
  模型触发滞后，ADR-0010）。
- **延迟**：句尾判定 ≈ 尾静音 100 ms + chunk 量化（32 ms），取代旧路径的「尾静音 +
  盲区 0.26 s + 轮询 0.3 s ≈ 0.96 s」——盲区与轮询都是 audio.cpp 逐请求 reset 服务
  逻辑的补丁，进程内连续状态后不再存在。实测（12 核 x86 单线程，官方 60s 测试语音）
  端到端 RTF 0.011，切出 19 个端点、语音合计 47.5 s。
- **降级**：模型缺失/加载失败 → 直接用能量 VAD（行为与 `ARIA_VAD=energy` 完全
  一致）；推理连续 3 次失败 → 本连接**永久降级**能量 VAD（日志 warning，重连或重启
  恢复）。能量路径参数（`ARIA_VAD_SPEECH_THRESHOLD` /
  `ARIA_VAD_TRAILING_SILENCE_FRAMES` / `ARIA_VAD_PRE_ROLL_FRAMES`）不变。
- **对照实验**：官方语音 wav 可通过 `ARIA_SILERO_WAV=<16kHz 单声道 wav>` 运行
  opt-in 测试（默认跳过）。

## ASR：真流式 live 路径与批量回退（ADR-0005）

批量路由（`POST /v1/audio/transcriptions`）必须等一句话说完、整段 WAV 上传后才
开始解码；live 路由（`POST /v1/audio/transcriptions/live`）允许在同一条连接上边
上传裸 PCM 边收 SSE 增量文本（`transcript.text.delta`），最终文本
（`transcript.text.done`）在客户端发出终止块（0 长度 chunk）后到达——实测
nemotron-3.5-asr-streaming 在终止块后约 40ms 出结果，即**出文本的时刻由端点检测
决定，而不是由识别耗时决定**。

- **会话生命周期**（`live_asr.LiveAsrTracker`，跑在采集泵里）：VAD 报告语音区间
  打开时开启一条会话，把端点缓冲里已有的语音（含前摇，与批量 PCM 同源）回填进
  去，此后每帧增量喂入；端点产生时把会话摘下交给识别 worker，worker 发终止块并
  等最终文本；VAD 把语音当噪声丢弃时直接断开连接（服务端随之释放模型占用）。
- **回退**：会话失败、`ARIA_ASR_LIVE_TIMEOUT_MS` 内没等到最终文本、或服务端
  拒绝流式模型时，worker 用同一份端点PCM 走批量转写，行为与关闭流式完全一致；
  回退前先关闭本句 live 连接并留 150ms 让服务端释放模型锁。「模型被占用」属并发
  瞬时状态，只降级当次、不计失败；连续 `ARIA_ASR_LIVE_MAX_FAILURES` 次失败后
  本路径**永久关闭**（日志 warning），重启模块才会重新启用。
- **空文本不回退**：live 协议成功但最终文本为空，是模型对过短/不可辨音频（实测
  <0.5s）的正常输出；同一模型对同一段音频的批量结果必然也是空（实测同为 `''`），
  回退只会白等。且 audio.cpp 的 ASR 模型**单实例串行**——请求存续期间任何别的
  live 会话（如下句刚开的流式会话）都会把批量请求挂起直到超时（实测 25s+ 无响
  应），因此批量兜底仅限流式路径真失败，且批量 HTTP 超时收敛到 8s，避免撞锁时
  拖垮端点队列。
- **实现约束**：httpx 无法在同一条 HTTP/1.1 连接上边发边收（发完请求体才开始读
  响应），live 客户端是裸 asyncio 的 chunked 上传 + 增量 SSE 解析；喂音频只入
  缓冲、由独立写任务刷出，网络背压不会卡住采集泵。live 路由语义以 audio.cpp
  `app/server/api.md` 的 live 一节与 `runtime.cpp` 为准：参数全在查询串、请求体
  是裸 PCM、不发终止块直接断连是错误而非说完、请求存续期间服务端一直持有模型
  占用。

## 说话人识别（Speaker ID）

识别在模块进程内**全内存**完成：不做注册、不做 consent、不加密、不持久化（ADR-0002）。每次进程启动后说话人重新从 `spk_1` 编号，配合核心全局记忆（waycast ADR-0005）在自称时建立 ID↔称呼关联。

- 每个 VAD 端点**一次嵌入、一次匹配**：短端点尾补静音到 1.5s 窗，长端点整段送（CAM++ 模型要求 16 kHz / 25ms / 10ms / 80-bin FBank，与模型训练前端一致）。
- 判定口径：端点 speech ≥ 500ms 才判身份；余弦最近邻 ≥ 复用阈值（`ARIA_SPEAKER_THRESHOLD`，默认 0.31，真实语音工作点，ADR-0009）且与次近邻差距 > 0.05 复用 ID；全部低于阈值立即铸新 ID（`spk_N` 单调递增）；歧义（次近邻差 ≤ 0.05）本句不标注也不铸新 ID；说话人运行均值最多保留最近 64 窗贡献，并入前余弦校验防污染。判定分数（best/second）以 DEBUG 日志输出，便于校准阈值。
- 命中 ID 时 input `content = 【说话人spk_N】 + ASR文本`，`metadata.stable_prefix` 保持纯 ASR 文本。
- 身份仅用于个性化（称呼、偏好），不构成任何鉴权 / 考勤 / 授权依据。
- 模型加载失败 → 降级无 ID 模式：文本照常发送（不带前缀），启动日志告警。
- 模型：CAM++ ONNX（组内导出，约 30 MB），随包分发 `src/aria/models/`，不读环境变量。
- 模块广告 status 能力（waycast A15/A19），状态工具可查询 `{speaker_enabled, model_revision, known_speaker_count, onnx_provider, denoise_enabled, denoise_model_revision}`。

## 开发

```sh
uv run --locked pytest
uv run --locked ruff check .
uv run --locked basedpyright
```

测试分层（ADR-0002）：T1 假 embedder 聚类单测默认离线跑；T2 真实模型闭环（同音频同 ID / 异音频异 ID / 短端点补窗），模型缺省时自动跳过；官方 golden fixture parity 为 opt-in（设 `MIC_CAMPP_OFFICIAL_GOLDEN=1` 并提供 fixture 路径后运行）。
