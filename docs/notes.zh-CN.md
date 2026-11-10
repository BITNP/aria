# 树莓娘语音输入模块开发记录

## 2026-10-30 仓库建立与首版实现

- 创建独立仓库 raspberrygirl-voice-input，作为树莓娘重构体系的语音输入模块。
- 明确模块形态：waycast 输入模块（capability.kind = "input"），按 ADR-0008 契约工作。
- 职责边界：采集 + VAD 端点检测 + ASR 识别 -> 发送 waycast input 帧；不参与话权/打断（ADR-0011）。
- 实现 config / capture / vad / asr / hub_client / main 源码模块与对应测试。
- 验证：pytest 23 用例通过，ruff check 通过，basedpyright strict 0 错误。
- 修正 README 配置表：移除不存在的 VOICE_IN_ASR_ENDPOINT_INCLUDES_VAD，补全 VAD 配置项。
- 移除对 waycast / openai / numpy 的包依赖：模块与核心按微服务边界仅通过网络通信，wire 契约在 src/voice_input/wire.py 本地实现（纯 JSON over WebSocket）。
- 验证更新：pytest 31 用例通过（新增 wire 协议测试 8 个），ruff check 通过，basedpyright strict 0 错误。

## 2026-10-30 说话人识别（ADR-0002）

- 按 ADR-0002 实现全内存态动态说话人识别：CAM++ ONNX 嵌入（模型从旧 mic 仓库复制进 `src/voice_input/models/`，约 30 MB，随包分发、不读环境变量）+ 受约束的 3D-Speaker FBank 前端（`fbank.py`）+ 1:N 余弦最近邻聚类与 `spk_N` 铸新/复用/歧义判定（`speaker.py`）。
- 数值落地：speech ≥ 500ms 才判身份；短端点补静音到 1.5s 窗；最近邻 ≥ 0.90 且与次近邻差 > 0.05 复用；全部 < 0.90 铸新；歧义不标注不铸新；并入均值前余弦 ≥ 0.90 防污染；均值保留最近 64 窗贡献。
- 管道接入：`_send_one` 命中说话人时 content 前缀 `【说话人spk_N】`，`metadata.stable_prefix` 保持纯 ASR 文本；嵌入异常时文本不带前缀照常发送；模型加载失败降级无 ID 模式。
- status 能力（waycast A15/A19）：hub_client 广告 `status.enabled`，监听 status_query 并以 `{speaker_enabled, model_revision, known_speaker_count, onnx_provider}` 应答 status_result。
- 依赖：新增 numpy、onnxruntime（正式依赖，uv.lock 锁定）；补 onnxruntime 最小类型桩。
- 测试三层：T1 假 embedder 聚类单测（离线默认跑）；T2 真实模型闭环（同音频同 ID / 异音频异 ID / 短端点补窗，模型缺省跳过）；官方 golden fixture parity 为 opt-in（`MIC_CAMPP_OFFICIAL_GOLDEN=1`）。
- 验证：pytest 69 用例通过（1 个 opt-in 跳过），ruff check 通过，basedpyright strict 0 错误；真实模型嵌入 192 维、单位范数、同音频余弦 1.0、440/660Hz 余弦 ≈ 0.897（< 0.90 铸新）。

## 2026-11-03 进程内 GTCRN 流式降噪（ADR-0006）

- 选型（在线核实）：audio.cpp 虽内置 GTCRN（builtin_audio_utils/s2s）但任务路
  为整段请求、状态逐请求清零且无 PCM 流式会话，逐帧不可行；sherpa-onnx
  OnlineSpeechDenoiser 依赖过重；选进程内 onnxruntime 直跑官方流式导出
  （Xiaobin-Rong/gtcrn，MIT；模型为 sherpa-onnx release
  `speech-enhancement-models/gtcrn_simple.onnx`，535,638 字节，sha256
  e77603ac0c23dac3227dd2d7135b3a585cbee2679048aecfa886657d3ae1b534，与官方
  checksum.txt 一致，入库 `src/voice_input/models/`）。
- 实现：`denoise.py` 宿主侧流式 STFT/ISTFT（512 窗/256 hop/sqrt-Hann，
  rfft/irfft 与 sherpa `online-speech-denoiser-stft-impl.h` 语义等价，首 hop
  预热不出样本）+ 三缓存逐帧回接 + 320 样本帧对齐输出（LCM 1280，稳态 4 进
  4 出）；推理走 `asyncio.to_thread` 护采集泵；图签名与元数据加载时严格校验。
- 降级语义：加载失败不降噪（build 返回 None）；单 hop 失败原始直通；连续 3 次
  永久旁路本连接；`VOICE_IN_DENOISE=0` 整体关闭（默认开）；status 新增
  denoise_enabled/denoise_model_revision。
- 实测（12 核 x86，单线程 ONNX CPU）：纯计算 RTF 0.178（3.55ms/帧）、含线程
  切换 0.218（4.35ms/帧）；官方带噪 wav 上非语音窗 RMS 显著衰减、语音窗保留。
  T2 断言：平稳噪声衰减至 ≤15%（实测 2%）、语音谐波串保留 ≥60%（实测 91%）；
  纯正弦被压至 1.2%（GTCRN 对非语音的预期行为，已写入 README）。
- 恒等会话测试：对称 sqrt-Hann WOLA 固有纹波实测 ≤18 LSB（模型导出同窗），
  测试容差 64 LSB；样本守恒 =（hops−1 预热）×256 下取整到 320 帧。
- 验证：pytest 117 通过（2 个 opt-in 跳过），ruff check 通过，basedpyright
  strict 0 错误（typings/onnxruntime 桩补 get_modelmeta）。

## 2026-11-03 端点检测改进程内 Silero VAD ONNX（ADR-0007，取代 0003/0004）

- 动机（用户决策）：audio.cpp 的 Silero 封装逐请求 reset、全缓冲重放、句尾判定
  ~0.96s，且 VAD 占用其 GPU 面；官方模型仅 2.3MB，进程内 CPU 足够。
- 选型（在线核实）：snakers4/silero-vad master（MIT）；README 确认 ONNX 路径只需
  onnxruntime+numpy（torch 仅为官方加载器依赖，不引入）。模型取官方默认导出
  src/silero_vad/data/silero_vad.onnx（v5/opset16，2,327,524 字节，sha256
  1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3；git blob
  80c5592ef1f4c9ede3e357bbd02eb863358a6a9d 与 master 树对象一致，入库
  src/voice_input/models/）。
- 图签名实测：input [1,576]（64 context+512 chunk，宿主维护 context）、state
  [2,1,128] 回接、sr int64 标量；输出概率 [1,1]+状态。动态维图 → 运行时形状守卫。
  后处理按官方 utils_vad 流式化：滞回 0.5/0.35、min_silence 100ms、pad 150ms
  （ADR-0010，官方默认 30ms 的前垫加强）、
  min_speech 沿用 250ms（严格大于）；官方纯 Python 后处理做 oracle 在官方
  tests/data/test.wav 上验证复刻前端（19 段）。
- 实现：silero_vad.py 原地重写（SileroVadModel + SileroEndpointer，
  asyncio.to_thread）；VOICE_IN_VAD=silero|energy（默认 silero）、
  VOICE_IN_VAD_THRESHOLD（默认 0.5）；删 SileroVadClient 与
  VOICE_IN_VAD_URL/MODEL/POLL_MS 与盲区/轮询逻辑；加载失败落能量、连续 3 次推理
  失败本连接永久降级；句尾判定 ~0.4s → ~0.13s。
- 实测：端到端 RTF 0.011（模型本体 0.005；60s 官方语音，12 核 x86 单线程）；
  19 端点/47.5s 语音；T1 脚本化概率模型 16 例锁滞回/切片/live 对齐/封顶/降级/
  校验；opt-in VOICE_IN_SILERO_WAV 通过。
- 验证：pytest 133 通过（3 opt-in 跳过）、ruff 通过、basedpyright strict 0 错误
  （typings/onnxruntime 桩补 SessionOptions）。
