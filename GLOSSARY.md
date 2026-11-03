# 树莓娘语音输入模块术语表

模块专属术语在本仓库定义（ADR-0001）；核心语义与跨模块契约以 [waycast/GLOSSARY.md](../waycast/GLOSSARY.md) 为准。模块内部与文档、日志统一使用下列术语，不发明替代表述。

## 基础

- **语音输入模块 / voice-input**：本仓库实现的 waycast 输入模块（capability.kind = "input"），负责麦克风采集 + VAD 端点检测 + ASR 识别（+ 说话人识别），把识别结果作为 input 帧发给核心。
- **端点 / endpoint**：VAD 把一段连续语音（含尾随静音与前摇）聚合成的一个完备音频段；一个端点对应一条 input 帧。
- **stable_prefix**：input 帧 metadata 中保持纯 ASR 文本的字段，说话人前缀只进 content，不改写该字段（ADR-0002）。

## ASR（ADR-0005）

- **流式路径 / live 路径**：语音在途时经 audio.cpp live 路由（`/v1/audio/transcriptions/live`）边上传裸 PCM 边收 SSE 增量文本的识别方式，为默认主路径。
- **流式会话 / live session**：一条语音区间对应一条 live 连接：区间打开时开启并回填语音起点音频，端点产生时交给识别 worker 收尾（发终止块等最终文本或断开）。
- **终止块 / terminating chunk**：chunked 编码里的 0 长度块，向服务端声明「说完了」；不发终止块直接断连在服务端语义里是错误而非语音结束。
- **批量转写 / batch 回退**：端点音频整段上传的 `POST /v1/audio/transcriptions` 路径；流式失败、超时或服务端拒绝流式时按端点回退到它，`VOICE_IN_ASR_LIVE=0` 时作为唯一路径。
- **流式路径永久关闭 / live path disabled**：连续 `VOICE_IN_ASR_LIVE_MAX_FAILURES` 次失败（或收到不支持信号）后本进程不再尝试流式，全部走批量；重启恢复。

## 降噪（ADR-0006）

- **降噪 / speech enhancement**：采集与端点检测之间的进程内 GTCRN 流式语音增
  强；VAD、live/批量 ASR 与说话人识别一律消费降噪后音频。`VOICE_IN_DENOISE=0`、
  模型不可用或被旁路时，管道消费原始麦克风音频。
- **增强器 / enhancer**：挂在采集泵上的降噪组件（`denoise.GtcrnEnhancer`），
  输入输出都是 20ms/320 样本帧；接口为 `push_async(block) -> 若干帧`。
- **保持 / hold-back**：增强器把音频向后压住再出帧的固有延迟（模型 512 样本
  ≈ 32ms 加帧对齐缓冲）；样本不丢弃，只是晚些交付。
- **预热 hop / priming hop**：流式前端处理的第一个 256 样本帧，只填充分析缓
  冲不产出样本（sherpa-onnx 语义）。
- **旁路 / bypass**：降噪的降级形态——模型加载失败则整段不降噪（构建期），推
  理连续 `MAX_ENHANCE_FAILURES` 次失败则原始音频直通帧缓冲（运行期，本连接
  内永久，重连或重启恢复）；两者管道行为与关闭开关完全一致。

## 说话人识别（ADR-0002）

- **说话人 ID / speaker ID**：进程内唯一的说话人编号，形如 `spk_1`、`spk_2`（`spk_N` 单调递增）；身份以 `【说话人spk_N】` 前缀随 content 进入 Agent。
- **动态说话人识别 / dynamic speaker identification**：不注册、不持久化、全内存态的 1:N 声纹识别；每个端点一次嵌入、一次匹配；进程重启后从 `spk_1` 重新编号（特性行为，非缺陷）。
- **自称称呼 / self-appellation**：说话人在对话中自报的称呼；本模块只负责给出 ID，ID↔称呼的关联与应答由核心 Agent 经全局记忆维护（模块在 capability 描述中自描述该行为）。
- **识别窗口 / ID window**：1.5 s（24000 采样）的嵌入输入窗；短端点尾补静音到该窗、长端点整段送。
- **运行均值窗口 / running-mean window**：每个 ID 保留的最近 64 个嵌入贡献，取均值作为该说话人的参考向量；并入前与均值余弦 ≥ 0.90 才允许合并（防污染）。
- **降级无 ID 模式 / degraded no-ID mode**：模型加载失败或本次识别异常时，文本不带说话人前缀照常发送。
- **复用 / 铸新 / 歧义**：最近邻余弦 ≥ 0.90 且与次近邻差 > 0.05 复用 ID；全部 < 0.90 铸新 ID；次近邻差 ≤ 0.05 视为歧义，本句不标注也不铸新 ID。
