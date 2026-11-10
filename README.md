# 树莓娘语音输入模块

本仓库是树莓娘（Raspberry Girl）重构体系中的**语音输入模块**：负责麦克风采集、VAD 端点检测与 ASR 识别，并把识别结果作为一个 waycast 输入帧发送给核心。核心是 [waycast](../waycast/README.md)；本模块是它的一个输入模块（capability.kind = "input"），完全按 waycast 的 wire 契约（ADR-0008）工作，不包含任何核心逻辑。模块与核心是独立的微服务：两者之间只通过网络（WebSocket + wire JSON 帧）通信，本仓库不依赖 waycast 的任何 Python 包。

本文是**用户文档**：如何安装、配置并运行本模块，以及各功能的对外行为。内部实现（模型来源与校验和、延迟实测、降级/回退策略细节、测试分层）见**开发者文档** [DEVELOPMENT.md](DEVELOPMENT.md)；术语见 [GLOSSARY.md](GLOSSARY.md)，决策见 [docs/adr/](docs/adr/)。

## 职责边界

- **采集**：持续从默认或指定麦克风采集 16 kHz 单声道 PCM16 音频块。
- **降噪（语音增强）**：采集后先用随包分发的 GTCRN 流式模型逐帧降噪（ADR-0006），
  VAD、ASR 与说话人识别一律消费降噪后的音频；`VOICE_IN_DENOISE=0` 或模型不可用时
  直接使用原始麦克风音频。注意 GTCRN 会压制「不像语音」的声音（提示音、哼唱等），
  非语音音频更难触发输入，属预期行为。
- **端点检测（VAD）**：默认用随包分发的 Silero VAD ONNX 模型进程内逐帧检测语音
  起止，把一段语音聚合成一个端点（ADR-0007）；`VOICE_IN_VAD=energy` 或模型不可用
  时改用能量 VAD。说完一句话并稍作停顿即产生一个端点。
- **ASR**：调用 OpenAI 兼容的语音识别服务，把端点音频转成文本。默认走**真流式
  live 路径**：语音在途时边采集边上传，出文本的时刻由端点检测决定而非识别耗时
  决定；流式会话失败或模型不支持时自动回退批量转写（ADR-0005，回退语义见
  ADR-0008）。
- **说话人识别**：对每个端点做一次声纹嵌入与内存态 1:N 动态匹配（ADR-0002）；
  命中的话在 content 前加 `【说话人spk_N】` 前缀。识别在模块进程内**全内存**完成：
  不做注册、不做 consent、不加密、不持久化；每次进程启动后说话人重新从 `spk_1`
  编号，配合核心全局记忆（waycast ADR-0005）在自称时建立 ID↔称呼关联。身份仅用于
  个性化（称呼、偏好），**不构成任何鉴权 / 考勤 / 授权依据**。
- **发送输入帧**：识别结果作为 waycast input 帧发送给 hub，content 为识别文本（命中说话人时带前缀），metadata 携带来源与端点信息。
- 模块**不参与话权与打断判断**（ADR-0011）：任何抵达 hub 的 input 帧都触发一次 Invocation；模块自身在静音分段处决定何时发送，但不标记 partial/final，也不做"这是部分还是最终"的判断。

模型缺失或加载失败一律自动降级（不降噪 / 能量 VAD / 无说话人 ID / 批量转写），
其余功能不受影响；降级细节见 [DEVELOPMENT.md](DEVELOPMENT.md)。

## 运行前提

- Python 3.12+ 与 uv（https://docs.astral.sh/uv/ ），Linux 下需要 PortAudio 运行库。
- 已启动且可达的 waycast hub（例如 ws://127.0.0.1:8000 ），以及分配给语音输入的连接令牌。
- 一个 OpenAI 兼容的 ASR 服务（地址 + 模型名 + 可选 API Key）。本机部署为 audio.cpp（`http://127.0.0.1:8080`，模型 `nemotron-3.5-asr-streaming`，与 presenter 的 TTS 共用同一服务；本地服务无需密钥）。
- 说话人识别（CAM++）、降噪（GTCRN）与端点检测（Silero VAD）模型均随包分发在 `src/voice_input/models/`，已随仓库提交、不读环境变量；来源与校验和见 [DEVELOPMENT.md](DEVELOPMENT.md)。

## 安装与配置

同步依赖（本仓库不依赖 waycast 包，仅需各自的运行时依赖）：

```sh
uv sync --locked
```

创建 .env（全部键与默认值见 [.env.example](.env.example)，最少填四项）：

```sh
cp .env.example .env
```

```dotenv
WAYCAST_HUB_URL=ws://127.0.0.1:8000
WAYCAST_HUB_TOKEN=填写核心分配给语音输入的连接令牌
VOICE_IN_ASR_ENDPOINT=http://127.0.0.1:8080/v1
VOICE_IN_ASR_MODEL=nemotron-3.5-asr-streaming
VOICE_IN_ASR_API_KEY=本地 audio.cpp 无需密钥，留空即可
```

可选配置项：

| 配置项 | 用途 |
| --- | --- |
| VOICE_IN_DEVICE | 指定麦克风设备名称；不填用系统默认输入设备 |
| VOICE_IN_VAD | 端点检测引擎（默认 `silero`：进程内 Silero VAD ONNX，ADR-0007；设 `energy` 强制能量 VAD） |
| VOICE_IN_VAD_THRESHOLD | Silero 语音判定概率阈值（默认 0.5；负阈值自动取 threshold−0.15） |
| VOICE_IN_VAD_MIN_SPEECH_MS | 短于此长度的语音段视为噪声丢弃（默认 250） |
| VOICE_IN_VAD_SPEECH_THRESHOLD | VAD 语音判定能量阈值（默认 300；能量 VAD 与降级路径使用） |
| VOICE_IN_VAD_TRAILING_SILENCE_FRAMES | 判定一句结束所需的连续静音帧数（默认 20） |
| VOICE_IN_VAD_MAX_FRAMES | 单段语音最大帧数（默认 1500） |
| VOICE_IN_VAD_PRE_ROLL_FRAMES | 语音起始前保留的前摇帧数（默认 5；能量 VAD 与降级路径使用） |
| VOICE_IN_ASR_LIVE | 真流式 ASR 路径开关（默认开；设 `0` 关闭，全部走批量转写） |
| VOICE_IN_ASR_LIVE_TIMEOUT_MS | 端点后等待流式最终文本的毫秒数（默认 10000；超时按失败回退批量） |
| VOICE_IN_ASR_LIVE_MAX_FAILURES | 连续失败多少次后永久关闭流式路径（默认 3） |
| VOICE_IN_DENOISE | GTCRN 降噪开关（默认开；设 `0` 关闭，直接用原始麦克风音频，ADR-0006） |
| VOICE_IN_SPEAKER_THRESHOLD | 说话人复用的余弦相似度阈值（默认 0.31，真实语音工作点，ADR-0009；调高更倾向铸新 ID，调低更倾向合并） |
| VOICE_IN_DUMP_WAV | 调试：填目录则每个端点发给 ASR 的音频额外落盘为该目录下 WAV（16 kHz 单声道，内容与批量上传逐字节一致；默认关闭不落盘） |
| VOICE_IN_LOG_LEVEL | 本模块日志级别（默认 `INFO`；设 `DEBUG` 输出端点、ASR 转写结果、说话人判定分数等调试信息） |
| VOICE_IN_DEPS_LOG_LEVEL | 第三方依赖日志级别（默认 `WARNING`，屏蔽 websockets/httpx/asyncio/onnxruntime 噪音） |
| WAYCAST_TLS_CA | waycast 使用私有 TLS 证书时填写 CA 证书路径 |

## 启动

先启动 waycast hub，然后运行：

```sh
uv run --locked voice-input
```

说完一句话并稍作停顿，识别文本就会作为输入帧发送给核心。按 Ctrl+C 结束。

## 相关链接

- 开发者文档：[DEVELOPMENT.md](DEVELOPMENT.md)（内部实现、模型来源与校验和、测试分层）。
- 本仓库的架构决策与模块专属术语见 [docs/adr/](docs/adr/)（0001 独立微服务项目、0002 说话人 ID、0003/0004 Silero 端点检测〔已被 0007 取代〕、0005 真流式 ASR、0006 进程内 GTCRN 降噪、0007 进程内 Silero VAD ONNX、0008 live 空文本不回退批量〔修正 0005 回退语义〕、0009 说话人复用阈值 0.31〔修正 0002 阈值〕）；本模块是独立微服务项目，与 waycast 仅经 wire 契约（ADR-0008）松耦合。
- 核心仓库：[waycast](../waycast/)，契约文档见 ADR-0008（waycast/docs/adr/0008-module-network-protocol.md）、ADR-0011（waycast/docs/adr/0011-agent-decides-turn-taking.md）。
- 旧语音输入实现（只读参考，不复用）：[bitnp-raspberrygirl-vtuber-mic](../bitnp-raspberrygirl-vtuber-mic/)。
