# 树莓娘语音输入模块

本仓库是树莓娘（Raspberry Girl）重构体系中的**语音输入模块**：负责麦克风采集、VAD 端点检测与 ASR 识别，并把识别结果作为一个 waycast 输入帧发送给核心。核心是 [waycast](../waycast/README.md)；本模块是它的一个输入模块（capability.kind = "input"），完全按 waycast 的 wire 契约（ADR-0008）工作，不包含任何核心逻辑。模块与核心是独立的微服务：两者之间只通过网络（WebSocket + wire JSON 帧）通信，本仓库不依赖 waycast 的任何 Python 包。

## 职责边界

- **采集**：持续从默认或指定麦克风采集 16 kHz 单声道 PCM16 音频块。
- **端点检测（VAD）**：识别一段语音的起止（静音分段），把一段语音聚合成一个端点。
- **ASR**：调用 OpenAI 兼容的语音识别服务，把端点音频转成文本。
- **说话人识别**：对每个端点做一次 CAM++ 声纹嵌入与内存态 1:N 动态匹配（ADR-0002）；命中的话在 content 前加 `【说话人spk_N】` 前缀，`stable_prefix` 保持纯 ASR 文本。
- **发送输入帧**：识别结果作为 waycast input 帧发送给 hub，content 为识别文本（命中说话人时带前缀），metadata 携带来源与端点信息。
- 模块**不参与话权与打断判断**（ADR-0011）：任何抵达 hub 的 input 帧都触发一次 Invocation；模块自身在静音分段处决定何时发送，但不标记 partial/final，也不做"这是部分还是最终"的判断。

## 与 waycast 的关系

- 模块连接 waycast hub 的 /ws/modules，首帧发送 capability 广告（input-only，另声明 status 查询通道），随后通过同一条连接发送 input 帧、接收并应答 status_query。
- 帧格式、capability 结构、连接令牌均来自 waycast wire 协议（waycast/src/waycast/wire.py）；本仓库在 src/voice_input/wire.py 本地实现该客户端侧契约（纯 JSON over WebSocket，无包依赖），不修改协议本身，互操作以 waycast 的实现为权威。
- 本模块属于 ADR-0010 之外的独立输入模块；语音输入与弹幕/IM 输入是两个独立模块，共用一个核心。

## 运行前提

- Python 3.12+ 与 uv（https://docs.astral.sh/uv/ ），Linux 下需要 PortAudio 运行库。
- 已启动且可达的 waycast hub（例如 ws://127.0.0.1:8000 ），以及分配给语音输入的连接令牌。
- 一个 OpenAI 兼容的 ASR 服务（地址 + 模型名 + 可选 API Key）。
- 说话人识别模型随包分发在 `src/voice_input/models/`（组内导出的 CAM++ ONNX，约 30 MB，已随仓库提交，不读环境变量）；模型缺失或加载失败时模块自动降级为无 ID 模式，其余功能不受影响。

## 安装与配置

同步依赖（本仓库不依赖 waycast 包，仅需各自的运行时依赖）：

```sh
uv sync --locked
```

创建 .env：

```dotenv
WAYCAST_HUB_URL=ws://127.0.0.1:8000
WAYCAST_HUB_TOKEN=填写核心分配给语音输入的连接令牌
VOICE_IN_ASR_ENDPOINT=https://asr.example.test/v1
VOICE_IN_ASR_MODEL=填写语音识别模型名称
VOICE_IN_ASR_API_KEY=填写语音识别服务的API密钥
```

可选配置项：

| 配置项 | 用途 |
| --- | --- |
| VOICE_IN_DEVICE | 指定麦克风设备名称；不填用系统默认输入设备 |
| VOICE_IN_VAD_SPEECH_THRESHOLD | VAD 语音判定能量阈值（默认 300） |
| VOICE_IN_VAD_TRAILING_SILENCE_FRAMES | 判定一句结束所需的连续静音帧数（默认 20） |
| VOICE_IN_VAD_MAX_FRAMES | 单段语音最大帧数（默认 1500） |
| VOICE_IN_VAD_PRE_ROLL_FRAMES | 语音起始前保留的前摇帧数（默认 5） |
| WAYCAST_TLS_CA | waycast 使用私有 TLS 证书时填写 CA 证书路径 |

## 启动

先启动 waycast hub，然后运行：

```sh
uv run --locked voice-input
```

说完一句话并稍作停顿，识别文本就会作为输入帧发送给核心。按 Ctrl+C 结束。

## 说话人识别（Speaker ID）

识别在模块进程内**全内存**完成：不做注册、不做 consent、不加密、不持久化（ADR-0002）。每次进程启动后说话人重新从 `spk_1` 编号，配合核心全局记忆（waycast ADR-0005）在自称时建立 ID↔称呼关联。

- 每个 VAD 端点**一次嵌入、一次匹配**：短端点尾补静音到 1.5s 窗，长端点整段送（CAM++ 模型要求 16 kHz / 25ms / 10ms / 80-bin FBank，与模型训练前端一致）。
- 判定口径：端点 speech ≥ 500ms 才判身份；余弦最近邻 ≥ 0.90 且与次近邻差距 > 0.05 复用 ID；全部 < 0.90 立即铸新 ID（`spk_N` 单调递增）；歧义（次近邻差 ≤ 0.05）本句不标注也不铸新 ID；说话人运行均值最多保留最近 64 窗贡献，并入前余弦校验防污染。
- 命中 ID 时 input `content = 【说话人spk_N】 + ASR文本`，`metadata.stable_prefix` 保持纯 ASR 文本。
- 身份仅用于个性化（称呼、偏好），不构成任何鉴权 / 考勤 / 授权依据。
- 模型加载失败 → 降级无 ID 模式：文本照常发送（不带前缀），启动日志告警。
- 模块广告 status 能力（waycast A15/A19），状态工具可查询 `{speaker_enabled, model_revision, known_speaker_count, onnx_provider}`。

## 开发

```sh
uv run --locked pytest
uv run --locked ruff check .
uv run --locked basedpyright
```

测试分层（ADR-0002）：T1 假 embedder 聚类单测默认离线跑；T2 真实模型闭环（同音频同 ID / 异音频异 ID / 短端点补窗），模型缺省时自动跳过；官方 golden fixture parity 为 opt-in（设 `MIC_CAMPP_OFFICIAL_GOLDEN=1` 并提供 fixture 路径后运行）。

## 相关链接

- 本仓库的架构决策与模块专属术语见 [docs/adr/](docs/adr/)（0001 独立微服务项目、0002 说话人 ID）；本模块是独立微服务项目，与 waycast 仅经 wire 契约（ADR-0008）松耦合。
- 核心仓库：[waycast](../waycast/)，契约文档见 ADR-0008（waycast/docs/adr/0008-module-network-protocol.md）、ADR-0011（waycast/docs/adr/0011-agent-decides-turn-taking.md）。
- 旧语音输入实现（只读参考，不复用）：[bitnp-raspberrygirl-vtuber-mic](../bitnp-raspberrygirl-vtuber-mic/)。
