# 树莓娘语音输入模块

本仓库是树莓娘（Raspberry Girl）重构体系中的**语音输入模块**：负责麦克风采集、VAD 端点检测与 ASR 识别，并把识别结果作为一个 waycast 输入帧发送给核心。核心是 [waycast](../waycast/README.md)；本模块是它的一个输入模块（capability.kind = "input"），完全按 waycast 的 wire 契约（ADR-0008）工作，不包含任何核心逻辑。模块与核心是独立的微服务：两者之间只通过网络（WebSocket + wire JSON 帧）通信，本仓库不依赖 waycast 的任何 Python 包。

## 职责边界

- **采集**：持续从默认或指定麦克风采集 16 kHz 单声道 PCM16 音频块。
- **端点检测（VAD）**：识别一段语音的起止（静音分段），把一段语音聚合成一个端点。
- **ASR**：调用 OpenAI 兼容的语音识别服务，把端点音频转成文本。默认走**真流式
  live 路径**（`POST /v1/audio/transcriptions/live`）：语音在途时边采集边上传，
  服务端边解码边回 SSE 增量文本，端点产生后再等最终结果（实测端点后约 40ms 出
  文本，而不是端点后才开始整段上传识别）；流式会话失败、超时或模型不支持时
  自动回退批量转写（ADR-0005）。
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
- 一个 OpenAI 兼容的 ASR 服务（地址 + 模型名 + 可选 API Key）。本机部署为 audio.cpp（`http://127.0.0.1:8080`，模型 `nemotron-3.5-asr-streaming`，与 presenter 的 TTS 共用同一服务；本地服务无需密钥）。
- 说话人识别模型随包分发在 `src/voice_input/models/`（组内导出的 CAM++ ONNX，约 30 MB，已随仓库提交，不读环境变量）；模型缺失或加载失败时模块自动降级为无 ID 模式，其余功能不受影响。

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
VOICE_IN_VAD_URL=http://127.0.0.1:8080
```

可选配置项：

| 配置项 | 用途 |
| --- | --- |
| VOICE_IN_DEVICE | 指定麦克风设备名称；不填用系统默认输入设备 |
| VOICE_IN_VAD_URL | audio.cpp 服务地址（如 `http://127.0.0.1:8080`）；填写后用 Silero VAD 做端点检测（ADR-0003），不填用能量 VAD |
| VOICE_IN_VAD_MODEL | audio.cpp 上的 VAD 模型 id（默认 `silero-vad`） |
| VOICE_IN_VAD_POLL_MS | Silero 分段轮询间隔毫秒（默认 300） |
| VOICE_IN_VAD_MIN_SPEECH_MS | 短于此长度的分段视为噪声丢弃（默认 250） |
| VOICE_IN_VAD_SPEECH_THRESHOLD | VAD 语音判定能量阈值（默认 300；能量 VAD 与降级路径使用） |
| VOICE_IN_VAD_TRAILING_SILENCE_FRAMES | 判定一句结束所需的连续静音帧数（默认 20） |
| VOICE_IN_VAD_MAX_FRAMES | 单段语音最大帧数（默认 1500） |
| VOICE_IN_VAD_PRE_ROLL_FRAMES | 语音起始前保留的前摇帧数（默认 5；能量 VAD 与降级路径使用） |
| VOICE_IN_ASR_LIVE | 真流式 ASR 路径开关（默认开；设 `0` 关闭，全部走批量转写） |
| VOICE_IN_ASR_LIVE_TIMEOUT_MS | 端点后等待流式最终文本的毫秒数（默认 10000；超时按失败回退批量） |
| VOICE_IN_ASR_LIVE_MAX_FAILURES | 连续失败多少次后永久关闭流式路径（默认 3） |
| WAYCAST_TLS_CA | waycast 使用私有 TLS 证书时填写 CA 证书路径 |

## 端点检测（VAD）

端点检测分两层（ADR-0003）：

- **Silero VAD（首选）**：`VOICE_IN_VAD_URL` 指向 audio.cpp 服务时启用。采集缓冲每
  `VOICE_IN_VAD_POLL_MS` 送一次 `POST /v1/tasks/run`（WAV base64，补齐到 512 样本的
  整窗），服务端在全新的 stream 上重放整窗得到语音分段；缓冲尾音超出最后分段末尾达到
  「尾静音时长 + 盲区余量（0.26s）」即判一句结束。盲区余量是源码核实的服务端语义
  （ADR-0004）：streaming 会话的参数在模型装载时固定（语音阈值 0.5 / 最短语音
  250ms / 最短静音 100ms / 尾垫 30ms），**逐请求 options 会被服务端解析但忽略**；
  ≥100ms 的停顿会切开分段，而缓冲末尾仍在进行的语音要超过 250ms 最短语音门限才会
  出现在结果里——这段可见性延迟就是盲区余量覆盖的对象，句中短暂停顿不会被误切，
  低音量语音也不再被能量阈值截断。
- **能量 VAD（兜底）**：默认路径；Silero 服务连续轮询失败后**永久降级**到它（日志
  warning），audio.cpp 挂掉不影响采集链路。恢复 Silero 需重启本模块。

两种路径共用 `VOICE_IN_VAD_TRAILING_SILENCE_FRAMES`（句尾静音）与
`VOICE_IN_VAD_MAX_FRAMES`（单句上限）语义；每个端点仍是一个 waycast input 帧。

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
- **回退**：会话失败、`VOICE_IN_ASR_LIVE_TIMEOUT_MS` 内没等到最终文本、或服务端
  拒绝流式模型时，worker 用同一份端点 PCM 走批量转写，行为与关闭流式完全一致；
  回退前先关闭 live 连接并留 150ms 让服务端释放模型锁，批量请求不会撞
  「model is busy」。「模型被占用」属并发瞬时状态，只降级当次、不计失败；连续
  `VOICE_IN_ASR_LIVE_MAX_FAILURES` 次失败后本路径**永久关闭**（日志 warning），
  重启模块才会重新启用。
- **实现约束**：httpx 无法在同一条 HTTP/1.1 连接上边发边收（发完请求体才开始读
  响应），live 客户端是裸 asyncio 的 chunked 上传 + 增量 SSE 解析；喂音频只入
  缓冲、由独立写任务刷出，网络背压不会卡住采集泵。live 路由语义以 audio.cpp
  `app/server/api.md` 的 live 一节与 `runtime.cpp` 为准：参数全在查询串、请求体
  是裸 PCM、不发终止块直接断连是错误而非说完、请求存续期间服务端一直持有模型
  占用。

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
