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
