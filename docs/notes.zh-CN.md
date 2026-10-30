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
