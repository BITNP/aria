# 说话人 ID：动态声纹识别随 content 前缀进 Agent，全内存态

## Status

accepted

## Context

树莓娘需要知道"谁在说话"，以便个性化应答。旧架构的 CAM++ 说话人识别栈（注册、AEAD 加密模板、consent 策略）随 waycast 重构（ADR-0009）被整体弃用；新的语音输入模块重建为"采集 + VAD + ASR"最小闭环，不含说话人识别。本 ADR 记录重新引入 CAM++ 的方式，并明确否决旧合规栈——新形态（无注册、无持久化）让旧策略失去作用对象，遵守 KISS。

## Decision

- 能力全部在本模块实现；waycast 核心与 wire 协议零改动。
- 身份随 input 的 content 前缀进入 Agent：`content = "【说话人spk_N】" + ASR文本`；`stable_prefix` 等 metadata 保持纯 ASR 文本。
- "说话人自称称呼时建立 ID↔称呼关联并据此应答"的行为指令写入模块 capability description，经 waycast 能力桥注入系统提示词；模块自描述行为，不依赖核心。
- 识别语义为 1:N 动态识别 + unknown 回退；按 VAD 端点一次嵌入、一次匹配（wire 无独立证据通道）。
- 不注册、不 consent、不加密、不持久化：全链路内存态。模块进程内维护 `{ID → 运行均值向量}`；Agent 全局记忆维护 `{ID → 自称称呼}`；进程重启均归零（与 ADR-0005 全局记忆一致）。
- 数值：端点 speech ≥ 500ms 才判身份；短端点尾补静音到 1.5s 窗、长端点整段送（VAD 上限 ≈30s）；余弦最近邻 ≥ 0.90 且与次近邻差距 > 0.05 才复用 ID；并入运行均值前须与均值余弦 ≥ 0.90（防污染）；均值最多保留最近 64 窗贡献；陌生（全部 < 0.90）立即铸新 ID（`spk_N` 单调递增，无缓刑）；歧义（次近邻差 ≤ 0.05）本句不标注也不铸新 ID；< 500ms 无前缀直发。
- 模型：使用组内专门导出的 CAM++ 模型（约十几 MB，公开渠道不可复现），随包分发进仓库（不读模型路径环境变量）；加载失败 → 降级为无 ID 模式 + 日志。
- 可选：声明 status 能力，上报 `{speaker_enabled, model_revision, known_speaker_count, onnx_provider}`。

## Consequences

- 身份是进程内临时概念：系统重启后说话人重新从 `spk_1` 编号，Agent 需重新通过自称建立称呼关联。这是特性行为，不是缺陷。
- 说话人 ID 只用于个性化（称呼、偏好），不构成任何鉴权 / 考勤 / 授权依据。
- 测试三层：T1 假 embedder 聚类单测（默认 CI 离线）；T2 真实模型对官方 golden fixture parity + 同音频同 ID / 异音频异 ID 闭环（默认 CI 离线）；T3 真麦克风 opt-in（同人两句→同 ID、换人→新 ID）。
