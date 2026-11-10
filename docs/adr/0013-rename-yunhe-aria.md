# 更名为云和（Aria）

## Status

accepted

## Context

树莓娘体系确立四个关联项目的诗意命名（waycast ADR-0015）：核心 waycast 中文名「道枢」，三个模块仓库取「一模块一首诗」的中英双名。本模块取 **云和 / Aria**：云和出自《周礼》庙堂乐章「云和之琴瑟」，兼取「云合成、和众声」的现代职能义；Aria 即咏叹调，一段呼吸般的语音唱成一句可懂的文本。命名全文见 README「命名来历」。

## Decision

本仓库自 2026-11-10 起使用下列标识符，旧标识符全部废止（不再用于新设计）：

| 层面 | 旧 | 新 |
| --- | --- | --- |
| 仓库 / 目录 | raspberrygirl-voice-input | aria |
| Python 包名与目录 | `voice_input`（src/voice_input） | `aria`（src/aria） |
| pyproject 项目名 | raspberrygirl-voice-input | aria |
| CLI 命令 | `voice-input` | `aria` |
| 环境变量前缀 | `VOICE_IN_` | `ARIA_` |
| wire 模块名（MODULE_NAME、input 帧 source） | `voice_input` | `aria` |

本 ADR 之前的历史文档（docs/adr/ 旧条目、docs/notes.zh-CN.md、git 历史）保持原文不改写，按本表读解旧标识符。

## Consequences

- 部署环境的 .env 需把 `VOICE_IN_*` 键改名为 `ARIA_*`（本机已完成）；`.env.example` 同步。
- wire 模块名变化会进入核心全局记忆与能力目录（Agent 可见的模块名变为 `aria`）；hub 侧无按模块名绑定的令牌，无需核心改动。
- GitHub 远端仓库改名是网页端手工操作，克隆 URL 需另行更新。
- waycast GLOSSARY 与各仓库间交叉引用同步改用新名。
