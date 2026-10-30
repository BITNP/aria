# 独立微服务项目：voice-input 自持文档与决策

## Status

accepted

## Context

本仓库是树莓娘语音输入模块，作为微服务与 waycast 核心松耦合：仅通过网络 + wire 契约（waycast ADR-0008）交互，不依赖 waycast 的任何 Python 包。旧约定（waycast ADR-0009）曾规定"领域术语、架构决策与各仓库工作规则的唯一维护位置都在 waycast 仓库"——这只适合核心单仓的集中式运营；在模块即独立项目的微服务形态下，模块自身的决策与术语属于本仓库，继续放在 waycast 只会造成跨仓库同步负担与事实源错位。

## Decision

- 本模块仓库是独立项目，自有 AGENTS.md / README / docs/adr/ 与模块术语；模块自身的架构决策以本仓库 `docs/adr/` 为准（如 0002 说话人 ID）。
- 与 waycast 的耦合面只有 wire 契约：帧信封、载荷模型、capability 语义（waycast ADR-0008 / `wire.py` / `protocols.py`）。waycast 仍是核心语义（Agent、Invocation、全局记忆等）与跨模块契约的唯一权威。
- 模块专属术语（说话人 ID、动态说话人识别、自称称呼等）在本仓库定义，不塞进 waycast GLOSSARY。

## Consequences

- 跨仓库决策若同时影响 core 与模块，仍需回到 waycast ADR；纯模块决策不再需要 waycast 放行。
- 旧约定中"唯一维护位置在 waycast"的表述已同步修订：工作区根 `AGENTS.md` / `CONTEXT.md`、waycast `ADR-0009`。
