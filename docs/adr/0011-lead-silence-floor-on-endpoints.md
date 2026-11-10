# 端点音频前置 150 ms 前导静音底：修复 ASR 吞掉零静音起头句子的首词

## Status

accepted

补齐 ADR-0010 未覆盖的一层：ADR-0010 保证端点音频**含得住**起音（前垫），本决策保证
ASR 前端**认得出**起音（静音底）。滞回、pad、闭合边界策略与 live 对齐不变式全部沿用。

## Context

用户报告句首丢失（2026-11-10）。ADR-0010 落地后实机仍复现（`Can you speak English`
转写为 `You speak English`），用 `VOICE_IN_DUMP_WAV` 落盘同一失败端点后定位：

- **端点音频内容是完整的**：失败样本落盘文件（`ep0001-205426-1376ms.wav`，触发点
  150ms 前垫完整覆盖 "Can you"）用户试听确认整句都在。
- **但 batch 与 live 两条路线都转写为 `You speak English`**——对落盘文件单独 curl
  batch 同样缺 "Can"。丢失不在传输、不在切割，在 ASR 模型（audio.cpp 的
  nemotron-3.5-asr-streaming）的前端。
- **根因实验（同一文件，服务端实测，2026-11-10）**：该文件第一帧帧 MAE=3365，
  零前导静音、从语音中间开始；前置数字静音 50/100/200/400 ms 四个变体，batch 全部
  转写为完整 `Can you speak English`；前置 100 ms 后走 live 路线（burst/paced 两种
  喂法）同样完整。成功样本（实机录音 seg0）天然带 120 ms 前导静音、转写完整。
  结论：该 ASR 对零静音起头的音频吞掉首音节，≥50 ms 前导静音即可完全恢复。
- 失败机制链：弱读功能词（"Can"）概率不过阈值 → VAD 在后续音节才触发（实测触发概率
  贴边 0.500-0.757）→ 前垫虽捞回 "Can you"，但捞回的全是语音、端点音频头部没有任何
  静音 → ASR 前端吞掉首词。ADR-0010 的 attack-latency 切头是相邻但不同的机制。

## Decision

1. **固定前导静音底**：一切端点音频与其 live 快照（`utterance_audio()`）头部前置
   `LEAD_SILENCE_SAMPLES = 2_400`（150 ms）数字静音，常量不开环境变量（与 ADR-0010
   pad 同款决策）；取 150 ms 为实测下限 50 ms 的 3 倍余量。
2. **落点在端点器发射面**（`vad.py` 提供 `with_lead_silence`）：`SileroEndpointer`
   的 `_close`/`utterance_audio` 与 `EnergyEndpointer` 的对应方法统一加底；
   `EndpointDetector` 原始面不加（保持既有语义）。批量上传的是端点音频、live 流的是
   其严格前缀，两处同位置加底后前缀不变式自动保持。
3. `duration_ms`、min_speech 门控、说话人 speech_ms 口径均不含静音底，全部不变。

## Consequences

- 每个端点多 2400 样本（4.8 KB）数字静音与最长 150 ms 的端到端识别延迟（流式路径
  仅回填阶段），可忽略。
- `VOICE_IN_DUMP_WAV` 落盘文件头部恒有 150 ms 静音，属预期，试听以语音起点为准。
- 说话人嵌入输入多 150 ms 静音：无实测危害（成功样本此前天然含 120 ms 前导静音，
  嵌入判定正常）。
- 短弱段流式转写为空的概率预期下降（模型获得安静攻击上下文）。
- 测试：T1 新增端点/live 快照双表面静音底用例（Silero 与能量各一），既有长度断言
  按 +2400 样本更新；实机端对实机失败样本前置静音的 batch/live 恢复实验留档
  /tmp/opencode/lead{50,100,200,400}.wav。
