# 硬封顶丢弃钳制在模型读指针：修复越界丢弃导致的静默降级能量 VAD

## Status

accepted

amends ADR-0010 决策 3 的「`max_frames` 硬封顶丢到 `end + pad`，相邻两段严格首尾相接」：
该措辞让丢弃越过模型读指针，是回归 bug 的成因。钳制后硬封顶相邻段最多重叠一个模型
chunk（≤32 ms）；其余边界语义（静音闭合丢到 `end`、不丢未发出语音、不留空洞）不变。

## Context

用户报告回归（2026-11-10）：ADR-0010/0011 落地后「一半语音不再触发 VAD，即使很大声」。
大声必使概率趋近 1.0，阈值解释不了；代码审查 + 最小复现（`ScriptedModel` 式脚本 +
带 shape guard 的假模型 + `max_frames=16` + 连续高概率脚本）定位：

- **触发只看每个 512 样本 chunk 的概率**，进入模型之前的音频流一旦损坏，触发即坏。
- **不变式**：环形缓冲头 `_pcm_start` 不得超过模型读指针 `_chunk_pos`——
  `_take_chunk` 以 `first = _chunk_pos − _pcm_start` 取 chunk，first 为负时 numpy
  负索引切片产生 0 长度/错位 chunk。
- **越界丢弃**：`_close` 在硬封顶闭合时丢到 `end + SPEECH_PAD_SAMPLES`，而硬封顶处
  `end == _chunk_pos`（闭合恰发生在封顶 chunk 消费完时），丢弃目标恒越过读指针一个
  pad；`_discard_before` 的缓冲长度截断（闭合瞬间 `fed − end < 512 < pad`）使实际丢弃
  落到 `fed`，`first` 变负。
- **后果链**：真实 ONNX 模型只接受固定 512 样本窗口（`[1,576]` = 64 上下文 + 512），
  坏 chunk 抛错 → 连续 3 次失败触发 `MAX_VAD_FAILURES` → **该连接永久降级能量 VAD**
  （仅 DEBUG 日志一条 WARNING）。降级后灵敏度换成了能量口径——「很大声也不触发」的
  触发特征变化即源于此；重启进程才恢复。
- **谱系**：基线（ADR-0010 之前）同样丢到 `end + 480`，越界仅在 `fed − end ≥ 480`
  的约 6% 窗口内成立（pad 480 < 512 chunk 的巧合），属罕见潜伏 bug；ADR-0010 将 pad
  增至 2400 后，缓冲长度截断恒生效，硬封顶几乎必然损坏音频流。测试漏检原因：
  `ScriptedModel` 不校验 chunk 长度，0 长度 chunk 照单全收。
- 静音闭合路径不受影响：闭合要求尾随静音 ≥ min_silence（≥2048 样本已被消费），
  `end + pad ≤ _chunk_pos` 恒成立；`_trim_idle_ring` 的目标本身低于读指针。

## Decision

1. **不变式落进 `_discard_before`**：丢弃目标一律钳制 `sample = min(sample, _chunk_pos)`
   ——绝不丢弃模型尚未消费的音频。钳制点选在共享入口而非硬封顶调用点，使所有现有与
   未来的丢弃调用方自动受保护。
2. **测试守卫对齐真实模型**：`ScriptedModel.run_chunks` 拒绝任何非 512 样本的 chunk
   （镜像 ONNX shape guard），坏 chunk 用例会像生产一样触发降级而非静默通过。
3. 硬封顶相邻段的 ≤32 ms 尾部重叠（上一段已发出的尾音被下一段回补重发）判定为无害：
   内容为语音，ASR 相邻段各自独立转写；相对「永久静默降级」的代价可忽略。

## Consequences

- 硬封顶后紧接的语音照常走 Silero 触发（回归用例：`test_hard_cap_close_is_contiguous`
  断言全部 chunk 长 512 且 `degraded is False`）。
- ADR-0010 决策 3 的「严格首尾相接、不重叠」在硬封顶场景放宽为「至多重叠一个模型
  chunk」；静音闭合与 idle trim 语义逐字节不变。
- 降级路径（`MAX_VAD_FAILURES`）保留：模型真的持续故障时仍需能量 VAD 兜底，但
  WARNING 日志自此只在真实模型故障时出现。
- 测试：145 passed；复现脚本（假 shape guard 模型）修复前 3 个 0 长度 chunk、
  degraded=True，修复后 0 坏 chunk、degraded=False、两段完整。
