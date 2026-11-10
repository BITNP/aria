# 语音垫增至 150 ms 并落实尾垫与闭合边界策略：修复首字被切

## Status

accepted

amends ADR-0007 决策 3/4 中「端点音频 = 语音起止各外扩 30 ms（官方默认）」的取值与
边界实现。滞回、min_silence、min_speech、threshold 默认值与 live 对齐不变式全部沿用。

## Context

用户报告语音输入经常丢失句首第一个字（2026-11-10），定位为首字起点被端点音频切掉。

事实（生产代码路径 + 随包模型 + 真实语音实测，2026-11-10）：

- Silero 的概率滞后于真实发声起点（attack latency）：1 s 静音 + 真实语音（工作区
  ref_audio.wav 转 16 kHz），正常音量下 `prob >= 0.5` 晚于起点 56 ms，衰减至 10%
  音量（模拟小声起音）晚 88 ms（概率在滞回带 0.45/0.50 徘徊两个 chunk）；端点音频
  只从触发 chunk 向前回补 30 ms，首音节被切 26 ms / 58 ms。汉语声母的关键声学线索
  （送气、VOT）约 30-80 ms，切掉后 ASR 对第一个字丢失或误识。
- 官方实现与本文复刻语义一致（snakers4/silero-vad master `utils_vad.py`，查阅
  2026-11-10）：`get_speech_timestamps` 的 `speech_start` 取触发 chunk 起点，
  `speech_pad_ms` 默认 30 ms 外扩；官方文档同时明确 threshold 应「tune for each
  dataset」——`VOICE_IN_VAD_THRESHOLD=0.35` 可作应急缓解（实测攻击延迟降至 24 ms），
  但默认参数应自身站得住。
- 空闲环形缓冲仅 `RING_SAMPLES=1600`（100 ms）：attack 超过「回补 + 环」的触达范围
  时，起音音频已被丢弃，加大 pad 也救不回。
- 代码与 ADR-0007 分歧（本次一并修复，按「代码与 ADR 不符时修复代码」原则）：
  ADR-0007 与模块 docstring 均写「语音起止各外扩 30 ms」，但 `_close` 只发到 `end`，
  尾垫从未发出、仅被 `end + pad` 丢弃。pad 增至 150 ms 后，若保持「闭合即丢
  end + pad」，min_silence 仅 100 ms 的即时续说会被吞掉至多 50 ms 起音——边界策略
  必须与 pad 取值同时定。
- GTCRN 降噪（ADR-0006）与采集路径实测无丢样（Gtcrn 仅固定 16 ms 流水线延迟），
  不在本决策范围。

## Decision

1. **前垫增至 150 ms**：`SPEECH_PAD_SAMPLES` 480 → 2400（覆盖实测最大 attack 88 ms
   并留余量），仍为常量（沿用 ADR-0007「pad 留常量、不开环境变量」的决策）；
   `RING_SAMPLES` 1600 → 4800（300 ms = 2×pad），保证触发时回补触达范围内确有音频。
2. **尾垫按 ADR-0007 字面落实**：端点音频 = `[start − pad, min(end + pad, fed))`，
   尾部不足按缓冲截断。live 对齐不变式不变：`utterance_audio()` 仍为
   `[start − pad, fed)`，是端点音频的严格前缀。
3. **闭合边界丢弃策略**：静音闭合丢到 `end`——尾随静音留在环形缓冲内，下一句的
   前垫可回补至此（min_silence 100 ms 后即时续说不再吃起音，重叠区只可能是静音）；
   `max_frames` 硬封顶丢到 `end + pad`——已发出的音频恰好被丢弃，相邻两段严格首尾
   相接，不重叠语音、不留空洞。
4. `duration_ms` 口径不变（语音跨度，不含垫），说话人识别 speech_ms 门控不受影响；
   滞回/min_silence/min_speech 语义与能量回退路径不变。

## Consequences

- 端点音频与 live 回填平均多约 120 ms 静音（起音前垫），对 ASR 无害；每连接缓冲
  上界约 +3.2 KB。
- 首字起点在实测（含 10% 音量小声起音）内完整保留：端点音频起点先于真实语音起点
  约 62 ms（150 − 88）；实测脚本（1 s 静音 + ref_audio 素材）切头量从 26/58 ms
  归零，T2 真实语音切分照常通过。
- 静音闭合的尾垫区 `[end, end + pad)` 会被下一段的回补再次覆盖——内容必为静音
  （闭合前提是 ≥100 ms 低于负阈值），ASR 无感。
- 测试：T1 新增前垫回补（起点 = 触发 chunk − pad）、闭合后续说起点回补至上一段
  `end`、封顶切分首尾相接三个用例；`test_endpoint_after_speech_then_silence` 与
  `test_idle_buffer_stays_bounded` 按新常量更新。
