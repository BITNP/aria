# 说话人复用阈值从 0.90 修正为 0.31：CAM++ 真实语音工作点

## Status

accepted

amends: ADR-0002 的判定口径数值（余弦最近邻 ≥ 0.90 复用 ID）。0.90 被实测否定：
该值对 CAM++ embedding 的同人跨句分数不可达，导致同一说话人每句都铸新 ID。
ADR-0002 的其余决策（全内存动态 ID、前缀标注、防污染、无缓刑铸新等）不变，
保留为历史记录。

## Context

真实麦克风闭环测试（2026-11-06）：同一人对麦克风说 4 句，`spk_1`→`spk_2`→
`spk_3`→`spk_4` 每句铸新。离线实测（随包 campp.onnx，9 s 同一说话人连续语音，
1.5 s 窗 pairwise cosine）：

| 窗口对 | cosine |
| --- | --- |
| 1.5 s 窗 vs 1.5 s 窗（同人异句） | 0.47 – 0.60 |
| 全段 9 s vs 1.5 s 窗 | 0.70 – 0.84 |
| 0.5 s 短句（零填充）vs 1.5 s 窗 | 0.36 – 0.78 |

同人异句的最高分 0.84 也够不到 0.90 —— `_decide` 必然每句 `_mint`。这与模型
分数分布一致：官方同架构 CAM++（`iic/speech_campplus_sv_zh-cn_16k-common`，
192 维）model card 的使用示例阈值即 `thr=0.31`，并注明"阈值越高，判定为同一人
的条件越严格"（https://www.modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common ，
查阅 2026-11-06）。随包模型为组内导出（ADR-0002），但训练分布与分数尺度同源，
官方工作点可直接采用为默认值。

## Decision

1. **默认复用阈值 0.90 → 0.31**（`speaker.DEFAULT_REUSE_THRESHOLD`）；
   `SpeakerIdentifier` 接受 `reuse_threshold` 构造参数，配置键
   `VOICE_IN_SPEAKER_THRESHOLD`（config 校验 0<v<1）。
2. 防污染并入（`_enroll`）使用同一阈值：并入均值前与均值余弦 < 阈值则不并入
   （行为不变，阈值随配置走）。
3. `_decide` 输出 DEBUG 日志（best/second 分数、阈值、决策），校准不再盲猜。
4. T2 真实模型测试的"异音频→铸新"用例改为**显式传入严格阈值 0.95**：440/660 Hz
   纯音对 cosine≈0.897 不是语音声纹，不能用来测默认工作点语义；新增 opt-in 真实
   语音回归（`VOICE_IN_CAMPP_SPEECH_WAV`）：同人两段 1.5 s 不相交窗在默认阈值下
   必须共享 ID。

## Consequences

- 同一说话人跨句稳定同 ID；歧义带宽（次近邻差 ≤ 0.05 不标注）语义不变。
- 阈值进入用户可调面：多人场景若出现过合并，上调阈值（如 0.4–0.5）比铸新更省
  ——日志里有分数，可数据化校准。
- 旧行为不回归测试：T1 单测全部按 0.31 边界重排（120° 均布铸新、cos 0.353 复用、
  cos 0.296 铸新、显式 0.95 严格阈值用例）。
