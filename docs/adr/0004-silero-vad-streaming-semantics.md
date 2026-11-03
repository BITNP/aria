# Silero VAD 服务端语义的源码核实修正：streaming 会话参数固定，盲区余量按 250ms 门限放大

## Status

accepted

supersedes: ADR-0003 中「请求 options 将服务端 `min_speech_duration_ms` 降到
100ms」与「盲区余量 0.2s」两处表述；ADR-0003 的其余决策（轮询式全缓冲重分段、
512 对齐补零、永久降级、配置键）不变。

## Context

ADR-0003 落地时，服务端语义是从响应行为反推的，其中「请求 `options.
min_speech_duration_ms=100` 能降低服务端最短语音门限」这一条经 audio.cpp
源码（v0.9.3，`src/models/silero_vad/{runtime,session}.cpp`、
`app/streaming/streaming.cpp`、`app/server/runtime.cpp`）逐行核实后被证伪：

- `/v1/tasks/run` 对 `mode=streaming` 的模型走 `run_streaming_model`：每个请求
  在会话上 `start_stream()`——基类实现**忽略 request（含 options）只做
  `reset()`**；随后 `process_audio_chunk`/`finalize` 一律使用会话构造时从
  模型装载配置（server.json 的 `session_options`）烘焙出的 `default_config_`。
  逐请求 options 被 JSON 层解析后没有任何消费者。实测证实：200ms 语音切片
  不带尾静音时，带 `min_speech_duration_ms=50` 与不带的响应同为空。
- 当前部署（server.json `silero-vad`，无 `session_options`）的生效参数即
  `SileroVADConfig` 默认值：`threshold=0.5`（负阈值 0.35）、
  `min_speech_duration_ms=250`、`min_silence_duration_ms=100`、
  `speech_pad_ms=30`。
- streaming 解码路径（`build_stream_event`/`finalize_stream`）与离线
  decode 语义不同：已闭合分段（停顿 ≥100ms 触发 SpeechEnd）**不做最短语音
  过滤**，任意短促噪声只要后续静音足够都会成为分段；250ms 最短语音门限只作用
  于 `finalize` 时仍在进行的尾部分段——尾部分段要 `(已喂样本 − 起点 + 30ms
  语音垫) > 4000 样本` 才出现。这正是「复说语音对轮询不可见」窗口的真实宽度：
  约 220–252ms（3520 样本 + 512 样本起点量化），大于 ADR-0003 假设的
  「100ms + 余量」。

由此，原盲区余量 3200 样本（0.2s）不足：句尾判定阈值 = 尾静音 + 盲区 = 9600
样本时，停顿约 0.38–0.6s（与轮询相位相关）的复说仍可能在未变得可见前被切断。

## Decision

- 请求体不再携带 `options`：对 streaming 会话是静默无效项，留着只会误导。
  需要改服务端参数时唯一的口子是 audio.cpp 的 server.json `session_options`
  （外部进程，本模块不配置它，只按默认值适配）。
- `BLIND_WINDOW_SAMPLES` 取 4160 样本（0.26s）：覆盖源码推出的可见性延迟
  上界 3520 + 量化 512，保证任何短于尾静音（默认 0.4s）的停顿在最早可能触发
  句尾判定的那次轮询上必然已可见；停顿超过「尾静音 + 盲区」（默认约 0.66s）
  判句尾，属预期语义。
- 其余（补零到 512 整窗、每请求 fresh reset、永久降级、配置键）经源码核实
  成立，不改。

## Consequences

- 端点判定延迟上界 ≈ 尾静音 + 盲区 + 轮询间隔（0.4 + 0.26 + 0.3 ≈ 0.96s，
  默认配置）。
- 本模块对服务端参数的假设收敛为一个数：`min_speech_duration_ms=250`
  （默认部署）。若部署方在 server.json 显式调小它，盲区只会偏保守（切得更晚），
  不会重新引入句中截断。
- 「已闭合分段不受最短语音过滤」意味着短噪声只要后接 ≥100ms 静音就会进分段；
  本模块的 `VOICE_IN_VAD_MIN_SPEECH_MS`（首末分段跨度）本地过滤承担噪声丢弃，
  语义不变。
