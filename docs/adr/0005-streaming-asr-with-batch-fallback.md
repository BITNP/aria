# 真流式 ASR 前置到语音在途：live 路由为主路径，批量转写为回退

## Status

accepted

## Context

批量 ASR（`POST /v1/audio/transcriptions`）的延迟结构是「端点检测延迟 + 整段
上传 + 整段解码」：识别只能在话说完、WAV 组装完并提交之后才开始，实测本地
audio.cpp 上识别本身约 260ms（RTF≈0.06），叠加排队与上传后「话说完→出文本」
约 0.3s 起步。而这个耗时本可以与「说话」重叠——`nemotron-3.5-asr-streaming`
在 audio.cpp 上以 `mode: "streaming"` 配置，服务提供专门的 live 路由
（`POST /v1/audio/transcriptions/live`，契约见 audio.cpp
`app/server/api.md` 的 live 一节，语义经 `app/server/runtime.cpp` 的
`handle_transcription_live` 源码核实）：请求体是 `Transfer-Encoding: chunked`
的裸 PCM，响应是同连接上的 SSE 增量事件；客户端发终止块（0 长度 chunk）表示
说完，服务端冲完解码回 `transcript.text.done`。实测（q_story.wav，实时喂入）：
增量 delta 在语音进行中持续流出（最后一个字在说出后约 90ms 内出现），done 在
终止块后约 40ms 到达——即出文本的时刻由端点检测决定，不再由识别耗时决定。

约束与事实（均经源码或实测核实）：

- live 请求存续期间服务端一直持有该模型的占用锁，**断开即释放**；不发终止块
  直接断连是错误（防截断误判为说完），stall 超过 idle 超时（30s）同理，均以
  SSE `error` 事件呈现。模型占用冲突（`model ... is busy`）同样发生在 SSE
  流内（响应头 200 已发出后才在流内获取锁），不是非 200 状态码。
- live 路由只接受查询串参数 + 裸 PCM 体；非 `mode=streaming` 模型在开流时以
  400 拒绝（消息含 `mode=streaming`）。
- **httpx 无法胜任该客户端**：HTTP/1.1 下 `client.stream()` 生成器请求体要
  全部发完才开始读响应（实测 delta 全部堆积在音频发完之后一次性到达），
  双向流式必须绕开 httpx。

## Decision

1. **新增 live 流式路径为主路径**（`src/voice_input/live_asr.py`，ADR 编号见
   仓库 ADR 序列）：`LiveTranscriptStream` 用裸 asyncio
   （`open_connection` + 手工 chunked 框定 + 增量 chunk 解码 + SSE 按行解析）
   实现单连接双工；`LiveAsrTracker` 挂接 VAD 的语音区间（`Endpointer` 协议
   为此新增 `utterance_open` / `utterance_audio()`，两个实现同源暴露端点
   缓冲）——区间打开时开一条会话并回填缓冲中的语音起点音频（含前摇，与批量
   PCM 完全同源），此后逐帧增量喂入；端点产生时把会话随队列项交给识别 worker，
   worker 发终止块等最终文本。
2. **回退语义**：会话失败（连接断、SSE `error`、`VOICE_IN_ASR_LIVE_TIMEOUT_MS`
   超时）或服务端以 400 拒绝流式模型时，识别 worker 用**同一份端点 PCM**走
   批量转写；回退前 live 连接必已关闭（并额外等 150ms 让服务端完成会话清理），
   批量请求不会撞上模型占用。连续 `VOICE_IN_ASR_LIVE_MAX_FAILURES`（默认 3）次
   失败**永久关闭**本路径（与 VAD 的永久降级同构）；「model is busy」是并发
   瞬时状态，只降级当次、不计失败。`VOICE_IN_ASR_LIVE=0` 整体关闭。
3. **采集泵不受网络影响**：`feed()` 只入缓冲并唤醒独立写任务，网络背压、
   模型加载都不阻塞音频帧循环；写失败记在会话上，由 `finish()` 统一暴露。
4. 说话人识别、端点→input 帧的既有语义不变：live 只替换「端点音频→文本」
   这一步的传输方式，文本仍以端点为粒度一帧一发（ADR-0011），
   `metadata.stable_prefix` 用最终文本。

## Consequences

- 「话说完→出文本」的识别段延迟从约 260ms 降到实测 40ms（端点→done）；且
  增量文本在语音在途即可供展示类消费者使用（当前无消费者，保留在会话对象上）。
- 每条语音区间会占用 ASR 模型锁直至 `finish()`/断开：端点→finish 的窗口极短，
  且模块串行处理队列，实测无并发冲突；即使撞上 busy 也只是当次回退批量。
- 双工 HTTP 客户端是本仓库自持有的裸 asyncio 代码（无法用 httpx 复用），协议
  细节（chunked 框定、SSE 行解析、终止块）由 `tests/test_live_asr.py` 的本地
  脚本化 live 服务器覆盖（真实 socket、真实线协议）；跨实现回归以 audio.cpp
  的 api.md 为权威。
- VAD 丢弃噪声区间时 live 连接被直接掐断（不发终止块），这是刻意为之：服务端
  按「未说完」报错并释放模型，模块不消费该结果。
