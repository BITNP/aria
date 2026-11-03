# 端点检测接入 audio.cpp 的 Silero VAD，能量 VAD 降级兜底

## Status

accepted

## Context

本模块此前的端点检测只有能量 VAD（20 ms 帧平均绝对能量阈值）。实测证明它在低音量
语音上会把一句话截断：整段文件的 ASR 识别能听出完整句子，而 VAD 只切出了前 2.5 s，
后续内容被当作静音丢弃——阈值调低则噪声误触发率上升，能量法在真实音量动态范围内不
可靠。

audio.cpp 推理服务（本工作区 TTS/ASR 的同一宿主，端口 8080）已部署 streaming 形态的
Silero VAD（模型 id `silero-vad`，task=vad）。其 HTTP 暴露面实测结论：

- 通用任务路由 `POST /v1/tasks/run`（`{"model": "silero-vad", "request": {"audio_base64": <WAV>}}`）
  返回 `segments: [{start_sample, end_sample, confidence}]`（16 kHz 样本索引），
  30 s 音频单次全程分段仅 6–31 ms，远超实时要求。
- 流式模型固定消费 512 样本（32 ms）窗口：WAV 样本数不是 512 的整数倍时服务直接报
  `Silero VAD streaming chunk must contain exactly 512 samples`，因此请求侧必须补零
  对齐。
- live 路由（`/v1/audio/transcriptions/live`）接受该模型但传输层会把多个 chunk 合并
  投递，破坏 512 样本窗口契约（实测报同样的错），不可用。

## Decision

- 端点检测实现为**轮询式模型分段**（`src/voice_input/silero_vad.py`）：语音缓冲每
  `VOICE_IN_VAD_POLL_MS`（默认 300 ms）把**当前全部缓冲**送一次 `/v1/tasks/run` 重新
  分段（补零到 512 样本整窗），不再依赖逐帧能量判定。句尾判定：缓冲尾音超过最后分段
  末尾达到尾静音时长（复用 `VOICE_IN_VAD_TRAILING_SILENCE_FRAMES`）即出端点；缓冲触
  顶 `VOICE_IN_VAD_MAX_FRAMES` 强制出端点。已沉降的起始静音随分段结果裁剪，检测不到
  语音时缓冲只保留 0.5 s 预滚窗，不随 silence 无限增长。
- 端点音频 = 首分段起点前 0.1 s 至末分段终点后 0.1 s；短于 `VOICE_IN_VAD_MIN_SPEECH_MS`
  （默认 250 ms）的分段视为噪声丢弃。每个端点仍是一个 waycast input 帧（ADR-0011 不
  变），每端点仍只做一次说话人嵌入（ADR-0002 不变）。
- 服务契约（模型名、路由、512 对齐）只在本模块适配层出现，audio.cpp 是外部进程，本
  模块不启动、不管理它。
- **降级**：轮询连续失败 `MAX_POLL_FAILURES`（3）次后**永久切换到能量 VAD**（现有
  `EndpointDetector`，参数不变），日志 warning。选择永久降级而非重试回切：服务恢复的
  收益低（能量 VAD 仍能完成采集），而双状态来回切换会在切换点产生重复/截断端点。
- 配置：`VOICE_IN_VAD_URL`（不设则纯能量 VAD，行为与旧版完全一致）、
  `VOICE_IN_VAD_MODEL`（默认 `silero-vad`）、`VOICE_IN_VAD_POLL_MS`、
  `VOICE_IN_VAD_MIN_SPEECH_MS`。
- 采集循环对端点器的接口统一为异步 `feed(frame, sample) -> SpeechSegment | None`
  （`vad.Endpointer` 协议）；能量路径经 `EnergyEndpointer` 适配，语音时长统计口径不变。

## Consequences

- 低音量语音不再被截断；端点判定延迟增加至约「尾静音 + 轮询间隔」（0.4 s + 0.3 s 默认）。
- 轮询请求体随语音长度线性增长（30 s ≈ 1 MB WAV），本机回环上开销可忽略（模型分段
  本身毫秒级）。
- 每 poll 一次全量分段意味着分段结果可能被后续 poll 修正（起点更准），端点音频以最新
  一次 poll 为准。
- 降级后本进程生命周期内不再使用模型 VAD；测试用注入 httpx transport 的桩服务，真实
  服务验证走 audio.cpp 实机（禁止假模型进库代码）。
