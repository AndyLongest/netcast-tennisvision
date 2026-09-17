const byId = (id) => document.getElementById(id);
const DEMO_VIDEO = '../assets/demo/demo.mp4';
const DEMO_REPORT = '../assets/demo/scene3d.json';
const MAX_MISSING_POLLS = 12;
const state = {
  mode: null, report: null, delivered: [], nextEvent: 0, activeRally: null,
  animationFrame: 0, flashTimer: 0, sessionId: null, eventCursor: 0,
  pollTimer: 0, missingPolls: 0, playbackUrl: '', playbackConnecting: false, nextPlaybackRetryAt: 0,
  routeCounters: {ingested: null, processed: null, relayUpdated: null}, routePulseAt: {}, routePulseTimers: {},
  pendingLiveEvents: [], latestLivePayload: null, liveSyncTimer: 0,
  calibration: {file: null, objectUrl: '', points: [], frameReady: false},
};

function rememberLiveSession(sessionId) {
  state.sessionId = sessionId;
  state.missingPolls = 0;
  if (sessionId) sessionStorage.setItem('netcastLiveSession', sessionId);
}

function restoreIdleLab(message = '') {
  clearTimeout(state.pollTimer);
  cancelAnimationFrame(state.animationFrame);
  closeLivePlayback();
  state.sessionId = null;
  state.mode = null;
  state.missingPolls = 0;
  sessionStorage.removeItem('netcastLiveSession');
  history.replaceState(null, '', window.location.pathname);
  resetTimeline();
  resetRouteFlow();
  setRoutePhase('idle');
  setRunning(false, '待开始');
  byId('sourceClock').textContent = '00:00.000';
  byId('resultClock').textContent = '00:00.000';
  byId('videoHudClock').textContent = '00:00.000';
  byId('sourceName').textContent = '尚未选择视频';
  byId('videoPlaceholder').hidden = false;
  byId('analysisEmpty').hidden = false;
  byId('analysisEmpty').querySelector('strong').textContent = '分析画面尚未开始';
  byId('analysisEmpty').querySelector('span').textContent = '落点只会在确认后出现，不会提前剧透';
  byId('playButton').disabled = true;
  byId('restartButton').disabled = true;
  byId('measuredBadge').classList.remove('live');
  byId('measuredBadge').textContent = '尚未开始实测';
  window.scrollTo({top: 0, left: 0, behavior: 'instant'});
  if (message) toast(message);
}

function formatClock(seconds) {
  const value = Math.max(0, Number(seconds) || 0);
  const minutes = Math.floor(value / 60);
  const wholeSeconds = Math.floor(value % 60);
  const millis = Math.floor((value - Math.floor(value)) * 1000);
  return `${String(minutes).padStart(2, '0')}:${String(wholeSeconds).padStart(2, '0')}.${String(millis).padStart(3, '0')}`;
}
function toast(message) {
  const node = byId('labToast');
  node.textContent = message;
  node.classList.add('show');
  clearTimeout(node.timer);
  node.timer = setTimeout(() => node.classList.remove('show'), 3200);
}
function eventTouchdownTime(event) {
  if (Number.isFinite(Number(event.t))) return Number(event.t);
  const fps = Number(state.report?.fps) || 30;
  if (Number.isFinite(Number(event.touchdown_frame_f))) return Number(event.touchdown_frame_f) / fps;
  return Number(event.frame / fps);
}
function baselineDecisionTime(event) {
  const fps = Number(state.report?.fps) || 30;
  const touchdownFrame = Number.isFinite(Number(event.touchdown_frame_f)) ? Number(event.touchdown_frame_f) : Number(event.frame);
  const reportedDecision = Number.isFinite(Number(event.decision_frame)) ? Number(event.decision_frame) : touchdownFrame;
  return Math.max(reportedDecision, touchdownFrame + 10) / fps;
}
function orderedBaselineEvents(report) {
  const bounces = Array.isArray(report?.bounces) ? report.bounces : [];
  const netHits = Array.isArray(report?.net_hits) ? report.net_hits.map((event) => ({...event, zone: '下网', net_hit: true})) : [];
  return [...bounces, ...netHits].sort((a, b) => baselineDecisionTime(a) - baselineDecisionTime(b));
}
function resetTimeline() {
  state.delivered = []; state.nextEvent = 0; state.activeRally = null; state.eventCursor = 0;
  state.pendingLiveEvents = []; state.latestLivePayload = null;
  clearTimeout(state.liveSyncTimer); state.liveSyncTimer = 0;
  byId('eventFeed').innerHTML = '<li class="empty-feed">等待首个落点确认…</li>';
  byId('landingCount').textContent = '0';
  byId('landingDetail').textContent = '当前回合 0 个';
  byId('latencyValue').textContent = '—';
  byId('latencyDetail').textContent = '等待真实链路';
  drawCourt();
}
function setVisualMode(mode) {
  state.mode = mode;
  byId('liveVideo').hidden = false;
  byId('liveFrame').hidden = true;
  byId('videoPlaceholder').hidden = true;
  byId('playButton').disabled = mode !== 'baseline';
  byId('restartButton').disabled = mode !== 'baseline';
  byId('measuredBadge').classList.toggle('live', mode === 'live');
  byId('measuredBadge').textContent = mode === 'live' ? '端到端实测中' : '离线基准 · 非链路实测';
}

function closeLivePlayback() {
  clearTimeout(state.liveSyncTimer); state.liveSyncTimer = 0;
  state.playbackConnecting = false;
  state.playbackUrl = '';
  state.nextPlaybackRetryAt = 0;
  const video = byId('liveVideo');
  video.onplaying = null;
  video.onerror = null;
  video.onstalled = null;
  video.pause();
  video.srcObject = null;
  video.removeAttribute('src');
  video.load();
}

function liveEventDecisionTime(event, payload) {
  const fps = Math.max(1, Number(payload?.fps) || 30);
  const frame = Number.isFinite(Number(event.decision_frame))
    ? Number(event.decision_frame)
    : Number(event.touchdown_frame_f ?? event.frame);
  return Math.max(0, frame / fps);
}

function displayedLiveSourceTime(payload) {
  const video = byId('liveVideo');
  if (video.hidden || video.readyState < 2 || video.paused) return null;
  const sourceTime = Math.max(0, Number(payload?.source_time) || 0);
  if (video.seekable?.length) {
    const edge = Number(video.seekable.end(video.seekable.length - 1));
    const current = Number(video.currentTime) || 0;
    if (Number.isFinite(edge) && edge >= current) {
      return Math.max(0, sourceTime - Math.min(60, edge - current));
    }
  }
  const current = Number(video.currentTime);
  return Number.isFinite(current) && current <= sourceTime + 2 ? Math.max(0, current) : null;
}

function scheduleLiveEventSync() {
  clearTimeout(state.liveSyncTimer);
  if (state.mode !== 'live' || !state.pendingLiveEvents.length) return;
  state.liveSyncTimer = setTimeout(flushLiveEventsAgainstPlayback, 80);
}

function flushLiveEventsAgainstPlayback() {
  const payload = state.latestLivePayload;
  if (!payload || state.mode !== 'live') return;
  const displayedTime = displayedLiveSourceTime(payload);
  if (displayedTime != null) {
    while (state.pendingLiveEvents.length) {
      const queued = state.pendingLiveEvents[0];
      if (liveEventDecisionTime(queued.event, payload) > displayedTime + 0.025) break;
      state.pendingLiveEvents.shift();
      const syncWaitMs = Math.max(0, performance.now() - queued.receivedAt);
      const visibleDelayMs = queued.algorithmDelayMs + syncWaitMs;
      deliverEvent(queued.event, visibleDelayMs, true);
      byId('latencyDetail').textContent = `算法 ${Math.round(queued.algorithmDelayMs)} 毫秒 · 画面同步 ${Math.round(syncWaitMs)} 毫秒`;
    }
  }
  scheduleLiveEventSync();
}

function queueLiveEvents(events, payload) {
  for (const event of events || []) {
    state.pendingLiveEvents.push({
      event,
      receivedAt: performance.now(),
      algorithmDelayMs: Math.max(0, Number(event.end_to_end_delay_ms) || 0),
    });
  }
  state.pendingLiveEvents.sort((a, b) => liveEventDecisionTime(a.event, payload) - liveEventDecisionTime(b.event, payload));
  flushLiveEventsAgainstPlayback();
}

async function connectLivePlayback(playbackUrl) {
  if (!playbackUrl || state.playbackConnecting || state.playbackUrl === playbackUrl) return;
  state.playbackConnecting = true;
  state.playbackUrl = playbackUrl;
  const video = byId('liveVideo');
  video.onplaying = () => {
    state.playbackConnecting = false;
    byId('videoPlaceholder').hidden = true;
    byId('measuredBadge').textContent = '连续直播 · HTTPS';
  };
  const retry = () => {
    state.playbackConnecting = false;
    state.playbackUrl = '';
    state.nextPlaybackRetryAt = Date.now() + 1200;
    byId('measuredBadge').textContent = '直播画面正在重连';
  };
  video.onerror = retry;
  video.onstalled = () => { byId('measuredBadge').textContent = '直播缓冲中'; };
  video.srcObject = null;
  video.src = playbackUrl;
  video.load();
  try { await video.play(); } catch (_) { retry(); }
}
function setRunning(running, label) {
  byId('liveState').classList.toggle('running', running);
  byId('enginePill').classList.toggle('running', running);
  byId('liveState').querySelector('b').textContent = label || (running ? '直播中' : '已暂停');
  byId('enginePill').lastChild.textContent = running ? ' 在线推理中' : ' 等待数据';
}

function setRoutePhase(stateName) {
  const phases = {
    queued: ['boot', '正在申请 L40S 算力'],
    preparing: ['boot', 'L40S 正在启动'],
    awaiting_stream: ['relay', 'ZLMediaKit 等待拉流'],
    connecting: ['relay', '正在建立实时链路'],
    running: ['live', '实时数据正在流动'],
    complete: ['complete', '本轮链路已完成'],
    stopped: ['idle', '实验已停止'],
    error: ['error', '实时链路已中断'],
    baseline: ['idle', '离线基准不经过实时链路'],
    idle: ['idle', '等待实验开始'],
  };
  const [phase, label] = phases[stateName] || phases.idle;
  byId('dataRoute').dataset.phase = phase;
  byId('routeStatus').textContent = label;
  byId('routeStatus').parentElement.classList.toggle('live', phase === 'live');
}

function resetRouteFlow() {
  state.routeCounters = {ingested: null, processed: null, relayUpdated: null};
  state.routePulseAt = {};
  Object.values(state.routePulseTimers).forEach(clearTimeout);
  state.routePulseTimers = {};
  document.querySelectorAll('.route-link').forEach((link) => link.classList.remove('flowing'));
  byId('cameraFlow').textContent = '30 FPS · RTMP';
  byId('relayFlow').textContent = 'RTMP / ZLMediaKit 直播流分发';
  byId('gpuFlow').textContent = '落点 · 球员 · 界内外';
  byId('resultRelayFlow').textContent = '等待推理结果';
  byId('flowAnalysis').textContent = '等待实测';
}

function pulseRouteLink(name, delay = 0) {
  const now = performance.now();
  if (now - Number(state.routePulseAt[name] || 0) < 360) return;
  state.routePulseAt[name] = now + delay;
  clearTimeout(state.routePulseTimers[name]);
  clearTimeout(state.routePulseTimers[`${name}End`]);
  state.routePulseTimers[name] = setTimeout(() => {
    const link = document.querySelector(`.link-${name}`);
    if (!link || byId('dataRoute').dataset.phase !== 'live') return;
    link.classList.remove('flowing');
    void link.offsetWidth;
    link.classList.add('flowing');
    state.routePulseTimers[`${name}End`] = setTimeout(() => link.classList.remove('flowing'), 560);
  }, delay);
}

function syncRouteFlow(payload) {
  const ingested = Math.max(0, Number(payload.ingested_frames) || 0);
  const processed = Math.max(0, Number(payload.processed_frames) || 0);
  const sourceSeconds = Math.max(0, Number(payload.source_time) || 0);
  const backlog = Math.max(0, Number(payload.backlog_seconds) || 0);
  const relayUpdated = Math.max(0, Number(payload.relay_updated_at) || 0);
  const ingestAdvanced = state.routeCounters.ingested == null
    ? ingested > 0 : ingested > state.routeCounters.ingested;
  const processAdvanced = state.routeCounters.processed == null
    ? processed > 0 : processed > state.routeCounters.processed;
  const relayAdvanced = state.routeCounters.relayUpdated == null
    ? relayUpdated > 0 : relayUpdated > state.routeCounters.relayUpdated;

  if (payload.state === 'running' && ingestAdvanced) {
    pulseRouteLink('upload');
    pulseRouteLink('pull', 80);
  }
  if (payload.state === 'running' && processAdvanced) pulseRouteLink('result', 150);
  if (payload.state === 'running' && relayAdvanced) pulseRouteLink('client', 230);

  state.routeCounters.ingested = ingested;
  state.routeCounters.processed = processed;
  state.routeCounters.relayUpdated = relayUpdated;
  byId('cameraFlow').textContent = `RTMP · ${formatClock(sourceSeconds).slice(0, 5)} 已推送`;
  byId('relayFlow').textContent = `${ingested.toLocaleString()} 帧已送达 L40S`;
  byId('gpuFlow').textContent = `${processed.toLocaleString()} 帧已完成分析`;
  byId('resultRelayFlow').textContent = `${Number(payload.event_cursor) || 0} 个落点已中继`;
  byId('flowAnalysis').textContent = `${Number(payload.event_cursor) || 0} 个落点 · ${backlog.toFixed(2)} 秒积压`;
}

function redrawLiveCalibration() {
  const canvas = byId('liveCalibrationCanvas');
  const video = byId('liveCalibrationVideo');
  if (!state.calibration.frameReady || !video.videoWidth) return;
  const maxWidth = Math.min(1280, video.videoWidth);
  const maxHeight = Math.round(maxWidth * video.videoHeight / video.videoWidth);
  if (canvas.width !== maxWidth || canvas.height !== maxHeight) {
    canvas.width = maxWidth; canvas.height = maxHeight;
    canvas.style.aspectRatio = `${maxWidth}/${maxHeight}`;
  }
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const points = state.calibration.points.map(([x, y]) => [x * canvas.width, y * canvas.height]);
  if (points.length > 1) {
    ctx.strokeStyle = '#c4f12c'; ctx.lineWidth = 3; ctx.setLineDash([10, 7]);
    ctx.beginPath(); points.forEach((point, index) => index ? ctx.lineTo(...point) : ctx.moveTo(...point));
    ctx.stroke(); ctx.setLineDash([]);
  }
  points.forEach((point, index) => {
    ctx.fillStyle = '#7f27d8'; ctx.strokeStyle = '#fff'; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(point[0], point[1], 10, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    ctx.fillStyle = '#fff'; ctx.font = 'bold 11px system-ui'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(String(index + 1), point[0], point[1]);
  });
}

function resetLiveCalibration() {
  state.calibration.points = [];
  byId('liveCalibrationStep').textContent = '第 1 步：点击近端左角';
  byId('liveCalibrationConfirm').disabled = true;
  redrawLiveCalibration();
}

function closeLiveCalibration() {
  const dialog = byId('liveCalibrationDialog');
  if (dialog.open) dialog.close();
  const video = byId('liveCalibrationVideo');
  video.removeAttribute('src'); video.load();
  if (state.calibration.objectUrl) URL.revokeObjectURL(state.calibration.objectUrl);
  state.calibration.objectUrl = '';
}

function openLiveCalibration(file = null) {
  closeLiveCalibration();
  state.calibration.file = file;
  state.calibration.frameReady = false;
  resetLiveCalibration();
  const dialog = byId('liveCalibrationDialog');
  const video = byId('liveCalibrationVideo');
  state.calibration.objectUrl = file ? URL.createObjectURL(file) : '';
  video.src = state.calibration.objectUrl || DEMO_VIDEO;
  byId('liveCalibrationStep').textContent = '正在读取球场画面…';
  dialog.showModal();
  const capture = () => {
    state.calibration.frameReady = true;
    resetLiveCalibration();
  };
  video.onseeked = capture;
  video.onloadeddata = () => {
    const target = Math.min(8, Math.max(0, (video.duration || 0) * .15));
    if (Math.abs(video.currentTime - target) > .05) video.currentTime = target;
    else capture();
  };
  video.onerror = () => { closeLiveCalibration(); toast('无法读取球场确认画面'); };
  video.load();
}

async function startLiveExperiment(corners) {
  clearTimeout(state.pollTimer);
  cancelAnimationFrame(state.animationFrame);
  closeLivePlayback();
  state.report = null;
  resetTimeline();
  setVisualMode('live');
  setRunning(true, '准备中');
  resetRouteFlow();
  setRoutePhase('preparing');
  byId('sourceName').textContent = 'Demo · L40S 实时实验';
  byId('analysisEmpty').hidden = false;
  byId('analysisEmpty').querySelector('strong').textContent = '正在建立真实媒体链路';
  byId('analysisEmpty').querySelector('span').textContent = '模型加载完成后才会开始原速推流';
  try {
    const response = await fetch('/api/live-lab/start', {
      method: 'POST', cache: 'no-store',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({court_corners: corners}),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || '无法启动端到端实验');
    rememberLiveSession(payload.session_id);
    state.eventCursor = 0;
    pollLiveExperiment();
  } catch (error) {
    setRunning(false, '启动失败');
    setRoutePhase('error');
    toast(error.message || '真实链路启动失败');
  }
}

async function uploadLiveExperiment(file, corners) {
  if (!file) return;
  clearTimeout(state.pollTimer);
  cancelAnimationFrame(state.animationFrame);
  closeLivePlayback();
  state.report = null;
  resetTimeline();
  setVisualMode('live');
  setRunning(true, state.sessionId ? '正在切换视频' : '正在上传');
  resetRouteFlow();
  setRoutePhase('preparing');
  byId('sourceName').textContent = file.name;
  byId('analysisEmpty').hidden = false;
  byId('analysisEmpty').querySelector('strong').textContent = '正在上传实验视频';
  byId('analysisEmpty').querySelector('span').textContent = state.sessionId
    ? '上传完成后会自动结束上一场并启动新实验'
    : '上传完成后会自动启动临时 L40S';
  byId('uploadLiveButton').disabled = true;
  try {
    const response = await fetch('/api/live-lab/upload', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/octet-stream',
        'X-Filename': encodeURIComponent(file.name),
        'X-Court-Corners': JSON.stringify(corners),
      },
      body: file,
      cache: 'no-store',
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || '视频上传失败');
    rememberLiveSession(payload.session_id);
    state.eventCursor = 0;
    byId('analysisEmpty').querySelector('strong').textContent = '正在启动 L40S';
    byId('analysisEmpty').querySelector('span').textContent = '冷启动与云端传送完成后按原始速度播放';
    pollLiveExperiment();
  } catch (error) {
    setRunning(false, '启动失败');
    setRoutePhase('error');
    toast(error.message || '上传实验没有启动');
  } finally {
    byId('uploadLiveButton').disabled = false;
    byId('liveUploadInput').value = '';
  }
}

async function pollLiveExperiment() {
  if (!state.sessionId || state.mode !== 'live') return;
  try {
    const url = `/api/live-lab/status?session_id=${encodeURIComponent(state.sessionId)}&after_event=${state.eventCursor}`;
    const response = await fetch(url, {cache: 'no-store'});
    const payload = await response.json();
    if (response.status === 404 && state.missingPolls < MAX_MISSING_POLLS) {
      state.missingPolls += 1;
      setRunning(true, '正在同步云端会话');
      state.pollTimer = setTimeout(pollLiveExperiment, 250);
      return;
    }
    if (response.status === 404) {
      restoreIdleLab('上一次实验已经结束，请开始新的测试');
      return;
    }
    if (!response.ok) throw new Error(payload.error || '实时状态读取失败');
    state.missingPolls = 0;
    state.latestLivePayload = payload;
    const running = ['queued', 'preparing', 'awaiting_stream', 'connecting', 'running'].includes(payload.state);
    setRunning(running, payload.stage || (running ? '直播中' : '已完成'));
    setRoutePhase(payload.state);
    syncRouteFlow(payload);
    byId('sourceClock').textContent = formatClock(payload.source_time);
    byId('videoHudClock').textContent = formatClock(payload.analysis_time);
    byId('resultClock').textContent = formatClock(payload.analysis_time);
    const backlog = Math.max(0, Number(payload.backlog_seconds) || 0);
    byId('latencyValue').textContent = `${backlog.toFixed(2)} 秒`;
    byId('latencyDetail').textContent = state.pendingLiveEvents.length
      ? '算法已确认 · 等待左侧画面到达确认时刻'
      : payload.latest_event_delay_ms == null
      ? '摄像头时间 − 算法处理时间'
      : `最近落点端到端 ${Math.round(payload.latest_event_delay_ms)} 毫秒`;
    if (!state.playbackUrl && !state.playbackConnecting) {
      byId('measuredBadge').textContent = payload.device ? '等待连续直播画面' : '正在准备模型';
    }
    if (payload.state === 'running' && payload.fmp4_playback_url
        && Date.now() >= state.nextPlaybackRetryAt) {
      connectLivePlayback(payload.fmp4_playback_url);
    }
    queueLiveEvents(payload.events, payload);
    state.eventCursor = Number(payload.event_cursor) || state.eventCursor;
    if ((payload.events || []).length) byId('analysisEmpty').hidden = true;
    if (payload.state === 'error') throw new Error(payload.error || '端到端实验中断');
    if (running) state.pollTimer = setTimeout(pollLiveExperiment, 120);
    else if (payload.state === 'complete') {
      byId('latencyValue').textContent = `${Number(payload.realtime_factor).toFixed(2)}×`;
      byId('latencyDetail').textContent = `最大积压 ${Number(payload.peak_backlog_seconds).toFixed(1)} 秒`;
      const comparison = payload.comparison;
      if (comparison) {
        byId('landingDetail').textContent = `基准 ${comparison.reference_events} 个 · 匹配 ${comparison.matched_events} 个`;
      }
      toast(`实测完成：总耗时 ${Number(payload.elapsed_seconds).toFixed(1)} 秒，实时倍率 ${Number(payload.realtime_factor).toFixed(2)}×`);
    }
  } catch (error) {
    setRunning(false, '实验中断');
    setRoutePhase('error');
    toast(error.message || '实时实验连接中断');
  }
}

async function useDemoBaseline() {
  clearTimeout(state.pollTimer);
  closeLivePlayback();
  try {
    const response = await fetch(DEMO_REPORT, {cache: 'no-store'});
    if (!response.ok) throw new Error('Demo 分析结果读取失败');
    state.report = await response.json();
    setVisualMode('baseline');
    resetRouteFlow();
    setRoutePhase('baseline');
    resetTimeline();
    const video = byId('liveVideo');
    video.src = DEMO_VIDEO;
    video.load();
    byId('sourceName').textContent = 'Demo 离线冻结结果基准';
    byId('analysisEmpty').hidden = true;
    byId('flowAnalysis').textContent = '固定 10 帧';
    await video.play();
  } catch (error) {
    toast(error.message || 'Demo 载入失败');
  }
}

function drawCourt() {
  const canvas = byId('landingCourt');
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const width = rect.width, height = rect.height;
  ctx.clearRect(0, 0, width, height);
  const courtWidth = 10.97, courtLength = 23.77;
  const courtH = height * .82, courtW = courtH * (courtWidth / courtLength);
  const marginX = (width - courtW) / 2, marginY = (height - courtH) / 2;
  const x = (metres) => marginX + metres / courtWidth * courtW;
  const y = (metres) => marginY + (courtLength - metres) / courtLength * courtH;
  ctx.fillStyle = 'rgba(142,54,235,.045)'; ctx.fillRect(marginX, marginY, courtW, courtH);
  ctx.strokeStyle = 'rgba(196,241,44,.78)'; ctx.lineWidth = 1.4; ctx.strokeRect(marginX, marginY, courtW, courtH);
  ctx.strokeStyle = 'rgba(255,255,255,.55)';
  ctx.beginPath(); ctx.moveTo(marginX, y(11.885)); ctx.lineTo(marginX + courtW, y(11.885)); ctx.stroke();
  [1.37, 9.6].forEach((value) => {ctx.beginPath(); ctx.moveTo(x(value), marginY); ctx.lineTo(x(value), marginY + courtH); ctx.stroke();});
  ctx.beginPath(); ctx.moveTo(x(5.485), y(5.485)); ctx.lineTo(x(5.485), y(18.285)); ctx.stroke();
  [5.485, 18.285].forEach((value) => {ctx.beginPath(); ctx.moveTo(x(1.37), y(value)); ctx.lineTo(x(9.6), y(value)); ctx.stroke();});
  state.delivered.filter((event) => event.rally_id === state.activeRally).forEach((event) => {
    const px = x(Number.isFinite(Number(event.x)) ? Number(event.x) : 5.485);
    const py = event.net_hit ? y(11.885) : y(Number.isFinite(Number(event.y)) ? Number(event.y) : 11.885);
    const lineCall = event.line_call || (event.zone === 'Out' ? 'out' : 'in');
    const out = lineCall === 'out' || event.outcome === 'out' || event.net_hit;
    if (out) {
      ctx.strokeStyle = '#ff5b6e'; ctx.lineWidth = 3; ctx.beginPath();
      ctx.moveTo(px - 7, py - 7); ctx.lineTo(px + 7, py + 7); ctx.moveTo(px + 7, py - 7); ctx.lineTo(px - 7, py + 7); ctx.stroke();
    } else {
      const color = event.player_id === 'A' ? '#b542f6' : event.player_id === 'B' ? '#2dd4bf' : '#c4f12c';
      ctx.shadowColor = color; ctx.shadowBlur = 14; ctx.fillStyle = color;
      ctx.beginPath(); ctx.arc(px, py, 6, 0, Math.PI * 2); ctx.fill(); ctx.shadowBlur = 0;
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; ctx.stroke();
    }
  });
}

function deliverEvent(event, latencyMs, measured) {
  if (state.activeRally !== event.rally_id) state.activeRally = event.rally_id;
  state.delivered.push(event);
  const lineCall = event.line_call || (event.zone === 'Out' ? 'out' : 'in');
  const out = lineCall === 'out' || event.outcome === 'out' || event.net_hit;
  const player = event.player_id === 'A' ? '球员 A' : event.player_id === 'B' ? '球员 B' : '未分配球员';
  const label = out ? (event.net_hit ? '下网' : '界外') : `${player} 落点`;
  const color = event.player_id === 'A' ? '#b542f6' : event.player_id === 'B' ? '#2dd4bf' : '#c4f12c';
  byId('landingCount').textContent = String(state.delivered.length);
  const currentRallyCount = state.delivered.filter((item) => item.rally_id === state.activeRally).length;
  byId('landingDetail').textContent = `当前回合 ${currentRallyCount} 个`;
  const feed = byId('eventFeed'); feed.querySelector('.empty-feed')?.remove();
  const li = document.createElement('li');
  li.innerHTML = `<time>${formatClock(eventTouchdownTime(event))}</time><b>${label}</b><em>${measured ? '实测 ' : '+'}${Math.round(latencyMs)} ms</em>`;
  li.querySelector('b').style.color = out ? '#ff5b6e' : color;
  feed.prepend(li); while (feed.children.length > 6) feed.lastElementChild.remove();
  byId('flashZone').textContent = label; byId('decisionFlash').classList.add('show');
  clearTimeout(state.flashTimer); state.flashTimer = setTimeout(() => byId('decisionFlash').classList.remove('show'), 1150);
  drawCourt();
}

function updateBaselineTimeline() {
  if (state.mode !== 'baseline') return;
  const video = byId('liveVideo'), current = Number(video.currentTime) || 0;
  byId('sourceClock').textContent = formatClock(current); byId('videoHudClock').textContent = formatClock(current);
  const events = orderedBaselineEvents(state.report);
  while (state.nextEvent < events.length && baselineDecisionTime(events[state.nextEvent]) <= current) {
    const event = events[state.nextEvent++];
    deliverEvent(event, Math.max(0, (baselineDecisionTime(event) - eventTouchdownTime(event)) * 1000), false);
  }
  byId('resultClock').textContent = formatClock(Math.max(0, current - 10 / (Number(state.report?.fps) || 30)));
  if (!video.paused && !video.ended) state.animationFrame = requestAnimationFrame(updateBaselineTimeline);
}

byId('startLiveButton').addEventListener('click', () => openLiveCalibration());
byId('uploadLiveButton').addEventListener('click', () => byId('liveUploadInput').click());
byId('liveUploadInput').addEventListener('change', (event) => {
  const file = event.target.files?.[0];
  if (file) openLiveCalibration(file);
});
byId('liveCalibrationCanvas').addEventListener('click', (event) => {
  if (!state.calibration.frameReady || state.calibration.points.length >= 4) return;
  const rect = event.currentTarget.getBoundingClientRect();
  state.calibration.points.push([
    Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)),
    Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height)),
  ]);
  const labels = ['近端左角', '近端右角', '远端右角', '远端左角'];
  const count = state.calibration.points.length;
  byId('liveCalibrationStep').textContent = count === 4
    ? '四个角点已标记，请确认后启动'
    : `第 ${count + 1} 步：点击${labels[count]}`;
  byId('liveCalibrationConfirm').disabled = count !== 4;
  redrawLiveCalibration();
});
byId('liveCalibrationReset').addEventListener('click', resetLiveCalibration);
byId('liveCalibrationClose').addEventListener('click', closeLiveCalibration);
byId('liveCalibrationDialog').addEventListener('cancel', (event) => {
  event.preventDefault(); closeLiveCalibration();
});
byId('liveCalibrationConfirm').addEventListener('click', () => {
  if (state.calibration.points.length !== 4) return;
  const file = state.calibration.file;
  const corners = state.calibration.points.map((point) => [...point]);
  closeLiveCalibration();
  if (file) uploadLiveExperiment(file, corners);
  else startLiveExperiment(corners);
});
byId('useDemoButton').addEventListener('click', useDemoBaseline);
byId('playButton').addEventListener('click', () => {
  if (state.mode !== 'baseline') return;
  const video = byId('liveVideo');
  if (video.paused) video.play().catch(() => toast('浏览器暂时无法播放这个视频')); else video.pause();
});
byId('restartButton').addEventListener('click', () => {
  if (state.mode !== 'baseline') return;
  const video = byId('liveVideo'); video.currentTime = 0; resetTimeline(); video.play().catch(() => {});
});
const video = byId('liveVideo');
video.addEventListener('play', () => {if (state.mode === 'baseline') {setRunning(true, '基准回放中'); cancelAnimationFrame(state.animationFrame); updateBaselineTimeline();}});
video.addEventListener('pause', () => {if (state.mode === 'baseline') setRunning(false, '已暂停');});
video.addEventListener('ended', () => {if (state.mode === 'baseline') setRunning(false, '已结束');});
window.addEventListener('resize', drawCourt);
drawCourt();

const resumeSession = new URLSearchParams(window.location.search).get('session_id')
  || sessionStorage.getItem('netcastLiveSession');
if (resumeSession) {
  if (window.location.hash) {
    history.replaceState(null, '', `${window.location.pathname}${window.location.search}`);
    window.scrollTo({top: 0, left: 0, behavior: 'instant'});
  }
  rememberLiveSession(resumeSession);
  resetTimeline();
  setVisualMode('live');
  setRunning(true, '正在恢复实验');
  resetRouteFlow();
  setRoutePhase('preparing');
  byId('analysisEmpty').hidden = false;
  byId('analysisEmpty').querySelector('strong').textContent = '正在重新连接云端实验';
  byId('analysisEmpty').querySelector('span').textContent = '刷新页面不会中断正在运行的 L40S';
  pollLiveExperiment();
} else {
  setRoutePhase('idle');
}
