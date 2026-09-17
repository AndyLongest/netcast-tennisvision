const byId = (id) => document.getElementById(id);
const DEMO_VIDEO = '../assets/demo/demo.mp4';
const DEMO_REPORT = '../assets/demo/scene3d.json';
const state = {
  mode: null, report: null, delivered: [], nextEvent: 0, activeRally: null,
  animationFrame: 0, flashTimer: 0, sessionId: null, eventCursor: 0,
  pollTimer: 0, playbackUrl: '', playbackConnecting: false, nextPlaybackRetryAt: 0,
};

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

async function startLiveExperiment() {
  clearTimeout(state.pollTimer);
  cancelAnimationFrame(state.animationFrame);
  closeLivePlayback();
  state.report = null;
  resetTimeline();
  setVisualMode('live');
  setRunning(true, '准备中');
  byId('sourceName').textContent = 'Demo · L40S 实时实验';
  byId('analysisEmpty').hidden = false;
  byId('analysisEmpty').querySelector('strong').textContent = '正在建立真实媒体链路';
  byId('analysisEmpty').querySelector('span').textContent = '模型加载完成后才会开始原速推流';
  try {
    const response = await fetch('/api/live-lab/start', {method: 'POST', cache: 'no-store'});
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || '无法启动端到端实验');
    state.sessionId = payload.session_id;
    state.eventCursor = 0;
    pollLiveExperiment();
  } catch (error) {
    setRunning(false, '启动失败');
    toast(error.message || '真实链路启动失败');
  }
}

async function uploadLiveExperiment(file) {
  if (!file) return;
  clearTimeout(state.pollTimer);
  cancelAnimationFrame(state.animationFrame);
  closeLivePlayback();
  state.report = null;
  resetTimeline();
  setVisualMode('live');
  setRunning(true, '正在上传');
  byId('sourceName').textContent = file.name;
  byId('analysisEmpty').hidden = false;
  byId('analysisEmpty').querySelector('strong').textContent = '正在上传实验视频';
  byId('analysisEmpty').querySelector('span').textContent = '上传完成后会自动启动临时 L40S';
  byId('uploadLiveButton').disabled = true;
  try {
    const response = await fetch('/api/live-lab/upload', {
      method: 'POST',
      headers: {'Content-Type': 'application/octet-stream', 'X-Filename': encodeURIComponent(file.name)},
      body: file,
      cache: 'no-store',
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || '视频上传失败');
    state.sessionId = payload.session_id;
    state.eventCursor = 0;
    byId('analysisEmpty').querySelector('strong').textContent = '正在启动 L40S';
    byId('analysisEmpty').querySelector('span').textContent = '冷启动与云端传送完成后按原始速度播放';
    pollLiveExperiment();
  } catch (error) {
    setRunning(false, '启动失败');
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
    if (!response.ok) throw new Error(payload.error || '实时状态读取失败');
    const running = ['preparing', 'connecting', 'running'].includes(payload.state);
    setRunning(running, payload.stage || (running ? '直播中' : '已完成'));
    byId('sourceClock').textContent = formatClock(payload.source_time);
    byId('videoHudClock').textContent = formatClock(payload.analysis_time);
    byId('resultClock').textContent = formatClock(payload.analysis_time);
    const backlog = Math.max(0, Number(payload.backlog_seconds) || 0);
    byId('latencyValue').textContent = `${backlog.toFixed(2)} 秒`;
    byId('latencyDetail').textContent = payload.latest_event_delay_ms == null
      ? '摄像头时间 − 算法处理时间'
      : `最近落点端到端 ${Math.round(payload.latest_event_delay_ms)} 毫秒`;
    byId('flowAnalysis').textContent = `${backlog.toFixed(2)} 秒积压`;
    if (!state.playbackUrl && !state.playbackConnecting) {
      byId('measuredBadge').textContent = payload.device ? '等待连续直播画面' : '正在准备模型';
    }
    if (payload.state === 'running' && payload.fmp4_playback_url
        && Date.now() >= state.nextPlaybackRetryAt) {
      connectLivePlayback(payload.fmp4_playback_url);
    }
    for (const event of payload.events || []) deliverEvent(event, Number(event.end_to_end_delay_ms) || 0, true);
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
    const review = lineCall === 'review';
    if (out) {
      ctx.strokeStyle = '#ff5b6e'; ctx.lineWidth = 3; ctx.beginPath();
      ctx.moveTo(px - 7, py - 7); ctx.lineTo(px + 7, py + 7); ctx.moveTo(px + 7, py - 7); ctx.lineTo(px - 7, py + 7); ctx.stroke();
    } else if (review) {
      ctx.strokeStyle = '#f3b941'; ctx.fillStyle = 'rgba(243,185,65,.16)'; ctx.lineWidth = 2;
      ctx.setLineDash([3, 2]); ctx.beginPath(); ctx.arc(px, py, 7, 0, Math.PI * 2); ctx.fill(); ctx.stroke(); ctx.setLineDash([]);
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
  const review = lineCall === 'review';
  const player = event.player_id === 'A' ? '球员 A' : event.player_id === 'B' ? '球员 B' : '未分配球员';
  const label = out ? (event.net_hit ? '下网' : '界外') : review ? '压线待复核' : `${player} 落点`;
  const color = review ? '#f3b941' : event.player_id === 'A' ? '#b542f6' : event.player_id === 'B' ? '#2dd4bf' : '#c4f12c';
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

byId('startLiveButton').addEventListener('click', startLiveExperiment);
byId('uploadLiveButton').addEventListener('click', () => byId('liveUploadInput').click());
byId('liveUploadInput').addEventListener('change', (event) => uploadLiveExperiment(event.target.files?.[0]));
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
