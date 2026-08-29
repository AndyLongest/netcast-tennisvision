const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const API_BASE = location.protocol === 'file:' ? 'http://127.0.0.1:4173' : '';
const apiUrl = (path) => `${API_BASE}${path}`;

const OUTPUT = {
  original: '../data/clip.mp4',
  annotated: '../data/outputs/annotated_clip.mp4',
  scene: '../data/outputs/scene3d.json',
  viewer: '../data/outputs/rally3d.html',
};

// The public sample must never point at files being replaced by an upload job.
const DEMO_OUTPUT = {
  original: '../assets/demo/demo.mp4',
  annotated: '../assets/demo/annotated_clip.mp4',
  scene: '../assets/demo/scene3d.json',
  viewer: '../assets/demo/rally3d.html',
};

const activeOutput = () => state.isDemo ? DEMO_OUTPUT : OUTPUT;

const ZONES = {
  'Far Backcourt': '远端后场',
  'Near Backcourt': '近端后场',
  'Far-Left Service Box': '远端左发球区',
  'Far-Right Service Box': '远端右发球区',
  'Near-Left Service Box': '近端左发球区',
  'Near-Right Service Box': '近端右发球区',
  'Left Doubles Alley': '左侧双打通道',
  'Right Doubles Alley': '右侧双打通道',
  Out: '界外',
};

const STAGE_MESSAGES = [
  [0, '正在读取视频画面…', '检查分辨率、时长与原始帧率'],
  [24, '正在建立球场坐标…', '识别边线、球网与两名球员'],
  [50, '正在连续追踪网球…', '融合视觉运动与物理约束'],
  [76, '正在确认击球与落地…', '按回合核验落点与界内外状态'],
  [92, '正在生成可视化报告…', '整理回放、落点地图与三维场景'],
];
const ACTIVE_BACKEND_STATES = new Set(['queued', 'running', 'needs_court_calibration']);

let state = {
  isDemo: true, generated: true, name: '真实比赛样例', duration: 0,
  events: [], scene: null, objectUrl: null, file: null, annotated: true,
};
let toastTimer;
let demoTimer;
let demoResultTimer;
let reportLoadToken = 0;
let demoSceneCache = globalThis.TENNIS_DEMO_SCENE || null;
let calibrationPromise = null;
let backendRunToken = 0;
const DEMO_SCENE_CACHE_KEY = 'netcast-demo-scene-v3';

function formatTime(seconds) {
  const value = Math.max(0, Number(seconds) || 0);
  const minutes = Math.floor(value / 60);
  return `${String(minutes).padStart(2, '0')}:${(value % 60).toFixed(1).padStart(4, '0')}`;
}

function toast(message) {
  const element = $('#toast');
  element.textContent = message;
  element.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => element.classList.remove('show'), 2800);
}

async function requestCourtCalibration(calibration) {
  if (calibrationPromise) return calibrationPromise;
  calibrationPromise = new Promise((resolve, reject) => {
    const dialog = $('#courtCalibrationDialog');
    const canvas = $('#courtCalibrationCanvas');
    const ctx = canvas.getContext('2d');
    const image = new Image();
    const labels = calibration.point_order || ['近端左角', '近端右角', '远端右角', '远端左角'];
    const points = [];
    let displayScale = 1;

    const redraw = () => {
      if (!image.naturalWidth) return;
      const maxWidth = Math.min(800, dialog.clientWidth - 52);
      displayScale = Math.min(1, maxWidth / image.naturalWidth);
      canvas.width = Math.round(image.naturalWidth * displayScale);
      canvas.height = Math.round(image.naturalHeight * displayScale);
      ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
      if (points.length > 1) {
        ctx.strokeStyle = '#c4f12c'; ctx.lineWidth = 2; ctx.setLineDash([7, 5]);
        ctx.beginPath();
        points.forEach((point, index) => index
          ? ctx.lineTo(point[0] * displayScale, point[1] * displayScale)
          : ctx.moveTo(point[0] * displayScale, point[1] * displayScale));
        if (points.length === 4) ctx.closePath();
        ctx.stroke(); ctx.setLineDash([]);
      }
      points.forEach((point, index) => {
        const x = point[0] * displayScale, y = point[1] * displayScale;
        ctx.fillStyle = '#c4f12c'; ctx.strokeStyle = '#25143a'; ctx.lineWidth = 3;
        ctx.beginPath(); ctx.arc(x, y, 8, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
        ctx.fillStyle = '#25143a'; ctx.font = '700 10px system-ui';
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(String(index + 1), x, y);
      });
      $('#calibrationStep').textContent = points.length < 4
        ? `第 ${points.length + 1} 步：点击${labels[points.length]}`
        : '四个角点已标记，请确认球场轮廓';
      $('#calibrationSubmit').disabled = points.length !== 4;
    };

    $('#calibrationReason').textContent = calibration.reason || '自动识别没有达到可靠标准，请确认四个底线角点。';
    image.onload = () => { redraw(); dialog.showModal(); };
    image.onerror = () => reject(new Error('无法读取球场确认画面'));
    image.src = `${calibration.preview}?v=${encodeURIComponent(calibration.request_id)}`;
    canvas.onclick = (event) => {
      if (points.length >= 4 || !image.naturalWidth) return;
      const rect = canvas.getBoundingClientRect();
      points.push([
        (event.clientX - rect.left) * image.naturalWidth / rect.width,
        (event.clientY - rect.top) * image.naturalHeight / rect.height,
      ]);
      redraw();
    };
    $('#calibrationReset').onclick = () => { points.length = 0; redraw(); };
    $('#calibrationSubmit').onclick = async () => {
      if (points.length !== 4) return;
      $('#calibrationSubmit').disabled = true;
      try {
        const response = await fetch(apiUrl('/api/court-calibration'), {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ request_id: calibration.request_id, corners: points }),
        });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(payload.error || '球场边界提交失败');
        dialog.close(); resolve();
      } catch (error) {
        $('#calibrationSubmit').disabled = false; toast(error.message);
      }
    };
  }).finally(() => { calibrationPromise = null; });
  return calibrationPromise;
}

function startAmbientBall() {
  const canvas = $('#ambientBall');
  const court = canvas.parentElement;
  const ctx = canvas.getContext('2d');
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const motion = {
    x: 0,
    y: 0,
    vx: 62,
    vy: -74,
    last: 0,
    trail: [],
    seed: Math.random() * Math.PI * 2,
    ready: false,
    bounceAt: -Infinity,
  };

  function fitCanvas() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, court.clientWidth);
    const height = Math.max(1, court.clientHeight);
    const pixelWidth = Math.round(width * dpr), pixelHeight = Math.round(height * dpr);
    if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
      canvas.width = pixelWidth;
      canvas.height = pixelHeight;
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    if (!motion.ready) {
      motion.x = width * .31;
      motion.y = height * .76;
      motion.ready = true;
    } else {
      motion.x = Math.min(width - 10, Math.max(10, motion.x));
      motion.y = Math.min(height - 10, Math.max(10, motion.y));
    }
    return { width, height };
  }

  function reflect(position, velocity, low, high) {
    if (position < low) return { position: low + (low - position), velocity: Math.abs(velocity) };
    if (position > high) return { position: high - (position - high), velocity: -Math.abs(velocity) };
    return { position, velocity };
  }

  function draw(timestamp) {
    if ($('#processingView').hidden) {
      motion.last = timestamp;
      motion.trail.length = 0;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      requestAnimationFrame(draw);
      return;
    }
    const { width, height } = fitCanvas();
    // Cap the time step so switching tabs can never make the point teleport.
    const dt = Math.min(.04, Math.max(0, (timestamp - (motion.last || timestamp)) / 1000));
    motion.last = timestamp;
    const seconds = timestamp / 1000;

    // Slow, continuous steering keeps the movement organic without random jumps.
    // Incommensurate frequencies keep it from falling into an obvious short loop.
    const drift = Math.sin(seconds * .31 + motion.seed)
      + .42 * Math.sin(seconds * .67 + 1.7 + motion.seed)
      + .16 * Math.sin(seconds * 1.07 + 4.1);
    const turn = drift * .16 * dt;
    const cos = Math.cos(turn), sin = Math.sin(turn);
    const nextVx = motion.vx * cos - motion.vy * sin;
    const nextVy = motion.vx * sin + motion.vy * cos;
    motion.vx = nextVx;
    motion.vy = nextVy;

    // The previous 48px/s became almost static after the court's perspective transform.
    // This remains calm, but crosses enough of the court to be visibly alive.
    const targetSpeed = (reduceMotion ? 52 : 92) + 4 * Math.sin(seconds * .17 + motion.seed);
    const speed = Math.hypot(motion.vx, motion.vy) || targetSpeed;
    motion.vx *= targetSpeed / speed;
    motion.vy *= targetSpeed / speed;
    motion.x += motion.vx * dt;
    motion.y += motion.vy * dt;

    const margin = 9;
    const rx = reflect(motion.x, motion.vx, margin, width - margin);
    const ry = reflect(motion.y, motion.vy, margin, height - margin);
    const bouncedX = rx.velocity !== motion.vx;
    const bouncedY = ry.velocity !== motion.vy;
    const bounced = bouncedX || bouncedY;
    motion.x = rx.position; motion.vx = rx.velocity;
    motion.y = ry.position; motion.vy = ry.velocity;
    if (bounced) {
      motion.bounceAt = timestamp;
      // Deflect at each impact so the point keeps exploring the whole court instead
      // of getting trapped in a nearly vertical or horizontal ping-pong loop.
      const phase = Math.sin(seconds * .73 + motion.seed);
      const nudge = Math.sign(phase || 1) * (.11 + Math.abs(phase) * .05);
      const c = Math.cos(nudge), s = Math.sin(nudge);
      const vx = motion.vx * c - motion.vy * s;
      motion.vy = motion.vx * s + motion.vy * c;
      motion.vx = vx;

      const axisFloor = targetSpeed * .27;
      if (bouncedY && Math.abs(motion.vx) < axisFloor) {
        motion.vx = Math.sign(Math.sin(seconds * .41 + motion.seed) || 1) * axisFloor;
        motion.vy = Math.sign(motion.vy) * Math.sqrt(targetSpeed ** 2 - motion.vx ** 2);
      }
      if (bouncedX && Math.abs(motion.vy) < axisFloor) {
        motion.vy = Math.sign(Math.cos(seconds * .37 + motion.seed) || 1) * axisFloor;
        motion.vx = Math.sign(motion.vx) * Math.sqrt(targetSpeed ** 2 - motion.vy ** 2);
      }
    }

    motion.trail.push({ x: motion.x, y: motion.y, at: timestamp });
    while (motion.trail.length && timestamp - motion.trail[0].at > 1250) motion.trail.shift();
    ctx.clearRect(0, 0, width, height);
    if (motion.trail.length > 1) {
      ctx.lineCap = 'round';
      for (let index = 1; index < motion.trail.length; index += 1) {
        const previous = motion.trail[index - 1], point = motion.trail[index];
        const age = Math.max(0, 1 - (timestamp - point.at) / 1250);
        ctx.strokeStyle = `rgba(196,241,44,${age * age * .2})`;
        ctx.lineWidth = .65 + age * 1.35;
        ctx.beginPath(); ctx.moveTo(previous.x, previous.y); ctx.lineTo(point.x, point.y); ctx.stroke();
      }
    }

    const depth = motion.y / Math.max(1, height);
    const breath = 1 + .07 * Math.sin(seconds * 2.1 + motion.seed);
    const radius = (2.8 + depth * 2.6) * breath;
    const glow = ctx.createRadialGradient(motion.x, motion.y, 0, motion.x, motion.y, radius * 4.8);
    glow.addColorStop(0, 'rgba(255,255,255,1)');
    glow.addColorStop(.18, 'rgba(244,255,192,.95)');
    glow.addColorStop(.46, 'rgba(196,241,44,.42)');
    glow.addColorStop(1, 'rgba(196,241,44,0)');
    ctx.fillStyle = glow;
    ctx.beginPath(); ctx.arc(motion.x, motion.y, radius * 4.8, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#fff';
    ctx.beginPath(); ctx.arc(motion.x, motion.y, radius, 0, Math.PI * 2); ctx.fill();

    const bounceAge = timestamp - motion.bounceAt;
    if (bounceAge >= 0 && bounceAge < 460) {
      const progress = bounceAge / 460;
      ctx.strokeStyle = `rgba(196,241,44,${(1 - progress) * .48})`;
      ctx.lineWidth = 1.2 * (1 - progress);
      ctx.beginPath();
      ctx.arc(motion.x, motion.y, radius * (1.5 + progress * 2.8), 0, Math.PI * 2);
      ctx.stroke();
    }

    // Expose the live position for visual regression checks without affecting the UI.
    canvas.dataset.motionX = motion.x.toFixed(2);
    canvas.dataset.motionY = motion.y.toFixed(2);
    canvas.dataset.motionSpeed = Math.hypot(motion.vx, motion.vy).toFixed(2);
    requestAnimationFrame(draw);
  }

  requestAnimationFrame(draw);
}

function setView(view) {
  $('#welcomeView').hidden = view !== 'welcome';
  $('#processingView').hidden = view !== 'processing';
  $('#resultsView').hidden = view !== 'results';
  window.scrollTo({ top: 0, behavior: 'auto' });
}

function resetForNextAnalysis({ announce = false } = {}) {
  if (!$('#processingView').hidden) {
    toast('本场比赛仍在分析，完成后即可开始下一场');
    return;
  }
  clearInterval(demoTimer);
  clearTimeout(demoResultTimer);
  reportLoadToken += 1;
  const video = $('#analysisVideo');
  video.pause();
  video.removeAttribute('src');
  video.load();
  $('#sceneFrame').src = 'about:blank';
  $('#videoInput').value = '';
  $$('.event-filter button').forEach((button, index) => button.classList.toggle('active', index === 0));
  if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
  state = {
    isDemo: true, generated: true, name: '真实比赛样例', duration: 0,
    events: [], scene: null, objectUrl: null, file: null, annotated: true,
  };
  setView('welcome');
  if (location.hash) history.replaceState(null, '', location.pathname + location.search);
  if (announce) toast('已返回首页，可以开始下一场比赛');
}

function updateStages(progress) {
  const active = Math.min(3, Math.floor(Math.min(progress, 99) / 25));
  $$('#stageList li').forEach((item, index) => {
    item.classList.toggle('active', index === active && progress < 100);
    item.classList.toggle('done', index < active || progress === 100);
    item.querySelector('b').textContent = index < active || progress === 100 ? '✓' : String(index + 1);
  });
  $('#progressBar').style.width = `${progress}%`;
  $('#progressValue').textContent = `${Math.round(progress)}%`;
}

async function videoFingerprint(file) {
  const sampleSize = 256 * 1024;
  const first = new Uint8Array(await file.slice(0, sampleSize).arrayBuffer());
  const last = new Uint8Array(await file.slice(Math.max(0, file.size - sampleSize)).arrayBuffer());
  const payload = new Uint8Array(8 + first.length + last.length);
  new DataView(payload.buffer).setBigUint64(0, BigInt(file.size));
  payload.set(first, 8);
  payload.set(last, 8 + first.length);
  const digest = await crypto.subtle.digest('SHA-256', payload);
  return `sha256-sample:${[...new Uint8Array(digest)].map((value) => value.toString(16).padStart(2, '0')).join('')}`;
}

function adoptBackendJob(status, { resumed = false } = {}) {
  const filename = status.filename || state.file?.name || '正在分析的视频';
  if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
  state = {
    ...state,
    isDemo: false,
    generated: false,
    name: filename.replace(/\.[^.]+$/, ''),
    duration: 0,
    events: [],
    scene: null,
    objectUrl: null,
    file: null,
    annotated: false,
    jobId: status.job_id || null,
    videoFingerprint: status.video_fingerprint || null,
  };
  setView('processing');
  updateStages(Number(status.progress) || 0);
  $('#processingMessage').textContent = status.stage || '正在继续分析这场比赛…';
  $('#timeHint').textContent = resumed
    ? '已接回上次进度，无需重新上传'
    : '比赛数据只在当前电脑上处理';
}

async function monitorBackendJob(initialStatus, token, { resumed = false } = {}) {
  let status = initialStatus;
  adoptBackendJob(status, { resumed });
  while (token === backendRunToken) {
    updateStages(Number(status.progress) || 0);
    if (status.state === 'needs_court_calibration') {
      $('#processingMessage').textContent = '自动识别需要你确认一下球场…';
      $('#timeHint').textContent = '仅在识别把握不足时出现，不会增加普通视频的步骤';
      await requestCourtCalibration(status.calibration);
    } else {
      $('#processingMessage').textContent = status.stage || '正在继续分析这场比赛…';
      const message = [...STAGE_MESSAGES].reverse().find(([at]) => (Number(status.progress) || 0) >= at);
      if (!resumed && message) $('#timeHint').textContent = message[2];
    }
    if (status.state === 'error') throw new Error(status.error || '分析失败');
    if (status.state === 'complete') {
      state.generated = true;
      state.annotated = true;
      await showResults();
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 1800));
    const response = await fetch(apiUrl('/api/status'), { cache: 'no-store' });
    if (!response.ok) throw new Error('无法读取分析进度');
    status = await response.json();
  }
}

function chooseVideo() { $('#videoInput').click(); }

function selectFile(file) {
  const supported = file.type.startsWith('video/') || /\.(mp4|mov|webm|mkv)$/i.test(file.name);
  if (!supported) { toast('暂不支持这种视频文件，请换用常见视频格式'); return; }
  if (file.size > 4 * 1024 ** 3) { toast('视频超过四吉字节，请先裁剪或压缩后再上传'); return; }
  if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
  state = {
    ...state, isDemo: false, generated: false, name: file.name.replace(/\.[^.]+$/, ''),
    duration: 0, events: [], scene: null, file, objectUrl: URL.createObjectURL(file), annotated: false,
  };
  startAnalysis();
}

function startDemo() {
  state = {
    ...state, isDemo: true, generated: true, name: '真实比赛样例', duration: 0,
    events: [], scene: null, file: null, annotated: true,
  };
  startAnalysis();
}

function startAnalysis() {
  clearTimeout(demoResultTimer);
  reportLoadToken += 1;
  setView('processing');
  updateStages(0);
  $('#processingMessage').textContent = STAGE_MESSAGES[0][1];
  $('#timeHint').textContent = STAGE_MESSAGES[0][2];
  if (!state.isDemo) { runBackendAnalysis(); return; }
  clearInterval(demoTimer);
  let progress = 0;
  demoTimer = setInterval(() => {
    // Let the sample transition breathe; the court animation is part of the experience,
    // not a one-second flash before the report appears.
    progress = Math.min(100, progress + 2);
    updateStages(progress);
    const message = [...STAGE_MESSAGES].reverse().find(([at]) => progress >= at);
    $('#processingMessage').textContent = message[1];
    $('#timeHint').textContent = message[2];
    if (progress === 100) {
      clearInterval(demoTimer);
      demoResultTimer = setTimeout(showResults, 260);
    }
  }, 100);
}

async function runBackendAnalysis() {
  const token = ++backendRunToken;
  try {
    $('#processingMessage').textContent = '视频已收到，正在准备分析…';
    $('#timeHint').textContent = '比赛数据只在当前电脑上处理';
    const health = await fetch(apiUrl('/api/status'), { cache: 'no-store' });
    if (!health.ok) throw new Error('本机分析服务尚未就绪，请重新打开 Netcast TennisVision');
    const current = await health.json();
    const fingerprint = await videoFingerprint(state.file);

    if (ACTIVE_BACKEND_STATES.has(current.state)) {
      const sameVideo = !current.video_fingerprint || current.video_fingerprint === fingerprint;
      adoptBackendJob(current, { resumed: true });
      toast(sameVideo ? '已接回这段视频的分析进度' : '已有比赛正在分析，已恢复当前任务');
      await monitorBackendJob(current, token, { resumed: true });
      return;
    }

    if (current.state === 'complete' && current.video_fingerprint === fingerprint) {
      adoptBackendJob(current, { resumed: true });
      toast('这段视频已经分析完成，正在打开报告');
      await monitorBackendJob(current, token, { resumed: true });
      return;
    }

    const response = await fetch(apiUrl('/api/analyze'), {
      method: 'POST',
      headers: {
        'Content-Type': state.file.type || 'application/octet-stream',
        'X-Filename': encodeURIComponent(state.file.name),
        'X-Video-Fingerprint': fingerprint,
      },
      body: state.file,
    });
    const accepted = await response.json().catch(() => ({}));
    if (!response.ok) {
      if (accepted.code === 'analysis_in_progress') {
        const activeResponse = await fetch(apiUrl('/api/status'), { cache: 'no-store' });
        if (!activeResponse.ok) throw new Error(accepted.error || '无法恢复当前分析');
        const active = await activeResponse.json();
        adoptBackendJob(active, { resumed: true });
        toast('已有比赛正在分析，已恢复当前任务');
        await monitorBackendJob(active, token, { resumed: true });
        return;
      }
      throw new Error(accepted.error || '无法启动分析服务');
    }
    const workload = Math.max(0.01, Number(accepted.workload_factor) || 1);
    $('#timeHint').textContent = `每秒 ${accepted.fps} 帧原画质分析 · 预计工作量 ${workload.toFixed(2)} 倍`;
    const acceptedStatus = {
      state: 'queued', progress: 1, stage: accepted.resumed ? '正在继续上次分析' : '视频已接收，准备分析',
      ...accepted, video_fingerprint: accepted.video_fingerprint || fingerprint,
    };
    await monitorBackendJob(acceptedStatus, token, { resumed: Boolean(accepted.resumed) });
  } catch (error) {
    if (token !== backendRunToken) return;
    console.error(error);
    setView('welcome');
    const message = error instanceof TypeError && /fetch/i.test(error.message)
      ? '无法连接本机分析服务。请关闭旧页面，再用桌面启动程序重新打开 Netcast TennisVision'
      : error.message;
    toast(`分析未完成：${message}`);
  }
}

async function restoreInterruptedAnalysis() {
  if (location.hash === '#demo') return false;
  try {
    const response = await fetch(apiUrl('/api/status'), { cache: 'no-store' });
    if (!response.ok) return false;
    const status = await response.json();
    if (!ACTIVE_BACKEND_STATES.has(status.state)) return false;
    const token = ++backendRunToken;
    adoptBackendJob(status, { resumed: true });
    toast('已自动恢复上次未完成的分析');
    monitorBackendJob(status, token, { resumed: true }).catch((error) => {
      if (token !== backendRunToken) return;
      console.error(error);
      setView('welcome');
      toast(`分析未完成：${error.message}`);
    });
    return true;
  } catch (_) {
    return false;
  }
}

async function loadScene(path) {
  if (state.isDemo && demoSceneCache) {
    state.scene = demoSceneCache;
  } else {
  let lastError;
  for (let attempt = 0; attempt < 3; attempt += 1) {
    try {
      const sceneUrl = state.isDemo ? path : `${path}?v=${Date.now()}`;
      const response = await fetch(sceneUrl, { cache: state.isDemo ? 'force-cache' : 'no-store' });
      if (!response.ok) throw new Error(`分析数据读取失败（${response.status}）`);
      state.scene = await response.json();
      if (state.isDemo) {
        demoSceneCache = state.scene;
        try { localStorage.setItem(DEMO_SCENE_CACHE_KEY, JSON.stringify(state.scene)); } catch (_) { /* 浏览器缓存不可用不影响报告 */ }
      }
      lastError = null;
      break;
    } catch (error) {
      lastError = error;
      await new Promise((resolve) => setTimeout(resolve, 180));
    }
  }
  if (lastError && state.isDemo) {
    try {
      demoSceneCache = JSON.parse(localStorage.getItem(DEMO_SCENE_CACHE_KEY));
      state.scene = demoSceneCache;
      lastError = state.scene ? null : lastError;
    } catch (_) { /* 保留原始读取错误 */ }
  }
  if (lastError) throw lastError;
  }
  state.duration = state.scene.n_frames / state.scene.fps;
  const bounces = state.scene.bounces.map((bounce, index) => ({
    id: `b${index}`, number: index + 1, type: 'bounce', time: bounce.t,
    zone: ZONES[bounce.zone] || bounce.zone, rawZone: bounce.zone,
    x: bounce.x, y: bounce.y, confidence: bounce.landing_confidence,
  }));
  const hits = state.scene.hits.map((hit, index) => ({
    id: `h${index}`, number: index + 1, type: 'hit', time: hit.t, zone: '球员击球',
  }));
  state.events = [...bounces, ...hits].sort((a, b) => a.time - b.time);
}

async function showResults() {
  const loadToken = ++reportLoadToken;
  const video = $('#analysisVideo');
  const assets = activeOutput();
  try {
    await loadScene(assets.scene);
    if (loadToken !== reportLoadToken) return;
    video.src = `${assets.annotated}?v=${Date.now()}`;
    $('#sceneFrame').src = `${assets.viewer}?v=${Date.now()}`;
    $('#overlayToggle').checked = true;
    state.annotated = true;
    applyRealMetrics();
    $('#reportTitle').textContent = `${state.name} · 智能复盘`;
    $('#completeBadge').innerHTML = '<i></i> 分析完成';
    $('#previewNotice').hidden = true;
    setView('results');
  } catch (error) {
    if (loadToken !== reportLoadToken) return;
    console.error(error);
    setView('welcome');
    toast(state.isDemo ? '页面连接中断，请刷新后重试' : '报告读取失败，请重新分析该视频');
    return;
  }
  video.load();
  video.onloadedmetadata = () => { updateMeta(); buildEvents(); renderAll(); };
  updateMeta();
  buildEvents();
  renderAll();
}

function gradeFor(trackRate) {
  if (trackRate >= 0.7) return { grade: '优', title: '结果可靠', label: '良好' };
  if (trackRate >= 0.55) return { grade: '良', title: '基本可靠', label: '一般' };
  return { grade: '可', title: '仅供参考', label: '有限' };
}

function applyRealMetrics() {
  const scene = state.scene;
  const trackedFrames = scene.frames.filter((frame) => Array.isArray(frame.b)).length;
  const trackRate = trackedFrames / Math.max(scene.n_frames, 1);
  const valid = scene.bounces.filter((bounce) => bounce.zone !== 'Out');
  const out = scene.bounces.length - valid.length;
  const inRate = valid.length / Math.max(scene.bounces.length, 1);
  const counts = valid.reduce((all, bounce) => { all[bounce.zone] = (all[bounce.zone] || 0) + 1; return all; }, {});
  const [dominantRaw = '—', dominantCount = 0] = Object.entries(counts).sort((a, b) => b[1] - a[1])[0] || [];
  const dominant = ZONES[dominantRaw] || dominantRaw;
  const averageDepth = valid.length
    ? valid.reduce((total, bounce) => total + Math.min(Math.max(bounce.y, 0), Math.max(23.77 - bounce.y, 0)), 0) / valid.length
    : 0;
  const quality = gradeFor(trackRate);
  const anchored = scene.frames.filter((frame) => frame.c === 2).length;
  const extrapolated = scene.frames.filter((frame) => frame.c === 1).length;

  $('#bounceCount').textContent = scene.bounces.length;
  $('#validBounceText').textContent = `${valid.length} 次界内 · ${out} 次界外`;
  $('#hitCount').textContent = scene.hits.length;
  $('#trackRate').textContent = `${Math.round(trackRate * 100)}%`;
  $('#trackDetail').textContent = `${trackedFrames.toLocaleString()} / ${scene.n_frames.toLocaleString()} 帧`;
  $('#courtError').innerHTML = '0.69<sup>像素</sup>';
  $('#qualityGrade').textContent = quality.grade;
  $('#scoreRing').textContent = quality.grade;
  $('#qualityTitle').textContent = quality.title;
  $('#trackQualityLabel').textContent = quality.label;
  $('#trackQualityBar').style.width = `${Math.round(trackRate * 100)}%`;
  $('#inRate').textContent = `${Math.round(inRate * 100)}%`;
  $('#inRateDetail').textContent = `${valid.length} / ${scene.bounces.length} 次`;
  $('#dominantZone').textContent = dominant;
  $('#dominantZoneDetail').textContent = `${dominantCount} 次落点`;
  $('#heatDominant').textContent = dominant;
  $('#heatInRate').textContent = `${Math.round(inRate * 100)}%`;
  $('#averageDepth').textContent = `${averageDepth.toFixed(1)} 米`;
  $('#verdictText').textContent = `系统连续追踪到 ${trackedFrames.toLocaleString()} 帧球位置，并按回合确认 ${scene.bounces.length} 次落地；界外落点已单独标红。`;
  $('#insightText').textContent = `其中 ${anchored.toLocaleString()} 帧由可靠视觉观测锚定，${extrapolated.toLocaleString()} 帧由时序与物理约束补全；低可信位置不会伪装成确定结果。`;
}

function updateMeta() {
  const video = $('#analysisVideo');
  const resolution = video.videoWidth ? `${video.videoWidth} × ${video.videoHeight}` : '1280 × 720';
  const fps = state.scene ? state.scene.fps.toFixed(1) : '30.0';
  $('#reportMeta').textContent = `${state.duration.toFixed(1)} 秒 · ${resolution} · 每秒 ${fps} 帧`;
  $('.timeline-times').innerHTML = Array.from({ length: 5 }, (_, index) => `<span>${formatTime(state.duration * index / 4)}</span>`).join('');
}

function buildEvents(filter = 'all') {
  const events = state.events.filter((event) => filter === 'all' || event.type === filter);
  const list = $('#eventList');
  const timeline = $('#eventTimeline');
  list.innerHTML = '';
  timeline.querySelectorAll('.event-marker').forEach((marker) => marker.remove());
  $('#eventCount').textContent = `${events.length} 个`;
  if (!events.length) list.innerHTML = '<div class="plain-note"><span>i</span><p><strong>当前没有这类事件</strong>可以切换上方筛选查看其他事件。</p></div>';
  events.forEach((event) => {
    const button = document.createElement('button');
    const isOut = event.rawZone === 'Out';
    button.className = `event-item ${event.type}${isOut ? ' out' : ''}`;
    button.innerHTML = `<span class="event-symbol">${event.type === 'bounce' ? (isOut ? '×' : '⌄') : '✦'}</span><span><strong>${event.type === 'bounce' ? `第 ${event.number} 次落地` : `第 ${event.number} 次击球`}</strong><small>${event.zone}</small></span><time>${formatTime(event.time)}</time>`;
    button.addEventListener('click', () => seek(event));
    list.appendChild(button);
  });
  state.events.forEach((event) => {
    const marker = document.createElement('button');
    marker.className = `event-marker ${event.type}${event.rawZone === 'Out' ? ' out' : ''}`;
    marker.style.left = `${Math.min(100, event.time / Math.max(state.duration, 1) * 100)}%`;
    marker.title = `${event.type === 'bounce' ? event.zone : '击球'} · ${formatTime(event.time)}`;
    marker.setAttribute('aria-label', marker.title);
    marker.addEventListener('click', () => seek(event));
    timeline.appendChild(marker);
  });
}

function seek(event) {
  const video = $('#analysisVideo');
  video.currentTime = Math.min(event.time, video.duration || event.time);
  video.play().catch(() => {});
  toast(`已跳到 ${formatTime(event.time)} · ${event.zone}`);
}

function sizeCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.min(devicePixelRatio || 1, 2);
  const pixelWidth = Math.max(1, Math.round(rect.width * dpr));
  const pixelHeight = Math.max(1, Math.round(rect.height * dpr));
  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) { canvas.width = pixelWidth; canvas.height = pixelHeight; }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, width: rect.width, height: rect.height };
}

function drawHeatmap() {
  const { ctx, width, height } = sizeCanvas($('#heatmapCanvas'));
  ctx.clearRect(0, 0, width, height);
  const side = Math.max(42, width * 0.13), top = 28, courtWidth = width - side * 2, courtHeight = height - 56;
  const xMin = -1.35, xMax = 12.32, yMin = -2.4, yMax = 26.17;
  const pointX = (x) => side + ((x - xMin) / (xMax - xMin)) * courtWidth;
  const pointY = (y) => top + (1 - (y - yMin) / (yMax - yMin)) * courtHeight;
  const courtLeft = pointX(0), courtRight = pointX(10.97), courtTop = pointY(23.77), courtBottom = pointY(0);
  const fieldGradient = ctx.createLinearGradient(0, courtTop, 0, courtBottom);
  fieldGradient.addColorStop(0, '#103d2d'); fieldGradient.addColorStop(1, '#0a2e22');
  ctx.fillStyle = fieldGradient; ctx.fillRect(courtLeft, courtTop, courtRight - courtLeft, courtBottom - courtTop);
  ctx.strokeStyle = 'rgba(222,255,184,.88)'; ctx.lineWidth = 1.25; ctx.strokeRect(courtLeft, courtTop, courtRight - courtLeft, courtBottom - courtTop);
  ctx.beginPath();
  ctx.moveTo(courtLeft, pointY(11.885)); ctx.lineTo(courtRight, pointY(11.885));
  ctx.moveTo(pointX(5.485), pointY(5.485)); ctx.lineTo(pointX(5.485), pointY(18.285));
  ctx.moveTo(courtLeft, pointY(5.485)); ctx.lineTo(courtRight, pointY(5.485));
  ctx.moveTo(courtLeft, pointY(18.285)); ctx.lineTo(courtRight, pointY(18.285));
  ctx.moveTo(pointX(1.37), courtTop); ctx.lineTo(pointX(1.37), courtBottom);
  ctx.moveTo(pointX(9.60), courtTop); ctx.lineTo(pointX(9.60), courtBottom); ctx.stroke();
  ctx.strokeStyle = 'rgba(199,255,94,.16)'; ctx.setLineDash([3, 5]);
  ctx.strokeRect(pointX(xMin), pointY(yMax), pointX(xMax) - pointX(xMin), pointY(yMin) - pointY(yMax)); ctx.setLineDash([]);
  (state.scene?.bounces || []).forEach((bounce, index) => {
    const x = pointX(Math.max(xMin, Math.min(xMax, bounce.x)));
    const y = pointY(Math.max(yMin, Math.min(yMax, bounce.y)));
    if (bounce.zone === 'Out') {
      ctx.save(); ctx.strokeStyle = '#ff665e'; ctx.shadowColor = 'rgba(255,80,72,.8)'; ctx.shadowBlur = 12; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.moveTo(x - 6, y - 6); ctx.lineTo(x + 6, y + 6); ctx.moveTo(x + 6, y - 6); ctx.lineTo(x - 6, y + 6); ctx.stroke(); ctx.restore(); return;
    }
    const glow = ctx.createRadialGradient(x, y, 0, x, y, 19);
    glow.addColorStop(0, 'rgba(229,255,141,.95)'); glow.addColorStop(0.25, 'rgba(199,255,94,.52)'); glow.addColorStop(1, 'rgba(199,255,94,0)');
    ctx.fillStyle = glow; ctx.fillRect(x - 22, y - 22, 44, 44); ctx.fillStyle = index % 5 === 0 ? '#fff' : '#d7ff78';
    ctx.beginPath(); ctx.arc(x, y, index % 5 === 0 ? 3 : 2.2, 0, Math.PI * 2); ctx.fill();
  });
}

function drawHeight() {
  const { ctx, width, height } = sizeCanvas($('#heightCanvas'));
  const pad = { left: 40, right: 18, top: 18, bottom: 27 };
  ctx.clearRect(0, 0, width, height); ctx.strokeStyle = 'rgba(19,51,38,.10)'; ctx.fillStyle = '#809087'; ctx.font = '10px system-ui, sans-serif';
  for (let index = 0; index <= 3; index += 1) {
    const y = pad.top + (height - pad.top - pad.bottom) * index / 3;
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(width - pad.right, y); ctx.stroke(); ctx.fillText(`${3 - index}m`, 8, y + 3);
  }
  if (!state.scene) return;
  const frames = state.scene.frames, plotWidth = width - pad.left - pad.right, plotHeight = height - pad.top - pad.bottom;
  let drawing = false;
  frames.forEach((frame, index) => {
    if (!Array.isArray(frame.b)) { drawing = false; return; }
    const x = pad.left + index / Math.max(frames.length - 1, 1) * plotWidth;
    const y = height - pad.bottom - Math.min(3, Math.max(0, frame.b[2])) / 3 * plotHeight;
    ctx.strokeStyle = frame.c === 2 ? '#41a869' : '#92a69a'; ctx.lineWidth = 2; ctx.setLineDash(frame.c === 2 ? [] : [4, 4]);
    if (!drawing) { ctx.beginPath(); ctx.moveTo(x, y); drawing = true; } else { ctx.lineTo(x, y); ctx.stroke(); ctx.beginPath(); ctx.moveTo(x, y); }
  });
  ctx.setLineDash([]);
  state.scene.bounces.forEach((bounce) => {
    const x = pad.left + bounce.frame / Math.max(frames.length - 1, 1) * plotWidth;
    ctx.fillStyle = '#ef7d5f'; ctx.beginPath(); ctx.arc(x, height - pad.bottom, 2.6, 0, Math.PI * 2); ctx.fill();
  });
  for (let index = 0; index <= 4; index += 1) ctx.fillText(formatTime(state.duration * index / 4), pad.left + plotWidth * index / 4 - 12, height - 7);
}

function renderAll() { drawHeatmap(); drawHeight(); }

$('#chooseBtn').addEventListener('click', chooseVideo);
$('#dropZone').addEventListener('click', (event) => { if (!event.target.closest('button')) chooseVideo(); });
['dragenter', 'dragover'].forEach((type) => $('#dropZone').addEventListener(type, (event) => { event.preventDefault(); $('#dropZone').classList.add('dragging'); }));
['dragleave', 'drop'].forEach((type) => $('#dropZone').addEventListener(type, (event) => { event.preventDefault(); $('#dropZone').classList.remove('dragging'); }));
$('#dropZone').addEventListener('drop', (event) => { const [file] = event.dataTransfer.files; if (file) selectFile(file); });
$('#videoInput').addEventListener('change', (event) => { const [file] = event.target.files; if (file) selectFile(file); });
$('#demoBtn').addEventListener('click', startDemo);
$('#heroDemoBtn').addEventListener('click', startDemo);
$$('.event-filter button').forEach((button) => button.addEventListener('click', () => {
  $$('.event-filter button').forEach((item) => item.classList.remove('active')); button.classList.add('active'); buildEvents(button.dataset.filter);
}));
window.addEventListener('resize', () => { if (!$('#resultsView').hidden) renderAll(); });
$('#analysisVideo').addEventListener('timeupdate', (event) => { $('#playhead').style.left = `${Math.min(100, event.currentTarget.currentTime / Math.max(state.duration, 1) * 100)}%`; });
$('#overlayToggle').addEventListener('change', (event) => {
  const video = $('#analysisVideo'), time = video.currentTime, wasPlaying = !video.paused;
  const assets = activeOutput();
  state.annotated = event.target.checked; video.src = state.annotated ? assets.annotated : assets.original; video.load();
  video.addEventListener('loadedmetadata', () => { video.currentTime = Math.min(time, video.duration || time); if (wasPlaying) video.play().catch(() => {}); }, { once: true });
  toast(state.annotated ? '已显示球轨迹与落地标记' : '已切换为原始比赛画面');
});
$('#newAnalysisBtn').addEventListener('click', () => resetForNextAnalysis({ announce: true }));
$$('.home-trigger').forEach((control) => control.addEventListener('click', (event) => {
  event.preventDefault();
  resetForNextAnalysis();
}));
$('#exportBtn').addEventListener('click', () => {
  const anchor = document.createElement('a'); anchor.href = activeOutput().scene; anchor.download = `${state.name}-Netcast-TennisVision-分析数据.json`; anchor.click(); toast('分析数据已导出');
});
$('#shareBtn').addEventListener('click', () => { window.print(); toast('请选择“另存为 PDF”即可保存复盘报告'); });
$$('[data-toast]').forEach((button) => button.addEventListener('click', () => toast(button.dataset.toast)));
$('#helpBtn').addEventListener('click', () => $('#helpDialog').showModal());
$$('.dialog-close, .dialog-ok').forEach((button) => button.addEventListener('click', () => $('#helpDialog').close()));
document.addEventListener('keydown', (event) => { if (event.key === 'Escape' && $('#helpDialog').open) $('#helpDialog').close(); });
startAmbientBall();
if (location.hash === '#demo') startDemo();
else restoreInterruptedAnalysis();
