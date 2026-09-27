// Presentation only. The shared Python classifier owns all thresholds.
globalThis.showCourtViewpoint = async function(element, corners, width, height, saved = null) {
  if (!element) return;
  if (saved?.version !== 2) saved = null;
  const ticket = {};
  element.viewpointTicket = ticket;
  if (!saved && (!corners || corners.length !== 4 || !width || !height)) {
    element.textContent = '视角：等待球场数据';
    return;
  }
  element.textContent = '正在判断球场视角…';
  try {
    let result = saved;
    if (!result) {
      const base = location.protocol === 'file:' ? 'http://127.0.0.1:4173' : '';
      const response = await fetch(`${base}/api/viewpoint`, {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({corners, width, height}),
      });
      if (!response.ok) throw new Error('viewpoint unavailable');
      result = await response.json();
    }
    if (element.viewpointTicket !== ticket) return;
    const label = {low: '平视角', high: '高视角'}[result.category] || '暂无数据';
    const ratio = Number.isFinite(result.height_ratio) ? ` · 底线间距 ${(result.height_ratio * 100).toFixed(1)}%` : '';
    element.textContent = `视角：${label}${ratio} · 沿用现有分析`;
    element.title = '根据原视频球场四角判断，不包含标注留白。仅作分类，不改变识别流程。';
  } catch (_) {
    if (element.viewpointTicket === ticket) element.textContent = '视角：暂时无法判断 · 沿用现有分析';
  }
};
