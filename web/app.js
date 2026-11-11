/* RouteShield dispatcher console — the full model runs client-side.
   No build step, no framework, no CDN dependency. Basemap tiles are an
   optional enhancement; everything works offline without them. */
'use strict';

// ---------------------------------------------------------------- utilities
const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));
const fmt = (x, d = 2) => (x === null || x === undefined || !isFinite(x)) ? '—' : Number(x).toFixed(d);
const pct = (x, d = 0) => fmt(100 * x, d) + '%';
const esc = (s) => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

const RISK_RAMP = ['#7a4a1f', '#9c5a20', '#bd6f23', '#dd8a2e', '#f0a94e', '#ffc978'];
const SERIES = ['#3987e5', '#d95926', '#199e70', '#c98500'];
const STATUS = { good: '#0ca30c', warning: '#fab219', serious: '#ec835a', critical: '#d03b3b' };

let DATA = null, GRAPH = null, BANDS = null;

// ----------------------------------------------------------- math: sampling
function randn() {
  let u = 0, v = 0;
  while (!u) u = Math.random();
  while (!v) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
/* Marsaglia–Tsang gamma sampler — the posterior over each segment's rate is
   Gamma(shape, 1/rate), so route-level uncertainty is a Monte Carlo sum. */
function gammaSample(shape, scale) {
  if (shape < 1) return gammaSample(shape + 1, scale) * Math.pow(Math.random(), 1 / shape);
  const d = shape - 1 / 3, c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let x, v;
    do { x = randn(); v = 1 + c * x; } while (v <= 0);
    v = v * v * v;
    const u = Math.random();
    if (u < 1 - 0.0331 * x * x * x * x) return d * v * scale;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v * scale;
  }
}
function quantile(sorted, q) {
  const i = (sorted.length - 1) * q, lo = Math.floor(i), hi = Math.ceil(i);
  return lo === hi ? sorted[lo] : sorted[lo] + (sorted[hi] - sorted[lo]) * (i - lo);
}

// --------------------------------------------------------- context (India)
const HOUR_MULT = (() => {
  const m = {};
  for (let h = 0; h < 24; h++) m[h] = 1.0;
  [22, 23, 0, 1, 2, 3, 4].forEach(h => m[h] = 1.55);
  [5, 21].forEach(h => m[h] = 1.20);
  [8, 9, 17, 18].forEach(h => m[h] = 1.15);
  m[19] = 1.25;
  return m;
})();
const WEATHER_MULT = { clear: 1.0, cloudy: 1.02, rain: 1.30, heavy_rain: 1.55, fog: 1.60, thunderstorm: 1.45 };
const COHORT_MULT = { none: 1.0, tenure_lt_1y: 1.45, tenure_1_3y: 1.00, tenure_gt_3y: 0.78 };

function contextParts() {
  const h = +$('#hour').value, w = $('#weather').value, c = $('#cohort').value;
  const parts = [
    { factor: 'Time of day', detail: String(h).padStart(2, '0') + ':00' + (h >= 22 || h < 5 ? ' (night window)' : ''), mult: HOUR_MULT[h] },
    { factor: 'Weather', detail: w.replace('_', ' '), mult: WEATHER_MULT[w] },
  ];
  if (c !== 'none') parts.push({ factor: 'Driver cohort', detail: c.replace(/_/g, ' '), mult: COHORT_MULT[c] });
  return parts;
}
const contextMultiplier = () => contextParts().reduce((a, p) => a * p.mult, 1);

// ------------------------------------------------------------------- graph
function buildGraph(d) {
  const n = d.nodes.length;
  const adj = Array.from({ length: n }, () => []);
  d.segments.forEach(s => adj[s.u].push([s.v, s.id]));
  const time = d.segments.map(s => s.min);
  const risk = d.segments.map(s => s.risk);
  return { adj, time, risk, n };
}

class Heap {
  constructor() { this.a = []; }
  push(x) { const a = this.a; a.push(x); let i = a.length - 1; while (i > 0) { const p = (i - 1) >> 1; if (a[p][0] <= a[i][0]) break; [a[p], a[i]] = [a[i], a[p]]; i = p; } }
  pop() { const a = this.a, top = a[0], last = a.pop(); if (a.length) { a[0] = last; let i = 0; for (;;) { const l = 2 * i + 1, r = l + 1; let m = i; if (l < a.length && a[l][0] < a[m][0]) m = l; if (r < a.length && a[r][0] < a[m][0]) m = r; if (m === i) break; [a[m], a[i]] = [a[i], a[m]]; i = m; } } return top; }
  get size() { return this.a.length; }
}

function dijkstra(cost, s, t) {
  const { adj, n } = GRAPH;
  const dist = new Float64Array(n).fill(Infinity);
  const prevN = new Int32Array(n).fill(-1), prevE = new Int32Array(n).fill(-1);
  dist[s] = 0;
  const pq = new Heap(); pq.push([0, s]);
  while (pq.size) {
    const [d, x] = pq.pop();
    if (d > dist[x]) continue;
    if (x === t) break;
    for (const [y, e] of adj[x]) {
      const nd = d + cost[e];
      if (nd < dist[y]) { dist[y] = nd; prevN[y] = x; prevE[y] = e; pq.push([nd, y]); }
    }
  }
  if (!isFinite(dist[t])) return null;
  const path = []; let x = t;
  while (prevN[x] !== -1) { path.push(prevE[x]); x = prevN[x]; }
  return path.reverse();
}

const LAMBDAS = [0, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 8, 13, 21, 34];

function candidateRoutes(s, t, mult) {
  const { time, risk } = GRAPH;
  const meanT = time.reduce((a, b) => a + b, 0) / time.length;
  const meanR = Math.max(risk.reduce((a, b) => a + b, 0) / risk.length, 1e-12);
  const scale = meanT / meanR;
  const seen = new Set(), out = [];
  const cost = new Float64Array(time.length);
  for (const lam of LAMBDAS) {
    for (let i = 0; i < time.length; i++) cost[i] = time[i] + lam * scale * risk[i] * mult;
    const r = dijkstra(cost, s, t);
    if (!r) continue;
    const key = r.join(',');
    if (!seen.has(key)) { seen.add(key); out.push(r); }
  }
  return out;
}

function routeStats(edges, mult) {
  const segs = DATA.segments;
  let timeMin = 0, km = 0, riskMean = 0;
  for (const e of edges) { timeMin += segs[e].min; km += segs[e].km; riskMean += segs[e].risk * mult; }
  return { edges, timeMin, km, riskMean };
}

function paretoFrontier(routes, mult) {
  const rows = routes.map(r => routeStats(r, mult));
  const keep = rows.filter(x => !rows.some(y =>
    y !== x && y.timeMin <= x.timeMin && y.riskMean <= x.riskMean &&
    (y.timeMin < x.timeMin || y.riskMean < x.riskMean)));
  return keep.sort((a, b) => a.timeMin - b.timeMin);
}

function recommend(front, budget, minCut) {
  const base = front[0];
  const affordable = front.filter(x => x.timeMin <= base.timeMin * (1 + budget));
  let best = affordable[0];
  for (const x of affordable) if (x.riskMean < best.riskMean) best = x;
  if (best === base) return { base, best: base, reason: 'the fastest route is already the safest available' };
  const cut = 1 - best.riskMean / Math.max(base.riskMean, 1e-12);
  if (cut >= minCut) {
    const dt = best.timeMin - base.timeMin;
    return { base, best, reason: `${pct(cut)} lower risk for ${dt >= 0 ? '+' : ''}${fmt(dt, 1)} min (${pct(dt / base.timeMin)} time)` };
  }
  return { base, best: base, reason: `no alternative cut risk by ${pct(minCut)} or more inside the time budget` };
}

/* Route risk with a credible interval. Summing per-segment means is exact;
   summing per-segment intervals is not, so the interval is sampled. */
function scoreRoute(edges, mult, nSamples = 1200) {
  const segs = DATA.segments;
  const totals = new Float64Array(nSamples);
  const shapes = [], scales = [];
  for (const e of edges) {
    const s = segs[e];
    // posterior mean = risk; cv recovered from the stored 90% interval
    const mean = Math.max(s.risk, 1e-12);
    const width = Math.max(s.hi - s.lo, 1e-12);
    const cv = Math.max(width / (3.29 * mean), 1e-3);     // ~2*1.645 sd for 90%
    const shape = 1 / (cv * cv);
    shapes.push(shape); scales.push(mean / shape);
  }
  for (let i = 0; i < nSamples; i++) {
    let tot = 0;
    for (let k = 0; k < shapes.length; k++) tot += gammaSample(shapes[k], scales[k]);
    totals[i] = tot * mult;
  }
  const sorted = Array.from(totals).sort((a, b) => a - b);
  const mean = sorted.reduce((a, b) => a + b, 0) / sorted.length;
  const lo = quantile(sorted, 0.05), hi = quantile(sorted, 0.95);
  const rel = (hi - lo) / Math.max(mean, 1e-12);
  const stats = routeStats(edges, mult);
  const confCount = { High: 0, Medium: 0, Low: 0 };
  edges.forEach(e => confCount[segs[e].conf]++);
  const worst = edges.map(e => ({ e, r: segs[e].risk * mult }))
    .sort((a, b) => b.r - a.r).slice(0, 5);
  return {
    edges, mean, lo, hi, rel,
    confidence: rel < 0.5 ? 'High' : rel < 1.0 ? 'Medium' : 'Low',
    timeMin: stats.timeMin, km: stats.km,
    per100: mean / Math.max(stats.km, 1e-9) * 100,
    confCount, worst, totalRisk: stats.riskMean,
  };
}

/* Exact additive attribution. For the log-linear prior the Shapley value of
   feature j is beta_j * (x_j - mean_j) in closed form — no approximation. */
function attribution(edgeIds) {
  const names = DATA.meta.feature_names, beta = DATA.meta.beta.slice(1);
  const segs = DATA.segments;
  const featOf = (s) => [s.curv, s.junc, 1 - s.lit, s.grad, s.speed / 100, 2];
  const mean = names.map((_, j) => {
    let t = 0; for (const s of segs) t += featOf(s)[j];
    return t / segs.length;
  });
  const totalKm = edgeIds.reduce((a, e) => a + segs[e].km, 0) || 1;
  const contrib = names.map((_, j) => {
    let t = 0;
    for (const e of edgeIds) t += beta[j] * (featOf(segs[e])[j] - mean[j]) * segs[e].km;
    return t / totalKm;
  });
  const LABELS = {
    curvature: 'Road curvature', junction_density: 'Junction density',
    unlit: 'No street lighting', gradient_pct: 'Gradient / ghat',
    maxspeed_100: 'Speed limit', lanes: 'Lane count',
  };
  return names.map((n, j) => ({ name: LABELS[n] || n, value: contrib[j] }))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
}

// -------------------------------------------------------------------- map
const MapView = {
  canvas: null, ctx: null, w: 0, h: 0,
  centre: { lat: 12.95, lon: 77.62 }, zoom: 10.4,
  tiles: new Map(), tilesOn: true, hover: null, routeEdges: null, altEdges: null,
  drag: null,

  init() {
    this.canvas = $('#map');
    this.ctx = this.canvas.getContext('2d');
    this.resize();
    window.addEventListener('resize', () => { this.resize(); this.draw(); });
    this.canvas.addEventListener('mousemove', e => this.onMove(e));
    this.canvas.addEventListener('mouseleave', () => { this.hover = null; $('#tooltip').style.display = 'none'; this.draw(); });
    this.canvas.addEventListener('mousedown', e => { this.drag = { x: e.clientX, y: e.clientY, c: { ...this.centre } }; });
    window.addEventListener('mouseup', () => { this.drag = null; });
    window.addEventListener('mousemove', e => {
      if (!this.drag) return;
      const s = this.scale();
      const dx = e.clientX - this.drag.x, dy = e.clientY - this.drag.y;
      this.centre.lon = this.drag.c.lon - dx / s;
      this.centre.lat = this.drag.c.lat + dy / (s * Math.cos(this.centre.lat * Math.PI / 180));
      this.draw();
    });
    this.canvas.addEventListener('wheel', e => {
      e.preventDefault();
      this.zoom = Math.max(8.5, Math.min(14, this.zoom - Math.sign(e.deltaY) * 0.35));
      this.draw();
    }, { passive: false });
    $('#zin').onclick = () => { this.zoom = Math.min(14, this.zoom + 0.6); this.draw(); };
    $('#zout').onclick = () => { this.zoom = Math.max(8.5, this.zoom - 0.6); this.draw(); };
    $('#zfit').onclick = () => { this.fit(); };
    $('#tiletoggle').onclick = (e) => {
      this.tilesOn = !this.tilesOn;
      e.target.classList.toggle('on', this.tilesOn);
      this.draw();
    };
  },
  resize() {
    const r = this.canvas.parentElement.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.w = r.width; this.h = r.height;
    this.canvas.width = r.width * dpr; this.canvas.height = r.height * dpr;
    this.canvas.style.width = r.width + 'px'; this.canvas.style.height = r.height + 'px';
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  },
  scale() { return 256 * Math.pow(2, this.zoom) / 360; },
  project(lat, lon) {
    const s = this.scale();
    const x = this.w / 2 + (lon - this.centre.lon) * s;
    const k = Math.cos(this.centre.lat * Math.PI / 180);
    const y = this.h / 2 - (lat - this.centre.lat) * s * k;
    return [x, y];
  },
  fit() {
    const segs = DATA.segments;
    // Fit to the bulk of the network, not its extremes — a couple of distant
    // rural endpoints would otherwise zoom the city out to a dot.
    const lats = [], lons = [];
    segs.forEach(s => s.geom.forEach(([a, b]) => { lats.push(a); lons.push(b); }));
    lats.sort((a, b) => a - b); lons.sort((a, b) => a - b);
    const minLat = quantile(lats, 0.02), maxLat = quantile(lats, 0.98);
    const minLon = quantile(lons, 0.02), maxLon = quantile(lons, 0.98);
    this.centre = { lat: (minLat + maxLat) / 2, lon: (minLon + maxLon) / 2 };
    const k = Math.cos(this.centre.lat * Math.PI / 180);
    const zx = Math.log2(this.w * 360 / (256 * (maxLon - minLon) * 1.1));
    const zy = Math.log2(this.h * 360 / (256 * (maxLat - minLat) * k * 1.15));
    this.zoom = Math.max(8.5, Math.min(14, Math.min(zx, zy)));
    this.draw();
  },

  // optional raster basemap: enhances the view online, absent offline
  drawTiles() {
    if (!this.tilesOn) return;
    const z = Math.max(0, Math.min(16, Math.round(this.zoom)));
    const n = Math.pow(2, z);
    const lon2x = (lon) => (lon + 180) / 360 * n;
    const lat2y = (lat) => {
      const r = lat * Math.PI / 180;
      return (1 - Math.log(Math.tan(r) + 1 / Math.cos(r)) / Math.PI) / 2 * n;
    };
    const x2lon = (x) => x / n * 360 - 180;
    const y2lat = (y) => { const m = Math.PI * (1 - 2 * y / n); return 180 / Math.PI * Math.atan(Math.sinh(m)); };
    const tl = [this.unproject(0, 0), this.unproject(this.w, this.h)];
    const x0 = Math.floor(lon2x(tl[0][1])), x1 = Math.ceil(lon2x(tl[1][1]));
    const y0 = Math.floor(lat2y(tl[0][0])), y1 = Math.ceil(lat2y(tl[1][0]));
    for (let x = x0; x <= x1; x++) {
      for (let y = y0; y <= y1; y++) {
        if (x < 0 || y < 0 || x >= n || y >= n) continue;
        const key = `${z}/${x}/${y}`;
        let img = this.tiles.get(key);
        if (img === undefined) {
          img = new Image(); img.crossOrigin = 'anonymous';
          img.onload = () => this.draw();
          img.onerror = () => { this.tiles.set(key, null); this.tileFail = (this.tileFail || 0) + 1;
            if (this.tileFail > 6 && this.tilesOn) {   // offline: stop asking
              this.tilesOn = false; $('#tiletoggle').classList.remove('on');
              $('#tiletoggle').title = 'Basemap unavailable offline — the network still renders';
              this.draw();
            } };
          img.src = `https://basemaps.cartocdn.com/dark_all/${z}/${x}/${y}.png`;
          this.tiles.set(key, img);
        }
        if (!img || !img.complete || !img.naturalWidth) continue;
        const [px, py] = this.project(y2lat(y), x2lon(x));
        const [qx, qy] = this.project(y2lat(y + 1), x2lon(x + 1));
        this.ctx.globalAlpha = 0.55;
        this.ctx.drawImage(img, px, py, qx - px, qy - py);
        this.ctx.globalAlpha = 1;
      }
    }
  },
  unproject(px, py) {
    const s = this.scale(), k = Math.cos(this.centre.lat * Math.PI / 180);
    return [this.centre.lat - (py - this.h / 2) / (s * k), this.centre.lon + (px - this.w / 2) / s];
  },

  riskColour(per100) {
    const i = BANDS.findIndex(b => per100 <= b);
    return RISK_RAMP[i === -1 ? RISK_RAMP.length - 1 : i];
  },

  draw() {
    const c = this.ctx;
    c.fillStyle = '#0f1013';
    c.fillRect(0, 0, this.w, this.h);
    this.drawTiles();

    const segs = DATA.segments;
    const show = $('#layer').value;

    // base network
    for (const s of segs) {
      if (!s.disp) continue;
      let col, width;
      if (show === 'risk') {
        const band = BANDS.findIndex(b => s.per100 <= b);
        const k = band === -1 ? RISK_RAMP.length - 1 : band;
        col = RISK_RAMP[k];
        width = 1.3 + k * 0.72;                       // thicker = riskier
      }
      else if (show === 'confidence') { col = { High: '#2f6f4f', Medium: '#8a6a22', Low: '#8c3b3b' }[s.conf]; width = 2.2; }
      else { col = '#3a3f47'; width = 1.8; }
      this.poly(s.geom, col, Math.min(width, 5), 0.92);
    }

    // hotspots
    if ($('#showHotspots').checked) {
      // Cap and de-overlap: leaf clustering can place several spots within a
      // few hundred metres, and stacked rings read as scribble, not data.
      const drawn = [];
      const maxScore = Math.max(...DATA.hotspots.map(h => h.weighted_score), 1);
      for (const h of DATA.hotspots.slice(0, 45)) {
        const [x, y] = this.project(h.lat, h.lon);
        if (x < -40 || y < -40 || x > this.w + 40 || y > this.h + 40) continue;
        if (drawn.some(d => Math.hypot(d[0] - x, d[1] - y) < 17)) continue;
        drawn.push([x, y]);
        if (drawn.length > 26) break;
        const r = 5 + 7 * Math.sqrt(h.weighted_score / maxScore);
        const worse = h.trend_label === 'worsening';
        c.beginPath(); c.arc(x, y, r, 0, 7);
        c.fillStyle = worse ? 'rgba(208,59,59,.22)' : 'rgba(250,178,25,.14)';
        c.fill();
        c.lineWidth = worse ? 1.8 : 1.2;
        c.strokeStyle = worse ? STATUS.critical : 'rgba(250,178,25,.75)';
        c.stroke();
      }
    }

    // alternative then chosen route
    if (this.altEdges) this.drawRoute(this.altEdges, '#8aa0b8', 3.2, [7, 5]);
    if (this.routeEdges) {
      this.drawRoute(this.routeEdges, '#ffffff', 6.5, null, 0.22);
      this.drawRoute(this.routeEdges, SERIES[0], 3.4, null);
      const first = segs[this.routeEdges[0]].geom[0];
      const lastSeg = segs[this.routeEdges[this.routeEdges.length - 1]].geom;
      this.marker(first, '#ffffff', 'A');
      this.marker(lastSeg[lastSeg.length - 1], SERIES[1], 'B');
    }

    if (this.hover !== null) {
      const s = segs[this.hover];
      this.poly(s.geom, '#ffffff', 5, 0.85);
    }
  },
  poly(geom, colour, width, alpha = 1, dash = null) {
    const c = this.ctx;
    c.save(); c.globalAlpha = alpha; c.strokeStyle = colour; c.lineWidth = width;
    c.lineJoin = 'round'; c.lineCap = 'round';
    if (dash) c.setLineDash(dash);
    c.beginPath();
    geom.forEach(([la, lo], i) => { const [x, y] = this.project(la, lo); i ? c.lineTo(x, y) : c.moveTo(x, y); });
    c.stroke(); c.restore();
  },
  drawRoute(edges, colour, width, dash, alpha = 1) {
    edges.forEach(e => this.poly(DATA.segments[e].geom, colour, width, alpha, dash));
  },
  marker([la, lo], colour, label) {
    const c = this.ctx, [x, y] = this.project(la, lo);
    c.beginPath(); c.arc(x, y, 8, 0, 7); c.fillStyle = colour; c.fill();
    c.lineWidth = 2; c.strokeStyle = '#0f1013'; c.stroke();
    c.fillStyle = '#0f1013'; c.font = 'bold 10px ui-sans-serif,system-ui';
    c.textAlign = 'center'; c.textBaseline = 'middle'; c.fillText(label, x, y + 0.5);
  },
  onMove(e) {
    const r = this.canvas.getBoundingClientRect();
    const px = e.clientX - r.left, py = e.clientY - r.top;
    let best = null, bestD = 14;
    for (const s of DATA.segments) {
      if (!s.disp) continue;
      for (let i = 0; i < s.geom.length - 1; i++) {
        const [ax, ay] = this.project(s.geom[i][0], s.geom[i][1]);
        const [bx, by] = this.project(s.geom[i + 1][0], s.geom[i + 1][1]);
        const vx = bx - ax, vy = by - ay, wx = px - ax, wy = py - ay;
        const den = vx * vx + vy * vy;
        let t = den > 0 ? (wx * vx + wy * vy) / den : 0;
        t = Math.max(0, Math.min(1, t));
        const dx = px - (ax + t * vx), dy = py - (ay + t * vy);
        const d = Math.hypot(dx, dy);
        if (d < bestD) { bestD = d; best = s.id; }
      }
    }
    if (best !== this.hover) { this.hover = best; this.draw(); }
    const tip = $('#tooltip');
    if (best === null) { tip.style.display = 'none'; return; }
    const s = DATA.segments[best];
    tip.innerHTML = `<b>${esc(s.name)}</b>
      <div class="trow"><span>Risk / 100 km</span><b>${fmt(s.per100, 3)}</b></div>
      <div class="trow"><span>90% interval</span><span>${fmt(s.lo / s.km * 100, 3)} – ${fmt(s.hi / s.km * 100, 3)}</span></div>
      <div class="trow"><span>Confidence</span><span class="badge b-${s.conf.toLowerCase()}">${s.conf}</span></div>
      <div class="trow"><span>Events (decayed)</span><span>${fmt(s.evt, 1)}</span></div>
      <div class="trow"><span>Exposure</span><span>${fmt(s.expo, 0)} veh-km</span></div>
      <div class="trow"><span>Road</span><span>${s.lit ? 'lit' : 'unlit'} · ${s.speed} km/h</span></div>`;
    tip.style.display = 'block';
    tip.style.left = Math.min(px + 16, this.w - 250) + 'px';
    tip.style.top = Math.min(py + 14, this.h - 170) + 'px';
  },
};

// ------------------------------------------------------------------ charts
function barChart(el, rows, { value, label, colour = SERIES[0], highlight = null, fmtV = (v) => fmt(v, 1) }) {
  const max = Math.max(...rows.map(value), 1e-9);
  el.innerHTML = rows.map(r => {
    const v = value(r), w = Math.max(1.5, 100 * v / max);
    const col = highlight && highlight(r) ? STATUS.warning : colour;
    return `<div class="bar-row" title="${esc(label(r))}: ${fmtV(v)}">
      <span class="bar-lab">${esc(label(r))}</span>
      <span class="bar-track"><span class="bar-fill" style="width:${w}%;background:${col}"></span></span>
      <span class="bar-val">${fmtV(v)}</span></div>`;
  }).join('');
}

function lineChart(el, series, { xkey, ykey, xlabel, ylabel, fmtY = (v) => fmt(v, 1) }) {
  const W = 520, H = 220, P = { t: 14, r: 16, b: 38, l: 52 };
  const n = series[0].rows.length;
  const ys = series.flatMap(s => s.rows.map(r => r[ykey]));
  const y0 = Math.min(0, ...ys), y1 = (Math.max(...ys) || 1) * 1.14;
  // x is categorical: position by index, label by value
  const px = i => n === 1 ? (P.l + W - P.r) / 2
    : P.l + i / (n - 1) * (W - P.l - P.r);
  const py = v => H - P.b - (v - y0) / Math.max(y1 - y0, 1e-9) * (H - P.t - P.b);
  let g = '';
  for (let i = 0; i <= 4; i++) {
    const v = y0 + (y1 - y0) * i / 4;
    g += `<line x1="${P.l}" x2="${W - P.r}" y1="${py(v).toFixed(1)}" y2="${py(v).toFixed(1)}" stroke="#262930" stroke-width="1"/>
          <text x="${P.l - 8}" y="${(py(v) + 4).toFixed(1)}" text-anchor="end" class="ax">${esc(fmtY(v))}</text>`;
  }
  const paths = series.map(s => {
    const d = s.rows.map((r, j) => `${j ? 'L' : 'M'}${px(j).toFixed(1)},${py(r[ykey]).toFixed(1)}`).join(' ');
    const dots = s.rows.map((r, j) =>
      `<circle cx="${px(j).toFixed(1)}" cy="${py(r[ykey]).toFixed(1)}" r="4" fill="${s.colour}" stroke="#15161a" stroke-width="2">
         <title>${esc(r[xkey])}: ${esc(fmtY(r[ykey]))}</title></circle>`).join('');
    return `<path d="${d}" fill="none" stroke="${s.colour}" stroke-width="2" stroke-linejoin="round"/>${dots}`;
  }).join('');
  const step = Math.max(1, Math.ceil(n / 10));
  const xt = series[0].rows.map((r, j) => j % step ? '' :
    `<text x="${px(j).toFixed(1)}" y="${H - 16}" text-anchor="middle" class="ax">${esc(r[xkey])}</text>`).join('');
  const legend = series.length > 1
    ? `<div class="legend">${series.map(s => `<span><i style="background:${s.colour}"></i>${esc(s.name)}</span>`).join('')}</div>` : '';
  el.innerHTML = `${legend}<svg viewBox="0 0 ${W} ${H}" class="chart" role="img" aria-label="${esc(ylabel)} by ${esc(xlabel)}">
      ${g}${paths}${xt}
      <text x="${W / 2}" y="${H - 2}" text-anchor="middle" class="ax dim">${esc(xlabel)}</text></svg>
    <div class="axlabel">${esc(ylabel)}</div>`;
}

// --------------------------------------------------------------- rendering
function renderScore(score, base, front, reason, chosenIsRec) {
  const m = contextMultiplier();
  const d = $('#result');
  const delta = base && score.edges !== base.edges;
  d.innerHTML = `
    <div class="score-head">
      <div>
        <div class="score-num">${fmt(score.mean, 3)}</div>
        <div class="score-sub">expected severity-weighted events · 90% CI ${fmt(score.lo, 3)}–${fmt(score.hi, 3)}</div>
      </div>
      <span class="badge b-${score.confidence.toLowerCase()}" title="Driven by how wide the interval is relative to the mean">${score.confidence} confidence</span>
    </div>
    <div class="kpis">
      <div><span>ETA</span><b>${fmt(score.timeMin, 0)} min</b></div>
      <div><span>Distance</span><b>${fmt(score.km, 1)} km</b></div>
      <div><span>Risk / 100 km</span><b>${fmt(score.per100, 3)}</b></div>
      <div><span>Context</span><b>×${fmt(m, 2)}</b></div>
    </div>
    <div class="rec ${chosenIsRec ? 'ok' : 'warn'}">
      <b>${chosenIsRec ? 'Recommended route' : 'Override — not the recommendation'}</b>
      <div>${esc(reason)}</div>
    </div>`;

  // context breakdown
  $('#ctxbreak').innerHTML = contextParts().map(p =>
    `<div class="trow"><span>${esc(p.factor)} <i class="dim">${esc(p.detail)}</i></span><b>×${fmt(p.mult, 2)}</b></div>`).join('')
    + `<div class="trow total"><span>Combined</span><b>×${fmt(m, 2)}</b></div>`;

  // why
  const attr = attribution(score.edges).slice(0, 5);
  const maxA = Math.max(...attr.map(a => Math.abs(a.value)), 1e-9);
  $('#why').innerHTML = attr.map(a => {
    const w = 100 * Math.abs(a.value) / maxA / 2;
    const pos = a.value >= 0;
    return `<div class="diverge" title="${a.value >= 0 ? 'raises' : 'lowers'} predicted risk on this route">
      <span class="dlab">${esc(a.name)}</span>
      <span class="dtrack"><span class="dfill ${pos ? 'pos' : 'neg'}" style="width:${w}%;${pos ? 'left:50%' : 'right:50%'}"></span><i class="dmid"></i></span>
      <span class="dval">${a.value >= 0 ? '+' : ''}${fmt(a.value, 2)}</span></div>`;
  }).join('') + `<p class="note">Exact additive contributions on the log-rate scale, length-weighted across the route. Positive values push risk up relative to the network average.</p>`;

  // worst segments
  $('#worst').innerHTML = score.worst.map(w => {
    const s = DATA.segments[w.e];
    return `<div class="trow"><span>${esc(s.name)} <i class="dim">${fmt(s.km, 1)} km</i></span>
      <span><b>${fmt(w.r, 3)}</b> <span class="badge mini b-${s.conf.toLowerCase()}">${s.conf}</span></span></div>`;
  }).join('');

  // hotspots on route
  const edgeSet = new Set(score.edges);
  const hits = DATA.hotspots.filter(h => h.segments.some(s => edgeSet.has(s))).slice(0, 4);
  $('#hotroute').innerHTML = hits.length
    ? hits.map(h => `<div class="trow"><span>#${h.rank} ${esc(h.road_names[0] || '')} <i class="dim">${h.n_events} events · ${esc(h.dominant_event.replace('_', ' '))}</i></span>
        <span class="badge mini ${h.trend_label === 'worsening' ? 'b-low' : 'b-medium'}">${h.trend_label}</span></div>`).join('')
    : `<p class="note">No clustered black spot on this route.</p>`;

  // frontier
  renderFrontier(front, score.edges);
  $('#segcount').textContent =
    `${score.edges.length} segments · ${score.confCount.High} high / ${score.confCount.Medium} medium / ${score.confCount.Low} low confidence`;
}

function renderFrontier(front, chosenEdges) {
  const key = chosenEdges.join(',');
  const W = 330, H = 190, P = { t: 12, r: 14, b: 32, l: 46 };
  const ts = front.map(f => f.timeMin), rs = front.map(f => f.riskMean);
  const t0 = Math.min(...ts) * 0.98, t1 = Math.max(...ts) * 1.02;
  const r0 = Math.min(...rs) * 0.9, r1 = Math.max(...rs) * 1.05;
  const px = v => P.l + (v - t0) / Math.max(t1 - t0, 1e-9) * (W - P.l - P.r);
  const py = v => H - P.b - (v - r0) / Math.max(r1 - r0, 1e-9) * (H - P.t - P.b);
  const line = front.map((f, i) => `${i ? 'L' : 'M'}${px(f.timeMin)},${py(f.riskMean)}`).join(' ');
  const pts = front.map((f, i) => {
    const sel = f.edges.join(',') === key;
    return `<circle cx="${px(f.timeMin)}" cy="${py(f.riskMean)}" r="${sel ? 7 : 5}"
      fill="${sel ? SERIES[0] : '#5a6475'}" stroke="#15161a" stroke-width="2"
      class="fpt" data-i="${i}"><title>${fmt(f.timeMin, 0)} min · risk ${fmt(f.riskMean, 3)}</title></circle>`;
  }).join('');
  $('#frontier').innerHTML = `<svg viewBox="0 0 ${W} ${H}" class="chart">
      <path d="${line}" fill="none" stroke="#39404c" stroke-width="2" stroke-dasharray="4 4"/>
      ${pts}
      <text x="${W / 2}" y="${H - 4}" text-anchor="middle" class="ax dim">travel time (min) →</text>
      <text x="12" y="${H / 2}" text-anchor="middle" class="ax dim" transform="rotate(-90 12 ${H / 2})">risk →</text>
    </svg>`;
  $('#altlist').innerHTML = front.map((f, i) => {
    const sel = f.edges.join(',') === key;
    const base = front[0];
    const dr = 1 - f.riskMean / Math.max(base.riskMean, 1e-12);
    const dt = f.timeMin - base.timeMin;
    return `<button class="alt ${sel ? 'sel' : ''}" data-i="${i}">
      <span class="altname">${i === 0 ? 'Fastest' : 'Option ' + (i + 1)}</span>
      <span>${fmt(f.timeMin, 0)} min <i class="dim">${dt > 0 ? '+' + fmt(dt, 1) : '—'}</i></span>
      <span>risk ${fmt(f.riskMean, 3)} <i class="${dr > 0.001 ? 'good' : 'dim'}">${dr > 0.001 ? '−' + pct(dr) : ''}</i></span>
    </button>`;
  }).join('');
  $$('#altlist .alt').forEach(b => b.onclick = () => selectRoute(+b.dataset.i));
}

// ----------------------------------------------------------------- actions
let STATE = { front: [], base: null, rec: null, reason: '', chosen: 0 };

function plan() {
  const s = +$('#origin').value, t = +$('#dest').value;
  if (s === t) { $('#result').innerHTML = `<p class="note">Pick two different places.</p>`; return; }
  const mult = contextMultiplier();
  const budget = +$('#budget').value / 100, minCut = +$('#mincut').value / 100;
  const cands = candidateRoutes(s, t, mult);
  if (!cands.length) { $('#result').innerHTML = `<p class="note">No route found between these points.</p>`; return; }
  const front = paretoFrontier(cands, mult);
  const { base, best, reason } = recommend(front, budget, minCut);
  STATE = { front, base, rec: best, reason, chosen: front.indexOf(best) };
  selectRoute(STATE.chosen);
}

function selectRoute(i) {
  const f = STATE.front[i];
  STATE.chosen = i;
  const mult = contextMultiplier();
  const score = scoreRoute(f.edges, mult);
  STATE.score = score;
  const isRec = f === STATE.rec;
  const reason = isRec ? STATE.reason
    : `This is not the recommended option — it carries ${pct(f.riskMean / Math.max(STATE.rec.riskMean, 1e-12) - 1)} more risk than the recommendation.`;
  MapView.routeEdges = f.edges;
  MapView.altEdges = (STATE.rec && STATE.rec !== f) ? STATE.rec.edges : null;
  MapView.draw();
  renderScore(score, STATE.base, STATE.front, reason, isRec);
  $('#assign').disabled = false;
  $('#override').style.display = isRec ? 'none' : 'block';
}

// trip log (localStorage — per-viewer, survives reloads)
const LOG_KEY = 'routeshield.triplog.v1';
const loadLog = () => { try { return JSON.parse(localStorage.getItem(LOG_KEY)) || []; } catch { return []; } };
const saveLog = (l) => { try { localStorage.setItem(LOG_KEY, JSON.stringify(l.slice(0, 200))); } catch {} };

function assign() {
  const s = STATE.score; if (!s) return;
  const isRec = STATE.front[STATE.chosen] === STATE.rec;
  const log = loadLog();
  log.unshift({
    trip_id: 'T' + Date.now().toString(36).toUpperCase(),
    at: new Date().toISOString(),
    from: DATA.nodes[+$('#origin').value].name,
    to: DATA.nodes[+$('#dest').value].name,
    hour: +$('#hour').value, weather: $('#weather').value, cohort: $('#cohort').value,
    risk: s.mean, lo: s.lo, hi: s.hi, conf: s.confidence,
    eta: s.timeMin, km: s.km,
    base_risk: STATE.base.riskMean, base_eta: STATE.base.timeMin,
    alternatives: STATE.front.length,
    recommended: isRec,
    override_reason: isRec ? null : $('#overrideReason').value,
  });
  saveLog(log); renderLog();
  const btn = $('#assign'); btn.textContent = 'Logged ✓';
  setTimeout(() => btn.textContent = 'Assign this route', 1400);
}

function renderLog() {
  const log = loadLog();
  const n = log.length, taken = log.filter(l => l.recommended).length;
  const riskSaved = log.reduce((a, l) => a + (l.base_risk - l.risk), 0);
  const timeCost = log.reduce((a, l) => a + (l.eta - l.base_eta), 0);
  $('#logstats').innerHTML = `
    <div><span>Trips logged</span><b>${n}</b></div>
    <div><span>Recommendation taken</span><b>${n ? pct(taken / n) : '—'}</b></div>
    <div><span>Risk avoided (cumulative)</span><b>${fmt(riskSaved, 3)}</b></div>
    <div><span>Extra minutes spent</span><b>${fmt(timeCost, 0)}</b></div>`;
  $('#logrows').innerHTML = n ? log.slice(0, 25).map(l => `<tr>
      <td><code>${esc(l.trip_id)}</code></td>
      <td>${esc(l.from)} → ${esc(l.to)}</td>
      <td>${String(l.hour).padStart(2, '0')}:00 · ${esc(l.weather)}</td>
      <td>${fmt(l.risk, 3)} <i class="dim">(${fmt(l.lo, 3)}–${fmt(l.hi, 3)})</i></td>
      <td><span class="badge mini b-${l.conf.toLowerCase()}">${l.conf}</span></td>
      <td>${fmt(l.eta, 0)} min</td>
      <td>${l.recommended ? '<span class="good">taken</span>' : '<span class="warn">' + esc(l.override_reason || 'override') + '</span>'}</td>
    </tr>`).join('')
    : `<tr><td colspan="7" class="note">No trips logged yet. Plan a route and press “Assign this route”.</td></tr>`;
}

// ------------------------------------------------------------------- tabs
function showTab(name) {
  $$('.tab').forEach(t => t.classList.toggle('on', t.dataset.tab === name));
  $$('.panel').forEach(p => p.classList.toggle('on', p.id === 'p-' + name));
  if (name === 'dispatch') { MapView.resize(); MapView.draw(); }
}

// ------------------------------------------------------------------- boot
async function boot() {
  DATA = await (await fetch('data/network.json')).json();
  GRAPH = buildGraph(DATA);

  const per = DATA.segments.filter(s => s.disp).map(s => s.per100).sort((a, b) => a - b);
  BANDS = [0.30, 0.52, 0.70, 0.84, 0.94].map(q => quantile(per, q));

  // header
  const meta = DATA.meta, ev = DATA.evaluation;
  $('#place').textContent = meta.place;
  $('#srcbadge').textContent = meta.source === 'OSM_OVERPASS' ? 'Live OpenStreetMap' : 'Geo-anchored network';
  $('#srcbadge').title = meta.honesty_note;
  $('#netstat').textContent = `${DATA.segments.length} segments · ${meta.network_km} km · ${DATA.hotspots.length} hotspots`;

  // origin/destination: named junctions only
  const junctions = DATA.nodes.filter(n => n.j).sort((a, b) => a.name.localeCompare(b.name));
  const opts = junctions.map(n => `<option value="${n.id}">${esc(n.name)}</option>`).join('');
  $('#origin').innerHTML = opts; $('#dest').innerHTML = opts;
  const pick = (name) => (junctions.find(n => n.name.includes(name)) || junctions[0]).id;
  $('#origin').value = pick('K R Puram');
  $('#dest').value = pick('Koramangala');

  // legend
  $('#legend').innerHTML = RISK_RAMP.map((c, i) => {
    const lo = i === 0 ? 0 : BANDS[i - 1];
    const hi = i < BANDS.length ? BANDS[i] : Infinity;
    return `<span><i style="background:${c}"></i>${i === RISK_RAMP.length - 1 ? '≥ ' + fmt(lo, 2) : fmt(lo, 2) + '–' + fmt(hi, 2)}</span>`;
  }).join('');

  MapView.init();
  MapView.fit();

  // controls
  ['origin', 'dest', 'weather', 'cohort'].forEach(id => $('#' + id).onchange = plan);
  $('#hour').oninput = e => { $('#hourval').textContent = String(e.target.value).padStart(2, '0') + ':00'; plan(); };
  $('#budget').oninput = e => { $('#budgetval').textContent = e.target.value + '%'; plan(); };
  $('#mincut').oninput = e => { $('#mincutval').textContent = e.target.value + '%'; plan(); };
  $('#layer').onchange = () => MapView.draw();
  $('#showHotspots').onchange = () => MapView.draw();
  $('#assign').onclick = assign;
  $('#swap').onclick = () => { const a = $('#origin').value; $('#origin').value = $('#dest').value; $('#dest').value = a; plan(); };
  $$('.tab').forEach(t => t.onclick = () => showTab(t.dataset.tab));

  renderAnalytics(); renderModel(); renderHotspots(); renderLog();
  plan();
}

function renderHotspots() {
  const rows = DATA.hotspots.slice(0, 30);
  $('#hotrows').innerHTML = rows.map(h => `<tr>
    <td>#${h.rank}</td>
    <td>${esc(h.road_names.slice(0, 2).join(', '))}</td>
    <td>${h.n_events}</td>
    <td>${fmt(h.weighted_score, 1)}</td>
    <td>${esc(h.dominant_event.replace('_', ' '))}</td>
    <td><span class="badge mini ${h.trend_label === 'worsening' ? 'b-low' : h.trend_label === 'improving' ? 'b-high' : 'b-medium'}">${h.trend_label}</span></td>
    <td>${fmt(h.radius_m, 0)} m</td>
  </tr>`).join('');
  $('#hotsummary').textContent =
    `${DATA.hotspots.length} clusters · ${DATA.hotspots.filter(h => h.trend_label === 'worsening').length} worsening`;
}

function renderAnalytics() {
  const a = DATA.analytics;
  $('#sumstats').innerHTML = Object.entries({
    'Segments': a.summary.segments, 'Network km': a.summary.network_km,
    'Matched events': a.summary.matched_events.toLocaleString(),
    'Hotspots': a.summary.hotspots, 'Worsening': a.summary.worsening_hotspots,
    'High confidence': a.summary.high_confidence_pct + '%',
    'Low confidence': a.summary.low_confidence_pct + '%',
    'Median risk/100km': a.summary.median_risk_per_100km,
  }).map(([k, v]) => `<div><span>${k}</span><b>${v}</b></div>`).join('');

  barChart($('#byroad'), a.by_road.slice(0, 12), {
    value: r => r.risk_per_100km, label: r => r.road,
    fmtV: v => fmt(v, 2),
  });
  barChart($('#byhour'), a.by_hour, {
    value: r => r.rate_index !== undefined ? r.rate_index : r.weighted_share_pct,
    label: r => String(r.hour).padStart(2, '0') + ':00',
    highlight: r => r.night_window, fmtV: v => '×' + fmt(v, 2),
  });
  barChart($('#bycohort'), a.by_cohort, {
    value: r => r.share_pct, label: r => r.cohort.replace(/_/g, ' '),
    fmtV: v => fmt(v, 1) + '%',
  });
  // Two single-series charts rather than one chart with two wildly different
  // scales: severe events are ~4% of the total, so on a shared axis they are a
  // flat line on the floor — and a second y-axis is never the answer.
  lineChart($('#monthly'), [
    { name: 'All events', colour: SERIES[0], rows: a.monthly.map(m => ({ x: 'M' + m.month, y: m.events })) },
  ], { xkey: 'x', ykey: 'y', xlabel: 'month of history', ylabel: 'events logged', fmtY: v => fmt(v, 0) });
  lineChart($('#monthlySevere'), [
    { name: 'Severe share', colour: SERIES[1],
      rows: a.monthly.map(m => ({ x: 'M' + m.month, y: 100 * m.severe_events / Math.max(m.events, 1) })) },
  ], { xkey: 'x', ykey: 'y', xlabel: 'month of history',
       ylabel: 'near-miss / accident, % of events', fmtY: v => fmt(v, 1) + '%' });
}

function renderModel() {
  const ev = DATA.evaluation, sm = ev.segment_model;
  $('#honesty').textContent = ev.honesty_note;

  const mk = (obj, fmtd = 3) => Object.entries(obj)
    .sort((a, b) => b[1] - a[1])
    .map(([k, v]) => ({ k: k.replace(/_/g, ' '), v }));

  barChart($('#aucTop'), mk(sm.roc_auc_top_decile_TRUE_rate || sm.roc_auc_top_decile_rate), {
    value: r => r.v, label: r => r.k, fmtV: v => fmt(v, 3),
    highlight: r => r.k === 'routeshield',
  });
  barChart($('#aucObs'), mk(sm.roc_auc_top_decile_rate), {
    value: r => r.v, label: r => r.k, fmtV: v => fmt(v, 3),
    highlight: r => r.k === 'routeshield',
  });

  const cf = ev.counterfactual, mm = ev.map_matching;
  const tr = sm.roc_auc_top_decile_TRUE_rate || {};
  $('#headline').innerHTML = [
    ['Rank true high-risk roads', fmt(tr.routeshield, 3) + ' AUC'],
    ['vs raw event counts', fmt(tr.raw_history_count, 3) + ' AUC'],
    ['Sparse roads, ρ vs truth', fmt((sm.spearman_vs_true_rate_sparse_only || {}).routeshield, 3)],
    ['90% interval coverage', pct(sm.latent_rate_interval_coverage_90, 1)],
    ['Fleet-wide risk reduction', pct((cf.fleet_true_risk_reduction_pct || 0) / 100, 1)],
    ['for fleet-wide time cost', '+' + pct((cf.fleet_time_cost_pct || 0) / 100, 1)],
  ].map(([k, v]) => `<div><span>${esc(k)}</span><b>${v}</b></div>`).join('');

  $('#matchstat').innerHTML = [
    ['Road-level accuracy', pct(mm.road_match_accuracy, 1)],
    ['Events retained', pct(mm.match_rate, 1)],
    ['Median snap distance', fmt(mm.median_snap_dist_m, 1) + ' m'],
    ['Ambiguity margin', mm.ambiguity_margin_m + ' m'],
  ].map(([k, v]) => `<div class="trow"><span>${esc(k)}</span><b>${v}</b></div>`).join('')
    + `<p class="note">Near a junction two roads are almost equidistant. Forcing those matches
       misattributes 4.6% of events — enough to make quiet side roads look like the worst in the
       city. Rejecting ambiguous points costs ${pct(1 - mm.match_rate, 0)} of the data and lifts
       road accuracy to ${pct(mm.road_match_accuracy, 1)}.</p>`;

  const cm = DATA.analytics.summary;
  $('#badgesep').innerHTML = `<div class="trow"><span>High confidence</span><b>${fmt(cm.high_confidence_pct, 0)}% of roads</b></div>
    <div class="trow"><span>Low confidence</span><b>${fmt(cm.low_confidence_pct, 0)}% of roads</b></div>
    <p class="note">Badges are thresholds on posterior uncertainty, calibrated against measured
    error: median |log| error runs 0.10 / 0.15 / 0.24 across High / Medium / Low, and median
    exposure falls 6500 → 1450 → 190 vehicle-km. A badge that does not separate accuracy is
    decoration.</p>`;
  barChart($('#aucSevere'), mk(sm.roc_auc_severe_event), {
    value: r => r.v, label: r => r.k, fmtV: v => fmt(v, 3),
    highlight: r => r.k === 'routeshield',
  });
  barChart($('#sparse'), mk(sm.spearman_vs_true_rate_sparse_only), {
    value: r => r.v, label: r => r.k, fmtV: v => fmt(v, 3),
    highlight: r => r.k === 'routeshield',
  });

  $('#calib').innerHTML = `<table class="tbl"><thead><tr><th>Decile</th><th>Expected</th><th>Observed</th><th>Obs / Exp</th></tr></thead><tbody>`
    + sm.calibration.map(c => `<tr><td>${c.bin}</td><td>${fmt(c.expected_events, 1)}</td><td>${fmt(c.observed_events, 1)}</td>
        <td><b>${c.ratio_obs_over_exp === null ? '—' : fmt(c.ratio_obs_over_exp, 2)}</b></td></tr>`).join('')
    + `</tbody></table>`;

  // Both series fleet-wide (every trip, not only the ones that switched) so
  // the two lines are comparable. Mixing populations overstates the cost.
  lineChart($('#tradeoff'), [
    { name: 'Risk reduction (fleet-wide)', colour: SERIES[0],
      rows: ev.tradeoff_curve.map(r => ({ x: r.time_budget_pct + '%', y: r.fleet_true_risk_reduction_pct })) },
    { name: 'Travel time cost (fleet-wide)', colour: SERIES[1],
      rows: ev.tradeoff_curve.map(r => ({ x: r.time_budget_pct + '%', y: r.fleet_time_cost_pct })) },
  ], { xkey: 'x', ykey: 'y', xlabel: 'time budget offered to the dispatcher',
       ylabel: '% change, averaged over every trip', fmtY: v => fmt(v, 1) + '%' });

  const cov = {
    'Posterior predictive (computable on real data)': sm.predictive_interval_coverage_90,
    'True latent rate — all roads': sm.latent_rate_interval_coverage_90,
    'True latent rate — sparse roads': sm.latent_rate_interval_coverage_90_sparse,
  };
  $('#coverage').innerHTML = Object.entries(cov).map(([k, v]) =>
    `<div class="trow"><span>${k}</span><b>${pct(v, 1)}</b></div>`).join('')
    + `<p class="note">Nominal level is 90%. A 90% interval that contains the truth 90% of the time is what makes the confidence badge trustworthy.</p>`;

  $('#setup').innerHTML = Object.entries({
    'Train / test': `${ev.setup.train_months} mo → ${ev.setup.test_months} mo (temporal holdout)`,
    'Events': `${ev.setup.train_events.toLocaleString()} train · ${ev.setup.test_events.toLocaleString()} test`,
    'Roads / segments': `${ev.setup.roads} / ${ev.setup.segments}`,
    'Prior model': ev.setup.prior === 'gbm' ? 'Poisson gradient boosting + GLM blend' : 'Poisson GLM',
    'Fitted dispersion a': ev.setup.fitted_dispersion_a,
    'Prior log-error σ': ev.setup.prior_log_error_sigma,
    'Map-match rate': pct(ev.map_matching.match_rate, 1),
    'Median snap distance': fmt(ev.map_matching.median_snap_dist_m, 1) + ' m',
  }).map(([k, v]) => `<div class="trow"><span>${k}</span><b>${v}</b></div>`).join('');

  $('#scal').innerHTML = Object.entries({
    'Shortest path': ev.scalability.shortest_path_ms_per_route + ' ms',
    'Score a route (1000 MC draws)': ev.scalability.score_route_ms_per_route_1000_samples + ' ms',
    'Full alternatives + Pareto': ev.scalability.alternatives_ms_per_request + ' ms',
    'Fit whole network': ev.timings_s.fit_risk_model_s + ' s',
    'Map-match all events': ev.timings_s.map_match_s + ' s',
    'Hotspot clustering': ev.timings_s.hotspots_s + ' s',
  }).map(([k, v]) => `<div class="trow"><span>${k}</span><b>${v}</b></div>`).join('')
    + `<p class="note">${esc(ev.scalability.note)}</p>`;
}

boot().catch(e => {
  document.body.innerHTML = `<div style="padding:40px;font:14px system-ui;color:#ddd">
    <h2>Could not load the demo data</h2><pre>${esc(e.message)}</pre>
    <p>Run <code>make demo</code> to regenerate <code>web/data/network.json</code>.</p></div>`;
});
