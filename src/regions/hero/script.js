(() => {
  "use strict";
  const S = window.__sem; if (!S) return;
  const { mulberry32, gauss, roll, rgba, clamp01, sfx, reduce } = S;
  const sec = document.getElementById("hero"); if (!sec) return;

  const easeIO = t => t < .5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
  const easeSnap = t => t < .5 ? 8 * t * t * t * t : 1 - Math.pow(-2 * t + 2, 4) / 2;
  const phone = () => innerWidth < 600;
  let colors = S.colors();

  /* ---------- the target: Swiss roll geometry (same parametrisation as core.js) ---------- */
  const T0 = 1.5 * Math.PI, T1 = 4.5 * Math.PI;
  const uq = p => Math.sqrt(T0 * T0 + (T1 * T1 - T0 * T0) * p) / (T1 - T0) - T0 / (T1 - T0);   // arc quantile -> u
  const arcFrac = u => { const t = T0 + (T1 - T0) * u; return (t * t - T0 * T0) / (T1 * T1 - T0 * T0); };
  function normalAt(u) {
    const [x0, y0] = roll(u), [x1, y1] = roll(Math.min(1, u + .002));
    const nx = -(y1 - y0), ny = x1 - x0, l = Math.hypot(nx, ny) || 1;
    return [nx / l, ny / l];
  }

  /* ---------- distance field around the roll: precision ("чёткость") and coverage, computed from points ---------- */
  const GR = 1.3, GN = 260, CELL = 2 * GR / GN, M_ARCS = 48;
  const EPS_SHARP = .03, EPS_COVER = .05;
  const DG = new Float32Array(GN * GN).fill(1), UB = new Uint8Array(GN * GN);
  (function buildField() {
    const R = 7, P = 2000;
    for (let j = 0; j < P; j++) {
      const u = j / (P - 1), [px, py] = roll(u), bin = Math.min(M_ARCS - 1, Math.floor(arcFrac(u) * M_ARCS));
      const ci = Math.floor((px + GR) / CELL), cj = Math.floor((py + GR) / CELL);
      for (let a = -R; a <= R; a++) for (let b = -R; b <= R; b++) {
        const i = ci + a, k = cj + b; if (i < 0 || k < 0 || i >= GN || k >= GN) continue;
        const cx = -GR + (i + .5) * CELL, cy = -GR + (k + .5) * CELL, d = Math.hypot(cx - px, cy - py);
        const g = k * GN + i; if (d < DG[g]) { DG[g] = d; UB[g] = bin; }
      }
    }
  })();
  const cellOf = (x, y) => { const i = Math.floor((x + GR) / CELL), k = Math.floor((y + GR) / CELL); return i < 0 || k < 0 || i >= GN || k >= GN ? -1 : k * GN + i; };
  const coveredBins = new Uint8Array(M_ARCS);
  function measure(pos) {
    let sharp = 0, cov = 0; coveredBins.fill(0);
    for (let i = 0; i < N; i++) {
      const g = cellOf(pos[2 * i], pos[2 * i + 1]); if (g < 0) continue;
      const d = DG[g];
      if (d < EPS_SHARP) sharp++;
      if (d < EPS_COVER && !coveredBins[UB[g]]) { coveredBins[UB[g]] = 1; cov++; }
    }
    return { sharp: sharp / N, cov: cov / M_ARCS };
  }

  /* ---------- samples ---------- */
  const N = 640, K = 50;                     // points, diffusion steps
  let seed = 7;
  const E = new Float32Array(N * 2), X0 = new Float32Array(N * 2), VT = new Float32Array(N * 2), GT = new Float32Array(N * 2);
  const DP = new Float32Array(N * (K + 1) * 2); // diffusion path per point: K+1 positions
  const D = new Float32Array(N);              // per-point delay jitter
  let order = [], autoList = [];
  const abar = s => Math.pow(Math.sin(s * Math.PI / 2), 2);   // cosine-like schedule, s = share of reverse process done

  function build() {
    const r = mulberry32(seed);
    // GAN mode collapse: 2-3 separated short arcs of the roll; they change with every new noise
    const k = 2 + (r() < .5 ? 1 : 0);
    let cs = [];
    for (let tries = 0; tries < 60 && cs.length < k; tries++) {
      const c = .1 + r() * .8; if (cs.every(o => Math.abs(o - c) > .2)) cs.push(c);
    }
    cs.sort((a, b) => a - b);
    const modes = cs.map(c => [c - .045, c + .045]);

    // shared noise x_T (= z for VAE / GAN)
    for (let i = 0; i < N; i++) { E[2 * i] = gauss(r) * .62; E[2 * i + 1] = gauss(r) * .62; D[i] = r(); }
    // generators are continuous maps: rank points by where their noise projects onto the roll,
    // then hand out arc-length quantiles in that order (nearby noise -> nearby output)
    const PL = 420, poly = Array.from({ length: PL }, (_, j) => roll(j / (PL - 1)));
    const proj = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      let best = 1e9, bj = 0; const ex = E[2 * i], ey = E[2 * i + 1];
      for (let j = 0; j < PL; j++) { const dx = poly[j][0] - ex, dy = poly[j][1] - ey, d = dx * dx + dy * dy; if (d < best) { best = d; bj = j; } }
      proj[i] = bj / (PL - 1) + r() * 1e-4;
    }
    order = Array.from({ length: N }, (_, i) => i).sort((a, b) => proj[a] - proj[b]);
    order.forEach((i, rank) => {
      const p = Math.min(.9995, (rank + r()) / N);
      // diffusion: lands exactly on the roll, whole length
      const u = uq(p), [rx, ry] = roll(u), [nx, ny] = normalAt(u), j0 = gauss(r) * .006;
      X0[2 * i] = rx + nx * j0; X0[2 * i + 1] = ry + ny * j0;
      // VAE: same place, smeared across and along the roll (Gaussian decoder + KL pressure)
      const uv = Math.min(1, Math.max(0, u + gauss(r) * .012)), [vx, vy] = roll(uv), [vnx, vny] = normalAt(uv), bl = gauss(r) * .085;
      VT[2 * i] = vx + vnx * bl; VT[2 * i + 1] = vy + vny * bl;
      // GAN: crisp, but the whole noise space is squeezed onto k arcs
      const mj = Math.min(k - 1, Math.floor(p * k)), loc = p * k - mj, m = modes[mj], [gx, gy] = roll(m[0] + (m[1] - m[0]) * loc);
      GT[2 * i] = gx + gauss(r) * .008; GT[2 * i + 1] = gy + gauss(r) * .008;
      // diffusion trajectory: x_s = sqrt(abar)·x0 + sqrt(1-abar)·x_T + fresh noise each step (zero at both ends)
      for (let s = 0; s <= K; s++) {
        const q = s / K, a = abar(q), sa = Math.sqrt(a), sb = Math.sqrt(1 - a), J = .065 * Math.sqrt(Math.sin(Math.PI * q));
        const o = (i * (K + 1) + s) * 2;
        DP[o] = sa * X0[2 * i] + sb * E[2 * i] + J * gauss(r);
        DP[o + 1] = sa * X0[2 * i + 1] + sb * E[2 * i + 1] + J * gauss(r);
      }
    });
    // auto-demo samples: moderate noise radius, spread along the roll
    const cand = order.filter(i => { const rr = Math.hypot(E[2 * i], E[2 * i + 1]); return rr > .38 && rr < .85; });
    autoList = Array.from({ length: 7 }, (_, j) => cand[Math.floor((j * 3 % 7 + .5) / 7 * cand.length)]).filter(i => i !== undefined);
  }

  /* ---------- one loop, seconds (slow enough to read every phase) ---------- */
  const HOLD0 = 1.6, VAE_T = 1.4, GAN_T = 1.0, DIF_T = 8.5, HOLD1 = 4.0, BACK = 2.0;
  const G1 = HOLD0 + DIF_T, R1 = G1 + HOLD1, LOOP = R1 + BACK;          // 10.1, 14.1, 16.1
  const SEGS = [[0, HOLD0], [HOLD0, G1], [G1, R1], [R1, LOOP]];
  const PHASE_NAMES = ["шум", "генерация", "результат", "обратно в шум"];
  const REST_T = G1 + 1.5;                                               // the still frame (reduced motion)

  /* position of point i for model m at loop time t -> out[o], out[o+1]; returns progress 0..1 */
  function place(m, i, t, out, o) {
    const ex = E[2 * i], ey = E[2 * i + 1], d = D[i];
    const rb = clamp01((t - R1) / BACK), bb = easeIO(clamp01((rb - d * .2) / .8));
    if (m === "dif") {
      if (t < R1) {
        const raw = clamp01((t - HOLD0) / DIF_T) * K;
        const k = Math.min(K - 1, Math.floor(raw)), f = raw >= K ? 1 : easeIO(clamp01((raw - k) / .55));
        const a = (i * (K + 1) + k) * 2, b = a + 2;
        out[o] = DP[a] + (DP[b] - DP[a]) * f; out[o + 1] = DP[a + 1] + (DP[b + 1] - DP[a + 1]) * f;
        return (k + f) / K;
      }
      const a = abar(1 - bb), sa = Math.sqrt(a), sb = Math.sqrt(1 - a);
      out[o] = sa * X0[2 * i] + sb * ex; out[o + 1] = sa * X0[2 * i + 1] + sb * ey;
      return 1 - bb;
    }
    const vae = m === "vae", T = vae ? VAE_T : GAN_T, lag = d * (vae ? .2 : .12);
    const raw = clamp01((t - HOLD0) / T);
    const pp = (vae ? easeIO : easeSnap)(clamp01((raw - lag) / .8)) * (1 - bb);
    const tg = vae ? VT : GT;
    out[o] = ex + (tg[2 * i] - ex) * pp; out[o + 1] = ey + (tg[2 * i + 1] - ey) * pp;
    return pp;
  }

  /* ---------- views ---------- */
  const views = [...sec.querySelectorAll(".sampler")].map(el => ({
    el, m: el.dataset.m, cv: el.querySelector("[data-canvas]"),
    step: el.querySelector("[data-step]"), stN: el.querySelector('[data-st="n"]'),
    stCov: el.querySelector('[data-st="cov"]'), stSharp: el.querySelector('[data-st="prec"]'),
    barCov: el.querySelector('[data-bar="cov"]'), barSharp: el.querySelector('[data-bar="prec"]'),
    ctx: null, w: 0, h: 0, dpr: 1, ghost: null,
    pos: new Float32Array(N * 2), prev: new Float32Array(N * 2), prog: new Float32Array(N), scr: new Float32Array(N * 2),
    hasPrev: false, last: {}
  }));
  const geo = v => { const Sc = Math.min(v.w, v.h) * .395; return { Sc, cx: v.w / 2, cy: v.h / 2 + v.h * .01 }; };

  function renderGhost(v) {
    if (!v.w) return;
    const g = v.ghost || (v.ghost = document.createElement("canvas"));
    g.width = v.cv.width; g.height = v.cv.height;
    const c = g.getContext("2d"); c.setTransform(v.dpr, 0, 0, v.dpr, 0, 0);
    const { Sc, cx, cy } = geo(v), gr = mulberry32(99);
    c.fillStyle = rgba(colors.fg, colors.dark ? .16 : .2);
    for (let j = 0; j < 420; j++) {
      const r = gr(), u = uq(r), [x, y] = roll(u);
      c.beginPath(); c.arc(cx + x * Sc, cy - y * Sc, 1.3, 0, 6.283); c.fill();
    }
  }
  function size(v) {
    const r = v.cv.getBoundingClientRect();
    v.dpr = Math.min(2, devicePixelRatio || 1);
    v.w = r.width; v.h = r.height;
    v.cv.width = Math.max(1, Math.round(r.width * v.dpr)); v.cv.height = Math.max(1, Math.round(r.height * v.dpr));
    v.ctx = v.cv.getContext("2d"); v.hasPrev = false;
    renderGhost(v);
  }

  /* cached fill styles: per colour, alpha quantised to 1/50 */
  const styleCache = new Map();
  const fill = (hex, a) => { const q = Math.round(Math.max(0, Math.min(1, a)) * 50); const key = hex + q; let s = styleCache.get(key); if (!s) { s = rgba(hex, q / 50); styleCache.set(key, s); } return s; };

  /* ---------- spotlight state ---------- */
  let spot = -1, spotUser = false, hovering = false, lastUser = 0, autoN = 0, hintUsed = false;
  const hint = document.getElementById("hero-hint");
  const hintT = hint && hint.querySelector(".hx-hint-t");
  if (hintT && matchMedia("(hover: none)").matches) hintT.textContent = "коснись точки — увидишь её путь из шума";

  const MONO = '500 11px "JetBrains Mono", ui-monospace, monospace', MONO_S = '500 8.5px "JetBrains Mono", ui-monospace, monospace';
  function label(c, v, x, y, parts, dirx, diry, alpha) {
    // parts: [[text, sub?], ...]; drawn with a background halo, pushed away from the point
    c.font = MONO;
    let w = 0; for (const [s, sub] of parts) { c.font = sub ? MONO_S : MONO; w += c.measureText(s).width; }
    let lx = x + dirx * 13, ly = y + diry * 13 + 4;
    if (Math.abs(dirx) < .35) lx -= w / 2; else if (dirx < 0) lx -= w;
    if (diry > .35) ly += 5;
    lx = Math.max(4, Math.min(v.w - w - 4, lx)); ly = Math.max(12, Math.min(v.h - 5, ly));
    c.lineJoin = "round"; c.lineWidth = 3.5; c.strokeStyle = fill(colors.bg, .85); c.fillStyle = fill(colors.fg, alpha);
    let cx = lx;
    for (const [s, sub] of parts) {
      c.font = sub ? MONO_S : MONO; const yy = sub ? ly + 3 : ly;
      c.strokeText(s, cx, yy); c.fillText(s, cx, yy); cx += c.measureText(s).width;
    }
  }

  function drawSpot(v, t, user) {
    const c = v.ctx, i = spot, col = colors[v.m], { Sc, cx, cy } = geo(v);
    const X = x => cx + x * Sc, Y = y => cy - y * Sc;
    const ex = E[2 * i], ey = E[2 * i + 1];
    const k = user ? 1 : .7;
    const cur = v.prog[i];
    c.lineCap = "round"; c.lineJoin = "round";
    let endX, endY;
    if (v.m === "dif") {
      const base = i * (K + 1) * 2;
      endX = DP[base + 2 * K]; endY = DP[base + 2 * K + 1];
      // whole jagged path, faint
      c.strokeStyle = fill(col, .3 * k); c.lineWidth = 1;
      c.beginPath(); for (let s = 0; s <= K; s++) { const o = base + 2 * s; s ? c.lineTo(X(DP[o]), Y(DP[o + 1])) : c.moveTo(X(DP[o]), Y(DP[o + 1])); } c.stroke();
      // travelled part, bright, with a dot per denoising step
      const back = t >= R1;
      if (!back && cur > 0) {
        const kk = Math.min(K, Math.floor(cur * K + 1e-6));
        c.strokeStyle = fill(col, .95 * k); c.lineWidth = 1.6;
        c.beginPath(); c.moveTo(X(DP[base]), Y(DP[base + 1]));
        for (let s = 1; s <= kk; s++) c.lineTo(X(DP[base + 2 * s]), Y(DP[base + 2 * s + 1]));
        c.lineTo(X(v.pos[2 * i]), Y(v.pos[2 * i + 1])); c.stroke();
        c.fillStyle = fill(col, .9 * k);
        for (let s = 1; s <= kk; s++) { c.beginPath(); c.arc(X(DP[base + 2 * s]), Y(DP[base + 2 * s + 1]), 1.5, 0, 6.283); c.fill(); }
      }
    } else {
      const tg = v.m === "vae" ? VT : GT;
      endX = tg[2 * i]; endY = tg[2 * i + 1];
      c.setLineDash([2, 4]); c.strokeStyle = fill(col, .45 * k); c.lineWidth = 1;
      c.beginPath(); c.moveTo(X(ex), Y(ey)); c.lineTo(X(endX), Y(endY)); c.stroke(); c.setLineDash([]);
      if (cur > 0 && t < R1) {
        c.strokeStyle = fill(col, .95 * k); c.lineWidth = 1.6;
        c.beginPath(); c.moveTo(X(ex), Y(ey)); c.lineTo(X(v.pos[2 * i]), Y(v.pos[2 * i + 1])); c.stroke();
      }
    }
    // start ring (the shared noise point) and end ring (where this model sends it)
    c.lineWidth = 1.2;
    c.strokeStyle = fill(colors.fg, .75 * k); c.beginPath(); c.arc(X(ex), Y(ey), 5, 0, 6.283); c.stroke();
    c.strokeStyle = fill(col, .9 * k); c.beginPath(); c.arc(X(endX), Y(endY), 6.5, 0, 6.283); c.stroke();
    // current position
    const px = X(v.pos[2 * i]), py = Y(v.pos[2 * i + 1]);
    c.fillStyle = fill(col, .22 * k); c.beginPath(); c.arc(px, py, 9, 0, 6.283); c.fill();
    c.fillStyle = fill(colors.dark ? "#ffffff" : col, 1); c.beginPath(); c.arc(px, py, 3, 0, 6.283); c.fill();
    c.strokeStyle = fill(col, 1); c.lineWidth = 1.5; c.beginPath(); c.arc(px, py, 3.6, 0, 6.283); c.stroke();
    // labels: pushed away from the centre
    // labels: start and end pushed apart along the start->end line (away from the centre if they coincide)
    const fromC = (x, y) => { const dx = X(x) - cx, dy = Y(y) - cy, l = Math.hypot(dx, dy); return l < 8 ? [1, -1] : [dx / l, dy / l]; };
    const ddx = X(endX) - X(ex), ddy = Y(endY) - Y(ey), dl = Math.hypot(ddx, ddy);
    const along = dl > 6 ? [ddx / dl, ddy / dl] : null;
    const la = user ? .95 : .7;
    const [sx, sy] = along ? [-along[0], -along[1]] : fromC(ex, ey);
    label(c, v, X(ex), Y(ey), v.m === "dif" ? [["x"], ["T", 1]] : [["z"]], sx, sy, la);
    const [ux, uy] = along || fromC(endX, endY);
    label(c, v, X(endX), Y(endY), v.m === "dif" ? [["x"], ["0", 1], [" · 50 шагов"]] : [["x · 1 шаг"]], ux, uy, la);
  }

  function drawView(v, t, streaks) {
    const c = v.ctx; if (!c || !v.w) return;
    const { w, h, dpr } = v, { Sc, cx, cy } = geo(v);
    c.setTransform(1, 0, 0, 1, 0, 0); c.clearRect(0, 0, v.cv.width, v.cv.height);
    if (v.ghost) c.drawImage(v.ghost, 0, 0);
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    const col = colors[v.m], dark = colors.dark, pos = v.pos, prev = v.prev, scr = v.scr;
    const dim = spot >= 0 && spotUser ? .38 : 1;
    // positions (+ last frame for motion streaks)
    if (streaks && v.hasPrev) prev.set(pos);
    for (let i = 0; i < N; i++) v.prog[i] = place(v.m, i, t, pos, 2 * i);
    c.globalCompositeOperation = dark ? "lighter" : "source-over";
    if (streaks && v.hasPrev) {
      // velocity streaks: a short tail behind each moving point (~4 frames of motion)
      const dif = v.m === "dif", SK = dif ? 1.4 : 4, CAP = dif ? 12 : 34;
      c.lineWidth = v.m === "vae" ? 2 : 1.2; c.lineCap = "round";
      c.strokeStyle = fill(col, (dif ? (dark ? .12 : .16) : dark ? .22 : .3) * dim);
      c.beginPath();
      for (let i = 0; i < N; i++) {
        const x = pos[2 * i], y = pos[2 * i + 1];
        let dx = (x - prev[2 * i]) * Sc * SK, dy = (y - prev[2 * i + 1]) * Sc * SK;
        const l = Math.hypot(dx, dy); if (l < 2) continue;
        if (l > CAP) { dx *= CAP / l; dy *= CAP / l; }
        const sx = cx + x * Sc, sy = cy - y * Sc;
        c.moveTo(sx, sy); c.lineTo(sx - dx, sy + dy);
      }
      c.stroke();
    }
    for (let i = 0; i < N; i++) {
      const x = cx + pos[2 * i] * Sc, y = cy - pos[2 * i + 1] * Sc, p = v.prog[i];
      scr[2 * i] = x; scr[2 * i + 1] = y;
      if (v.m === "vae") {
        c.fillStyle = fill(col, ((dark ? .16 : .2) + (1 - p) * .3) * dim);
        c.beginPath(); c.arc(x, y, 1.6 + 2.6 * p, 0, 6.283); c.fill();
      } else if (v.m === "gan") {
        c.fillStyle = fill(col, (dark ? .32 + .2 * p : .55 + .35 * p) * dim);
        c.beginPath(); c.arc(x, y, 1.5 + .3 * p, 0, 6.283); c.fill();
      } else {
        const a = abar(p);
        c.fillStyle = fill(col, (dark ? .38 + .22 * a : .5 + .4 * a) * dim);
        c.beginPath(); c.arc(x, y, 1.4 + .3 * a, 0, 6.283); c.fill();
      }
    }
    c.globalCompositeOperation = "source-over";
    v.hasPrev = true;
    if (spot >= 0) drawSpot(v, t, spotUser);

    // header + step counter
    const L = v.last;
    let n;
    if (v.m === "dif") {
      const raw = t < HOLD0 ? 0 : t < R1 ? clamp01((t - HOLD0) / DIF_T) * K : 0;
      n = t >= R1 + BACK * .5 ? 0 : t >= R1 ? K : raw > 0 ? Math.min(K, Math.floor(raw) + 1) : 0;
      const rawT = t < R1 ? raw : K * (1 - easeIO(clamp01((t - R1) / BACK)));
      const tl = Math.round(1000 - 1000 * rawT / K);
      if (tl !== L.t) { v.step.textContent = tl; L.t = tl; }
    } else {
      n = t > HOLD0 && t < R1 + BACK * .5 ? 1 : 0;
    }
    if (n !== L.n) { v.stN.textContent = n; L.n = n; }
  }

  function updateStats(v) {
    const { sharp, cov } = measure(v.pos), L = v.last;
    const c = Math.round(cov * 100), s = Math.round(sharp * 100);
    if (c !== L.c) { v.stCov.textContent = c + "%"; v.barCov.style.transform = `scaleX(${cov})`; L.c = c; }
    if (s !== L.s) { v.stSharp.textContent = s + "%"; v.barSharp.style.transform = `scaleX(${sharp})`; L.s = s; }
  }

  /* ---------- clock (phases, playhead, lanes, timer, scrubber) ---------- */
  const grid = document.getElementById("hero-clock");
  const head = grid.querySelector(".clk-head");
  const phEls = [...grid.querySelectorAll(".clk-phases > span")];
  const segB = [...grid.querySelectorAll(".clk-track b")];
  const lanes = ["vae", "gan", "dif"].map(m => { const el = grid.querySelector(`[data-lane="${m}"]`); return { m, el, b: el.querySelector(".ln-bar b"), last: -1 }; });
  const scrub = document.getElementById("hero-scrub");
  const tEl = sec.querySelector("[data-clk-t]");
  sec.querySelector("[data-clk-total]").textContent = LOOP.toFixed(1);
  scrub.max = String(LOOP);
  let gridW = grid.getBoundingClientRect().width, lastPh = -1, lastTxt = "";
  new ResizeObserver(() => { gridW = grid.getBoundingClientRect().width; updateClock(loopT, true); }).observe(grid);
  const phaseOf = t => t < HOLD0 ? 0 : t < G1 ? 1 : t < R1 ? 2 : 3;

  function updateClock(t, force) {
    head.style.transform = `translateX(${(t / LOOP * gridW).toFixed(1)}px)`;
    SEGS.forEach(([a, b], i) => { segB[i].style.transform = `scaleX(${clamp01((t - a) / (b - a)).toFixed(3)})`; });
    const ph = phaseOf(t);
    if (ph !== lastPh || force) { phEls.forEach((e, i) => e.classList.toggle("on", i === ph)); lastPh = ph; }
    const fade = t >= R1 ? 1 - clamp01((t - R1) / BACK) : 1;
    for (const ln of lanes) {
      let p;
      if (ln.m === "dif") { const raw = clamp01((t - HOLD0) / DIF_T) * K; p = t >= HOLD0 ? Math.min(K, Math.floor(raw) + 1) / K : 0; }
      else p = clamp01((t - HOLD0) / (ln.m === "vae" ? VAE_T : GAN_T));
      const key = Math.round(p * 1000) + "|" + Math.round(fade * 50);
      if (key !== ln.last || force) {
        ln.b.style.clipPath = `inset(0 ${((1 - p) * 100).toFixed(2)}% 0 0)`;
        ln.b.style.opacity = fade.toFixed(2);
        ln.el.classList.toggle("done", p >= 1 && fade > .5);
        ln.last = key;
      }
    }
    const txt = t.toFixed(1).padStart(4, "0");
    if (txt !== lastTxt) {
      tEl.textContent = txt; lastTxt = txt;
      if (!scrubbing) scrub.value = t.toFixed(1);
      scrub.setAttribute("aria-valuetext", `${PHASE_NAMES[ph]}, ${t.toFixed(1)} с из ${LOOP.toFixed(1)}`);
    }
  }

  /* ---------- loop ---------- */
  let loopT = reduce ? REST_T : 0, paused = reduce, scrubbing = false, scrubAt = 0;
  let running = false, visible = false, lastNow = 0, lastStats = 0, raf = 0, queued = false;
  const playBtn = document.getElementById("hero-play");
  function setPausedUI() {
    playBtn.classList.toggle("paused", paused);
    playBtn.setAttribute("aria-label", paused ? "Продолжить" : "Пауза");
  }
  setPausedUI();

  function nextAuto() { spot = autoList.length ? autoList[autoN++ % autoList.length] : -1; }
  function sounds(a, b) {
    // fire the loop's sounds on the edges crossed between a and b (same loop)
    if (a < HOLD0 && b >= HOLD0) sfx.vae();
    if (a < HOLD0 + .15 && b >= HOLD0 + .15) sfx.gan();
    if (b >= HOLD0 && a < G1) {
      const ka = Math.floor(clamp01((a - HOLD0) / DIF_T) * K - 1e-9), kb = Math.floor(clamp01((b - HOLD0) / DIF_T) * K);
      if (kb > ka && kb < K) sfx.dif(kb, K);
    }
    if (a < R1 && b >= R1) sfx.back();
  }

  function render(now) {
    const streaks = !paused && !scrubbing && !phone() && !reduce;
    for (const v of views) drawView(v, loopT, streaks);
    if (!now || now - lastStats > (phone() ? 250 : 120) || paused || scrubbing) { for (const v of views) updateStats(v); lastStats = now || 0; }
    updateClock(loopT);
  }
  function frame(now) {
    raf = 0;
    if (!running) return;
    const dt = Math.min(.1, (now - lastNow) / 1000); lastNow = now;
    if (scrubbing && now - scrubAt > 5000) scrubbing = false;            // resume after 5 s idle
    if (spotUser && !hovering && now - lastUser > 5000) { spotUser = false; nextAuto(); }
    if (!paused && !scrubbing) {
      const a = loopT; loopT += dt;
      if (loopT >= LOOP) { loopT -= LOOP; if (!spotUser) nextAuto(); for (const v of views) v.hasPrev = false; }
      else sounds(a, loopT);
    }
    render(now);
    raf = requestAnimationFrame(frame);
  }
  function sync() {
    const want = visible && !document.hidden && (!paused || scrubbing || spotUser);
    if (want && !running) { running = true; lastNow = performance.now(); raf = requestAnimationFrame(frame); }
    else if (!want && running) { running = false; if (raf) cancelAnimationFrame(raf); raf = 0; }
  }
  function kick() {                       // one redraw when the loop is not running
    if (running || queued) return; queued = true;
    requestAnimationFrame(() => { queued = false; render(0); });
  }

  /* ---------- pointer: spotlight the nearest sample, same noise point in all three panels ---------- */
  let lastSnd = 0;
  function spotSound(m) {
    const now = performance.now(); if (now - lastSnd < 110) return; lastSnd = now;
    const f = { vae: 523.25, gan: 659.25, dif: 783.99 }[m];
    sfx.play(({ t, tone, noise }) => {
      if (m === "dif") { noise(t, .14, "bandpass", 3200, 1500, 5, .05); tone(t + .05, f, .26, "sine", .07); }
      else if (m === "gan") { tone(t, f, .16, "triangle", .06); }
      else { noise(t, .12, "lowpass", 1800, 600, .7, .04); tone(t, f, .24, "sine", .06); }
    });
  }
  function takeOver(i, m) {
    lastUser = performance.now();
    if (!spotUser || i !== spot) { spotSound(m); }
    spot = i; spotUser = true;
    if (!hintUsed && hint) { hintUsed = true; hint.classList.add("used"); }
    sync(); kick();
  }
  views.forEach(v => {
    const nearest = (ev) => {
      const r = v.cv.getBoundingClientRect(), x = ev.clientX - r.left, y = ev.clientY - r.top;
      let best = 1e12, bi = -1;
      for (let i = 0; i < N; i++) { const dx = v.scr[2 * i] - x, dy = v.scr[2 * i + 1] - y, d = dx * dx + dy * dy; if (d < best) { best = d; bi = i; } }
      return bi;
    };
    v.cv.addEventListener("pointermove", ev => { if (ev.pointerType === "mouse" || ev.pointerType === "pen" || ev.buttons) { hovering = true; takeOver(nearest(ev), v.m); } });
    v.cv.addEventListener("pointerdown", ev => { hovering = ev.pointerType !== "touch"; takeOver(nearest(ev), v.m); });
    v.cv.addEventListener("pointerup", ev => { if (ev.pointerType === "touch") { hovering = false; lastUser = performance.now(); } });
    v.cv.addEventListener("pointerleave", () => { hovering = false; lastUser = performance.now(); });
    v.cv.addEventListener("focus", () => { if (spot < 0) nextAuto(); takeOver(spot, v.m); hovering = true; });
    v.cv.addEventListener("blur", () => { hovering = false; lastUser = performance.now(); });
    v.cv.addEventListener("keydown", ev => {
      const d = ev.key === "ArrowRight" || ev.key === "ArrowUp" ? 1 : ev.key === "ArrowLeft" || ev.key === "ArrowDown" ? -1 : 0;
      if (!d) return; ev.preventDefault();
      const r = order.indexOf(spot), nr = ((r < 0 ? 0 : r) + d * 8 + N) % N;   // walk along the roll
      takeOver(order[nr], v.m); hovering = true;
    });
  });

  /* ---------- controls ---------- */
  let lastScrubSnd = 0;
  scrub.addEventListener("input", () => {
    scrubbing = true; scrubAt = performance.now();
    loopT = Math.min(LOOP - .001, Math.max(0, parseFloat(scrub.value) || 0));
    for (const v of views) v.hasPrev = false;
    const now = performance.now();
    if (now - lastScrubSnd > 45) { lastScrubSnd = now; sfx.scrub(loopT < R1 ? clamp01((loopT - HOLD0) / DIF_T) : 1 - clamp01((loopT - R1) / BACK)); }
    sync(); kick();
  });
  scrub.addEventListener("change", () => { scrubAt = performance.now(); });
  playBtn.addEventListener("click", () => {
    paused = !paused; scrubbing = false; setPausedUI();
    sfx.play(({ t, tone }) => tone(t, paused ? 523.25 : 659.25, .14, "triangle", .07, paused ? 392 : 783.99));
    sync(); kick();
  });
  document.getElementById("resample").addEventListener("click", () => {
    seed = (seed * 1103515245 + 12345) >>> 0; build();
    sfx.back();
    autoN = 0; if (!spotUser) nextAuto(); else spot = Math.min(spot, N - 1);
    if (!reduce) { loopT = 0; scrubbing = false; }
    for (const v of views) v.hasPrev = false;
    sync(); kick();
  });

  /* ---------- boot ---------- */
  build(); nextAuto();
  views.forEach(size);
  const ro = new ResizeObserver(() => { views.forEach(size); kick(); });
  views.forEach(v => ro.observe(v.cv));
  render(0);
  new IntersectionObserver(([e]) => { visible = e.isIntersecting; sync(); }, { rootMargin: "60px 0px" }).observe(sec);
  document.addEventListener("visibilitychange", sync);
  S.onTheme(() => { colors = S.colors(); styleCache.clear(); views.forEach(renderGhost); kick(); });
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(kick);
})();
