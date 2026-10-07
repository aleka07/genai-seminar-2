(() => {
  "use strict";
  const S = window.__sem; if (!S) return;
  const { sfx, mulberry32, gauss, roll, rollU, rgba, ease, clamp01, reduce } = S;
  const fig = document.getElementById("dif-lab"); if (!fig) return;
  const panel = fig.closest('[role="tabpanel"]') || fig.parentElement;
  const $ = id => document.getElementById(id);
  const NS = "http://www.w3.org/2000/svg";

  /* ================= model: cosine schedule, data, exact denoiser ================= */
  const T = 1000, SC = 2;                       // data ×2: roughly unit variance per coordinate, like normalised data
  const A = t => Math.cos(t / T * Math.PI / 2);  // a_t, so a_t^2 = cos^2(pi/2 * t/T)
  const B = t => Math.sin(t / T * Math.PI / 2);  // b_t = sqrt(1 - a_t^2)
  const M = 500, N = 500, BMIN = 1e-3;
  const X0 = new Float64Array(2 * M);            // training set: points on the Swiss roll
  { const r = mulberry32(2024); for (let i = 0; i < M; i++) { const [x, y] = roll(rollU(r)); X0[2 * i] = x * SC; X0[2 * i + 1] = y * SC; } }

  // optimal denoiser for the empirical data set: x0hat = sum_i w_i x0_i, w_i ~ exp(-|x - a x0_i|^2 / 2b^2)
  const LW = new Float64Array(M);
  function den(x, y, a, b, out, o) {
    const inv = 1 / (2 * b * b); let mx = -Infinity;
    for (let i = 0, j = 0; i < M; i++, j += 2) {
      const dx = x - a * X0[j], dy = y - a * X0[j + 1], l = -(dx * dx + dy * dy) * inv;
      LW[i] = l; if (l > mx) mx = l;
    }
    const cut = mx - 36; let s = 0, sx = 0, sy = 0;
    for (let i = 0, j = 0; i < M; i++, j += 2) {
      const l = LW[i]; if (l < cut) continue;
      const w = Math.exp(l - mx); s += w; sx += w * X0[j]; sy += w * X0[j + 1];
    }
    out[o] = sx / s; out[o + 1] = sy / s;
  }
  // deterministic DDIM update t -> t2: x_t2 = a_t2 * x0hat + b_t2 * epshat
  const HAT = new Float64Array(2 * N);
  function ddim(Xc, t, t2, Xn) {
    const a = A(t), b = Math.max(B(t), BMIN), a2 = t2 <= 0 ? 1 : A(t2), b2 = t2 <= 0 ? 0 : B(t2);
    for (let j = 0; j < 2 * N; j += 2) {
      den(Xc[j], Xc[j + 1], a, b, HAT, j);
      const ex = (Xc[j] - a * HAT[j]) / b, ey = (Xc[j + 1] - a * HAT[j + 1]) / b;
      Xn[j] = a2 * HAT[j] + b2 * ex; Xn[j + 1] = a2 * HAT[j + 1] + b2 * ey;
    }
  }
  const grid = K => Array.from({ length: K + 1 }, (_, k) => Math.round(T * (1 - k / K)));
  function noise(seed) { const r = mulberry32(seed * 7919 + 13), z = new Float64Array(2 * N); for (let i = 0; i < 2 * N; i++) z[i] = gauss(r); return z; }

  /* ---------- metrics: distance to the roll, coverage, memorisation ---------- */
  const P = 1200, RP = new Float64Array(2 * P), RS = new Float64Array(P);
  for (let k = 0; k < P; k++) {
    const [x, y] = roll(k / (P - 1)); RP[2 * k] = x * SC; RP[2 * k + 1] = y * SC;
    if (k) RS[k] = RS[k - 1] + Math.hypot(RP[2 * k] - RP[2 * k - 2], RP[2 * k + 1] - RP[2 * k - 1]);
  }
  const LEN = RS[P - 1], COVR = .02 * SC, MEM = .001 * SC, NC = 240;
  // coverage probes: NC points evenly spaced by arc length along the roll
  const CP = new Float64Array(2 * NC);
  for (let i = 0, k = 0; i < NC; i++) {
    const s = (i + .5) / NC * LEN; while (k < P - 2 && RS[k + 1] < s) k++;
    const u = (s - RS[k]) / (RS[k + 1] - RS[k] || 1);
    CP[2 * i] = RP[2 * k] + u * (RP[2 * k + 2] - RP[2 * k]); CP[2 * i + 1] = RP[2 * k + 1] + u * (RP[2 * k + 3] - RP[2 * k + 1]);
  }
  function segD(px, py, k) {        // distance from a point to roll segment k..k+1
    const ax = RP[2 * k], ay = RP[2 * k + 1], bx = RP[2 * k + 2] - ax, by = RP[2 * k + 3] - ay;
    const L2 = bx * bx + by * by, u = L2 ? clamp01(((px - ax) * bx + (py - ay) * by) / L2) : 0;
    return Math.hypot(px - ax - u * bx, py - ay - u * by);
  }
  function nearRoll(px, py) {
    let best = 1e9, bk = 0;
    for (let k = 0; k < P; k++) { const dx = px - RP[2 * k], dy = py - RP[2 * k + 1], d = dx * dx + dy * dy; if (d < best) { best = d; bk = k; } }
    let r = Math.sqrt(best);
    if (bk > 0) r = Math.min(r, segD(px, py, bk - 1));
    if (bk < P - 1) r = Math.min(r, segD(px, py, bk));
    return r;
  }
  // d: mean distance to the roll (roll radius = 1); cov: share of roll probes with a sample within COVR;
  // mem: share of samples sitting on a training point (within MEM)
  function metrics(X, full) {
    let sd = 0;
    for (let j = 0; j < 2 * N; j += 2) sd += nearRoll(X[j], X[j + 1]);
    const out = { d: sd / N / SC, cov: 0, mem: 0 };
    if (!full) return out;
    let c = 0;
    for (let i = 0; i < 2 * NC; i += 2) {
      for (let j = 0; j < 2 * N; j += 2) { const dx = X[j] - CP[i], dy = X[j + 1] - CP[i + 1]; if (dx * dx + dy * dy < COVR * COVR) { c++; break; } }
    }
    let m = 0;
    for (let j = 0; j < 2 * N; j += 2) {
      for (let i = 0; i < 2 * M; i += 2) { const dx = X[j] - X0[i], dy = X[j + 1] - X0[i + 1]; if (dx * dx + dy * dy < MEM * MEM) { m++; break; } }
    }
    out.cov = c / NC; out.mem = m / N;
    return out;
  }

  /* ================= DOM ================= */
  const cv = $("dif-cv"), chainEl = $("dif-chain"), schedEl = $("dif-sched"), tIn = $("dif-t");
  const rT = $("dif-r-t"), rA = $("dif-r-a"), rB = $("dif-r-b"), modeEl = $("dif-mode"), demoEl = $("dif-demo");
  const verdict = $("dif-verdict"), runBtn = $("dif-run"), epsBtn = $("dif-eps"), lgEps = $("dif-lg-eps");
  const kBtns = [...fig.querySelectorAll(".difl-k")];
  const KS = [2, 5, 20, 100];
  const word = k => k % 10 >= 2 && k % 10 <= 4 && (k % 100 < 10 || k % 100 >= 20) ? "шага" : k % 10 === 1 && k % 100 !== 11 ? "шаг" : "шагов";
  const f2 = v => v.toFixed(2), f3 = v => v.toFixed(3), pct = v => Math.round(v * 100) + "%";

  /* ================= state ================= */
  let K = 20, seed = 1, Z = noise(seed);
  let mode = "q", tView = reduce ? 360 : 0, preview = null;
  let run = null, ramp = null, hold = null, fadeE = -1, eps = false;
  let demo = false, lastUser = -1e9, idleT = 0;
  const IDLE = 6000;
  const best = {};                    // last metrics per K (table cells)

  /* ---------- sounds ---------- */
  const play = fn => sfx.play(fn);
  let lastSnd = 0;
  const throttled = f => { const n = performance.now(); if (n - lastSnd > 42) { lastSnd = n; f(); } };
  const sndK = k => play(({ t, tone }) => { const f = 392 * [1, 1.1225, 1.26, 1.4983][KS.indexOf(k)]; tone(t, f, .18, "sine", .13); tone(t, f * 2, .1, "sine", .03); });
  const sndStart = () => play(({ t, noise }) => noise(t, .34, "bandpass", 3800, 1100, .9, .2));
  const sndEps = on => play(({ t, tone }) => tone(t, on ? 587.33 : 880, .16, "triangle", .1, on ? 880 : 587.33));
  function sndChord(d) {
    play(({ t, tone }) => {
      if (d > .12) { tone(t, 392, .9, "sine", .12); tone(t, 415.3, .9, "sine", .07); tone(t, 277.18, .9, "sine", .05); }       // clouded: semitone clash
      else if (d > .04) { tone(t, 392, 1, "sine", .13); tone(t, 587.33, 1, "sine", .08); }                                         // open fifth
      else { tone(t, 392, 1.3, "sine", .14); tone(t + .05, 493.88, 1.2, "sine", .09); tone(t + .1, 587.33, 1.2, "sine", .08); tone(t + .15, 783.99, 1, "sine", .04); } // G major
    });
  }

  /* ================= chain x0 ... xT ================= */
  const CT = [0, 200, 400, 600, 800, 1000];
  let cw = 0, cx0 = 0, cx1 = 0, cR = 22, thumbs = [], mkG = null, tkG = null, pLbl = null;
  const chainX = t => cx0 + t / T * (cx1 - cx0);
  function el(name, attrs, parent) { const e = document.createElementNS(NS, name); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; }
  function buildChain() {
    const w = chainEl.clientWidth; if (!w) return; cw = w;
    chainEl.setAttribute("viewBox", `0 0 ${w} 130`); chainEl.textContent = "";
    const pad = 6; cR = Math.max(13, Math.min(22, (w - 2 * pad) / 6 * .36));
    cx0 = pad + cR + 4; cx1 = w - pad - cR - 4;
    const top = el("text", { class: "lbl", x: cx0, y: 12, "text-anchor": "middle" }, chainEl); top.innerHTML = 'x<tspan dy="3" font-size="9">0</tspan>';
    const tT = el("text", { class: "lbl", x: cx1, y: 12, "text-anchor": "middle" }, chainEl); tT.innerHTML = 'x<tspan dy="3" font-size="9">T</tspan>';
    el("text", { class: "lbl", x: (cx0 + cx1) / 2, y: 12, "text-anchor": "middle" }, chainEl).textContent = w < 420 ? "q: добавляем шум →" : "q(xₜ | x₀): добавляем шум →";
    el("path", { class: "ln", d: `M${cx0 + 12} 22 H${cx1 - 12}` }, chainEl);
    el("path", { class: "hh", d: `M${cx0 - 2} 102 l7 -3.5 v7 z` }, chainEl);
    tkG = el("g", {}, chainEl);
    mkG = el("g", {}, chainEl);
    el("line", { class: "mk", x1: 0, x2: 0, y1: 24, y2: 36 }, mkG);
    el("path", { class: "mkh", d: "M-4 22 L4 22 L0 28 Z" }, mkG);
    thumbs = CT.map((t, i) => {
      const x = chainX(t);
      const g = el("g", { class: "th", tabindex: 0, role: "button", "aria-label": `Перейти к шагу t = ${t}` }, chainEl);
      el("title", {}, g).textContent = `t = ${t}`;
      el("circle", { class: "bg", cx: x, cy: 60, r: cR }, g);
      el("circle", { class: "ring", cx: x, cy: 60, r: cR + 4, fill: "none" }, g);
      const dots = el("path", { class: "dots" }, g);
      el("text", { class: "tv", x, y: 96, "text-anchor": "middle" }, g).textContent = "t=" + t;
      el("rect", { x: x - cR - 6, y: 32, width: 2 * cR + 12, height: 68, fill: "transparent" }, g);   // generous hit area
      g.addEventListener("click", () => { user(); jumpTo(t); });
      g.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); user(); jumpTo(t); } });
      g.addEventListener("pointerenter", e => { if (e.pointerType !== "mouse") return; user(); preview = t; throttled(() => sfx.scrub(t / T)); kick(); });
      g.addEventListener("pointerleave", () => { if (preview !== null) { preview = null; kick(); } });
      return { g, dots, t, x };
    });
    pLbl = el("text", { class: "lbl c", x: (cx0 + cx1) / 2, y: 124, "text-anchor": "middle" }, chainEl);
    chainDots(); chainTicks(); chainMark(curT());
  }
  function chainDots() {
    const s = (cR - 2.5) / 3.1;
    thumbs.forEach(th => {
      const a = A(th.t), b = B(th.t); let d = "";
      for (let j = 0; j < 2 * N; j += 12) {
        const x = (a * X0[j] + b * Z[j]) * s, y = (a * X0[j + 1] + b * Z[j + 1]) * s;
        if (x * x + y * y > (cR - 2) * (cR - 2)) continue;
        d += `M${(th.x + x - 1).toFixed(1)} ${(60 - y).toFixed(1)}a1 1 0 1 0 2 0a1 1 0 1 0-2 0`;
      }
      th.dots.setAttribute("d", d);
    });
  }
  let hops = [], hopOn = -2;
  function chainTicks() {
    if (!tkG) return; tkG.textContent = ""; hopOn = -2;
    const g = grid(K);
    hops = g.slice(0, -1).map((t, k) => {
      const x1 = chainX(t), x2 = chainX(g[k + 1]), h = Math.min(9, (x1 - x2) * .3 + .8);
      return el("path", { class: "hop", d: `M${x1.toFixed(1)} 102 Q${((x1 + x2) / 2).toFixed(1)} ${(102 + 2 * h).toFixed(1)} ${x2.toFixed(1)} 102` }, tkG);
    });
    pLbl.innerHTML = `← p<tspan dy="3" font-size="9">θ</tspan><tspan dy="-3">: убираем шум · ${K} ${word(K)} DDIM</tspan>`;
  }
  function chainHops(s) {          // s = number of finished hops, -1 = idle (all neutral)
    if (s === hopOn) return; hopOn = s;
    hops.forEach((h, i) => h.setAttribute("class", s < 0 ? "hop" : i < s ? "hop done" : "hop todo"));
  }
  let lastMk = -1;
  function chainMark(t) {
    if (!mkG || t === lastMk) return; lastMk = t;
    mkG.setAttribute("transform", `translate(${chainX(t).toFixed(1)} 0)`);
    const on = Math.round(t / 200);
    thumbs.forEach((th, i) => th.g.classList.toggle("on", i === on));
  }

  /* ================= schedule plot ================= */
  let sMk = null, sDa = null, sDb = null;
  const sx = t => 14 + t / T * 168, sy = v => 50 - v * 42;
  function buildSched() {
    schedEl.textContent = "";
    el("path", { class: "ax", d: `M14 50 H182 M14 8 V50` }, schedEl);
    let da = "", db = "";
    for (let t = 0; t <= T; t += 20) { da += (t ? "L" : "M") + sx(t).toFixed(1) + " " + sy(A(t)).toFixed(1); db += (t ? "L" : "M") + sx(t).toFixed(1) + " " + sy(B(t)).toFixed(1); }
    el("path", { class: "ca", d: da }, schedEl); el("path", { class: "cb", d: db }, schedEl);
    el("text", { class: "tl a", x: 2, y: 12 }, schedEl).textContent = "a";
    el("text", { class: "tl", x: 187, y: 11 }, schedEl).textContent = "b";
    el("text", { class: "tl", x: 2, y: 53 }, schedEl).textContent = "0";
    el("text", { class: "tl", x: 14, y: 62, "text-anchor": "middle" }, schedEl).textContent = "0";
    el("text", { class: "tl", x: 98, y: 62, "text-anchor": "middle" }, schedEl).textContent = "t";
    el("text", { class: "tl", x: 182, y: 62, "text-anchor": "middle" }, schedEl).textContent = "1000";
    sMk = el("line", { class: "mk", y1: 6, y2: 50 }, schedEl);
    sDa = el("circle", { class: "da", r: 3 }, schedEl); sDb = el("circle", { class: "db", r: 3 }, schedEl);
  }
  buildSched();

  /* ================= readouts ================= */
  let lastRead = -1;
  function readout(t) {
    const tv = Math.round(t);
    if (tv === lastRead) return; lastRead = tv;
    const a = A(tv), b = B(tv);
    rT.textContent = tv; rA.textContent = f2(a); rB.textContent = f2(b);
    const x = sx(tv).toFixed(1);
    sMk.setAttribute("x1", x); sMk.setAttribute("x2", x);
    sDa.setAttribute("cx", x); sDa.setAttribute("cy", sy(a).toFixed(1));
    sDb.setAttribute("cx", x); sDb.setAttribute("cy", sy(b).toFixed(1));
    const sv = Math.round(tv / 10) * 10;
    if (+tIn.value !== sv) tIn.value = sv;
    tIn.setAttribute("aria-valuetext", `t = ${tv}, сигнал ${f2(a)}, шум ${f2(b)}`);
    chainMark(tv);
  }
  let lastMode = "";
  function setMode(html) { if (html !== lastMode) { lastMode = html; modeEl.innerHTML = html; } }

  function setCell(k, m) {
    best[k] = m;
    const mEl = $("dif-m" + k), bEl = $("dif-b" + k); if (!mEl) return;
    mEl.innerHTML = `<span>${f3(m.d)}</span><span class="c">${pct(m.cov)}</span>`;
    bEl.style.width = pct(m.cov);
    $("dif-k" + k).setAttribute("aria-label", `${k} ${word(k)}: расстояние до спирали ${f3(m.d)}, покрыто ${pct(m.cov)} спирали`);
  }
  function setK(k) { K = k; kBtns.forEach(b => b.setAttribute("aria-pressed", String(+b.dataset.k === k))); chainTicks(); }

  function finalText(k, m) {
    const head = `<span class="n">${k} ${word(k)}</span> = ${k} ${k < 5 ? "прогона" : "прогонов"} сети: расстояние до спирали <span class="n">${f3(m.d)}</span>, покрыто <span class="n">${pct(m.cov)}</span>. `;
    let tail;
    if (m.d > .12) tail = "Быстро, но мимо: крупный шаг берёт усреднённый x̂<sub>0</sub>, и точки оседают между витками.";
    else if (m.d > .04) tail = "Спираль угадывается, но витки размыты: шагов ещё мало.";
    else if (k <= 30) tail = "Точки на спирали за умеренную цену: на этом компромиссе и живут быстрые сэмплеры.";
    else tail = `Точно на спирали, но дорого, и <span class="n">${pct(m.mem)}</span> точек легли прямо на обучающие: это уже запоминание.`;
    return head + tail;
  }

  /* ================= canvas ================= */
  let ctx = null, W = 0, H = 0, kpx = 1;
  function sizeCanvas() {
    const r = cv.getBoundingClientRect(); if (!r.width) return false;
    const dpr = Math.min(2, devicePixelRatio || 1);
    W = r.width; H = r.height; cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
    ctx = cv.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    kpx = H / (2 * 2.95); fieldKey = "";
    return true;
  }
  const PX = x => W / 2 + x * kpx, PY = y => H / 2 - y * kpx;

  // positions currently on screen (into out); returns current t
  const POS = new Float64Array(2 * N);
  function curT() {
    if (preview !== null) return preview;
    if (mode === "q" || !run) return tView;
    if (run.re < run.ren) return T;
    if (run.done || reduce) return 0;
    const s = Math.min(run.K, Math.floor(run.e / run.sd));
    if (s >= run.K) return 0;
    const f = ease(clamp01((run.e - s * run.sd) / run.sd));
    return run.g[s] + (run.g[s + 1] - run.g[s]) * f;
  }
  function fwd(t, out) { const a = A(t), b = B(t); for (let j = 0; j < 2 * N; j++) out[j] = a * X0[j] + b * Z[j]; }
  function positions(out) {
    if (preview !== null || mode === "q" || !run) { fwd(preview !== null ? preview : tView, out); return; }
    if (run.re < run.ren) { const f = ease(run.re / run.ren); for (let j = 0; j < 2 * N; j++) out[j] = run.from[j] + (Z[j] - run.from[j]) * f; return; }
    if (run.done) { out.set(run.X[run.K]); return; }
    const s = Math.min(run.K, Math.floor(run.e / run.sd));
    if (s >= run.K || !run.X[s + 1]) { out.set(run.X[Math.min(s, run.X.length - 1)]); return; }
    const f = ease(clamp01((run.e - s * run.sd) / run.sd)), X1 = run.X[s], X2 = run.X[s + 1];
    for (let j = 0; j < 2 * N; j++) out[j] = X1[j] + (X2[j] - X1[j]) * f;
  }

  // ε̂ field on a grid: arrow from x toward a_t * x0hat(x)  (= -b_t * epshat)
  let field = null, fieldKey = "";
  function computeField(t) {
    const key = `${Math.round(t)}|${W}|${H}`; if (key === fieldKey) return field; fieldKey = key;
    const cols = Math.max(12, Math.round(W / 27)), cell = W / cols, rows = Math.max(1, Math.round(H / cell)), ch = H / rows;
    const a = A(t), b = Math.max(B(t), .01), o = new Float64Array(2); const arr = []; let vmax = 1e-9;
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      const px = (c + .5) * cell, py = (r + .5) * ch, x = (px - W / 2) / kpx, y = (H / 2 - py) / kpx;
      den(x, y, a, b, o, 0);
      const vx = a * o[0] - x, vy = a * o[1] - y, m = Math.hypot(vx, vy);
      if (m > vmax) vmax = m; arr.push([px, py, vx, vy, m]);
    }
    field = { arr, vmax, cell: Math.min(cell, ch) };
    return field;
  }
  function drawField(C, t) {
    const F = computeField(t), L = F.cell * .62, buckets = [[], [], [], [], [], []];
    for (const [px, py, vx, vy, m] of F.arr) {
      const rel = m / F.vmax, len = Math.min(L, m * kpx); if (len < 1.5) continue;
      buckets[Math.min(5, Math.floor(Math.sqrt(rel) * 6))].push([px, py, vx / m, -vy / m, len]);
    }
    ctx.lineWidth = 1.15; ctx.lineCap = "round";
    buckets.forEach((bk, i) => {
      if (!bk.length) return;
      ctx.strokeStyle = rgba(C.dif, (C.dark ? .08 : .12) + i * (C.dark ? .085 : .1));
      ctx.beginPath();
      for (const [px, py, ux, uy, len] of bk) {
        const x1 = px - ux * len / 2, y1 = py - uy * len / 2, x2 = px + ux * len / 2, y2 = py + uy * len / 2, h = Math.min(4, len * .45);
        ctx.moveTo(x1, y1); ctx.lineTo(x2, y2);
        ctx.moveTo(x2 - ux * h - uy * h * .6, y2 - uy * h + ux * h * .6); ctx.lineTo(x2, y2); ctx.lineTo(x2 - ux * h + uy * h * .6, y2 - uy * h - ux * h * .6);
      }
      ctx.stroke();
    });
  }
  function dots(X, r, style) {
    ctx.fillStyle = style; ctx.beginPath();
    for (let j = 0; j < X.length; j += 2) { const x = PX(X[j]), y = PY(X[j + 1]); ctx.moveTo(x + r, y); ctx.arc(x, y, r, 0, 6.2832); }
    ctx.fill();
  }
  function trail(list, alpha, C) {   // list: arrays of positions, newest first
    ctx.lineWidth = 1; ctx.lineCap = "round";
    for (let s = 0; s + 1 < list.length; s++) {
      const p = list[s], q = list[s + 1], al = alpha * [.5, .26, .12][s];
      if (al < .01) continue;
      ctx.strokeStyle = rgba(C.dif, C.dark ? al : Math.min(1, al * 1.3)); ctx.beginPath();
      for (let j = 0; j < 2 * N; j += 2) { ctx.moveTo(PX(p[j]), PY(p[j + 1])); ctx.lineTo(PX(q[j]), PY(q[j + 1])); }
      ctx.stroke();
    }
  }
  function trailAlpha() {
    if (!run || !run.done) return 1;
    if (reduce) return .55;
    if (fadeE < 0) return 0;
    return 1 - clamp01((fadeE - 900) / 900);
  }
  function draw() {
    if (!ctx || !W) return;
    const C = S.colors(), t = curT();
    ctx.globalCompositeOperation = "source-over"; ctx.clearRect(0, 0, W, H);
    dots(X0, 1.25, rgba(C.fg, C.dark ? .2 : .26));                               // p_data: the training set
    if (eps) drawField(C, t);
    positions(POS);
    if (C.dark) ctx.globalCompositeOperation = "lighter";
    const pr = W < 480 ? 1.6 : 1.8;
    // trails during / right after sampling
    if (preview === null && mode === "p" && run && run.re >= run.ren) {
      const ta = trailAlpha();
      if (ta > 0) {
        let s = run.done || reduce ? run.K : Math.min(run.K, Math.floor(run.e / run.sd));
        s = Math.min(s, run.X.length - 1);
        const list = run.done || reduce ? [] : [POS];
        for (let i = s; i >= 0 && list.length < 4; i--) list.push(run.X[i]);
        trail(list, ta, C);
      }
    }
    if (ramp && ramp.prev && ramp.e < 600) {                                        // demo loop: dissolve the last result
      const f = ramp.e / 600;
      dots(ramp.prev, pr, rgba(C.dif, (C.dark ? .75 : .85) * (1 - f)));
      dots(POS, pr, rgba(C.dif, (C.dark ? .75 : .85) * f));
    } else dots(POS, pr, rgba(C.dif, C.dark ? .75 : .85));
    ctx.globalCompositeOperation = "source-over";
    // labels / readouts
    readout(t);
    if (preview !== null) setMode(`q(x<sub>t</sub> | x<sub>0</sub>) · предпросмотр t = ${preview}`);
    else if (mode === "q" || !run) setMode(`q(x<sub>t</sub> | x<sub>0</sub>) · данные + шум`);
    else if (run.re < run.ren) setMode(`p<sub>θ</sub> · новый шум 𝒩(0, I)`);
    else if (run.done) setMode(`p<sub>θ</sub> · DDIM, ${run.K} ${word(run.K)} · готово`);
    else setMode(`p<sub>θ</sub> · DDIM, шаг ${Math.min(run.K, Math.floor(run.e / run.sd) + 1)} из ${run.K}`);
    chainHops(preview !== null || mode === "q" || !run || run.done ? -1 : run.re < run.ren ? 0 : Math.min(run.K, Math.floor(run.e / run.sd) + 1));
  }

  /* ================= sampler runs ================= */
  const stepDur = k => k <= 2 ? 1150 : k <= 5 ? 640 : k <= 20 ? 200 : 50;
  function startRun(k, fresh, isDemo) {
    setK(k);
    const from = new Float64Array(2 * N); positions(from);
    if (fresh) { seed++; Z = noise(seed); chainDots(); }
    let same = true; for (let j = 0; j < 2 * N && same; j++) if (Math.abs(from[j] - Z[j]) > 1e-6) same = false;
    run = { K: k, g: grid(k), X: [Z], e: 0, re: 0, ren: same || reduce ? 0 : 420, sd: stepDur(k), k: -1, done: false, demo: !!isDemo, from };
    mode = "p"; preview = null; ramp = null; hold = null; fadeE = -1;
    verdict.setAttribute("aria-busy", "true");
    verdict.innerHTML = reduce ? "Считаю…" : `Старт: 500 точек чистого шума <span class="mo">𝒩(0, I)</span>, ${k} ${word(k)} до данных.`;
    if (!run.demo) sndStart();
    kick();
  }
  function computeNext(r) { const Xn = new Float64Array(2 * N), k = r.X.length - 1; ddim(r.X[k], r.g[k], r.g[k + 1], Xn); r.X.push(Xn); }
  function onStep(r, s) {
    if (s >= r.K) return;
    if (!r.demo) throttled(() => sfx.dif(Math.min(37, Math.round(37 * s / r.K)), 40));
    const m = metrics(r.X[s], false);
    verdict.innerHTML = `Шаг <span class="n">${s + 1}</span> из ${r.K}, t: ${r.g[s]} → ${r.g[s + 1]}. Сейчас точки в среднем в <span class="n">${f3(m.d)}</span> от спирали.`;
  }
  function finish(r) {
    r.done = true; fadeE = 0;
    const m = metrics(r.X[r.K], true); setCell(r.K, m);
    verdict.setAttribute("aria-busy", "false");
    verdict.innerHTML = finalText(r.K, m);
    cv.setAttribute("aria-label", `Результат сэмплера за ${r.K} ${word(r.K)}: расстояние до спирали ${f3(m.d)}, покрыто ${pct(m.cov)} спирали`);
    if (!r.demo) { sndChord(m.d); lastUser = performance.now(); clearTimeout(idleT); idleT = setTimeout(tryDemo, IDLE + 60); }
    if (r.demo && demo) hold = { e: 0, dur: 3200 };
  }
  function startRamp() {   // demo loop: data -> noise along q, then sample back
    const prev = run && run.done ? run.X[run.K] : null;
    seed++; Z = noise(seed); chainDots();
    run = null; mode = "q"; tView = 0; hold = null; preview = null;
    ramp = { e: 0, prev, dur: 2600 };
    verdict.setAttribute("aria-busy", "false");
    verdict.innerHTML = "Прямой процесс: к обучающим точкам подмешивается шум, к t = 1000 от спирали ничего не остаётся.";
  }

  /* ================= loop ================= */
  let raf = 0, last = 0, inView = false;
  const visible = () => inView && !panel.hidden && !document.hidden;
  const busy = () => !!(ramp || hold || (run && !run.done) || (run && run.done && !reduce && fadeE >= 0 && fadeE < 1800));
  function update(dt) {
    if (ramp) {
      ramp.e += dt;
      tView = Math.round(ease(clamp01((ramp.e - 600) / ramp.dur)) * T);
      if (ramp.e >= ramp.dur + 1000) { ramp = null; if (demo) startRun(K, false, true); }
    }
    if (run && !run.done) {
      if (run.re < run.ren) run.re = Math.min(run.ren, run.re + dt);
      else if (reduce) {
        const t0 = performance.now();
        while (run.X.length <= run.K && performance.now() - t0 < 8) computeNext(run);
        if (run.X.length > run.K) finish(run);
      } else {
        run.e += dt;
        const s = Math.min(run.K, Math.floor(run.e / run.sd));
        while (run.X.length <= Math.min(run.K, s + 1)) computeNext(run);
        if (s !== run.k) { for (let q = run.k + 1; q <= s; q++) onStep(run, q); run.k = s; }
        if (s >= run.K) finish(run);
      }
    } else if (run && run.done && fadeE >= 0) fadeE += dt;
    if (hold) { hold.e += dt; if (hold.e >= hold.dur) { hold = null; if (demo) startRamp(); } }
  }
  function loop(now) {
    raf = 0;
    if (!visible()) { last = 0; return; }
    const dt = last ? Math.min(64, now - last) : 16; last = now;
    update(dt); draw();
    if (busy()) raf = requestAnimationFrame(loop); else last = 0;
  }
  function kick() { if (!raf && visible()) raf = requestAnimationFrame(loop); }

  /* ================= demo / interaction ================= */
  const demoAllowed = () => !reduce && visible() && performance.now() - lastUser > IDLE;
  function startDemo() {
    demo = true; demoEl.hidden = false;
    if (run && !run.done) run.demo = true; else startRamp();
    kick();
  }
  function stopDemo() {
    if (!demo) return;
    demo = false; demoEl.hidden = true;
    hold = null;
    if (ramp) { ramp = null; verdict.innerHTML = "Демо остановлено. Потяни t или выбери число шагов."; }
    if (run) run.demo = false;
  }
  function tryDemo() {
    clearTimeout(idleT);
    if (reduce) return;
    if (demoAllowed() && !(run && !run.done) && preview === null) { if (!demo) startDemo(); }
    else idleT = setTimeout(tryDemo, 1000);
  }
  function user() {
    lastUser = performance.now(); fig.classList.add("used");
    stopDemo();
    clearTimeout(idleT); if (!reduce) idleT = setTimeout(tryDemo, IDLE + 60);
  }
  fig.addEventListener("pointerdown", user, true);
  fig.addEventListener("keydown", e => { if (e.key !== "Tab" && e.key !== "Shift") user(); }, true);

  const ABORT = "Сэмплер остановлен. Сейчас на холсте прямой процесс q: данные плюс шум уровня t.";
  function toQ(t) {
    if (run && !run.done) { verdict.setAttribute("aria-busy", "false"); verdict.textContent = ABORT; }
    run = null; ramp = null; hold = null; preview = null; mode = "q"; tView = t;
    throttled(() => sfx.scrub(t / T));
    kick();
  }
  const jumpTo = t => toQ(t);
  tIn.addEventListener("input", () => { user(); toQ(+tIn.value); });
  kBtns.forEach(b => b.addEventListener("click", () => { user(); const k = +b.dataset.k; sndK(k); startRun(k, true, false); }));
  runBtn.addEventListener("click", () => { user(); startRun(K, true, false); });
  epsBtn.addEventListener("click", () => {
    user(); eps = !eps;
    epsBtn.setAttribute("aria-pressed", String(eps)); lgEps.hidden = !eps; sndEps(eps);
    kick();
  });

  /* ---------- precompute the step-count table in small idle chunks ---------- */
  (function precompute() {
    const queue = [20, 2, 5, 100]; let cur = null;
    const idle = window.requestIdleCallback || (f => setTimeout(() => f({ timeRemaining: () => 8 }), 16));
    function pump(dl) {
      const t0 = performance.now();
      while (performance.now() - t0 < 7) {
        if (!cur) { const k = queue.shift(); if (k === undefined) return; if (best[k]) continue; cur = { K: k, g: grid(k), X: [noise(1)] }; }
        if (cur.X.length <= cur.K) computeNext(cur);
        else { if (!best[cur.K]) setCell(cur.K, metrics(cur.X[cur.K], true)); cur = null; }
      }
      idle(pump, { timeout: 300 });
    }
    idle(pump, { timeout: 600 });
  })();

  /* ---------- sizing / visibility ---------- */
  function refresh() { if (sizeCanvas()) { buildChain(); draw(); } }
  new ResizeObserver(() => { const r = cv.getBoundingClientRect(); if (r.width && (r.width !== W || r.height !== H)) { sizeCanvas(); draw(); } }).observe(cv);
  new ResizeObserver(() => { if (chainEl.clientWidth && chainEl.clientWidth !== cw) buildChain(); }).observe(chainEl);
  function onVis() {
    if (!visible()) return;
    if (!W) refresh(); else draw();
    if (!demo && demoAllowed() && !(run && !run.done)) startDemo();
    kick();
  }
  new IntersectionObserver(es => { inView = es[es.length - 1].isIntersecting; onVis(); }, { threshold: .05 }).observe(fig);
  document.addEventListener("visibilitychange", onVis);
  document.addEventListener("sem:tab", e => { if (e.detail === "dif") requestAnimationFrame(() => { refresh(); onVis(); }); });
  S.onTheme(() => { fieldKey = ""; draw(); });
  readout(tView);
})();
