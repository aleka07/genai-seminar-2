(() => {
  "use strict";
  const S = window.__sem; if (!S) return;
  const { sfx, rgba, reduce } = S;
  const $ = id => document.getElementById(id);
  const fig = $("gan-fig"); if (!fig) return;
  const panel = fig.closest('[role="tabpanel"]') || fig;

  /* ================= 1. The model: a real 1D GAN =================
     p_data = 0.5 N(-1.3, 0.32) + 0.5 N(+1.3, 0.32).
     G: x = mu + s*z (one narrow bell; or two bells chosen by a coin flip of z).
     D(x) = sigmoid(b + sum_k w_k phi_k(x)), phi_k = RBF bumps (7 of them, width 0.9):
       a low-capacity discriminator, trained by SGD on minibatches -> it sees a blurred world and lags.
     Each iteration: one D ascent step on mean log D(x) + mean log(1 - D(G(z))),
     then one G step (with momentum) on mean log D(G(z)) (non-saturating loss);
     d/dmu log D(mu + s z) = (1 - D) f'(x)  -- the reparameterisation trick. */
  /*SIM-BEGIN*/
  const P = { K: 7, w: .9, lrD: .15, lrG: .04, mom: .8, s: .26, wd: .002, Nb: 128, sep: 1.3, sd: .32 };
  const CEN = Array.from({ length: P.K }, (_, k) => -3.2 + 6.4 * k / (P.K - 1));
  const IW2 = 1 / (P.w * P.w);
  const sig = z => 1 / (1 + Math.exp(-z));
  const npdf = (x, m, s) => Math.exp(-.5 * ((x - m) / s) ** 2) / (s * 2.5066283);
  const pdata = x => .5 * npdf(x, -P.sep, P.sd) + .5 * npdf(x, P.sep, P.sd);
  const HMAX = 600;
  function makeSim(rng, gauss, two) {
    const W = new Float64Array(P.K), ph = new Float64Array(P.K), gW = new Float64Array(P.K);
    const real = new Float64Array(P.Nb), fake = new Float64Array(P.Nb), comp = new Uint8Array(P.Nb);
    const m = two ? [-.25, .35] : [.25], v = m.map(() => 0);
    const H = { dr: [], df: [], m0: [], m1: [] };
    let b = 0;
    const st = { it: 0, dr: .5, df: .5, ld: 2 * Math.LN2, lg: Math.LN2, two, m, v, real, fake, comp, H };
    function f(x) { let s = b; for (let k = 0; k < P.K; k++) { const d = x - CEN[k]; s += W[k] * Math.exp(-.5 * d * d * IW2); } return s; }
    function fp(x) { let s = 0; for (let k = 0; k < P.K; k++) { const d = x - CEN[k]; s -= W[k] * d * IW2 * Math.exp(-.5 * d * d * IW2); } return s; }
    function feat(x) { let s = b; for (let k = 0; k < P.K; k++) { const d = x - CEN[k]; ph[k] = Math.exp(-.5 * d * d * IW2); s += W[k] * ph[k]; } return s; }
    st.D = x => sig(f(x));
    st.step = () => {
      const N = P.Nb;
      for (let i = 0; i < N; i++) {
        real[i] = (rng() < .5 ? -P.sep : P.sep) + P.sd * gauss(rng);
        const c = two && rng() < .5 ? 1 : 0; comp[i] = c;
        fake[i] = m[c] + P.s * gauss(rng);
      }
      gW.fill(0); let gb = 0, dr = 0, df = 0, ld = 0, lg = 0;
      for (let i = 0; i < N; i++) {
        const D = sig(feat(real[i])); dr += D; ld -= Math.log(D + 1e-9);
        for (let k = 0; k < P.K; k++) gW[k] += (1 - D) * ph[k]; gb += 1 - D;
      }
      for (let i = 0; i < N; i++) {
        const D = sig(feat(fake[i])); df += D; ld -= Math.log(1 - D + 1e-9); lg -= Math.log(D + 1e-9);
        for (let k = 0; k < P.K; k++) gW[k] -= D * ph[k]; gb -= D;
      }
      for (let k = 0; k < P.K; k++) W[k] += P.lrD * (gW[k] / N - P.wd * W[k]);
      b += P.lrD * gb / N;
      const gm = [0, 0], n = [0, 0];
      for (let i = 0; i < N; i++) { const x = fake[i], c = comp[i]; gm[c] += (1 - sig(f(x))) * fp(x); n[c]++; }
      for (let c = 0; c < m.length; c++) {
        v[c] = P.mom * v[c] + P.lrG * gm[c] / Math.max(1, n[c]);
        m[c] = Math.max(-2.8, Math.min(2.8, m[c] + v[c]));
      }
      st.it++; st.dr = dr / N; st.df = df / N; st.ld = ld / N; st.lg = lg / N;
      H.dr.push(st.dr); H.df.push(st.df); H.m0.push(m[0]); H.m1.push(two ? m[1] : NaN);
      if (H.dr.length > HMAX) { H.dr.shift(); H.df.shift(); H.m0.shift(); H.m1.shift(); }
    };
    return st;
  }
  /*SIM-END*/

  /* ---- events: settle on a mode / leave / hop (single-bell G only) ---- */
  let seed = 11, sim, ev;
  function newEv() { return { where: 0, last: 0, since: 0, settled: false, settledAt: -1e9, hopAt: -1e9, from: 0, to: 0, leftAt: -1e9, hops: 0, sliding: false }; }
  function reset(two) {
    sim = makeSim(S.mulberry32(seed++), S.gauss, two);
    ev = newEv(); dirty = true;
  }
  let dirty = true;
  function stepOnce(quiet) {
    sim.step();
    dirty = true;
    if (!quiet) tickSound();
    if (sim.two) return;
    const mu = sim.m[0], vel = sim.v[0], it = sim.it;
    const where = mu > .75 ? 1 : mu < -.75 ? -1 : 0;
    if (where !== ev.where) {
      if (ev.where !== 0 && where === 0) { ev.leftAt = it; if (ev.settled && !quiet) jumpSound(-ev.where); }
      if (where !== 0 && ev.last !== 0 && where !== ev.last) { ev.hopAt = it; ev.from = ev.last; ev.to = where; ev.hops++; ev.sliding = false; }
      if (where !== 0) ev.sliding = false;
      if (where !== 0) ev.last = where;
      ev.where = where; ev.since = it; ev.settled = false;
    }
    if (where !== 0 && !ev.settled && it - ev.since > 20 && Math.abs(vel) < .012 && Math.abs(Math.abs(mu) - P.sep) < .35) {
      ev.settled = true; ev.settledAt = it; if (!quiet) landSound();
    }
    if (ev.settled && where !== 0 && !ev.sliding && it - ev.settledAt > 60 && Math.abs(vel) > .02) ev.sliding = true;
  }

  /* ================= 2. Sounds (only when the page sound is on) ================= */
  let lastTick = 0;
  function tickSound() {
    if (!sfx.on) return;
    const now = performance.now(); if (now - lastTick < 120) return; lastTick = now;
    const df = sim.df;
    sfx.play(({ t, tone }) => tone(t, 1200 + 1000 * df, .025, "sine", .02));
  }
  function landSound() {   // E + F rub, then F resolves up to B: a fifth over E
    sfx.play(({ t, tone }) => {
      tone(t, 329.63, .95, "triangle", .085);
      tone(t, 349.23, .26, "triangle", .07);
      tone(t + .22, 493.88, .75, "sine", .075);
    });
  }
  function jumpSound(dir) {
    sfx.play(({ t, tone, noise }) => {
      const a = dir > 0 ? 261.63 : 523.25, b = dir > 0 ? 523.25 : 261.63;
      tone(t, a, .5, "sine", .11, b);
      noise(t, .4, "bandpass", dir > 0 ? 500 : 2600, dir > 0 ? 2600 : 500, 1.4, .05);
    });
  }
  const blockPitch = { z: 261.63, g: 329.63, xf: 392, x: 440, d: 523.25, v: 587.33, grad: 659.25 };
  let lastHover = 0;
  function hoverSound(k) {
    const now = performance.now(); if (now - lastHover < 60) return; lastHover = now;
    sfx.play(({ t, tone }) => { tone(t, blockPitch[k] || 392, .28, "sine", .07); });
  }

  /* ================= 3. Diagram: live flow, verdict, gradient pulse, explanations ================= */
  const svg = $("gan-svg"), NS = "http://www.w3.org/2000/svg";
  const paths = ["gan-p1", "gan-p2", "gan-p3", "gan-p4", "gan-p5", "gan-pb"].map($);
  const pulse = $("gan-pulse"), vd = $("gan-vd"), dbox = $("gan-dbox"), parts = $("gan-parts");
  const blocks = [...svg.querySelectorAll(".ganl-b")];
  const gBlock = blocks.find(b => b.dataset.k === "g");
  const dots = [];
  for (let i = 0; i < 7; i++) {
    const c = document.createElementNS(NS, "circle");
    c.setAttribute("r", i === 6 ? 3 : 3.6);
    c.setAttribute("class", i < 3 || i === 6 ? "d-c" : "");
    if (i >= 3 && i < 6) c.setAttribute("fill", "var(--fg)");
    c.style.opacity = 0; parts.appendChild(c); dots.push(c);
  }
  let lens = null;
  function pt(i, u) {
    if (!lens) lens = paths.map(p => p.getTotalLength());
    const q = paths[i].getPointAtLength(Math.max(0, Math.min(1, u)) * lens[i]); return q;
  }
  const PERIOD = 2.5;  // seconds per shown batch (the simulator itself runs far faster)
  let cycleT0 = 0, lastPhase = 0, shown = .5, from = .5, target = .5, tweenT = -1;
  function place(c, p, a) { c.setAttribute("cx", p.x.toFixed(1)); c.setAttribute("cy", p.y.toFixed(1)); c.style.opacity = a; }
  const sm = u => u <= 0 ? 0 : u >= 1 ? 1 : u * u * (3 - 2 * u);
  function animDiagram(now) {
    const tt = (now - cycleT0) / 1000;
    const ph = tt % PERIOD;
    if (ph < lastPhase) { /* new batch */ }
    // fakes: z -> G (hidden inside) -> x~ -> D
    for (let j = 0; j < 3; j++) {
      const p = ph - j * .1, c = dots[j];
      if (p < 0 || p > 1.55) c.style.opacity = 0;
      else if (p < .4) place(c, pt(0, sm(p / .4)), .95);
      else if (p < .7) c.style.opacity = 0;
      else if (p < 1.05) place(c, pt(1, sm((p - .7) / .35)), .95);
      else place(c, pt(2, sm((p - 1.05) / .5)), .95);
    }
    // reals: x -> D
    for (let j = 0; j < 3; j++) {
      const p = ph - 1.05 - j * .1, c = dots[3 + j];
      if (p < 0 || p > .5) c.style.opacity = 0; else place(c, pt(3, sm(p / .5)), .85);
    }
    // verdict leaves D
    const pv = ph - 1.65, cv = dots[6];
    if (pv < 0 || pv > .25) cv.style.opacity = 0; else place(cv, pt(4, sm(pv / .25)), 1);
    if (lastPhase < 1.65 && ph >= 1.65) {
      setVerdict(sim.df, true);
      dbox.classList.add("flash"); setTimeout(() => dbox.classList.remove("flash"), 260);
    }
    // gradient pulse back to G
    const pg = ph - 1.85;
    if (pg >= 0 && pg <= .6) {
      const L = lens[5], u = sm(pg / .6);
      pulse.style.strokeDasharray = `38 ${L + 60}`;
      pulse.style.strokeDashoffset = String(-u * (L - 38));
      pulse.style.opacity = String(pg < .5 ? .95 : .95 * (1 - (pg - .5) / .1));
    } else pulse.style.opacity = 0;
    if (lastPhase < 2.45 && ph >= 2.45) { gBlock.classList.add("gflash"); setTimeout(() => gBlock.classList.remove("gflash"), 300); }
    lastPhase = ph;
    // verdict number tween
    if (tweenT >= 0) {
      const u = Math.min(1, (now - tweenT) / 380); shown = from + (target - from) * sm(u);
      writeVerdict(shown); if (u >= 1) tweenT = -1;
    }
  }
  function writeVerdict(x) { vd.textContent = x.toFixed(2); vd.classList.toggle("lo", x < .4); }
  function setVerdict(x, tween) {
    target = x;
    if (tween && !reduce) { from = shown; tweenT = performance.now(); }
    else { shown = x; writeVerdict(x); }
    if (cur === "v") showExp("v", true);
  }
  function hideDots() { dots.forEach(d => d.style.opacity = 0); pulse.style.opacity = 0; }

  const EXP = {
    z: ["z", "Шум z ~ 𝒩(0, I). Каждый новый z даёт новый образец: случайность G превращает в объект."],
    g: ["G", "Генератор: нейросеть, которая делает из шума z объект x̃ = G(z). Реальных данных он не видит."],
    xf: ["x̃", "Подделка x̃ = G(z). Её показывают дискриминатору вперемешку с настоящими примерами."],
    x: ["x", "Реальные примеры из обучающей выборки. Их видит только D, генератор — никогда."],
    d: ["D", "Дискриминатор: выдаёт вероятность того, что вход настоящий. Учится ловить подделки."],
    v: ["D(x̃)", () => `Ответ D на подделки, сейчас ${shown.toFixed(2)}. 1 — «настоящее», 0 — «подделка», 0.5 — D не может отличить.`],
    grad: ["∇", "Градиент проходит сквозь D обратно в G: генератор сдвигается так, чтобы D чаще говорил «настоящее»."]
  };
  const ORDER = ["z", "g", "xf", "x", "d", "v", "grad"];
  const expLine = $("gan-exp-line"), expK = $("gan-exp-k"), expT = $("gan-exp-t");
  let cur = "", engaged = false, idleTimer = 0, cycI = 2;
  function showExp(k, silent) {
    const e = EXP[k]; if (!e) return;
    const txt = typeof e[1] === "function" ? e[1]() : e[1];
    if (k === cur) { expT.textContent = txt; }
    else {
      cur = k;
      blocks.forEach(b => b.classList.toggle("on", b.dataset.k === k));
      if (reduce || silent) { expK.textContent = e[0]; expT.textContent = txt; }
      else {
        expLine.classList.add("swap");
        setTimeout(() => { expK.textContent = e[0]; expT.textContent = typeof e[1] === "function" ? e[1]() : e[1]; expLine.classList.remove("swap"); }, 180);
      }
    }
  }
  function engage(k) {
    engaged = true; clearTimeout(idleTimer); fig.classList.add("used-d");
    if (k !== cur) hoverSound(k);
    showExp(k);
  }
  function release() { clearTimeout(idleTimer); idleTimer = setTimeout(() => { engaged = false; }, 5000); }
  blocks.forEach(b => {
    const k = b.dataset.k;
    b.addEventListener("pointerenter", () => engage(k));
    b.addEventListener("pointerleave", release);
    b.addEventListener("focus", () => engage(k));
    b.addEventListener("blur", release);
    b.addEventListener("click", () => { engage(k); release(); });
    b.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); engage(k); release(); } });
  });
  showExp("g", true);
  setInterval(() => {
    if (reduce || engaged || !active()) return;
    showExp(ORDER[cycI++ % ORDER.length]);
  }, 3400);

  /* ================= 4. Simulator canvas ================= */
  const cv = $("gan-cv"), sp1 = $("gan-sp1"), sp2 = $("gan-sp2");
  const itEl = $("gan-it"), r1 = $("gan-r1"), r2 = $("gan-r2");
  const cvs = [cv, sp1, sp2].map(c => ({ c, ctx: null, w: 0, h: 0 }));
  function size(o) {
    const b = o.c.getBoundingClientRect(); if (!b.width) return false;
    const dpr = Math.min(2, devicePixelRatio || 1);
    o.w = b.width; o.h = b.height;
    o.c.width = Math.round(o.w * dpr); o.c.height = Math.round(o.h * dpr);
    o.ctx = o.c.getContext("2d"); o.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return true;
  }
  let muted = "#98A3B3";
  function readMuted() { const m = getComputedStyle(fig).getPropertyValue("--muted").trim(); if (m) muted = m; }
  readMuted();
  const MONO = '"JetBrains Mono", ui-monospace, Menlo, monospace';
  const NG = 181, X0 = -3, X1 = 3;
  const gx = Array.from({ length: NG }, (_, i) => X0 + (X1 - X0) * i / (NG - 1));
  const pdv = gx.map(pdata);
  const fmt = (x, d = 2) => (x < 0 ? "−" : x > 0 ? "+" : "") + Math.abs(x).toFixed(d);

  function halo(ctx, txt, x, y, col, C) {
    ctx.lineJoin = "round"; ctx.lineWidth = 4; ctx.strokeStyle = rgba(C.bg, .88); ctx.strokeText(txt, x, y);
    ctx.fillStyle = col; ctx.fillText(txt, x, y);
  }
  function subLabel(ctx, main, sub, x, y, col, C) {
    ctx.textAlign = "left";
    ctx.font = `500 12px ${MONO}`; const w = ctx.measureText(main).width;
    halo(ctx, main, x, y, col, C);
    ctx.font = `400 9.5px ${MONO}`; halo(ctx, sub, x + w + 1, y + 3, col, C);
    return w + 1 + ctx.measureText(sub).width;
  }
  function drawMain() {
    const o = cvs[0], ctx = o.ctx; if (!ctx || !o.w) return;
    const C = S.colors(), dark = C.dark, W = o.w, Hh = o.h;
    const narrow = W < 460;
    const L = 30, R = W - 8, T = 30, B = Hh - 50;
    const X = x => L + (x - X0) / (X1 - X0) * (R - L);
    const DY = d => T + (1 - d) * (B - T);
    const PY = p => B - p / 1.7 * (B - T);
    ctx.clearRect(0, 0, W, Hh);
    ctx.font = `400 10px ${MONO}`; ctx.textBaseline = "alphabetic";
    // frame: D axis 0 / 0.5 / 1
    ctx.strokeStyle = rgba(C.fg, dark ? .1 : .12); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(L, DY(1) + .5); ctx.lineTo(R, DY(1) + .5); ctx.stroke();
    ctx.strokeStyle = rgba(C.fg, dark ? .3 : .34); ctx.setLineDash([3, 4]);
    ctx.beginPath(); ctx.moveTo(L, Math.round(DY(.5)) + .5); ctx.lineTo(R, Math.round(DY(.5)) + .5); ctx.stroke(); ctx.setLineDash([]);
    ctx.strokeStyle = rgba(C.fg, dark ? .28 : .34);
    ctx.beginPath(); ctx.moveTo(L, B + .5); ctx.lineTo(R, B + .5); ctx.stroke();
    ctx.fillStyle = muted; ctx.textAlign = "right";
    ctx.fillText("1", L - 7, DY(1) + 3.5); ctx.fillText("0.5", L - 5, DY(.5) + 3.5); ctx.fillText("0", L - 7, B + 3.5);
    // x ticks
    ctx.textAlign = "center"; ctx.fillStyle = muted;
    for (let x = -2; x <= 2; x++) {
      ctx.fillText(x === 0 ? "0" : (x < 0 ? "−" : "") + Math.abs(x), X(x), Hh - 4);
      ctx.strokeStyle = rgba(C.fg, .25); ctx.beginPath(); ctx.moveTo(X(x) + .5, B); ctx.lineTo(X(x) + .5, B + 4); ctx.stroke();
    }
    ctx.textAlign = "right"; ctx.fillText("x", R, Hh - 4);
    // p_data: filled, fg-tinted
    ctx.beginPath(); ctx.moveTo(X(X0), B);
    for (let i = 0; i < NG; i++) ctx.lineTo(X(gx[i]), PY(pdv[i]));
    ctx.lineTo(X(X1), B); ctx.closePath();
    ctx.fillStyle = rgba(C.fg, dark ? .1 : .1); ctx.fill();
    ctx.beginPath(); for (let i = 0; i < NG; i++) { const y = PY(pdv[i]); i ? ctx.lineTo(X(gx[i]), y) : ctx.moveTo(X(gx[i]), y); }
    ctx.strokeStyle = rgba(C.fg, dark ? .5 : .55); ctx.lineWidth = 1.2; ctx.stroke();
    // p_g: filled, GAN hue
    const m = sim.m, two = sim.two;
    const pg = x => two ? .5 * npdf(x, m[0], P.s) + .5 * npdf(x, m[1], P.s) : npdf(x, m[0], P.s);
    ctx.beginPath(); ctx.moveTo(X(X0), B);
    for (let i = 0; i < NG; i++) ctx.lineTo(X(gx[i]), Math.max(T - 6, PY(pg(gx[i]))));
    ctx.lineTo(X(X1), B); ctx.closePath();
    ctx.fillStyle = rgba(C.gan, dark ? .26 : .2); ctx.fill();
    ctx.beginPath(); for (let i = 0; i < NG; i++) { const y = Math.max(T - 6, PY(pg(gx[i]))); i ? ctx.lineTo(X(gx[i]), y) : ctx.moveTo(X(gx[i]), y); }
    ctx.strokeStyle = C.gan; ctx.lineWidth = 1.6; ctx.stroke();
    // D(x) curve
    const dv = gx.map(sim.D);
    ctx.beginPath(); for (let i = 0; i < NG; i++) { const y = DY(dv[i]); i ? ctx.lineTo(X(gx[i]), y) : ctx.moveTo(X(gx[i]), y); }
    ctx.strokeStyle = C.fg; ctx.lineWidth = 2; ctx.lineJoin = "round"; ctx.stroke();
    // D at the two modes: a dot + its value
    ctx.font = `500 10.5px ${MONO}`;
    for (const c of [-P.sep, P.sep]) {
      const d = sim.D(c), x = X(c), y = DY(d);
      ctx.fillStyle = C.fg; ctx.beginPath(); ctx.arc(x, y, 3, 0, 6.283); ctx.fill();
      ctx.textAlign = "center";
      const above = y - 8 > T + 2;
      halo(ctx, "D " + d.toFixed(2), x, above ? y - 8 : y + 15, d < .42 ? C.gan : C.fg, C);
    }
    // label of the 0.5 guide: try right / left / centre and keep the spot where neither D(x) nor p_g crosses it
    {
      ctx.font = `400 10px ${MONO}`;
      const gl = narrow ? "0.5: наугад" : "0.5: D угадывает наугад", tw = ctx.measureText(gl).width + 6;
      const invX = px => X0 + (px - L) / (R - L) * (X1 - X0), y5 = DY(.5);
      let best = null;
      for (const x0 of [R - tw, L + 4, X(0) - tw / 2]) {
        let score = 1e9, sum = 0, n = 0;
        for (let px = x0; px <= x0 + tw; px += 4) {
          const x = invX(px), d = sim.D(x); sum += d; n++;
          score = Math.min(score, Math.abs(DY(d) - y5));
          for (const mu of m) if (Math.abs(x - mu) < 2 * P.s) score = 0;
        }
        if (!best || score > best.score + 2) best = { x0, score, up: sum / n < .5 };
      }
      ctx.textAlign = "left";
      halo(ctx, gl, best.x0 + 3, best.up ? y5 - 5 : y5 + 13, muted, C);
    }
    // curve labels
    const muG = two ? m[1] : m[0];
    const freeMode = two ? -P.sep : (muG > 0 ? -P.sep : P.sep);
    // "D(x)" sits clear of the curve over x in [-3, -2.2]: above its highest point, or below its lowest
    let dMax = 0, dMin = 1;
    for (let i = 0; i <= Math.round(.8 / 6 * (NG - 1)); i++) { dMax = Math.max(dMax, dv[i]); dMin = Math.min(dMin, dv[i]); }
    ctx.font = `500 12px ${MONO}`; ctx.textAlign = "left";
    halo(ctx, "D(x)", X(-2.9), DY(dMax) - 8 > T + 4 ? DY(dMax) - 8 : DY(dMin) + 17, C.fg, C);
    // p_data: on the outer flank of the free mode, at mid-height (clear of the "D 0.xx" mode label)
    const out = freeMode < 0 ? -1 : 1, pdy = PY(.5 * pdata(freeMode)) + 4, pdx = X(freeMode + out * .42) + out * 6;
    ctx.font = `500 12px ${MONO}`; const pw = ctx.measureText("p").width; ctx.font = `400 9.5px ${MONO}`;
    const pdw = pw + 1 + ctx.measureText("data").width;
    subLabel(ctx, "p", "data", out < 0 ? pdx - pdw : pdx, pdy, rgba(C.fg, .85), C);
    const gTop = Math.max(T - 6, PY(two ? .5 * npdf(0, 0, P.s) : npdf(0, 0, P.s)));
    const gxl = X(muG) + (muG > 1.9 ? -44 : 14);
    subLabel(ctx, "p", "g", gxl, gTop + (two ? 2 : 22), C.gan, C);
    // rugs: the actual minibatch D sees this iteration
    const ry1 = B + 11, ry2 = B + 22;
    ctx.font = `400 9.5px ${MONO}`; ctx.textAlign = "right"; ctx.fillStyle = muted;
    ctx.fillText("x", L - 7, ry1 + 6); ctx.fillText("G(z)", L - 4, ry2 + 6);
    ctx.lineWidth = 1;
    ctx.strokeStyle = rgba(C.fg, dark ? .32 : .4); ctx.beginPath();
    for (let i = 0; i < 64; i++) { const x = sim.real[i]; if (x > X0 && x < X1) { const px = Math.round(X(x)) + .5; ctx.moveTo(px, ry1); ctx.lineTo(px, ry1 + 7); } }
    ctx.stroke();
    ctx.strokeStyle = rgba(C.gan, dark ? .6 : .7); ctx.beginPath();
    for (let i = 0; i < 64; i++) { const x = sim.fake[i]; if (x > X0 && x < X1) { const px = Math.round(X(x)) + .5; ctx.moveTo(px, ry2); ctx.lineTo(px, ry2 + 7); } }
    ctx.stroke();
    // live annotation
    let msg = "", ax = X(muG);
    const it = sim.it;
    if (two) {
      if (it > 120 && Math.abs(sim.df - .5) < .08) { msg = narrow ? "обе моды накрыты, D около 0.5" : "G накрыл обе моды: D около 0.5, равновесие"; ax = X(0); }
      else { msg = "два колокола расходятся по модам"; ax = X(0); }
    } else if (it - ev.hopAt < 110 && ev.hopAt > 0) {
      msg = "G перепрыгнул на другую моду";
      // arc from the old mode to the new one, fading
      const a = 1 - (it - ev.hopAt) / 110, x0 = X(ev.from * P.sep), x1 = X(ev.to * P.sep), yb = T + 26, peak = T + 6;
      ctx.strokeStyle = rgba(C.gan, .9 * a); ctx.lineWidth = 1.5; ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(x0, yb); ctx.quadraticCurveTo((x0 + x1) / 2, peak - 18, x1, yb); ctx.stroke(); ctx.setLineDash([]);
      const dir = Math.sign(x1 - x0);
      ctx.fillStyle = rgba(C.gan, .9 * a); ctx.beginPath();
      ctx.moveTo(x1, yb + 1); ctx.lineTo(x1 - dir * 8, yb - 5); ctx.lineTo(x1 - dir * 2, yb - 8); ctx.closePath(); ctx.fill();
      ax = (x0 + x1) / 2;
    } else if (ev.sliding) {
      msg = narrow ? "D раскусил моду: G уходит" : "D раскусил эту моду: G соскальзывает";
    } else if (ev.settled) {
      msg = narrow ? "mode collapse: G выдаёт одну моду" : "mode collapse: G выдаёт только одну моду";
    } else msg = "G ищет, где D ещё верит";
    if (msg) {
      ctx.font = `500 11px ${MONO}`; const tw = ctx.measureText(msg).width;
      const tx = Math.max(L + tw / 2, Math.min(R - tw / 2, ax));
      ctx.textAlign = "center"; halo(ctx, msg, tx, 13, C.gan, C);
    }
  }
  function spark(o, series, lo, hi, guides, C) {
    const ctx = o.ctx; if (!ctx || !o.w) return;
    const W = o.w, Hh = o.h, pad = 6, L = 30;
    const Y = v => pad + (1 - (v - lo) / (hi - lo)) * (Hh - 2 * pad);
    const Xi = i => L + i / (HMAX - 1) * (W - L - 2);
    ctx.clearRect(0, 0, W, Hh);
    ctx.font = `400 9.5px ${MONO}`; ctx.textAlign = "right"; ctx.lineWidth = 1;
    for (const [g, lbl, dash] of guides) {
      ctx.strokeStyle = rgba(C.fg, C.dark ? .2 : .24); ctx.setLineDash(dash ? [3, 4] : []);
      ctx.beginPath(); ctx.moveTo(L, Math.round(Y(g)) + .5); ctx.lineTo(W, Math.round(Y(g)) + .5); ctx.stroke();
      ctx.setLineDash([]); ctx.fillStyle = muted; ctx.fillText(lbl, L - 5, Y(g) + 3.2);
    }
    for (const [arr, col, lw] of series) {
      ctx.strokeStyle = col; ctx.lineWidth = lw; ctx.lineJoin = "round"; ctx.beginPath();
      let pen = false;
      for (let i = 0; i < arr.length; i++) { const v = arr[i]; if (Number.isNaN(v)) { pen = false; continue; } const x = Xi(i), y = Y(v); pen ? ctx.lineTo(x, y) : ctx.moveTo(x, y); pen = true; }
      ctx.stroke();
      if (arr.length && !Number.isNaN(arr[arr.length - 1])) { ctx.fillStyle = col; ctx.beginPath(); ctx.arc(Xi(arr.length - 1), Y(arr[arr.length - 1]), 2.4, 0, 6.283); ctx.fill(); }
    }
  }
  const nf = n => String(n).replace(/\B(?=(\d{3})+(?!\d))/g, " ");
  function drawAll() {
    if (!sim) return;
    const C = S.colors();
    drawMain();
    const H = sim.H;
    spark(cvs[1], [[H.dr, rgba(C.fg, .85), 1.4], [H.df, C.gan, 1.6]], 0, 1, [[.5, "0.5", true], [1, "1", false], [0, "0", false]], C);
    spark(cvs[2], [[H.m0, C.gan, 1.6], [H.m1, rgba(C.gan, .6), 1.4]], -2.8, 2.8, [[P.sep, "+1.3", true], [-P.sep, "−1.3", true]], C);
    itEl.innerHTML = `итерация <b>${nf(sim.it)}</b>`;
    r1.innerHTML = `реальные <b>${sim.dr.toFixed(2)}</b> · подделки <b>${sim.df.toFixed(2)}</b>`;
    r2.innerHTML = sim.two
      ? `μ <b>${fmt(sim.m[0])}</b> и <b>${fmt(sim.m[1])}</b>`
      : `μ <b>${fmt(sim.m[0])}</b> · прыжков между модами <b>${ev.hops}</b>`;
    if (reduce || !playing) setVerdict(sim.df, false);
    dirty = false;
  }
  function sizeAll() { let ok = false; cvs.forEach(o => { if (size(o)) ok = true; }); if (ok) drawAll(); }

  /* ================= 5. Loop, visibility, controls ================= */
  let playing = !reduce, speed = 1, acc = 0, last = 0, raf = 0, onScreen = false;
  function active() { return onScreen && !document.hidden && !panel.hidden; }
  function kick() { if (!raf && active() && playing) { last = performance.now(); raf = requestAnimationFrame(frame); } }
  function frame(now) {
    raf = 0;
    if (!active() || !playing) { hideDots(); return; }
    const dt = Math.min(.1, (now - last) / 1000); last = now;
    acc += dt * 60 * speed;
    let n = Math.min(40, Math.floor(acc)); acc -= n;
    while (n-- > 0) stepOnce(false);
    if (!reduce) animDiagram(now);
    if (dirty) drawAll();
    raf = requestAnimationFrame(frame);
  }
  const playBtn = $("gan-play"), playL = $("gan-play-l"), hintSt = $("gan-hint-st");
  function setPlaying(on, user) {
    playing = on;
    playBtn.setAttribute("aria-pressed", String(on));
    playL.textContent = on ? "Пауза" : "Обучать";
    hintSt.textContent = on
      ? (sim && sim.two ? "смотри, как два колокола расходятся по модам и D приходит к 0.5" : "смотри, как G прыгает между модами; пауза и «Шаг» покажут по одной итерации")
      : "нажми «Обучать», чтобы запустить, или «Шаг», чтобы сделать одну итерацию";
    if (user) fig.classList.add("used-s");
    if (on) { cycleT0 = performance.now(); lastPhase = 0; kick(); }
    else { hideDots(); setVerdict(sim.df, false); }
  }
  playBtn.addEventListener("click", () => {
    const on = !playing;
    sfx.play(({ t, tone }) => {
      if (on) { tone(t, 659.25, .07, "sine", .12); tone(t + .07, 987.77, .1, "sine", .1); }
      else { tone(t, 987.77, .07, "sine", .1); tone(t + .07, 659.25, .1, "sine", .1); }
    });
    setPlaying(on, true);
  });
  $("gan-step").addEventListener("click", () => {
    if (playing) setPlaying(false, true);
    fig.classList.add("used-s");
    stepOnce(true);
    sfx.play(({ t, tone }) => { tone(t, 880, .05, "sine", .1); tone(t + .03, 1200 + 1000 * sim.df, .03, "sine", .05); });
    drawAll(); setVerdict(sim.df, false);
  });
  $("gan-reset").addEventListener("click", () => {
    fig.classList.add("used-s"); sfx.back();
    reset(sim.two); if (reduce && !playing) prerun(); drawAll(); setVerdict(sim.df, false);
  });
  const segs = [...fig.querySelectorAll(".ganl-seg")];
  segs.forEach(b => b.addEventListener("click", () => {
    speed = +b.dataset.v; fig.classList.add("used-s");
    segs.forEach(s => s.setAttribute("aria-pressed", String(s === b)));
    sfx.play(({ t, tone }) => tone(t, { 1: 523.25, 3: 659.25, 10: 783.99 }[speed], .09, "sine", .1));
  }));
  const twoBtn = $("gan-two");
  twoBtn.addEventListener("click", () => {
    const two = !sim.two; fig.classList.add("used-s");
    twoBtn.setAttribute("aria-pressed", String(two));
    sfx.gan();
    reset(two); if (reduce && !playing) prerun(); drawAll(); setVerdict(sim.df, false);
    setPlaying(playing, false);
  });
  // Reduced motion: no autoplay, but show a meaningful still frame (G collapsed on a mode, one hop in the history).
  function prerun() {
    const lim = sim.two ? 500 : 2400;
    while (sim.it < lim) {
      stepOnce(true);
      if (!sim.two && sim.it > 650 && ev.hops >= 1 && ev.settled && sim.it - ev.settledAt > 70) break;
    }
  }
  reset(false);
  if (reduce) prerun();
  setPlaying(playing, false);

  if ("ResizeObserver" in window) { const ro = new ResizeObserver(() => sizeAll()); cvs.forEach(o => ro.observe(o.c)); }
  if ("IntersectionObserver" in window) {
    new IntersectionObserver(es => { onScreen = es[es.length - 1].isIntersecting; if (onScreen) { sizeAll(); kick(); } }, { threshold: 0 }).observe(fig);
  } else onScreen = true;
  document.addEventListener("visibilitychange", kick);
  document.addEventListener("sem:tab", e => { if (e.detail === "gan") requestAnimationFrame(() => { sizeAll(); kick(); }); });
  S.onTheme(() => { readMuted(); drawAll(); });
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => drawAll());
  addEventListener("resize", () => { lens = null; });
})();
