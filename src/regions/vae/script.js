(() => {
  "use strict";
  const S = window.__sem; if (!S) return;
  const fig = document.getElementById("vae-fig"); if (!fig) return;
  const panel = document.getElementById("pan-vae");
  const { mulberry32, gauss, roll, rollU, rgba, clamp01, sfx, reduce } = S;
  const TAU = Math.PI * 2;
  const MONO = '"JetBrains Mono", ui-monospace, Menlo, monospace';
  const easeIO = t => t < .5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
  const smooth = t => t * t * (3 - 2 * t);
  const fmt = v => (v < -.005 ? "−" : "") + Math.abs(v).toFixed(2);
  let C = S.colors(), MUTED = "#98A3B3";
  function readTheme() {
    C = S.colors();
    MUTED = getComputedStyle(document.documentElement).getPropertyValue("--muted").trim() || MUTED;
  }
  readTheme();

  /* =====================================================================
     1. Pipeline diagram: packets on a loop + hover / focus explanations
     ===================================================================== */
  const dg = document.getElementById("vae-dg");
  const parts = [...dg.querySelectorAll(".vp")];
  const partBy = Object.fromEntries(parts.map(g => [g.dataset.p, g]));
  const say = document.getElementById("vae-say");
  const sayK = say.querySelector(".vae-say-k"), sayT = say.querySelector(".vae-say-t");
  const dgHint = document.getElementById("vae-dg-hint");
  const ORDER = ["x", "enc", "ms", "eps", "z", "dec", "xh", "elbo"];
  const NOTE = { x: 261.63, enc: 293.66, ms: 329.63, eps: 349.23, z: 392, dec: 440, xh: 523.25, elbo: 196 };
  const TEXT = {
    x: ["x", "Объект из обучающей выборки. Здесь это точка Swiss roll, в жизни — картинка или строка таблицы."],
    enc: ["энкодер q<sub>φ</sub>(z|x)", "Нейросеть, которая по x выдаёт не точку z, а параметры гауссова распределения над z."],
    ms: ["μ, σ", "Среднее и разброс q(z|x). KL-слагаемое тянет их к <span class=\"m\">μ = 0, σ = 1</span>, то есть к <span class=\"m\">𝒩(0, I)</span>."],
    eps: ["ε ~ 𝒩(0, I)", "Вся случайность вынесена в ε, поэтому <span class=\"m\">z = μ + σ ⊙ ε</span> дифференцируемо по μ и σ. Это трюк репараметризации."],
    z: ["z", "Точка латентного пространства. При генерации энкодер не нужен: z берут прямо из <span class=\"m\">𝒩(0, I)</span>."],
    dec: ["декодер p<sub>θ</sub>(x|z)", "Нейросеть, которая по z выдаёт среднее гауссова распределения над x."],
    xh: ["x̂", "Реконструкция. Для гауссова декодера слагаемое реконструкции сводится к <span class=\"m\">‖x − x̂‖²</span>: модель выгодно усреднять, отсюда размытость."],
    elbo: ["ELBO", "Нижняя оценка log p(x). Её максимизируют вместо самого правдоподобия, которое напрямую не посчитать."]
  };
  let hl = null, hlUser = false, lastHLUser = -1e9, lastCycle = 0, cycleIdx = -1;
  function setHL(p, user) {
    if (p === hl) return;
    hl = p;
    for (const g of parts) g.classList.toggle("hl", g.dataset.p === p);
    dg.classList.toggle("has-hl", !!p);
    if (p) {
      sayK.innerHTML = TEXT[p][0]; sayT.innerHTML = TEXT[p][1];
      say.classList.remove("swap"); void say.offsetWidth; say.classList.add("swap");
    }
    if (user && p) {
      dgHint.classList.add("used");
      sfx.play(({ t, tone }) => { tone(t, NOTE[p], .32, "sine", .09); tone(t, NOTE[p] * 2, .18, "sine", .025); });
    }
  }
  for (const g of parts) {
    const p = g.dataset.p;
    const on = () => { hlUser = true; lastHLUser = performance.now(); setHL(p, true); };
    const off = () => { hlUser = false; lastHLUser = performance.now(); };
    g.addEventListener("pointerenter", on);
    g.addEventListener("pointerleave", off);
    g.addEventListener("focus", on);
    g.addEventListener("blur", off);
    g.addEventListener("click", on);
    g.addEventListener("keydown", e => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); on(); }
    });
  }
  function cycleHL(now) {
    if (reduce || hlUser || now - lastHLUser < 5000) return;
    if (now - lastCycle < 2600) return;
    lastCycle = now;
    cycleIdx = (ORDER.indexOf(hl) + 1) % ORDER.length;
    setHL(ORDER[cycleIdx], false);
  }

  // packets: x -> encoder -> (μ, σ) -> z (+ε from below) -> decoder -> x̂
  const pkG = dg.querySelector(".vae-pk");
  const NS = "http://www.w3.org/2000/svg";
  const TRK = [
    { a: .00, b: .13, p: [[90, 170], [126, 170]], fin: 1, fout: 1 },
    { a: .19, b: .29, p: [[236, 160], [260, 146]], fin: 1 },
    { a: .19, b: .29, p: [[236, 180], [260, 194]], fin: 1 },
    { a: .33, b: .45, p: [[304, 145], [333, 162]], fout: 0 },
    { a: .33, b: .45, p: [[304, 195], [333, 178]], fout: 0 },
    { a: .25, b: .45, p: [[350, 304], [350, 193]], fin: 1 },
    { a: .53, b: .61, p: [[372, 170], [406, 170]], fout: 1 },
    { a: .72, b: .82, p: [[514, 170], [550, 170]], fin: 1 }
  ];
  const BUSY = [["enc", .12, .2], ["ms", .28, .34], ["z", .44, .54], ["dec", .6, .73], ["xh", .81, .96]];
  const PK_T = 3.4;
  for (const tr of TRK) {
    tr.halo = document.createElementNS(NS, "circle"); tr.halo.setAttribute("r", "7"); tr.halo.setAttribute("class", "halo");
    tr.core = document.createElementNS(NS, "circle"); tr.core.setAttribute("r", "3.1");
    pkG.append(tr.halo, tr.core);
    tr.halo.style.opacity = tr.core.style.opacity = 0;
  }
  let busyNow = "";
  function drawPackets(now) {
    const ph = (now / 1000 / PK_T) % 1;
    for (const tr of TRK) {
      const k = (ph - tr.a) / (tr.b - tr.a);
      if (k < 0 || k > 1) { if (tr.vis) { tr.core.style.opacity = 0; tr.halo.style.opacity = 0; tr.vis = 0; } continue; }
      const e = easeIO(k);
      const x = tr.p[0][0] + (tr.p[1][0] - tr.p[0][0]) * e, y = tr.p[0][1] + (tr.p[1][1] - tr.p[0][1]) * e;
      let o = 1;
      if (tr.fin) o = Math.min(o, clamp01(k / .25));
      if (tr.fout) o = Math.min(o, clamp01((1 - k) / .3));
      tr.core.setAttribute("cx", x.toFixed(1)); tr.core.setAttribute("cy", y.toFixed(1));
      tr.halo.setAttribute("cx", x.toFixed(1)); tr.halo.setAttribute("cy", y.toFixed(1));
      tr.core.style.opacity = o.toFixed(2); tr.halo.style.opacity = (o * .22).toFixed(2); tr.vis = 1;
    }
    let b = "";
    for (const [p, a, z] of BUSY) if (ph >= a && ph < z) b = p;
    if (b !== busyNow) {
      if (busyNow) partBy[busyNow].classList.remove("busy");
      if (b) partBy[b].classList.add("busy");
      busyNow = b;
    }
  }

  /* =====================================================================
     2. Toy VAE on the Swiss roll (honest, closed form)
     Data: x = roll(u) + small normal jitter. Encoder q(z|x) = N(μ(x), diag σ²).
     Low β: KL barely matters, encoder lays the roll out as three pieces wherever
     reconstruction likes (holes between them). High β: μ₁ = c·Φ⁻¹(rank of u), μ₂ → 0,
     the aggregate posterior matches N(0, I), z₂ collapses. Posterior width follows the
     β-VAE optimum for a Gaussian decoder: σ_q ≈ √β·σ_x / |dx/dz|.
     Decoder mean = E[x | z] under the posteriors (the optimal decoder for this
     encoder); where no posterior has mass it falls back to a global linear trend,
     i.e. an untrained region. Decoder spread σ_dec ∝ √β (β·KL ≡ noisier decoder).
     ===================================================================== */
  function probit(p) {
    const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
    const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
    const c = [-.007784894002430293, -.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
    const d = [.007784695709041462, .3224671290700398, 2.445134137142996, 3.754408661907416];
    const tail = q => (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
    if (p < .02425) return tail(Math.sqrt(-2 * Math.log(p)));
    if (p > .97575) return -tail(Math.sqrt(-2 * Math.log(1 - p)));
    const q = p - .5, r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  }
  // HD: decoder smoothness (a net cannot be sharper than its data); HQ: smoothing for the q(z) picture
  const N = 360, THICK = .022, HD = .06, HQ = .12, LAM = .0008;
  const P_EDGE = Math.exp(-.5 * 2.3 * 2.3) / TAU;     // prior density at 2.3σ: holes are counted inside it
  const U = new Float32Array(N), Dn = new Float32Array(N), X = new Float32Array(2 * N), G = new Float32Array(N), L0 = new Float32Array(2 * N);
  {
    const r = mulberry32(2013), tmp = [];
    for (let i = 0; i < N; i++) tmp.push([rollU(r), gauss(r)]);
    tmp.sort((a, b) => a[0] - b[0]);
    // low-β layout: three pieces of the roll, scattered around (and outside) the prior rings
    const SEG = [[-2.25, 1.45, -.35, 1.7, 1], [2.35, .95, 1.2, 1.6, -1], [.35, -2.35, .15, 2.3, 1]];
    for (let i = 0; i < N; i++) {
      const [u, d] = tmp[i];
      U[i] = u; Dn[i] = d;
      const [x0, y0] = roll(u), [x1, y1] = roll(Math.min(1, u + .001)), [xa, ya] = roll(Math.max(0, u - .001));
      let nx = -(y1 - ya), ny = x1 - xa; const nl = Math.hypot(nx, ny) || 1; nx /= nl; ny /= nl;
      X[2 * i] = x0 + nx * THICK * d; X[2 * i + 1] = y0 + ny * THICK * d;
      G[i] = probit((i + .5) / N);
      const k = Math.min(2, Math.floor(i * 3 / N)), s = (i - k * N / 3) / (N / 3);
      const [cx, cy, an, len, dir] = SEG[k], ca = Math.cos(an), sa = Math.sin(an);
      const t = (s - .5) * len * dir;
      L0[2 * i] = cx + t * ca - .1 * d * sa; L0[2 * i + 1] = cy + t * sa + .1 * d * ca;
    }
  }
  const M = { beta: 1, w: 1, s1: .05, s2: .97, k1: .08, k2: .97, q1: .13, q2: .98, sdec: .055, mu: new Float32Array(2 * N), la: [0, 0, 0, 0, 0, 0], holeFrac: 0 };
  function setModel(beta) {
    M.beta = beta;
    const lb = Math.log10(beta);
    const w = M.w = smooth(clamp01((lb + 1) / 1.05));
    const s1 = M.s1 = .05 * Math.sqrt(beta);
    const s2 = M.s2 = s1 + (.97 - s1) * w;
    const c1 = Math.sqrt(1 - s1 * s1);
    M.k1 = Math.sqrt(s1 * s1 + HD * HD); M.k2 = Math.sqrt(s2 * s2 + HD * HD);
    M.q1 = Math.sqrt(s1 * s1 + HQ * HQ); M.q2 = Math.sqrt(s2 * s2 + HQ * HQ);
    M.sdec = .055 * Math.sqrt(beta);
    const mu = M.mu;
    let mz1 = 0, mz2 = 0, mx = 0, my = 0;
    for (let i = 0; i < N; i++) {
      const a = L0[2 * i] * (1 - w) + c1 * G[i] * w, b = L0[2 * i + 1] * (1 - w) + .1 * Dn[i] * w;
      mu[2 * i] = a; mu[2 * i + 1] = b; mz1 += a; mz2 += b; mx += X[2 * i]; my += X[2 * i + 1];
    }
    mz1 /= N; mz2 /= N; mx /= N; my /= N;
    // global linear trend x ≈ a + B z (ridge): what an untrained region of a smooth decoder looks like
    let s11 = .3 * N, s12 = 0, s22 = .3 * N, x1 = 0, x2 = 0, y1 = 0, y2 = 0;
    for (let i = 0; i < N; i++) {
      const a = mu[2 * i] - mz1, b = mu[2 * i + 1] - mz2, dx = X[2 * i] - mx, dy = X[2 * i + 1] - my;
      s11 += a * a; s12 += a * b; s22 += b * b; x1 += dx * a; x2 += dx * b; y1 += dy * a; y2 += dy * b;
    }
    const det = s11 * s22 - s12 * s12;
    const bx1 = (x1 * s22 - x2 * s12) / det, bx2 = (x2 * s11 - x1 * s12) / det;
    const by1 = (y1 * s22 - y2 * s12) / det, by2 = (y2 * s11 - y1 * s12) / det;
    M.la = [mx - bx1 * mz1 - bx2 * mz2, bx1, bx2, my - by1 * mz1 - by2 * mz2, by1, by2];
  }
  const dec = { x: 0, y: 0, u: 0, q: 0, p: 0, hole: false };
  function decode(z1, z2, o) {
    const mu = M.mu, i1 = 1 / (M.k1 * M.k1), i2 = 1 / (M.k2 * M.k2), j1 = 1 / (M.q1 * M.q1), j2 = 1 / (M.q2 * M.q2);
    let sw = 0, sx = 0, sy = 0, su = 0, sq = 0;
    for (let i = 0; i < N; i++) {
      const a = z1 - mu[2 * i], b = z2 - mu[2 * i + 1], aa = a * a, bb = b * b;
      const eq = aa * j1 + bb * j2;
      if (eq > 40) continue;
      sq += Math.exp(-.5 * eq);
      const e = aa * i1 + bb * i2;
      if (e > 40) continue;
      const k = Math.exp(-.5 * e);
      sw += k; sx += k * X[2 * i]; sy += k * X[2 * i + 1]; su += k * U[i];
    }
    // q: aggregate posterior density at z; where it is ~0 the decoder was never trained
    const q = sq / (N * TAU * M.q1 * M.q2), p = Math.exp(-.5 * (z1 * z1 + z2 * z2)) / TAU;
    const L = M.la, lx = L[0] + L[1] * z1 + L[2] * z2, ly = L[3] + L[4] * z1 + L[5] * z2;
    const mx = sw > 0 ? sx / sw : lx, my = sw > 0 ? sy / sw : ly;
    o.x = (q * mx + LAM * lx) / (q + LAM); o.y = (q * my + LAM * ly) / (q + LAM);
    o.u = sw > 0 ? su / sw : NaN; o.q = q; o.p = p;
    o.hole = p > P_EDGE && q < .25 * p;
    return o;
  }
  function density(z1, z2) {
    const mu = M.mu, i1 = 1 / (M.q1 * M.q1), i2 = 1 / (M.q2 * M.q2);
    let sw = 0;
    for (let i = 0; i < N; i++) {
      const a = z1 - mu[2 * i], b = z2 - mu[2 * i + 1], e = a * a * i1 + b * b * i2;
      if (e < 40) sw += Math.exp(-.5 * e);
    }
    return sw / (N * TAU * M.q1 * M.q2);
  }
  // fixed prior draws for the "holes" meter
  const PRI = new Float32Array(800);
  { const r = mulberry32(77); for (let i = 0; i < 800; i++) PRI[i] = gauss(r); }
  function holeFraction() {
    let h = 0;
    for (let i = 0; i < 400; i++) {
      const a = PRI[2 * i], b = PRI[2 * i + 1], p = Math.exp(-.5 * (a * a + b * b)) / TAU;
      if (p > P_EDGE && density(a, b) < .25 * p) h++;
    }
    return h / 400;
  }

  /* =====================================================================
     3. Canvases
     ===================================================================== */
  const latEl = document.getElementById("vae-lat"), datEl = document.getElementById("vae-dat");
  const lat = { el: latEl, ctx: latEl.getContext("2d"), w: 0, h: 0, dpr: 1, bg: document.createElement("canvas") };
  const dat = { el: datEl, ctx: datEl.getContext("2d"), w: 0, h: 0, dpr: 1, bg: document.createElement("canvas") };
  const R = 3.4;
  function size(cv) {
    const w = cv.el.clientWidth, h = cv.el.clientHeight;
    if (!w || !h) return false;
    const dpr = Math.min(2, devicePixelRatio || 1);
    if (w === cv.w && h === cv.h && dpr === cv.dpr) return true;
    cv.w = w; cv.h = h; cv.dpr = dpr;
    cv.el.width = cv.bg.width = Math.round(w * dpr); cv.el.height = cv.bg.height = Math.round(h * dpr);
    return true;
  }
  // latent mapping
  const LS = () => Math.min(lat.w, lat.h) / (2 * R);
  const zX = z => lat.w / 2 + z * LS(), zY = z => lat.h / 2 - z * LS();
  // data mapping (centre of the roll's bounding box)
  const DS = () => Math.min(dat.w, dat.h) / 2 / 1.1;
  const xX = x => dat.w / 2 + (x - .11) * DS(), xY = y => dat.h / 2 - (y - .1) * DS();

  const GW = 52, qCv = document.createElement("canvas"), hCv = document.createElement("canvas"), tmpCv = document.createElement("canvas");
  qCv.width = hCv.width = GW; qCv.height = hCv.height = GW;
  function renderLatBg() {
    if (!lat.w) return;
    const c = lat.bg.getContext("2d"), { w, h, dpr } = lat, sc = LS();
    const Rx = w / 2 / sc, Ry = h / 2 / sc;
    c.setTransform(1, 0, 0, 1, 0, 0); c.clearRect(0, 0, lat.bg.width, lat.bg.height);
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    // aggregate posterior q(z) and the holes (prior mass with no encoded data)
    const qi = qCv.getContext("2d").createImageData(GW, GW), hi = hCv.getContext("2d").createImageData(GW, GW);
    const n = parseInt(C.vae.slice(1), 16), vr = n >> 16 & 255, vg = n >> 8 & 255, vb = n & 255;
    const qa = C.dark ? 120 : 105;
    for (let j = 0; j < GW; j++) {
      const z2 = Ry - (j + .5) / GW * 2 * Ry;
      for (let i = 0; i < GW; i++) {
        const z1 = -Rx + (i + .5) / GW * 2 * Rx;
        const q = density(z1, z2), p = Math.exp(-.5 * (z1 * z1 + z2 * z2)) / TAU, o = 4 * (j * GW + i);
        qi.data[o] = vr; qi.data[o + 1] = vg; qi.data[o + 2] = vb;
        qi.data[o + 3] = qa * Math.pow(Math.min(1, q / .159), .6);
        const inside = clamp01((p - P_EDGE * .7) / (P_EDGE * .6));
        const hole = clamp01((.3 * p - q) / (.2 * p + 1e-9)) * inside;
        hi.data[o + 3] = 255 * hole;
      }
    }
    qCv.getContext("2d").putImageData(qi, 0, 0);
    hCv.getContext("2d").putImageData(hi, 0, 0);
    c.imageSmoothingEnabled = true;
    c.drawImage(qCv, 0, 0, w, h);
    // hatch, masked by the hole map
    tmpCv.width = lat.bg.width; tmpCv.height = lat.bg.height;
    const t = tmpCv.getContext("2d");
    t.setTransform(dpr, 0, 0, dpr, 0, 0);
    t.strokeStyle = rgba(C.fg, C.dark ? .3 : .34); t.lineWidth = 1;
    t.beginPath();
    for (let x = -h; x < w; x += 6) { t.moveTo(x, h); t.lineTo(x + h, 0); }
    t.stroke();
    t.globalCompositeOperation = "destination-in"; t.imageSmoothingEnabled = true;
    t.drawImage(hCv, 0, 0, w, h);
    c.setTransform(1, 0, 0, 1, 0, 0); c.drawImage(tmpCv, 0, 0); c.setTransform(dpr, 0, 0, dpr, 0, 0);
    // axes
    c.strokeStyle = rgba(C.fg, C.dark ? .13 : .16); c.lineWidth = 1;
    c.beginPath(); c.moveTo(0, zY(0) + .5); c.lineTo(w, zY(0) + .5); c.moveTo(zX(0) + .5, 0); c.lineTo(zX(0) + .5, h); c.stroke();
    // prior rings 1σ, 2σ
    c.setLineDash([3, 4]); c.strokeStyle = rgba(C.fg, C.dark ? .5 : .55); c.lineWidth = 1.1;
    for (const r of [1, 2]) { c.beginPath(); c.arc(zX(0), zY(0), r * sc, 0, TAU); c.stroke(); }
    c.setLineDash([]);
    c.font = `400 10.5px ${MONO}`; c.fillStyle = MUTED; c.textBaseline = "middle";
    const d45 = Math.SQRT1_2;
    c.textAlign = "left";
    c.fillText("1σ", zX(d45) + 4, zY(d45) - 6);
    c.fillText("2σ", zX(2 * d45) + 4, zY(2 * d45) - 6);
    c.fillText("z₁", w - 18, zY(0) - 9);
    c.fillText("z₂", zX(0) + 6, 10);
    c.textAlign = "right";
    c.fillText("𝒩(0, I)", zX(-2 * d45) - 4, zY(-2 * d45) + 9);
    // posterior means of the training data
    c.globalCompositeOperation = C.dark ? "lighter" : "source-over";
    c.fillStyle = rgba(C.vae, C.dark ? .7 : .75);
    const mu = M.mu;
    for (let i = 0; i < N; i++) { c.beginPath(); c.arc(zX(mu[2 * i]), zY(mu[2 * i + 1]), 1.3, 0, TAU); c.fill(); }
    c.globalCompositeOperation = "source-over";
  }
  const GH = mulberry32(99), GHOST = Array.from({ length: 520 }, () => roll(rollU(GH)));
  function renderDatBg() {
    if (!dat.w) return;
    const c = dat.bg.getContext("2d"), { dpr } = dat;
    c.setTransform(1, 0, 0, 1, 0, 0); c.clearRect(0, 0, dat.bg.width, dat.bg.height);
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.fillStyle = rgba(C.fg, C.dark ? .2 : .26);
    for (const [x, y] of GHOST) { c.beginPath(); c.arc(xX(x), xY(y), 1.35, 0, TAU); c.fill(); }
  }

  /* =====================================================================
     4. State: z (drag / keys / idle Lissajous), β, prior samples
     ===================================================================== */
  const z = { x: .42, y: -1.1 }, zGoal = { x: .42, y: -1.1 };
  let dragging = false, lastUserZ = -1e9, wanderT = 0, zMoved = true, lastZSound = 0;
  const lissa = t => [2.3 * Math.sin(.33 * t + .184), 1.55 * Math.sin(.51 * t - .789)];
  const zHint = document.getElementById("vae-z-hint");
  const trail = [];  // recent { z1, z2, x, y, hole }
  const cloud = Array.from({ length: 44 }, (_, i) => ({ ex: 0, ey: 0, ph: i / 44, gen: -1 }));
  const cloudRng = mulberry32(5);
  const smp = { n: 0, z: new Float32Array(400), e: new Float32Array(400), x: new Float32Array(400), hole: new Uint8Array(200), born: 0 };
  let betaT = 1, betaShow = 1, betaDirty = true, lastBetaSound = 0, firstSampled = false;

  function setZUser(x, y, immediate) {
    zGoal.x = Math.max(-3.2, Math.min(3.2, x)); zGoal.y = Math.max(-3.2, Math.min(3.2, y));
    if (immediate || reduce) { z.x = zGoal.x; z.y = zGoal.y; }
    lastUserZ = performance.now(); zMoved = true;
    zHint.classList.add("used");
    if (!running) drawAll(performance.now());
  }
  function evZ(e) {
    const r = latEl.getBoundingClientRect(), sc = LS();
    return [(e.clientX - r.left - latEl.clientLeft - lat.w / 2) / sc, -(e.clientY - r.top - latEl.clientTop - lat.h / 2) / sc];
  }
  function nearHandle(cx, cy) {
    const r = latEl.getBoundingClientRect();
    return Math.hypot(cx - r.left - latEl.clientLeft - zX(z.x), cy - r.top - latEl.clientTop - zY(z.y)) < 34;
  }
  let tap = null;
  latEl.addEventListener("touchstart", e => { const t = e.touches[0]; if (t && nearHandle(t.clientX, t.clientY)) e.preventDefault(); }, { passive: false });
  latEl.addEventListener("pointerdown", e => {
    if (e.pointerType === "mouse" && e.button !== 0) return;
    if (e.pointerType === "mouse" || nearHandle(e.clientX, e.clientY)) {
      dragging = true; latEl.classList.add("drag");
      try { latEl.setPointerCapture(e.pointerId); } catch (_) { /* ignore */ }
      const [a, b] = evZ(e); setZUser(a, b, e.pointerType !== "mouse" ? true : false);
      e.preventDefault();
    } else tap = { x: e.clientX, y: e.clientY };
  });
  latEl.addEventListener("pointermove", e => {
    if (dragging) { const [a, b] = evZ(e); setZUser(a, b, true); return; }
    if (e.pointerType === "mouse") latEl.classList.toggle("grab", nearHandle(e.clientX, e.clientY));
    if (tap && Math.hypot(e.clientX - tap.x, e.clientY - tap.y) > 10) tap = null;
  });
  const endDrag = e => {
    if (dragging) { dragging = false; latEl.classList.remove("drag"); lastUserZ = performance.now(); }
    else if (tap && e.type === "pointerup") { const [a, b] = evZ(e); setZUser(a, b, false); }
    tap = null;
  };
  latEl.addEventListener("pointerup", endDrag);
  latEl.addEventListener("pointercancel", endDrag);
  latEl.addEventListener("keydown", e => {
    const st = e.shiftKey ? .4 : .1;
    const d = { ArrowLeft: [-st, 0], ArrowRight: [st, 0], ArrowUp: [0, st], ArrowDown: [0, -st] }[e.key];
    if (e.key === "Home") { e.preventDefault(); setZUser(0, 0, true); return; }
    if (!d) return;
    e.preventDefault();
    setZUser(zGoal.x + d[0], zGoal.y + d[1], true);
  });

  const betaEl = document.getElementById("vae-beta"), betaOut = document.getElementById("vae-beta-out");
  const mHole = document.getElementById("vae-m-hole"), mHoleV = document.getElementById("vae-m-hole-v");
  const mBlur = document.getElementById("vae-m-blur"), mBlurV = document.getElementById("vae-m-blur-v");
  const verdict = document.getElementById("vae-verdict"), readEl = document.getElementById("vae-read");
  const sliderToBeta = v => .1 * Math.pow(40, v / 1000);
  function onBetaInput() {
    const v = +betaEl.value;
    betaT = sliderToBeta(v);
    const s = "β = " + betaT.toFixed(2);
    betaOut.textContent = s; betaEl.setAttribute("aria-valuetext", s);
    betaEl.style.setProperty("--p", (v / 10) + "%");
    betaDirty = true;
    if (reduce || !running) { betaShow = betaT; applyBeta(); drawAll(performance.now()); }
    const now = performance.now();
    if (now - lastBetaSound > 55) {
      lastBetaSound = now;
      const sharp = 1 - (Math.sqrt(betaT) - Math.sqrt(.1)) / (2 - Math.sqrt(.1));
      sfx.play(({ t, noise }) => { const f = 380 + 5200 * sharp * sharp; noise(t, .13, "lowpass", f, f * .7, .9, .1 + .06 * sharp); });
    }
  }
  betaEl.addEventListener("input", onBetaInput);

  function decodeSamples() {
    for (let j = 0; j < smp.n; j++) {
      decode(smp.z[2 * j], smp.z[2 * j + 1], dec);
      smp.x[2 * j] = dec.x + M.sdec * smp.e[2 * j]; smp.x[2 * j + 1] = dec.y + M.sdec * smp.e[2 * j + 1];
      smp.hole[j] = dec.hole ? 1 : 0;
    }
  }
  function applyBeta() {
    setModel(betaShow);
    M.holeFrac = holeFraction();
    decodeSamples();
    renderLatBg();
    updateTexts();
    zMoved = true;
  }
  function updateTexts() {
    const hf = Math.round(M.holeFrac * 100);
    mHole.style.width = hf + "%"; mHoleV.textContent = hf + "%";
    mBlur.style.width = Math.min(100, M.sdec / .11 * 100).toFixed(1) + "%"; mBlurV.textContent = M.sdec.toFixed(3);
    const b = betaT;
    const regime = b < .4 ? "low" : b <= 1.6 ? "mid" : "high";
    if (regime === verdict.dataset.r && Math.abs(hf - (+verdict.dataset.h || 0)) < 1) return;
    verdict.dataset.r = regime; verdict.dataset.h = hf;
    const hs = `<span class="m">${hf}%</span>`;
    verdict.innerHTML = regime === "low"
      ? `<span class="k">мало KL</span>Энкодер раскладывает данные кусками, как удобно реконструкции, и центр 𝒩(0, I) остаётся пустым. Облака декодера узкие, восстановление чёткое. Но ${hs} точек из 𝒩(0, I) попадают в дыры, декодер выдаёт там точки мимо данных, а интерполяция между кусками рвётся.`
      : regime === "mid"
        ? `<span class="k">баланс</span>Закодированные данные заполняют 𝒩(0, I) почти без дыр (${hs}), и случайное z декодируется в правдоподобную точку. Веди z вдоль оси z<sub>1</sub>: x̂ плавно скользит по роллу, отсюда гладкая интерполяция. Цена — облака декодера уже заметно размыты.`
        : `<span class="k">много KL</span>KL тянет каждое q(z|x) к 𝒩(0, I), и закодированные данные ровно заполняют априорное распределение: дыр ${hs}. Цена — точность: облака q(z|x) соседних точек перекрываются, декодер их усредняет, образцы размыты. Ось z<sub>2</sub> вообще перестаёт что-либо кодировать (posterior collapse по одной оси).`;
  }

  const sampleBtn = document.getElementById("vae-sample");
  function sample(silent) {
    const r = mulberry32((Math.random() * 1e9) | 0);
    smp.n = 200;
    for (let j = 0; j < 400; j++) { smp.z[j] = gauss(r); smp.e[j] = gauss(r); }
    smp.born = reduce ? -1e9 : performance.now();
    decodeSamples();
    if (!silent) {
      sampleBtn.classList.add("pressed"); setTimeout(() => sampleBtn.classList.remove("pressed"), 160);
      // soft cascade: one voice per 20 samples; a batch that lands mostly in holes plays a muted tick
      const scale = [261.63, 293.66, 329.63, 392, 440, 523.25, 587.33, 659.25, 783.99, 880];
      sfx.play(({ t, tone, noise }) => {
        for (let k = 0; k < 10; k++) {
          let h = 0; for (let j = k * 20; j < k * 20 + 20; j++) h += smp.hole[j];
          const tk = t + k * .085;
          if (h > 10) noise(tk, .07, "bandpass", 900, 700, 3, .08);
          else tone(tk, scale[k], .42, "sine", .06 - k * .003);
        }
      });
    }
    if (!running) drawAll(performance.now());
  }
  sampleBtn.addEventListener("click", () => sample(false));

  /* =====================================================================
     5. Drawing
     ===================================================================== */
  function label(c, text, x, y, align, w, h) {
    c.font = `500 11px ${MONO}`;
    const tw = c.measureText(text).width, pad = 5;
    let lx = align === "right" ? x - tw - pad * 2 : x;
    lx = Math.max(4, Math.min(w - tw - pad * 2 - 4, lx));
    const ly = Math.max(4, Math.min(h - 22, y));
    c.fillStyle = rgba(C.bg, .86); c.fillRect(lx, ly, tw + pad * 2, 18);
    c.strokeStyle = rgba(C.fg, .55); c.lineWidth = 1; c.strokeRect(lx + .5, ly + .5, tw + pad * 2 - 1, 17);
    c.fillStyle = C.fg; c.textAlign = "left"; c.textBaseline = "middle";
    c.fillText(text, lx + pad, ly + 9.5);
  }
  function drawLat(now) {
    const c = lat.ctx, { w, h, dpr } = lat; if (!w) return;
    c.setTransform(1, 0, 0, 1, 0, 0); c.clearRect(0, 0, lat.el.width, lat.el.height);
    c.drawImage(lat.bg, 0, 0);
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    const lighter = C.dark ? "lighter" : "source-over";
    // prior samples
    if (smp.n) {
      const age = now - smp.born;
      for (let j = 0; j < smp.n; j++) {
        const a = clamp01((age - j * 4.5) / 260); if (a <= 0) continue;
        const px = zX(smp.z[2 * j]), py = zY(smp.z[2 * j + 1]);
        if (smp.hole[j]) {
          c.strokeStyle = rgba(C.fg, (C.dark ? .55 : .6) * a); c.lineWidth = 1;
          c.beginPath(); c.arc(px, py, 2.6, 0, TAU); c.stroke();
        } else {
          c.fillStyle = rgba(C.fg, (C.dark ? .6 : .7) * a);
          c.beginPath(); c.arc(px, py, 1.9, 0, TAU); c.fill();
        }
        if (a < 1) { c.strokeStyle = rgba(C.vae, .6 * (1 - a)); c.beginPath(); c.arc(px, py, 2 + 10 * a, 0, TAU); c.stroke(); }
      }
    }
    // trail of z
    c.lineCap = "round";
    for (let i = 1; i < trail.length; i++) {
      const a = i / trail.length;
      c.strokeStyle = rgba(C.vae, a * .55); c.lineWidth = 1.6;
      c.beginPath(); c.moveTo(zX(trail[i - 1].z1), zY(trail[i - 1].z2)); c.lineTo(zX(trail[i].z1), zY(trail[i].z2)); c.stroke();
    }
    // handle
    const hx = zX(z.x), hy = zY(z.y);
    c.globalCompositeOperation = lighter;
    const g = c.createRadialGradient(hx, hy, 0, hx, hy, 22);
    g.addColorStop(0, rgba(C.vae, C.dark ? .55 : .35)); g.addColorStop(1, rgba(C.vae, 0));
    c.fillStyle = g; c.beginPath(); c.arc(hx, hy, 22, 0, TAU); c.fill();
    c.globalCompositeOperation = "source-over";
    c.fillStyle = C.vae; c.beginPath(); c.arc(hx, hy, 4.2, 0, TAU); c.fill();
    c.strokeStyle = C.fg; c.lineWidth = 1.5; c.beginPath(); c.arc(hx, hy, 9.5, 0, TAU); c.stroke();
    if (dec.hole) {
      c.setLineDash([3, 3]); c.strokeStyle = C.fg; c.lineWidth = 1;
      c.beginPath(); c.arc(hx, hy, 16, 0, TAU); c.stroke(); c.setLineDash([]);
      label(c, "дыра в латенте", hx + 20, hy - 30, hx > w * .6 ? "right" : "left", w, h);
    } else {
      c.font = `500 12px ${MONO}`; c.fillStyle = C.fg; c.textAlign = "left"; c.textBaseline = "middle";
      c.fillText("z", hx + 13, hy - 12);
    }
  }
  function drawDat(now) {
    const c = dat.ctx, { w, h, dpr } = dat; if (!w) return;
    c.setTransform(1, 0, 0, 1, 0, 0); c.clearRect(0, 0, dat.el.width, dat.el.height);
    c.drawImage(dat.bg, 0, 0);
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    const lighter = C.dark ? "lighter" : "source-over";
    // decoded prior samples
    if (smp.n) {
      const age = now - smp.born;
      c.globalCompositeOperation = lighter;
      for (let j = 0; j < smp.n; j++) {
        const a = clamp01((age - 140 - j * 4.5) / 320); if (a <= 0) continue;
        const px = xX(smp.x[2 * j]), py = xY(smp.x[2 * j + 1]);
        if (smp.hole[j]) {
          c.strokeStyle = rgba(C.vae, (C.dark ? .75 : .85) * a); c.lineWidth = 1.1;
          c.beginPath(); c.arc(px, py, 2.8, 0, TAU); c.stroke();
        } else {
          c.fillStyle = rgba(C.vae, (C.dark ? .55 : .7) * a);
          c.beginPath(); c.arc(px, py, 2, 0, TAU); c.fill();
        }
      }
      c.globalCompositeOperation = "source-over";
    }
    // trail of decoded means: smooth glide on the roll, or jumps across holes
    c.lineCap = "round";
    for (let i = 1; i < trail.length; i++) {
      const a = i / trail.length;
      c.strokeStyle = rgba(C.vae, a * .6); c.lineWidth = 1.6;
      c.beginPath(); c.moveTo(xX(trail[i - 1].x), xY(trail[i - 1].y)); c.lineTo(xX(trail[i].x), xY(trail[i].y)); c.stroke();
    }
    // u tick on the roll (where the decoder "aims")
    if (!dec.hole && dec.u === dec.u) {
      const [rx, ry] = roll(Math.max(0, Math.min(1, dec.u)));
      c.strokeStyle = C.fg; c.lineWidth = 1.2;
      c.beginPath(); c.arc(xX(rx), xY(ry), 5, 0, TAU); c.stroke();
    }
    // decoder cloud p_θ(x | z) = N(m(z), σ_dec² I), resampled slowly
    const sc = DS(), mx = xX(dec.x), my = xY(dec.y), sd = M.sdec * sc;
    const rr = Math.max(7, 2.6 * sd);
    c.globalCompositeOperation = lighter;
    const g = c.createRadialGradient(mx, my, 0, mx, my, rr);
    g.addColorStop(0, rgba(C.vae, C.dark ? .5 : .32)); g.addColorStop(.45, rgba(C.vae, C.dark ? .2 : .14)); g.addColorStop(1, rgba(C.vae, 0));
    c.fillStyle = g; c.beginPath(); c.arc(mx, my, rr, 0, TAU); c.fill();
    const tt = reduce ? .5 : now / 1000 * .7;
    for (const p of cloud) {
      const life = (tt + p.ph) % 1, gen = Math.floor(tt + p.ph);
      if (gen !== p.gen) { p.gen = gen; p.ex = gauss(cloudRng); p.ey = gauss(cloudRng); }
      const a = reduce ? .8 : Math.sin(Math.PI * life);
      c.fillStyle = rgba(C.vae, (C.dark ? .75 : .85) * a);
      c.beginPath(); c.arc(mx + p.ex * sd, my - p.ey * sd, 1.7, 0, TAU); c.fill();
    }
    c.globalCompositeOperation = "source-over";
    c.fillStyle = C.vae; c.beginPath(); c.arc(mx, my, 3.4, 0, TAU); c.fill();
    c.strokeStyle = C.fg; c.lineWidth = 1.2; c.beginPath(); c.arc(mx, my, 3.4, 0, TAU); c.stroke();
    if (dec.hole) {
      c.setLineDash([3, 3]); c.strokeStyle = C.fg; c.lineWidth = 1;
      c.beginPath(); c.arc(mx, my, rr + 6, 0, TAU); c.stroke(); c.setLineDash([]);
      label(c, "из дыры: мимо данных", mx + rr + 8, my - rr - 20, mx > w * .55 ? "right" : "left", w, h);
    } else {
      c.font = `500 12px ${MONO}`; c.fillStyle = C.fg; c.textAlign = "left"; c.textBaseline = "middle";
      c.fillText("x̂", mx + rr * .7 + 6, my - rr * .7 - 6);
    }
  }
  let lastRead = 0, readStr = "";
  function updateRead(now, force) {
    if (!force && now - lastRead < 90) return;
    lastRead = now;
    const zs = `z = (<b>${fmt(z.x)}</b>, <b>${fmt(z.y)}</b>) → `;
    const s = dec.hole
      ? zs + `<span class="hole">дыра в латенте</span>: данных рядом нет, декодер угадывает`
      : zs + `u = <b>${clamp01(dec.u).toFixed(2)}</b> на ролле`;
    if (s !== readStr) { readStr = s; readEl.innerHTML = s; }
  }
  function drawAll(now) {
    decode(z.x, z.y, dec);
    drawLat(now); drawDat(now); updateRead(now, !running);
  }

  /* =====================================================================
     6. Loop: only while the VAE panel is shown, on screen and the tab visible
     ===================================================================== */
  let running = false, onScreen = false, prev = 0;
  function frame(now) {
    if (!running) return;
    const dt = Math.min(.05, (now - prev) / 1000 || .016); prev = now;
    drawPackets(now);
    cycleHL(now);
    // β eases in log space so the latent visibly re-arranges
    if (Math.abs(Math.log(betaShow / betaT)) > .002) {
      betaShow = Math.exp(Math.log(betaShow) + (Math.log(betaT) - Math.log(betaShow)) * (1 - Math.exp(-dt / .09)));
      applyBeta();
    } else if (betaDirty) { betaShow = betaT; applyBeta(); betaDirty = false; }
    // z: user goal, or the idle Lissajous wander
    if (!dragging && now - lastUserZ > 5000) {
      wanderT += dt;
      const [a, b] = lissa(wanderT); zGoal.x = a; zGoal.y = b;
    }
    const k = 1 - Math.exp(-dt / (now - lastUserZ > 5000 ? .5 : .08));
    const ox = z.x, oy = z.y;
    z.x += (zGoal.x - z.x) * k; z.y += (zGoal.y - z.y) * k;
    decode(z.x, z.y, dec);
    trail.push({ z1: z.x, z2: z.y, x: dec.x, y: dec.y });
    if (trail.length > 70) trail.shift();
    // sound: a soft sine that follows z₁ while the user moves z; a tritone ghost inside a hole
    if (now - lastUserZ < 200 && Math.hypot(z.x - ox, z.y - oy) > .004 && now - lastZSound > 70) {
      lastZSound = now;
      const f = 261.63 * Math.pow(2, z.x / 3), hole = dec.hole;
      sfx.play(({ t, tone }) => { tone(t, f, .16, "sine", .06); if (hole) tone(t, f * 1.414, .16, "triangle", .025); });
    }
    drawLat(now); drawDat(now); updateRead(now);
    requestAnimationFrame(frame);
  }
  function sync() {
    const want = onScreen && !panel.hidden && !document.hidden && !reduce;
    if (want && !running) {
      running = true; prev = performance.now();
      if (!firstSampled) { firstSampled = true; sample(true); }
      requestAnimationFrame(frame);
    } else if (!want) running = false;
  }
  function resizeAll() {
    const a = size(lat), b = size(dat);
    if (a) renderLatBg();
    if (b) renderDatBg();
    if (!running) drawAll(performance.now());
  }
  new ResizeObserver(resizeAll).observe(latEl);
  new ResizeObserver(resizeAll).observe(datEl);
  new IntersectionObserver(es => { onScreen = es[es.length - 1].isIntersecting; sync(); }, { rootMargin: "80px" }).observe(fig);
  document.addEventListener("visibilitychange", sync);
  document.addEventListener("sem:tab", e => { if (e.detail === "vae") { resizeAll(); } sync(); });
  S.onTheme(() => { readTheme(); renderLatBg(); renderDatBg(); if (!running) drawAll(performance.now()); });
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => { renderLatBg(); if (!running) drawAll(performance.now()); });

  // initial, complete frame at rest
  setModel(1); M.holeFrac = holeFraction(); updateTexts();
  if (reduce) { firstSampled = true; sample(true); }
  resizeAll();
})();
