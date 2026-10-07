(() => {
  "use strict";
  const S = window.__sem; if (!S) return;
  const { sfx, mulberry32, gauss, roll, rollU, rgba, ease, clamp01, reduce } = S;
  const tri = document.getElementById("tri-box"), fig = document.getElementById("tri-fig");
  if (!tri || !fig) return;
  const cv = document.getElementById("tri-cv"), hub = document.getElementById("tri-hub");
  const statusT = document.getElementById("tri-status-t");
  const legend = [...tri.querySelectorAll("#tri-legend button")];
  const btnOf = Object.fromEntries(legend.map(b => [b.dataset.m, b]));
  const table = document.getElementById("cmp"), cmpWrap = document.getElementById("cmp-wrap"), cmpHint = document.getElementById("cmp-hint");

  const ORDER = ["dif", "gan", "vae"];                 // auto-cycle order
  const NAME = { dif: "Diffusion", gan: "GAN", vae: "VAE" };
  const FREQ = { vae: 261.63, gan: 329.63, dif: 392 }; // C major triad, same as core
  const DWELL = 3200, IDLE = 5000;                     // ms per model in auto mode; idle before resuming
  const now = () => performance.now();

  /* ---------- state ---------- */
  let shown = null;          // model currently highlighted (sticky after hover ends)
  let hover = null, hoverSrc = null, pinned = null, hubOpen = false, beforeHub = null;
  let lastInteract = -Infinity, entered = reduce, inView = false;
  let autoOn = false, autoIdx = 0, autoT0 = 0;
  let lastNoteT = 0;

  /* ---------- sounds (all no-ops until the user turns sound on) ---------- */
  function note(m, g) { const t = now(); if (t - lastNoteT < 40) return; lastNoteT = t; sfx.note(m, g); }
  const chord = () => sfx.play(({ t, tone }) => {
    [261.63, 329.63, 392].forEach((f, i) => tone(t + i * .035, f, 1.2, "sine", .07));
    tone(t + .1, 523.25, 1.0, "sine", .03);
  });
  const arpeggio = () => sfx.play(({ t, tone }) => {
    [261.63, 329.63, 392, 523.25].forEach((f, i) => tone(t + i * .055, f, .32, "triangle", .06));
  });
  const pinSound = (m, on) => sfx.play(({ t, tone }) => {
    const f = FREQ[m] * 2;
    tone(t, on ? f : f * 1.5, .09, "sine", .07);
    tone(t + .07, on ? f * 1.5 : f, .14, "sine", .06);
  });
  const tick = m => sfx.play(({ t, tone }) => tone(t, FREQ[m] * 2, .05, "sine", .07));

  /* ---------- highlight ---------- */
  function show(m, src) {
    const changed = m !== shown;
    shown = m;
    if (m) tri.dataset.hl = m; else delete tri.dataset.hl;
    legend.forEach(b => b.parentElement.classList.toggle("is-hl", b.dataset.m === m));
    if (table) {
      table.dataset.tri = m || "";
      table.dataset.col = m && src !== "auto" && src !== "quiet" ? m : "";
    }
    if (changed && m) {
      thumbs[m].restart();
      if (src !== "quiet") note(m, src === "auto" ? .55 : 1);
    }
    if (changed) legend.forEach(b => b.style.setProperty("--p", 0));
    kick();
  }
  function setStatus() {
    const mode = reduce ? "off" : pinned ? "pin" : autoOn || !entered ? "auto" : "pause";
    tri.dataset.mode = mode;
    statusT.textContent = mode === "pin" ? "закреплено: " + NAME[pinned] : mode === "auto" ? "автопоказ" : "пауза";
    tri.classList.toggle("is-auto", autoOn);
  }
  function stopAuto() { autoOn = false; legend.forEach(b => b.style.setProperty("--p", 0)); }
  function interact() {
    lastInteract = now();
    tri.classList.add("used");
    if (autoOn) stopAuto();
    setStatus();
  }
  function enter(m, src) { hover = m; hoverSrc = src; interact(); if (!hubOpen) show(m, src); }
  function leave() {
    const src = hoverSrc; hover = null; hoverSrc = null; interact();
    if (pinned) show(pinned, "quiet");
    else if (src === "table" && table) table.dataset.col = "";
  }
  function togglePin(m) {
    if (hubOpen) hubSet(false);
    pinned = pinned === m ? null : m;
    legend.forEach(b => b.setAttribute("aria-pressed", String(b.dataset.m === pinned)));
    interact();
    pinSound(m, pinned === m);
    show(pinned || m, "pin");
  }
  function hubSet(open) {
    if (open === hubOpen) return;
    hubOpen = open;
    tri.classList.toggle("hub-open", open);
    hub.setAttribute("aria-expanded", String(open));
    interact();
    if (open) { beforeHub = shown; arpeggio(); show(null, "hub"); }
    else show(pinned || hover || beforeHub, "quiet");
  }

  /* ---------- wiring: legend, edges + thumbnails, hub ---------- */
  legend.forEach(b => {
    const m = b.dataset.m;
    b.addEventListener("pointerenter", e => { if (e.pointerType === "mouse") enter(m, "legend"); });
    b.addEventListener("pointerleave", e => { if (e.pointerType === "mouse" && hover === m) leave(); });
    b.addEventListener("focus", () => { if (b.matches(":focus-visible")) enter(m, "legend"); });
    b.addEventListener("blur", () => { if (hover === m) leave(); });
    b.addEventListener("click", () => togglePin(m));
  });
  tri.querySelectorAll(".t-m").forEach(g => {
    const m = g.dataset.m;
    g.addEventListener("pointerenter", e => { if (e.pointerType === "mouse") enter(m, "tri"); });
    g.addEventListener("pointerleave", e => { if (e.pointerType === "mouse" && hover === m) leave(); });
    g.addEventListener("click", () => togglePin(m));
  });
  let lastPT = "mouse";
  hub.addEventListener("pointerdown", e => { lastPT = e.pointerType; });
  hub.addEventListener("pointerenter", e => { tri.classList.add("hub-hover"); if (e.pointerType === "mouse") hubSet(true); });
  hub.addEventListener("pointerleave", e => { tri.classList.remove("hub-hover"); if (e.pointerType === "mouse") hubSet(false); });
  hub.addEventListener("click", e => {
    // mouse already opened it on hover; touch and keyboard toggle
    if (e.detail > 0 && lastPT === "mouse") hubSet(true); else hubSet(!hubOpen);
    lastPT = "mouse";
  });
  hub.addEventListener("blur", () => hubSet(false));
  hub.addEventListener("keydown", e => { if (e.key === "Escape") hubSet(false); });
  document.addEventListener("pointerdown", e => { if (hubOpen && e.target !== hub) hubSet(false); }, { passive: true });

  /* ---------- thumbnails: each family's typical output on the Swiss roll ---------- */
  // Positions in SVG viewBox units (560 × 580); each box sits outside its own edge.
  const BOX = 112, VBW = 560, CTR = { dif: [89, 183], gan: [471, 183], vae: [280, 482] };
  const NPT = 220, DIF_K = 24, DIF_T = 1.9, LOOP = DWELL / 1000;
  const abar = s => Math.pow(Math.sin(s * Math.PI / 2), 2);
  const easeOut = t => 1 - Math.pow(1 - t, 3);
  const ghost = (() => { const g = mulberry32(99); return Array.from({ length: 170 }, () => roll(rollU(g))); })();
  let seedCtr = 11;
  function makeSet(m, seed) {
    const r = mulberry32(seed);
    let modes = null;
    if (m === "gan") { // mode collapse: 2–3 short arcs, different every time
      const k = 2 + (r() < .5 ? 1 : 0);
      modes = Array.from({ length: k }, () => { const c = .08 + r() * .84; return [c - .045, c + .045]; });
    }
    return Array.from({ length: NPT }, () => {
      const u = rollU(r), [x0, y0] = roll(u);
      let tgt;
      if (m === "vae") { // decoder mean + Gaussian smear: lands on the roll, but blurred
        const [tx, ty] = roll(Math.min(1, u + .002));
        const nx = -(ty - y0), ny = tx - x0, nl = Math.hypot(nx, ny) || 1;
        const blur = gauss(r) * .085, along = gauss(r) * .05;
        tgt = [x0 + nx / nl * blur + along, y0 + ny / nl * blur + gauss(r) * .05];
      } else if (m === "gan") {
        const mm = modes[Math.floor(r() * modes.length)];
        const [gx, gy] = roll(mm[0] + (mm[1] - mm[0]) * r());
        tgt = [gx + gauss(r) * .008, gy + gauss(r) * .008];
      } else tgt = [x0 + gauss(r) * .006, y0 + gauss(r) * .006];
      return { e: [gauss(r) * .62, gauss(r) * .62], tgt, f: [[gauss(r), gauss(r)], [gauss(r), gauss(r)], [gauss(r), gauss(r)]], d: r() };
    });
  }
  const thumbs = {};
  ORDER.forEach(m => {
    thumbs[m] = {
      m, a: .85, period: 0, t0: 0, set: makeSet(m, 7 + m.length),
      restart() { this.t0 = now(); this.period = 0; this.set = makeSet(m, seedCtr++ * 7919); }
    };
  });

  const ctx = cv.getContext("2d");
  let cw = 0, ch = 0, dpr = 1, colors = S.colors();
  function size() {
    const r = fig.getBoundingClientRect();
    dpr = Math.min(2, devicePixelRatio || 1);
    cw = r.width; ch = r.height;
    cv.width = Math.round(cw * dpr); cv.height = Math.round(ch * dpr);
  }

  function drawThumb(th, t, dt) {
    const k = cw / VBW, [ux, uy] = CTR[th.m];
    const cx = ux * k, cy = uy * k, half = BOX / 2 * k, R = half * .9, sc = half / 56;
    const live = shown === th.m && !hubOpen;
    const target = hubOpen ? .5 : !shown ? .85 : live ? 1 : .26;
    th.a = reduce || dt == null ? target : th.a + (target - th.a) * (1 - Math.exp(-dt / 110));
    const dark = colors.dark, fg = colors.fg, col = colors[th.m];
    const X = x => cx + x * R, Y = y => cy - y * R;

    // crop marks frame the sample like a figure inset
    ctx.globalAlpha = .55 + .45 * th.a;
    ctx.strokeStyle = rgba(fg, live ? .55 : .24); ctx.lineWidth = 1;
    const L = 7 * sc, x0 = cx - half, y0 = cy - half, x1 = cx + half, y1 = cy + half;
    ctx.beginPath();
    ctx.moveTo(x0, y0 + L); ctx.lineTo(x0, y0); ctx.lineTo(x0 + L, y0);
    ctx.moveTo(x1 - L, y0); ctx.lineTo(x1, y0); ctx.lineTo(x1, y0 + L);
    ctx.moveTo(x1, y1 - L); ctx.lineTo(x1, y1); ctx.lineTo(x1 - L, y1);
    ctx.moveTo(x0 + L, y1); ctx.lineTo(x0, y1); ctx.lineTo(x0, y1 - L);
    ctx.stroke();

    // p_data: the faint grey roll, same as in the hero
    ctx.globalAlpha = 1;
    ctx.fillStyle = rgba(fg, dark ? .15 : .2);
    const gr = Math.max(.6, .85 * sc);
    for (const [x, y] of ghost) ctx.fillRect(X(x) - gr, Y(y) - gr, gr * 2, gr * 2);

    // the model's sample: live (animated generation) when highlighted, final frame otherwise
    let tau = LOOP, fade = 1;
    if (live && !reduce) {
      const el = (t - th.t0) / 1000, per = Math.floor(el / LOOP);
      if (per !== th.period) { th.period = per; th.set = makeSet(th.m, seedCtr++ * 7919); }
      tau = el - per * LOOP;
      fade = clamp01(tau / .15);
      if (!autoOn) fade *= 1 - clamp01((tau - (LOOP - .25)) / .25); // about to resample the same model
    }
    ctx.globalCompositeOperation = dark ? "lighter" : "source-over";
    ctx.globalAlpha = th.a * fade;
    const pts = th.set;
    if (th.m === "dif") {
      const raw = clamp01(tau / DIF_T) * DIF_K, kk = Math.min(DIF_K, Math.floor(raw));
      const s = Math.min(1, (kk + ease(clamp01((raw - kk) / .55))) / DIF_K);
      const a = abar(s), sa = Math.sqrt(a), sb = Math.sqrt(1 - a), fi = kk % 3;
      ctx.fillStyle = rgba(col, dark ? .42 + .25 * a : .55 + .35 * a);
      const rr = Math.max(.7, (1.05 + .2 * a) * sc);
      for (const q of pts) {
        const j = sb * .07;
        const x = sa * q.tgt[0] + sb * q.e[0] + q.f[fi][0] * j, y = sa * q.tgt[1] + sb * q.e[1] + q.f[fi][1] * j;
        ctx.beginPath(); ctx.arc(X(x), Y(y), rr, 0, 6.283); ctx.fill();
      }
    } else {
      const soft = th.m === "vae";
      const p = soft ? ease(clamp01(tau / .7)) : easeOut(clamp01(tau / .35));
      for (const q of pts) {
        const pp = clamp01((p - q.d * .12) / .88);
        const x = q.e[0] + (q.tgt[0] - q.e[0]) * pp, y = q.e[1] + (q.tgt[1] - q.e[1]) * pp;
        if (soft) {
          ctx.fillStyle = rgba(col, (dark ? .17 : .22) + (1 - pp) * .3);
          ctx.beginPath(); ctx.arc(X(x), Y(y), Math.max(.8, (1.2 + 1.9 * pp) * sc), 0, 6.283); ctx.fill();
        } else {
          ctx.fillStyle = rgba(col, dark ? .36 + .24 * pp : .58 + .32 * pp);
          ctx.beginPath(); ctx.arc(X(x), Y(y), Math.max(.7, (1.05 + .2 * pp) * sc), 0, 6.283); ctx.fill();
        }
      }
    }
    ctx.globalCompositeOperation = "source-over";
    ctx.globalAlpha = 1;
  }
  function draw(t, dt) {
    if (!cw) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cw, ch);
    for (const m of ORDER) drawThumb(thumbs[m], t, dt);
  }

  /* ---------- auto-cycle: Diffusion → GAN → VAE while in view and idle ---------- */
  function autoStep(t) {
    const can = !reduce && entered && inView && !hover && !pinned && !hubOpen && t - lastInteract > IDLE;
    if (!can) { if (autoOn) { stopAuto(); setStatus(); } return; }
    if (!autoOn) {
      autoOn = true; autoT0 = t;
      autoIdx = shown ? ORDER.indexOf(shown) : 0;
      if (shown !== ORDER[autoIdx]) show(ORDER[autoIdx], "auto"); else if (table) table.dataset.col = "";
      setStatus();
    }
    let p = (t - autoT0) / DWELL;
    if (p >= 1) {
      autoIdx = (autoIdx + 1) % ORDER.length; autoT0 = t; p = 0;
      if (autoIdx === 0) { chord(); show(ORDER[0], "quiet"); } // a full round: soft C-major chord
      else show(ORDER[autoIdx], "auto");
    }
    btnOf[ORDER[autoIdx]].style.setProperty("--p", p.toFixed(4));
  }

  /* ---------- loop: only while the figure is on screen and the tab is visible ---------- */
  let running = false, lastT = 0, kickQueued = false;
  function frame(t) {
    if (!running) return;
    const dt = Math.min(64, t - lastT); lastT = t;
    autoStep(t);
    draw(t, dt);
    requestAnimationFrame(frame);
  }
  function setRunning() {
    const want = inView && !document.hidden && !reduce;
    if (want && !running) { running = true; lastT = now(); requestAnimationFrame(frame); }
    else if (!want && running) { running = false; if (autoOn) { stopAuto(); setStatus(); } }
  }
  function kick() {
    if (running || kickQueued) return;
    kickQueued = true;
    requestAnimationFrame(t => { kickQueued = false; draw(t, null); });
  }

  new ResizeObserver(() => { size(); kick(); }).observe(fig);
  size();
  new IntersectionObserver(([e]) => { inView = e.isIntersecting; setRunning(); }, { rootMargin: "0px 0px -10% 0px" }).observe(fig);
  document.addEventListener("visibilitychange", setRunning);
  S.onTheme(() => { colors = S.colors(); kick(); });

  /* ---------- entrance: edges draw in, vertices pop, labels fade ---------- */
  if (!reduce && "IntersectionObserver" in window) {
    tri.classList.add("pre");
    const io = new IntersectionObserver(es => {
      if (!es.some(e => e.isIntersecting)) return;
      io.disconnect();
      requestAnimationFrame(() => {
        tri.classList.remove("pre"); tri.classList.add("entering");
        setTimeout(() => { tri.classList.remove("entering"); entered = true; }, 1700);
      });
    }, { threshold: .3 });
    io.observe(fig);
  } else entered = true;
  setStatus();
  draw(now(), null);

  /* ---------- table: column ↔ triangle sync, meters, sticky column hint ---------- */
  if (table) {
    const COLS = ["", "vae", "gan", "dif"];
    table.querySelectorAll("tr").forEach(tr => [...tr.children].forEach((c, i) => {
      if (!i) return;
      c.dataset.c = COLS[i];
      c.classList.add("m-" + COLS[i]);
    }));
    table.addEventListener("pointerover", e => {
      const c = e.target.closest("td, th"); if (!c) return;
      const m = c.dataset.c;
      if (m) { if (hover !== m) enter(m, "table"); }
      else if (hover && hoverSrc === "table") leave();
    });
    table.addEventListener("pointerleave", () => { if (hover && hoverSrc === "table") leave(); });

    // meters fill dot by dot (row by row, left to right) when the table comes into view
    const dots = [...table.querySelectorAll(".meter i.on")];
    if (!reduce && "IntersectionObserver" in window) {
      table.classList.add("pre");
      const io = new IntersectionObserver(es => {
        if (!es.some(e => e.isIntersecting)) return;
        io.disconnect();
        dots.forEach((d, i) => setTimeout(() => {
          d.classList.add("lit");
          tick(d.closest("td").dataset.c);
          if (i === dots.length - 1) setTimeout(() => table.classList.remove("pre"), 400);
        }, 250 + i * 75));
      }, { threshold: .35 });
      io.observe(table.querySelector("tbody tr:nth-child(3)") || table);
    }
  }
  if (cmpWrap && cmpHint) {
    const check = () => cmpHint.classList.toggle("show", cmpWrap.scrollWidth > cmpWrap.clientWidth + 2);
    new ResizeObserver(check).observe(cmpWrap);
    let lastScrollSnd = 0;
    cmpWrap.addEventListener("scroll", () => {
      cmpWrap.classList.toggle("is-scrolled", cmpWrap.scrollLeft > 2);
      if (cmpWrap.scrollLeft > 24) cmpHint.classList.add("used");
      const t = now(); if (t - lastScrollSnd > 120) { lastScrollSnd = t; sfx.play(({ t: at, noise }) => noise(at, .05, "bandpass", 1800, 1500, 4, .05)); }
    }, { passive: true });
  }
})();
