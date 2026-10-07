(() => {
  "use strict";
  const S = window.__sem;
  const sd = document.getElementById("end-sd");
  if (!sd) return;
  const NS = "http://www.w3.org/2000/svg";
  const svg = document.getElementById("end-svg");
  const num = document.getElementById("end-num");
  const lat = document.getElementById("end-lat");
  const lnum = document.getElementById("end-lnum");
  const ldim = document.getElementById("end-ldim");
  const F = { vae: 261.63, gan: 329.63, dif: 392 };

  /* ---------- static drawing: grid (step = one latent side), a Swiss roll as the "picture", latent noise ---------- */
  const el = (tag, attrs, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); parent.appendChild(e); return e; };
  const grid = document.getElementById("end-grid");
  for (let i = 1; i < 8; i++) {
    el("line", { x1: 30 + i * 25, y1: 46, x2: 30 + i * 25, y2: 246 }, grid);
    el("line", { x1: 30, y1: 46 + i * 25, x2: 230, y2: 46 + i * 25 }, grid);
  }
  let d = "";
  for (let i = 0; i <= 220; i++) {
    const [x, y] = S.roll(i / 220);
    d += (i ? "L" : "M") + (130 + x * 78).toFixed(1) + " " + (146 - y * 78).toFixed(1);
  }
  document.getElementById("end-roll").setAttribute("d", d);
  setNum(document.getElementById("end-inum"), 786432); setNum(lnum, 16384);
  const noise = document.getElementById("end-noise"), rng = S.mulberry32(48);
  for (let i = 0; i < 14; i++) el("circle", { cx: (472 + rng() * 21).toFixed(1), cy: (130 + rng() * 21).toFixed(1), r: 1.25 }, noise);

  /* ---------- count-up 1 -> 48, the latent shrinks from image size to 64x64x4 ---------- */
  const FIN = { x: 491.5, y: 131.5 }, IMG = { x: 144, y: 132 };
  // grouped digits as tspans: a monospace space is too wide between thousands
  function setNum(t, n) {
    const g = String(Math.round(n)).replace(/\B(?=(\d{3})+(?!\d))/g, " ").split(" ");
    if (t.childElementCount === g.length) { g.forEach((s, i) => { t.children[i].textContent = s; }); return; }
    t.textContent = "";
    g.forEach((s, i) => { const sp = document.createElementNS(NS, "tspan"); if (i) sp.setAttribute("dx", "0.28em"); sp.textContent = s; t.appendChild(sp); });
  }
  function setK(k, q) {
    const s = Math.sqrt(196608 / k) / 64;           // latent side relative to 64 px: volume 4*L^2 = 786432/k
    const cx = IMG.x + (FIN.x - IMG.x) * q, cy = IMG.y + (FIN.y - IMG.y) * q;
    lat.setAttribute("transform", `translate(${cx.toFixed(2)} ${cy.toFixed(2)}) scale(${s.toFixed(4)}) translate(${-FIN.x} ${-FIN.y})`);
    setNum(lnum, 786432 / k);
  }
  function finalState() {
    num.textContent = "48"; lat.removeAttribute("transform"); lat.classList.remove("moving");
    setNum(lnum, 16384); ldim.style.opacity = "";
  }
  let counted = false, done = false, lastTick = 0, lastK = 1;
  function tick(k) {
    const now = performance.now();
    if (now - lastTick < 70) return;
    lastTick = now;
    S.sfx.play(({ t, tone }) => tone(t, 420 + k * 9, .035, "triangle", .028));
  }
  function triad() {
    S.sfx.play(({ t, tone }) => { ["vae", "gan", "dif"].forEach((m, i) => { tone(t + i * .08, F[m], 1.0, "sine", .09); tone(t + i * .08, F[m] * 2, .5, "sine", .02); }); });
  }
  function countUp() {
    if (counted) return; counted = true;
    if (S.reduce) { finalState(); done = true; return; }
    const T = 1900, t0 = performance.now();
    lat.classList.add("moving"); ldim.style.opacity = "0";
    const step = now => {
      const p = Math.min(1, (now - t0) / T);
      const e = 1 - Math.pow(1 - p, 3);
      const k = 1 + 47 * e;
      const ki = Math.round(k);
      if (ki !== lastK) { lastK = ki; num.textContent = String(ki); tick(ki); }
      setK(k, e);
      if (p < 1) requestAnimationFrame(step);
      else { finalState(); triad(); done = true; setTimeout(startCycle, 900); }
    };
    requestAnimationFrame(step);
  }
  // below the fold at load: start from the "1x" frame; otherwise keep the final, complete frame
  const below = sd.getBoundingClientRect().top > innerHeight * .8;
  if (below && !S.reduce) { num.textContent = "1"; setK(1, 0); lat.classList.add("moving"); ldim.style.opacity = "0"; }
  else { counted = true; done = true; }

  /* ---------- the three ideas <-> diagram ---------- */
  const ideas = sd.querySelector(".en-ideas"), partsUl = sd.querySelector(".en-parts");
  const parts = [...sd.querySelectorAll(".en-part")];
  let pinned = null, hover = null, cyc = null, lastNote = 0;
  function apply() {
    const hl = hover || pinned || cyc;
    if (hl) svg.dataset.hl = hl; else delete svg.dataset.hl;
    parts.forEach(b => {
      const on = b.dataset.part === hl;
      b.classList.toggle("on", on);
      b.setAttribute("aria-pressed", String(b.dataset.part === pinned));
    });
    partsUl.classList.toggle("has-on", !!hl);
  }
  function noteFor(m) {
    const now = performance.now();
    if (now - lastNote < 120) return; lastNote = now;
    S.sfx.note(m, .7);
  }

  // idle auto-cycle of highlights, until the reader takes over; resumes after a pause
  const CYCLE = ["vae", "gan", "dif", null];
  let ci = 0, cycTimer = 0, idleTimer = 0, visible = false, userActive = false, cycling = false;
  function stepCycle() {
    cycTimer = 0;
    if (userActive || !visible || document.hidden) { cycling = false; return; }
    cyc = CYCLE[ci]; ci = (ci + 1) % CYCLE.length; apply();
    cycTimer = setTimeout(stepCycle, cyc ? 2400 : 1600);
  }
  function startCycle() {
    if (S.reduce || userActive || cycling || !visible || !done) return;
    cycling = true; stepCycle();
  }
  function takeOver() {
    userActive = true; ideas.classList.add("touched");
    clearTimeout(cycTimer); cycTimer = 0; cycling = false; cyc = null;
    clearTimeout(idleTimer);
    idleTimer = setTimeout(() => { if (!pinned && !hover) { userActive = false; startCycle(); } }, 6000);
  }
  parts.forEach(b => {
    const m = b.dataset.part;
    b.addEventListener("pointerenter", e => { if (e.pointerType === "touch") return; takeOver(); hover = m; apply(); noteFor(m); });
    b.addEventListener("pointerleave", e => { if (e.pointerType === "touch") return; hover = null; apply(); takeOver(); });
    b.addEventListener("focus", () => { takeOver(); hover = m; apply(); });
    b.addEventListener("blur", () => { hover = null; apply(); });
    b.addEventListener("click", () => { takeOver(); pinned = pinned === m ? null : m; apply(); S.sfx.note(m, pinned ? 1 : .5); });
  });
  sd.querySelectorAll(".en-p").forEach(g => {
    const m = g.dataset.part;
    g.addEventListener("pointerenter", e => { if (e.pointerType === "touch") return; takeOver(); hover = m; apply(); noteFor(m); });
    g.addEventListener("pointerleave", () => { hover = null; apply(); });
  });

  new IntersectionObserver(es => {
    visible = es[0].isIntersecting;
    if (visible && es[0].intersectionRatio >= .3) countUp();
    if (visible && counted) startCycle();
  }, { threshold: [0, .3] }).observe(sd);
  document.addEventListener("visibilitychange", () => { if (!document.hidden && visible && counted) startCycle(); });
  if (S.reduce) { finalState(); }

  /* ---------- summary: staged reveal ---------- */
  const concl = document.getElementById("end-concl");
  [...concl.children].forEach((li, i) => li.style.setProperty("--i", i));
  if (!S.reduce && concl.getBoundingClientRect().top > innerHeight * .92) {
    concl.classList.add("pre");
    const io = new IntersectionObserver(es => {
      if (es[0].isIntersecting) {
        concl.classList.remove("pre"); concl.classList.add("in"); io.disconnect();
        [0, 1, 2, 3].forEach(i => setTimeout(() => S.sfx.play(({ t, tone }) => tone(t, [523.25, 587.33, 659.25, 783.99][i], .12, "sine", .03)), i * 170));
      }
    }, { threshold: .2 });
    io.observe(concl);
  }

  /* ---------- questions: accordion ---------- */
  sd.parentElement.querySelectorAll(".en-q").forEach(q => {
    const a = document.getElementById(q.getAttribute("aria-controls"));
    q.addEventListener("click", () => {
      const open = q.getAttribute("aria-expanded") !== "true";
      q.setAttribute("aria-expanded", String(open));
      a.classList.toggle("open", open);
      q.closest(".en-qs").parentElement.classList.add("touched");
      S.sfx.play(({ t, tone, noise }) => {
        if (open) { tone(t, 392, .16, "sine", .08); tone(t + .07, 587.33, .3, "sine", .07); }
        else { tone(t, 587.33, .12, "sine", .06); tone(t + .06, 392, .22, "sine", .05); }
        noise(t, .12, "bandpass", open ? 1800 : 3200, open ? 3200 : 1800, 2, .05);
      });
    });
  });
})();
