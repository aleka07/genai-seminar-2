(() => {
  "use strict";
  const S = window.__sem; if (!S) return;
  const sec = document.getElementById("time"); if (!sec) return;
  const { sfx, reduce, clamp01 } = S;

  const stage = sec.querySelector(".tm-stage"), ol = stage.querySelector(".tl");
  const card = document.getElementById("time-card"), cin = document.getElementById("time-cin");
  const head = stage.querySelector(".tm-head");
  const playBtn = document.getElementById("time-play"), playLbl = playBtn.querySelector(".tm-plbl");
  const pst = document.getElementById("time-pst");
  const chips = [...sec.querySelectorAll(".tm-chip")];
  const yrs = [...ol.children];
  const evs = [...ol.querySelectorAll(".ev")];
  const btns = evs.map(e => e.querySelector(".ev-b"));
  const dets = evs.map(e => e.querySelector(".ev-d"));
  const fams = evs.map(e => e.dataset.m.split(" "));
  const yrOf = evs.map(e => e.closest(".yr"));
  const yearOf = evs.map(e => e.closest(".yr").querySelector(".y").textContent.trim());
  const N = evs.length;
  const mq = matchMedia("(max-width: 1023px)");
  let phone = mq.matches;

  /* ---------- sounds (all no-ops while the page sound is off) ---------- */
  let lastSnd = 0;
  function famNote(i, g) {
    const now = performance.now(); if (now - lastSnd < 60) return; lastSnd = now;
    const f = fams[i]; f.forEach(m => sfx.note(m, g / Math.sqrt(f.length)));
  }
  const softClick = (hz = 1500) => sfx.play(({ t, tone }) => { tone(t, hz, .035, "sine", .07); });
  // playhead: a C-major scale climbing with time (event k = scale degree k), so history literally rises
  const SC = [0, 2, 4, 5, 7, 9, 11];
  function stepNote(i) {
    const f = 261.63 * Math.pow(2, (SC[i % 7] + 12 * Math.floor(i / 7)) / 12);
    sfx.play(({ t, tone }) => {
      tone(t, f, .9, "sine", .1);
      tone(t, f * 2, .35, "sine", .02);
      if (fams[i].includes("gan")) tone(t, f, .25, "triangle", .03);
    });
  }
  const endChord = () => sfx.play(({ t, tone }) => { [523.25, 659.25, 783.99].forEach((f, k) => tone(t + k * .05, f, 1.2, "sine", .06)); });

  /* ---------- counts on the filter chips ---------- */
  sec.querySelectorAll(".tm-n").forEach(n => {
    const f = n.dataset.n; n.textContent = f === "all" ? N : fams.filter(x => x.includes(f)).length;
  });

  /* ---------- hints dim after the first real interaction ---------- */
  let used = false;
  function markUsed() { if (used) return; used = true; sec.classList.add("t-used"); stage.classList.remove("t-wave"); }

  /* ---------- roving focus: one tab stop, arrows walk through the events ---------- */
  let fi = 0;
  function rove(i) { fi = i; btns.forEach((b, k) => { b.tabIndex = k === i ? 0 : -1; }); }
  rove(0);

  /* ---------- filter ---------- */
  let filter = "all";
  const match = i => filter === "all" || fams[i].includes(filter);
  function setFilter(f) {
    filter = f; stage.dataset.f = f;
    chips.forEach(c => c.setAttribute("aria-pressed", String(c.dataset.f === f)));
    evs.forEach((e, i) => e.classList.toggle("is-dim", !match(i)));
    yrs.forEach(y => {
      const any = [...y.querySelectorAll(".ev")].some(e => !e.classList.contains("is-dim"));
      y.classList.toggle("is-dim", !any);
      y.classList.toggle("is-hit", any && f !== "all");
    });
    if (cur >= 0 && !match(cur)) close();
    if (!match(fi)) { const k = evs.findIndex((_, i) => match(i)); if (k >= 0) rove(k); }
  }
  chips.forEach(c => c.addEventListener("click", () => {
    markUsed();
    const f = c.dataset.f;
    softClick(f === "all" ? 1500 : 1300);
    if (f !== "all") sfx.note(f, .25);
    setFilter(f);
  }));

  /* ---------- open / close: popover card on desktop, inline expansion on phone ---------- */
  let cur = -1, how = null, tClose = 0, tHide = 0, tHover = 0;
  const el = (tag, cls, txt) => { const n = document.createElement(tag); n.className = cls; if (txt != null) n.textContent = txt; return n; };

  function fillCard(i) {
    cin.textContent = "";
    const h = el("p", "tm-ch");
    h.append(btns[i].querySelector(".ev-dots").cloneNode(true), el("span", "tm-cn", btns[i].querySelector(".ev-t").textContent.trim()));
    cin.append(h, ...[...dets[i].firstElementChild.children].map(n => n.cloneNode(true)));
  }
  function placeCard(i) {
    const sr = stage.getBoundingClientRect(), r = btns[i].getBoundingClientRect();
    const W = stage.clientWidth, w = card.offsetWidth, h = card.offsetHeight, gap = 16;
    let x = r.right - sr.left + gap, side = "r", y = r.top - sr.top - 6;
    if (x + w > W) { x = r.left - sr.left - w - gap; side = "l"; }
    if (x < 0) { x = Math.max(0, Math.min(W - w, r.left - sr.left)); side = "n"; y = r.bottom - sr.top + 8; }
    else y = Math.max(0, Math.min(y, stage.clientHeight - h + 70));
    card.dataset.side = side;
    card.style.setProperty("--cx", Math.round(x) + "px");
    card.style.setProperty("--cy", Math.round(y) + "px");
    card.style.setProperty("--ay", Math.round(r.top - sr.top + 17 - y) + "px");
  }
  function showCard(i) {
    clearTimeout(tHide);
    const wasHidden = card.hidden || !card.classList.contains("show");
    card.hidden = false;
    if (wasHidden) {
      fillCard(i); card.classList.add("snap"); placeCard(i);
      void card.offsetWidth; card.classList.remove("snap"); card.classList.add("show");
    } else if (reduce) { fillCard(i); placeCard(i); }
    else {
      // glide to the new event and cross-fade the text
      cin.classList.add("swap");
      setTimeout(() => { if (cur !== i) return; fillCard(i); placeCard(i); cin.classList.remove("swap"); }, 120);
    }
  }
  function hideCard() {
    card.classList.remove("show");
    clearTimeout(tHide); tHide = setTimeout(() => { if (cur < 0) card.hidden = true; }, 220);
  }
  function expandInline(i) {
    const d = dets[i]; d.hidden = false;
    void d.offsetWidth; evs[i].classList.add("is-x");
  }
  function collapseInline(i) {
    evs[i].classList.remove("is-x");
    const d = dets[i];
    setTimeout(() => { if (!evs[i].classList.contains("is-x")) d.hidden = true; }, reduce ? 0 : 340);
  }
  function open(i, src) {
    clearTimeout(tClose);
    if (cur === i) { if (src !== "hover") how = src; return; }
    if (cur >= 0) { evs[cur].classList.remove("is-open"); btns[cur].setAttribute("aria-expanded", "false"); if (phone) collapseInline(cur); }
    cur = i; how = src;
    evs[i].classList.add("is-open"); btns[i].setAttribute("aria-expanded", "true");
    if (phone) expandInline(i); else showCard(i);
  }
  function close() {
    clearTimeout(tClose); clearTimeout(tHover);
    if (cur < 0) return;
    const i = cur; cur = -1; how = null;
    evs[i].classList.remove("is-open"); btns[i].setAttribute("aria-expanded", "false");
    if (phone) collapseInline(i); else hideCard();
  }
  const scheduleClose = () => { clearTimeout(tClose); tClose = setTimeout(() => { if (how === "hover") close(); }, 260); };

  btns.forEach((b, i) => {
    b.addEventListener("pointerenter", e => {
      if (e.pointerType !== "mouse") return;
      stopPlay();
      famNote(i, .3);
      if (phone || how === "pin" || how === "focus") return;
      clearTimeout(tClose); clearTimeout(tHover);
      tHover = setTimeout(() => { open(i, "hover"); markUsed(); }, cur >= 0 ? 0 : 110);
    });
    b.addEventListener("pointerleave", e => {
      if (e.pointerType !== "mouse") return;
      clearTimeout(tHover);
      if (how === "hover") scheduleClose();
    });
    b.addEventListener("click", () => {
      markUsed(); rove(i);
      if (cur === i && (how === "pin" || how === "focus" || phone)) { close(); softClick(1100); return; }
      open(i, "pin"); famNote(i, .6);
    });
    b.addEventListener("focus", () => {
      rove(i);
      if (!phone && b.matches(":focus-visible") && how !== "pin") open(i, "focus");
    });
  });
  card.addEventListener("pointerenter", () => clearTimeout(tClose));
  card.addEventListener("pointerleave", e => { if (e.pointerType === "mouse" && how === "hover") scheduleClose(); });
  stage.addEventListener("focusout", e => {
    if (how !== "focus") return;
    const to = e.relatedTarget;
    if (!to || !stage.contains(to)) close();
  });

  ol.addEventListener("keydown", e => {
    const i = btns.indexOf(e.target); if (i < 0) return;
    const live = evs.map((_, k) => k).filter(match);
    const p = Math.max(0, live.indexOf(i));
    let to = -1;
    if (e.key === "ArrowRight" || e.key === "ArrowDown") to = live[Math.min(live.length - 1, p + 1)];
    else if (e.key === "ArrowLeft" || e.key === "ArrowUp") to = live[Math.max(0, p - 1)];
    else if (e.key === "Home") to = live[0];
    else if (e.key === "End") to = live[live.length - 1];
    else if (e.key === "Escape" && cur >= 0) { e.preventDefault(); close(); return; }
    if (to == null || to < 0) return;
    e.preventDefault(); markUsed();
    if (to === i) return;
    if (how === "pin") how = "focus"; // arrows carry the open card along
    rove(to); btns[to].focus();
    if (phone) open(to, "focus");
    famNote(to, .45);
  });
  document.addEventListener("keydown", e => { if (e.key === "Escape" && cur >= 0 && !phone) { const i = cur; close(); if (stage.contains(document.activeElement)) btns[i].focus(); } });
  document.addEventListener("pointerdown", e => {
    if (cur < 0 || how === "hover" || phone) return;
    if (card.contains(e.target) || btns.some(b => b === e.target || b.contains(e.target))) return;
    close();
  });

  /* ---------- aria-controls follows the active presentation ---------- */
  function syncMode() {
    btns.forEach((b, i) => b.setAttribute("aria-controls", phone ? dets[i].id : "time-card"));
  }

  /* ---------- playhead: walks through the (filtered) events, one card at a time ---------- */
  let play = null;
  const STEP = 2700;
  function headTo(i, snap) {
    const y = yrOf[i], yl = y.querySelector(".y");
    let x, yy, p;
    if (!phone) {
      x = y.offsetLeft;
      yy = ol.offsetTop + yrs[1].offsetTop;              // the axis = top of the lower lane
      p = (x + 1) / ol.clientWidth;
    } else {
      x = 0;
      yy = ol.offsetTop + y.offsetTop + yl.offsetTop + 15;
      p = (y.offsetTop + yl.offsetTop + 15 - 6) / Math.max(1, ol.clientHeight - 6);
    }
    if (snap) head.classList.add("snap");
    head.style.setProperty("--hx", x + "px"); head.style.setProperty("--hy", yy + "px");
    if (snap) { void head.offsetWidth; head.classList.remove("snap"); }
    ol.style.setProperty("--tp", clamp01(p).toFixed(4));
  }
  function ensureVisible(i) {
    const r = evs[i].getBoundingClientRect();
    if (r.top < 90 || r.bottom + 220 > innerHeight) window.scrollBy({ top: r.top - innerHeight * .3, behavior: reduce ? "auto" : "smooth" });
  }
  function setPlayUI(on) {
    playBtn.setAttribute("aria-pressed", String(on));
    playLbl.textContent = on ? "Остановить" : "Проиграть историю";
    stage.classList.toggle("is-play", on);
  }
  function startPlay() {
    const list = evs.map((_, i) => i).filter(match);
    if (!list.length) return;
    markUsed(); finishEntrance(); close();
    play = { list, k: -1, t: 0 };
    setPlayUI(true);
    headTo(list[0], true);
    play.t = setTimeout(next, 260);
  }
  function next() {
    if (!play) return;
    play.k++;
    if (play.k >= play.list.length) { endChord(); stopPlay(); pst.textContent = "конец истории"; setTimeout(() => { if (!play) pst.textContent = ""; }, 2400); return; }
    const i = play.list[play.k];
    open(i, "play"); rove(i); headTo(i);
    pst.innerHTML = `<b>${yearOf[i]}</b> · ${play.k + 1} / ${play.list.length}`;
    stepNote(i);
    if (phone) ensureVisible(i);
    play.t = setTimeout(next, STEP + (play.k === play.list.length - 1 ? 600 : 0));
  }
  function stopPlay() {
    if (!play) return;
    clearTimeout(play.t); play = null;
    setPlayUI(false); pst.textContent = "";
    if (how === "play") close();
  }
  playBtn.addEventListener("click", () => { if (play) { stopPlay(); softClick(1100); } else startPlay(); });
  // any other interaction hands control back to the reader
  document.addEventListener("pointerdown", e => { if (play && !playBtn.contains(e.target)) stopPlay(); }, true);
  document.addEventListener("keydown", e => {
    if (!play) return;
    if (e.target === playBtn && (e.key === "Enter" || e.key === " ")) return;
    if (e.key === "Shift" || e.key === "Meta" || e.key === "Control" || e.key === "Alt") return;
    stopPlay();
  }, true);
  document.addEventListener("visibilitychange", () => { if (document.hidden) stopPlay(); });

  /* ---------- entrance ---------- */
  let entered = reduce, ap = 0, onScroll = null;
  function finishEntrance() {
    entered = true;
    stage.classList.remove("t-pre", "t-scroll");
    yrs.forEach(y => y.classList.add("in"));
    if (onScroll) { removeEventListener("scroll", onScroll); onScroll = null; }
  }
  function prepEntrance() {
    // delays follow the axis: year i is reached at ~150 + i*105 ms
    yrs.forEach((y, i) => {
      y.style.setProperty("--d", 150 + i * 105);
      [...y.querySelectorAll(".ev")].forEach((e, j) => { e.style.setProperty("--d", 150 + i * 105 + 170 + j * 95); e.style.setProperty("--j", j); });
    });
    evs.forEach((e, k) => e.style.setProperty("--k", k));
    const [eg, ed] = stage.querySelectorAll(".tm-eb");
    eg.style.setProperty("--d", 150 + 1 * 105); eg.style.setProperty("--dur", "560ms");
    ed.style.setProperty("--d", 150 + 6 * 105); ed.style.setProperty("--dur", "460ms");
    stage.classList.add(phone ? "t-scroll" : "t-pre");
    if (phone) stage.classList.add("t-pre");
  }
  function runDesktop() {
    stage.classList.add("t-run");
    requestAnimationFrame(() => requestAnimationFrame(() => stage.classList.remove("t-pre")));
    setTimeout(() => { stage.classList.remove("t-run"); finishEntrance(); startWave(); }, 2600);
  }
  function phoneProgress() {
    const r = ol.getBoundingClientRect();
    const p = clamp01((innerHeight * .8 - r.top) / Math.max(1, r.height));
    if (p > ap) { ap = p; ol.style.setProperty("--ap", ap.toFixed(4)); }
    const lim = ap * r.height;
    yrs.forEach(y => { if (!y.classList.contains("in") && y.offsetTop + 10 < lim) y.classList.add("in"); });
    if (ap > .995) { setTimeout(finishEntrance, 900); if (onScroll) { removeEventListener("scroll", onScroll); onScroll = null; } startWave(); }
  }
  if (!reduce) prepEntrance();
  else evs.forEach((e, k) => e.style.setProperty("--k", k));

  let inView = false;
  new IntersectionObserver(es => {
    for (const en of es) {
      inView = en.isIntersecting;
      if (!inView) { stopPlay(); continue; }
      if (entered) continue;
      if (!phone) { if (!stage.classList.contains("t-run")) runDesktop(); }
      else if (!onScroll) {
        let raf = 0;
        onScroll = () => { if (!raf) raf = requestAnimationFrame(() => { raf = 0; phoneProgress(); }); };
        addEventListener("scroll", onScroll, { passive: true });
        phoneProgress();
      }
    }
  }, { threshold: [0, .15] }).observe(stage);

  /* ---------- idle cue: a wave runs through the event dots until the reader touches anything ---------- */
  let waveT = 0;
  function startWave() {
    if (reduce || used || waveT) return;
    const tick = () => {
      if (used) { waveT = 0; return; }
      if (inView && !play && cur < 0 && !document.hidden) {
        stage.classList.add("t-wave");
        setTimeout(() => stage.classList.remove("t-wave"), 2400);
      }
      waveT = setTimeout(tick, 7000);
    };
    waveT = setTimeout(tick, 1200);
  }

  /* ---------- layout changes ---------- */
  function relayout() {
    if (!phone && cur >= 0 && !card.hidden) placeCard(cur);
    if (play && cur >= 0) headTo(cur, true);
  }
  new ResizeObserver(relayout).observe(stage);
  mq.addEventListener("change", () => {
    stopPlay(); close(); card.hidden = true; card.classList.remove("show");
    evs.forEach((e, i) => { e.classList.remove("is-x"); dets[i].hidden = true; });
    phone = mq.matches; syncMode();
    if (!entered) finishEntrance();
  });
  syncMode();
  setFilter("all");
})();
