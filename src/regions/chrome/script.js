(() => {
  "use strict";
  const S = window.__sem; if (!S) return;
  const root = document.documentElement;
  const reduce = S.reduce;
  const bar = document.getElementById("chrome-bar");
  if (!bar) return;

  /* ---------- bar height -> --chrome-bar-h (scroll-margin for sections) ---------- */
  const setBarH = () => root.style.setProperty("--chrome-bar-h", Math.round(bar.getBoundingClientRect().height) + "px");
  setBarH();
  new ResizeObserver(setBarH).observe(bar);

  /* ---------- sounds: nav jump, sheet open/close, sound-on arpeggio ---------- */
  const NAV_NOTES = [523.25, 587.33, 659.25, 783.99, 880]; // C major pentatonic, one per section
  let lastNavSnd = 0;
  function navSound(i) {
    const now = performance.now(); if (now - lastNavSnd < 60) return; lastNavSnd = now;
    S.sfx.play(({ t, tone, noise }) => {
      noise(t, .28, "bandpass", 2600, 900, 1.4, .06);              // a soft "travel" swish
      tone(t + .05, NAV_NOTES[i] || 659.25, .32, "sine", .1);
      tone(t + .05, (NAV_NOTES[i] || 659.25) * 2, .14, "sine", .025);
    });
  }
  const sheetSound = open => S.sfx.play(({ t, tone }) => tone(t, open ? 392 : 523.25, .18, "triangle", .07, open ? 523.25 : 392));

  /* ---------- smooth scrolling for in-page anchors (nav, sheet, cites) ---------- */
  const SECS = ["mech", "tri", "time", "pick", "end"];
  function jump(id, viaKeyboard) {
    const el = document.getElementById(id); if (!el) return false;
    el.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" });
    try { history.replaceState(null, "", "#" + id); } catch (e) { /* sandboxed frame */ }
    if (viaKeyboard) {
      if (!el.hasAttribute("tabindex")) el.setAttribute("tabindex", "-1");
      el.focus({ preventScroll: true });
    }
    return true;
  }
  document.addEventListener("click", e => {
    const a = e.target.closest && e.target.closest('a[href^="#"]');
    if (!a || e.defaultPrevented || e.metaKey || e.ctrlKey || e.shiftKey || e.button) return;
    const id = decodeURIComponent(a.getAttribute("href").slice(1));
    if (!id) return;
    if (jump(id, e.detail === 0)) {
      e.preventDefault();
      if (a.hasAttribute("data-ch-link")) navSound(Math.max(0, SECS.indexOf(id)));
      if (sheetOpen && a.closest("#chrome-sheet")) closeSheet(false);
    }
  });

  /* ---------- active section (IntersectionObserver) + sliding indicator ---------- */
  const nav = document.getElementById("chrome-nav");
  const navLinks = [...bar.querySelectorAll("a[data-sec]")];
  const ind = nav.querySelector(".ch-ind");
  const tocLbl = document.getElementById("chrome-toc-lbl");
  const NAMES = { mech: "Устройство", tri: "Сравнение", time: "История", pick: "Выбор", end: "Выводы" };
  const watched = ["hero", ...SECS, "refs"].map(id => document.getElementById(id)).filter(Boolean);
  const mapId = id => id === "refs" ? "end" : id === "hero" ? null : id;
  let active;  // undefined first, so the first set always renders
  function placeIndicator() {
    const a = nav.querySelector('a[aria-current="true"]');
    if (!a || !nav.offsetWidth) return;
    // measure the text inside the link (not the padded hit area)
    const lr = a.getBoundingClientRect(), nr = nav.getBoundingClientRect();
    const pad = parseFloat(getComputedStyle(a).paddingLeft) || 0;
    ind.style.setProperty("--x", (lr.left - nr.left + pad) + "px");
    ind.style.setProperty("--w", (lr.width - 2 * pad) + "px");
  }
  function setActive(id) {
    if (id === active) return;
    active = id;
    for (const a of navLinks) a.setAttribute("aria-current", String(a.dataset.sec === id));
    nav.classList.toggle("has-active", !!id);
    if (id) placeIndicator();
    tocLbl.textContent = id ? NAMES[id] : "Содержание";
  }
  const vis = new Set();
  const pick = () => {
    let cur = null;
    for (const el of watched) if (vis.has(el)) cur = el;   // last visible in document order wins
    if (cur) setActive(mapId(cur.id));
  };
  let io;
  function observeSections() {
    if (io) io.disconnect();
    vis.clear();
    const top = Math.round(bar.getBoundingClientRect().height) + 4;
    // a section is "current" once its top has crossed 45% of the viewport
    io = new IntersectionObserver(es => {
      for (const e of es) e.isIntersecting ? vis.add(e.target) : vis.delete(e.target);
      pick();
    }, { rootMargin: `-${top}px 0px -55% 0px`, threshold: 0 });
    watched.forEach(el => io.observe(el));
  }
  observeSections();
  addEventListener("resize", () => { placeIndicator(); }, { passive: true });
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(placeIndicator);
  new ResizeObserver(placeIndicator).observe(nav);
  // enable the slide only after the first placement, so the line doesn't fly in from the left on load
  setTimeout(() => nav.classList.add("ind-anim"), 400);

  /* ---------- reading progress (1px line) ---------- */
  const prog = bar.querySelector(".ch-prog i");
  let maxScroll = 1, progQueued = false;
  const measure = () => { maxScroll = Math.max(1, root.scrollHeight - innerHeight); };
  function progress() {
    progQueued = false;
    const p = Math.min(1, Math.max(0, scrollY / maxScroll));
    prog.style.transform = `scaleX(${p.toFixed(4)})`;
  }
  const queue = () => { if (!progQueued) { progQueued = true; requestAnimationFrame(progress); } };
  measure(); progress();
  addEventListener("scroll", queue, { passive: true });
  addEventListener("resize", () => { measure(); queue(); }, { passive: true });
  new ResizeObserver(() => { measure(); queue(); }).observe(document.body);

  /* ---------- phone: «Содержание» sheet ---------- */
  const tocBtn = document.getElementById("chrome-toc");
  const sheet = document.getElementById("chrome-sheet");
  let sheetOpen = false;
  function openSheet() {
    sheetOpen = true; sheet.hidden = false;
    tocBtn.setAttribute("aria-expanded", "true");
    requestAnimationFrame(() => sheet.classList.add("open"));
    const cur = sheet.querySelector('a[aria-current="true"]') || sheet.querySelector("a");
    cur && cur.focus({ preventScroll: true });
    sheetSound(true);
  }
  function closeSheet(focusBtn) {
    if (!sheetOpen) return;
    sheetOpen = false; sheet.classList.remove("open");
    tocBtn.setAttribute("aria-expanded", "false");
    setTimeout(() => { if (!sheetOpen) sheet.hidden = true; }, reduce ? 0 : 260);
    if (focusBtn) tocBtn.focus();
    sheetSound(false);
  }
  tocBtn.addEventListener("click", () => sheetOpen ? closeSheet(false) : openSheet());
  document.addEventListener("keydown", e => { if (e.key === "Escape" && sheetOpen) closeSheet(true); });
  document.addEventListener("pointerdown", e => { if (sheetOpen && !bar.contains(e.target)) closeSheet(false); });
  matchMedia("(min-width: 900px)").addEventListener("change", e => { if (e.matches) closeSheet(false); });

  /* ---------- sound: first-visit hint (pulse + tooltip), arpeggio when switched on ---------- */
  const sndWrap = bar.querySelector(".ch-snd"), snd = document.getElementById("snd");
  const KEY = "sem2-snd-hint";
  let seen = false;
  try { seen = localStorage.getItem(KEY) === "1"; } catch (e) { /* storage blocked */ }
  let hintT1, hintT2;
  const endHint = () => { clearTimeout(hintT1); clearTimeout(hintT2); sndWrap.classList.remove("hint"); };
  if (!seen) {
    hintT1 = setTimeout(() => {
      if (S.sfx.on || scrollY > 300) return;
      sndWrap.classList.add("hint");
      try { localStorage.setItem(KEY, "1"); } catch (e) { /* ignore */ }
      hintT2 = setTimeout(endHint, 6500);
      // the hint belongs to the first screen: scrolling away dismisses it
      addEventListener("scroll", function off() { if (scrollY > 300) { endHint(); removeEventListener("scroll", off); } }, { passive: true });
    }, 2400);
  }
  snd.addEventListener("click", () => {
    endHint();
    try { localStorage.setItem(KEY, "1"); } catch (e) { /* ignore */ }
    // core.js toggles first (its listener was bound earlier); greet with the three model notes
    if (S.sfx.on) S.sfx.play(({ t, tone }) => {
      tone(t + .08, 261.63, .5, "sine", .1); tone(t + .2, 329.63, .5, "sine", .1); tone(t + .32, 392, .7, "sine", .12);
    });
  });

  /* ---------- global scroll-reveal ---------- */
  function initReveal() {
    // every section head gets an entrance (eyebrow, h2, intro), even if its region adds nothing
    document.querySelectorAll(".sec-head").forEach(h => {
      if (h.hasAttribute("data-reveal") || h.hasAttribute("data-reveal-stagger")) return;
      let i = 0;
      h.querySelectorAll(".eyebrow, h2, .intro").forEach(el => {
        if (!el.hasAttribute("data-reveal") && !el.closest("[data-reveal-stagger]")) {
          el.setAttribute("data-reveal", ""); el.setAttribute("data-reveal-delay", String(i++));
        }
      });
    });
    if (reduce || !("IntersectionObserver" in window)) return null;   // no hidden state at all

    const done = new WeakSet();
    const vh = () => innerHeight || root.clientHeight;
    function show(el, instant) {
      if (done.has(el)) return; done.add(el);
      if (instant) { el.classList.add("rv-in", "rv-done"); return; }
      el.classList.add("rv-in");
      const d = (parseFloat(getComputedStyle(el).getPropertyValue("--rv-d")) || 0) * 70;
      // drop our transition/transform once played, so region styles own the element again
      setTimeout(() => el.classList.add("rv-done"), 1100 + d);
    }
    const ioR = new IntersectionObserver(es => {
      for (const e of es) {
        if (!e.isIntersecting) continue;
        const r = e.boundingClientRect;
        // >=15% visible, or (tall elements) a quarter of the viewport filled
        if (e.intersectionRatio >= .15 || e.intersectionRect.height >= vh() * .25 || r.bottom < vh()) {
          show(e.target); ioR.unobserve(e.target);
        }
      }
    }, { threshold: [0, .15, .3] });

    function collect(scope) {
      const els = [];
      (scope || document).querySelectorAll("[data-reveal]").forEach(el => {
        el.style.setProperty("--rv-d", el.getAttribute("data-reveal-delay") || "0"); els.push(el);
      });
      (scope || document).querySelectorAll("[data-reveal-stagger]").forEach(p => {
        [...p.children].forEach((c, i) => {
          const base = parseFloat(p.getAttribute("data-reveal-delay")) || 0;
          c.style.setProperty("--rv-d", String(base + i)); els.push(c);
        });
      });
      return els.filter(el => !done.has(el));
    }
    function scan(scope) {
      const els = collect(scope), h = vh();
      for (const el of els) {
        const r = el.getBoundingClientRect();
        if (r.top < h && r.bottom > 0 || r.bottom <= 0) show(el, true);   // in view or already passed: no animation
        else ioR.observe(el);
      }
    }
    scan();
    root.classList.add("rv-ready");
    // safety net: after 6 s (and on every later scroll pause) nothing at or above the viewport stays hidden
    const sweep = () => {
      const h = vh();
      collect().forEach(el => { if (el.getBoundingClientRect().top < h) { show(el, true); ioR.unobserve(el); } });
    };
    setTimeout(sweep, 6000);
    let st; addEventListener("scroll", () => { clearTimeout(st); st = setTimeout(sweep, 700); }, { passive: true });
    addEventListener("beforeprint", () => collect().forEach(el => show(el, true)));
    return { scan, revealAll: () => collect().forEach(el => show(el, true)) };
  }
  // run after every region script has executed (they may add data-reveal attributes)
  const go = () => { S.reveal = initReveal() || { scan() {}, revealAll() {} }; };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", go); else setTimeout(go, 0);
})();
