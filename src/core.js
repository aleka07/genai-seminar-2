(() => {
  "use strict";
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  /* ---------- seeded randomness ---------- */
  function mulberry32(a) { return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
  function gauss(r) { let u = 0, v = 0; while (!u) u = r(); while (!v) v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }

  /* ---------- the data: 2D Swiss roll ---------- */
  const T0 = 1.5 * Math.PI, T1 = 4.5 * Math.PI;
  function roll(u) { const t = T0 + (T1 - T0) * u; return [t * Math.cos(t) / 14.5, t * Math.sin(t) / 14.5]; }
  // Arc-length-ish sampling so dots spread evenly along the roll.
  function rollU(r) { return Math.sqrt(T0 * T0 + (T1 * T1 - T0 * T0) * r()) / (T1 - T0) - T0 / (T1 - T0); }
  const ease = t => t < .5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
  const clamp01 = t => t < 0 ? 0 : t > 1 ? 1 : t;
  let colors = {};
  function readColors() {
    const cs = getComputedStyle(document.documentElement);
    const g = n => cs.getPropertyValue(n).trim();
    colors = { fg: g("--fg"), vae: g("--vae"), gan: g("--gan"), dif: g("--dif"), bg: g("--bg"), dark: g("color-scheme").includes("dark") };
  }
  function rgba(hex, a) {
    const h = hex.replace("#", "");
    const n = parseInt(h.length === 3 ? h.replace(/./g, c => c + c) : h, 16);
    return `rgba(${n >> 16 & 255},${n >> 8 & 255},${n & 255},${a})`;
  }
  /* ---------- coded SFX (Web Audio, synthesized, no files) ---------- */
  const sfx = (() => {
    let ac = null, master = null, noiseBuf = null, on = false;
    function init() {
      if (ac) return;
      ac = new (window.AudioContext || window.webkitAudioContext)();
      master = ac.createGain(); master.gain.value = .18;
      const comp = ac.createDynamicsCompressor(); comp.threshold.value = -18; comp.ratio.value = 6;
      master.connect(comp).connect(ac.destination);
      noiseBuf = ac.createBuffer(1, ac.sampleRate * 2, ac.sampleRate);
      const d = noiseBuf.getChannelData(0); for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
    }
    function env(g, t, a, peak, dcy) { g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(peak, t + a); g.gain.exponentialRampToValueAtTime(0.0001, t + a + dcy); }
    function noise(t, dur, type, f0, f1, q, peak) {
      const s = ac.createBufferSource(); s.buffer = noiseBuf;
      const f = ac.createBiquadFilter(); f.type = type; f.Q.value = q;
      f.frequency.setValueAtTime(f0, t); f.frequency.exponentialRampToValueAtTime(f1, t + dur);
      const g = ac.createGain(); env(g, t, Math.min(.08, dur * .3), peak, dur);
      s.connect(f).connect(g).connect(master); s.start(t, Math.random()); s.stop(t + dur + .2);
    }
    function tone(t, freq, dur, type, peak, glideTo) {
      const o = ac.createOscillator(); o.type = type; o.frequency.setValueAtTime(freq, t);
      if (glideTo) o.frequency.exponentialRampToValueAtTime(glideTo, t + dur * .6);
      const g = ac.createGain(); env(g, t, .006, peak, dur);
      o.connect(g).connect(master); o.start(t); o.stop(t + dur + .1);
    }
    return {
      get on() { return on; },
      // generic hook for section sounds: fn({ ac, t, tone, noise, master }) runs only while sound is on
      play(fn) { if (!on) return; try { fn({ ac, t: ac.currentTime, tone, noise, master }); } catch (e) { /* a sound must never break the page */ } },
      toggle() { init(); on = !on; if (on) { ac.resume(); this.click(); } return on; },
      click() { if (!on) return; const t = ac.currentTime; tone(t, 1800, .04, "sine", .25); },
      // VAE: soft low-passed exhale through the bottleneck
      vae() { if (!on) return; const t = ac.currentTime; noise(t, .55, "lowpass", 2400, 380, .8, .5); tone(t + .05, 220, .5, "sine", .12, 196); },
      // GAN: two detuned voices argue, then snap together
      gan() { if (!on) return; const t = ac.currentTime; tone(t, 330, .16, "sawtooth", .07, 440); tone(t, 347, .16, "square", .05, 415); tone(t + .17, 440, .22, "triangle", .22); },
      // Diffusion step k of K: hiss narrows into a pure tone as noise is removed
      dif(k, K) {
        if (!on) return; const t = ac.currentTime, s = k / K;
        const q = .7 + s * s * 28, f = 900 + s * 700;
        noise(t, .09, "bandpass", f * 1.6, f, q, .35 * (1 - s) + .05);
        if (s > .55) tone(t, 587.33, .09 + s * .1, "sine", .05 + (s - .55) * .3);
        if (k === K - 1) { tone(t + .08, 587.33, 1.1, "sine", .22); tone(t + .08, 880, 1.1, "sine", .08); }
      },
      // tab chime for diffusion: noise resolving into a fifth
      chime() { if (!on) return; const t = ac.currentTime; noise(t, .35, "bandpass", 1400, 900, 6, .25); tone(t + .12, 587.33, .7, "sine", .16); tone(t + .12, 880, .7, "sine", .07); },
      // slider scrub: s = t/T; more noise = wider, louder hiss; clean = tonal ping
      scrub(s) { if (!on) return; const t = ac.currentTime; noise(t, .07, "bandpass", 700 + (1 - s) * 900, 700 + (1 - s) * 900, .6 + (1 - s) * (1 - s) * 30, .08 + s * .25); },
      // one note per model (C major triad), for hover on the trilemma and timeline
      note(m, g = 1) { if (!on) return; const f = { vae: 261.63, gan: 329.63, dif: 392 }[m] || 392; tone(ac.currentTime, f, .35, "sine", .14 * g); tone(ac.currentTime, f * 2, .2, "sine", .03 * g); },
      // back to noise: reverse whoosh
      back() { if (!on) return; const t = ac.currentTime; noise(t, 1.0, "bandpass", 300, 5200, 1.2, .35); }
    };
  })();
  readColors();
  const themeCbs = [];
  const onThemeChange = () => { readColors(); themeCbs.forEach(f => f()); };
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", onThemeChange);
  new MutationObserver(onThemeChange).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  const S = window.__sem = { sfx, reduce, mulberry32, gauss, roll, rollU, rgba, ease, clamp01, colors: () => colors, onTheme: f => themeCbs.push(f) };
  const snd = document.getElementById("snd"), sndLbl = document.getElementById("snd-lbl");
  snd.addEventListener("click", () => {
    const on = sfx.toggle();
    snd.setAttribute("aria-pressed", String(on));
    sndLbl.textContent = on ? "Звук вкл" : "Звук выкл";
  });
  // one soft click when the pointer enters a control, not on every child element inside it
  document.addEventListener("pointerover", e => {
    const el = e.target.closest && e.target.closest("button, a, [role=tab]");
    if (!el || (e.relatedTarget && el.contains(e.relatedTarget))) return;
    sfx.click();
  });
  /* ---------- Tabs (mechanism) ---------- */
  const tabs = [...document.querySelectorAll('.tab[role="tab"]')];
  function select(tab, focus) {
    for (const t of tabs) {
      const on = t === tab;
      t.setAttribute("aria-selected", String(on));
      t.tabIndex = on ? 0 : -1;
      const p = document.getElementById(t.getAttribute("aria-controls"));
      p.hidden = !on;
      if (on) { p.classList.remove("enter"); void p.offsetWidth; p.classList.add("enter"); }
    }
    if (focus) tab.focus();
    const m = tab.dataset.m;
    if (m === "vae") sfx.vae(); else if (m === "gan") sfx.gan(); else sfx.chime();
    document.dispatchEvent(new CustomEvent("sem:tab", { detail: m }));
  }
  tabs.forEach((t, i) => {
    t.addEventListener("click", () => select(t));
    t.addEventListener("keydown", e => {
      const d = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
      if (e.key === "Home") { e.preventDefault(); select(tabs[0], true); }
      else if (e.key === "End") { e.preventDefault(); select(tabs[tabs.length - 1], true); }
      else if (d) { e.preventDefault(); select(tabs[(i + d + tabs.length) % tabs.length], true); }
    });
  });

})();
