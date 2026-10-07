(() => {
  "use strict";
  const S = window.__sem;
  const root = document.getElementById("pick-chooser");
  if (!root) return;

  /* ---------- the scoring model: one table, shown to the reader on demand ---------- */
  const MODELS = ["vae", "gan", "dif"];
  const NAME = { vae: "VAE", gan: "GAN", dif: "Diffusion" };
  const REQS = ["rt", "sharp", "rare", "lat", "cheap"];
  // points: 2 = strength, 1 = with caveats, 0 = weak spot
  const PTS = {
    rt:    { vae: 2, gan: 2, dif: 0 },
    sharp: { vae: 0, gan: 2, dif: 2 },
    rare:  { vae: 2, gan: 0, dif: 2 },
    lat:   { vae: 2, gan: 1, dif: 0 },
    cheap: { vae: 2, gan: 1, dif: 0 }
  };
  const ROW = {
    rt:    ["Реальное время", "VAE и GAN генерируют за один прогон сети. Диффузии нужны десятки шагов."],
    sharp: ["Максимум чёткости", "Гауссов декодер VAE усредняет варианты, отсюда размытость. GAN и диффузия дают чёткие образцы."],
    rare:  ["Важны редкие случаи", "GAN склонен к mode collapse. VAE и диффузия учатся по правдоподобию и покрывают все моды."],
    lat:   ["Нужен латентный вектор", "У VAE есть энкодер. У GAN латент есть, но z для данного x ищут инверсией. У диффузии латент размером с картинку."],
    cheap: ["Мало вычислений на обучение", "VAE: одна сеть и стабильный лосс. GAN: две сети и капризное равновесие. Диффузия учится долго, по всем шагам t."]
  };
  // what a model brings when it scores 2 on a requirement
  const GOOD = {
    rt: "один прогон сети", sharp: "чёткие образцы", rare: "покрытие всех мод",
    lat: "энкодер с осмысленным латентом", cheap: "дешёвое стабильное обучение"
  };
  // what a model pays with: per weak cell, and its default weakness
  const PAY = {
    vae: { sharp: "чёткостью: образцы размыты", _: "чёткостью: образцы размыты" },
    gan: { rare: "разнообразием: редкие моды теряются (mode collapse)", _: "разнообразием: редкие моды теряются (mode collapse)" },
    dif: { rt: "скоростью: десятки прогонов сети на один образец", lat: "компактным латентом: он размером с картинку",
           cheap: "бюджетом: обучение долгое", _: "скоростью: десятки прогонов сети на один образец" }
  };
  const PRESETS = [
    { on: ["sharp", "rare"], t: "картинка по тексту" },
    { on: ["rt", "sharp"], t: "фильтр в камере телефона" },
    { on: ["lat", "cheap"], t: "поиск аномалий" },
    { on: ["rt", "sharp", "rare"], t: "всё и сразу" }
  ];

  const btns = [...root.querySelectorAll(".pk-req")];
  const bars = {};
  root.querySelectorAll(".pk-bar").forEach(b => {
    const m = b.dataset.m, track = b.querySelector(".pk-track");
    const segs = {};
    REQS.forEach(r => {
      const s = document.createElement("span");
      s.className = "pk-seg" + (PTS[r][m] === 1 ? " half" : "");
      track.appendChild(s); segs[r] = s;
    });
    bars[m] = { el: b, segs, num: b.querySelector(".pk-sc b"), of: b.querySelector(".pk-of") };
  });
  const verdict = document.getElementById("pick-verdict");
  const barsBox = document.getElementById("pick-bars");
  const demoK = document.getElementById("pick-demo-k"), demoT = document.getElementById("pick-demo-t"), prog = document.getElementById("pick-prog");
  const rows = [...document.querySelectorAll("#pick-list li")];
  const how = document.getElementById("pick-how"), table = document.getElementById("pick-table"), tbody = document.getElementById("pick-tbody");

  // table body from the same data that drives the bars
  const trs = {};
  REQS.forEach(r => {
    const tr = document.createElement("tr");
    let h = `<th scope="row">${ROW[r][0]}<span class="n">${ROW[r][1]}</span></th>`;
    MODELS.forEach(m => {
      const p = PTS[r][m];
      h += `<td class="m-${m}"><span class="pk-pts" aria-label="${NAME[m]}: ${p} из 2"><i class="${p > 0 ? "f" : ""}"></i><i class="${p > 1 ? "f" : ""}"></i><b>${p}</b></span></td>`;
    });
    tr.innerHTML = h; tbody.appendChild(tr); trs[r] = tr;
  });

  let sel = new Set(PRESETS[0].on);
  let lastKey = "";

  const list = a => a.length < 2 ? a.join("") : a.slice(0, -1).join(", ") + " и " + a[a.length - 1];
  const mName = m => `<span class="m-${m}">${NAME[m]}</span>`;
  function payOf(m) {
    for (const r of REQS) if (sel.has(r) && PTS[r][m] === 0 && PAY[m][r]) return PAY[m][r];
    return PAY[m]._;
  }

  function compute() {
    const sc = {}; MODELS.forEach(m => { sc[m] = 0; sel.forEach(r => { sc[m] += PTS[r][m]; }); });
    const max = 2 * sel.size;
    const best = Math.max(...MODELS.map(m => sc[m]));
    const win = sel.size ? MODELS.filter(m => sc[m] === best) : [];
    const tri = sel.has("rt") && sel.has("sharp") && sel.has("rare");
    return { sc, max, win, tri };
  }

  function verdictHTML(st) {
    const { win, tri } = st;
    if (!sel.size) return "Отметь хотя бы одно требование, и таблица назовёт модель.";
    if (tri) return `<span class="pk-tri">Трилемма</span>Скорость, чёткость и редкие моды сразу не даёт ни одна из трёх моделей, каждая берёт два свойства из трёх, поэтому на практике выбирают гибриды: дистиллированную или латентную диффузию, Denoising Diffusion GAN.<a class="cite" href="#ref-7">[7]</a>`;
    if (win.length === 1) {
      const m = win[0];
      const good = REQS.filter(r => sel.has(r) && PTS[r][m] === 2).map(r => GOOD[r]);
      return `Выбор: ${mName(m)}${good.length ? ` — ${list(good)}` : ""}, но платишь ${payOf(m)}.`;
    }
    const common = REQS.filter(r => sel.has(r) && win.every(m => PTS[r][m] === 2)).map(r => GOOD[r]);
    const pays = win.map((m, i) => `${mName(m)}${i ? " —" : " платит"} ${payOf(m).split(":")[0]}`);
    return `Ничья: ${list(win.map(mName))}${common.length ? ` дают ${list(common)}` : " набирают поровну"}; разница в цене: ${pays.join(", ")}.`;
  }

  /* ---------- sound ---------- */
  const F = { vae: 261.63, gan: 329.63, dif: 392 };
  function chord(ms, gain = 1) {
    S.sfx.play(({ t, tone }) => {
      ms.forEach((m, i) => {
        const f = F[m];
        tone(t + i * .07, f, .9, "sine", .1 * gain);
        tone(t + i * .07, f * 1.5, .7, "sine", .035 * gain);
      });
      if (ms.length === 1) tone(t + .12, F[ms[0]] * 2, .6, "sine", .04 * gain);
    });
  }

  /* ---------- render ---------- */
  function render(fromUser) {
    const st = compute();
    btns.forEach(b => b.setAttribute("aria-pressed", String(sel.has(b.dataset.r))));
    MODELS.forEach(m => {
      const b = bars[m];
      let x = 0;
      REQS.forEach(r => {
        const w = sel.has(r) && st.max ? PTS[r][m] / st.max * 100 : 0;
        b.segs[r].style.left = x + "%"; b.segs[r].style.width = w + "%"; x += w;
      });
      b.num.textContent = st.sc[m];
      b.of.textContent = " / " + st.max;
      const isWin = st.win.includes(m) && !st.tri;
      b.el.classList.toggle("win", isWin && st.win.length === 1);
      b.el.classList.toggle("lose", !!st.win.length && !st.win.includes(m) && !st.tri);
    });
    barsBox.setAttribute("aria-label", "Баллы: " + MODELS.map(m => `${NAME[m]} ${st.sc[m]} из ${st.max}`).join(", "));
    root.dataset.trilemma = String(st.tri);
    REQS.forEach(r => { trs[r].classList.toggle("on", sel.has(r)); trs[r].classList.toggle("off", sel.size > 0 && !sel.has(r)); });
    rows.forEach(li => {
      const hit = sel.size > 0 && li.dataset.r.split(" ").some(r => sel.has(r));
      li.classList.toggle("hit", hit);
      li.classList.toggle("dim", sel.size > 0 && !hit);
    });
    const key = (st.tri ? "tri" : st.win.join("+")) + "|" + [...sel].sort().join(",");
    const html = verdictHTML(st);
    if (verdict.innerHTML !== html) {
      verdict.innerHTML = html;
      if (!S.reduce) { verdict.classList.remove("swap"); void verdict.offsetWidth; verdict.classList.add("swap"); }
    }
    const winKey = st.tri ? "tri" : st.win.join("+");
    if (fromUser && winKey !== lastKey.split("|")[0] && st.win.length) chord(st.tri ? MODELS : st.win);
    lastKey = key;
    return st;
  }

  /* ---------- auto-demo: cycle presets while idle and on screen ---------- */
  const DWELL = 3600, IDLE = 9000;
  let mode = S.reduce ? "still" : "demo";
  let pi = 0, t0 = 0, raf = 0, visible = false, idleTimer = 0, pointerInside = false;
  root.dataset.mode = mode;

  function showPreset(i) {
    pi = i; sel = new Set(PRESETS[i].on);
    demoK.textContent = "пример"; demoT.textContent = PRESETS[i].t;
    render(false);
  }
  function loop(now) {
    raf = 0;
    if (mode !== "demo" || !visible || document.hidden) return;
    if (!t0) t0 = now;
    const p = (now - t0) / DWELL;
    if (p >= 1) { t0 = now; showPreset((pi + 1) % PRESETS.length); prog.style.transform = "scaleX(0)"; }
    else prog.style.transform = `scaleX(${p.toFixed(3)})`;
    raf = requestAnimationFrame(loop);
  }
  function kick() { if (!raf && mode === "demo" && visible && !document.hidden) { t0 = 0; raf = requestAnimationFrame(loop); } }
  function takeOver() {
    if (mode === "demo") { mode = "user"; root.dataset.mode = mode; demoK.textContent = "твой выбор"; demoT.textContent = ""; }
    root.classList.add("touched");
    clearTimeout(idleTimer);
    if (!S.reduce) idleTimer = setTimeout(resume, IDLE);
  }
  function resume() {
    if (pointerInside || root.contains(document.activeElement)) { idleTimer = setTimeout(resume, IDLE); return; }
    mode = "demo"; root.dataset.mode = mode;
    showPreset((pi + 1) % PRESETS.length); kick();
  }

  btns.forEach(b => {
    b.addEventListener("click", () => {
      const was = mode === "demo";
      takeOver();
      if (was) demoT.textContent = "";
      const r = b.dataset.r;
      if (sel.has(r)) sel.delete(r); else sel.add(r);
      const on = sel.has(r);
      const st = render(true);
      // note of the requirement's best model (C major: VAE C, GAN E, Diffusion G)
      const fav = MODELS.filter(m => PTS[r][m] === 2);
      const m = st.win.length === 1 ? st.win[0] : fav[0];
      S.sfx.note(m, on ? 1 : .55);
    });
  });
  how.addEventListener("click", () => {
    const open = how.getAttribute("aria-expanded") !== "true";
    how.setAttribute("aria-expanded", String(open));
    table.classList.toggle("open", open);
    takeOver();
    S.sfx.play(({ t, tone }) => { tone(t, open ? 523.25 : 659.25, .18, "sine", .08, open ? 659.25 : 523.25); });
  });
  root.addEventListener("pointerenter", () => { pointerInside = true; });
  root.addEventListener("pointerleave", () => { pointerInside = false; });
  root.addEventListener("keydown", e => { if (e.key === "Tab") return; takeOver(); });

  new IntersectionObserver(es => {
    visible = es[0].isIntersecting;
    if (visible) kick();
  }, { threshold: .25 }).observe(root);
  document.addEventListener("visibilitychange", () => { if (!document.hidden) kick(); });

  /* ---------- list rows: staggered reveal ---------- */
  const ul = document.getElementById("pick-list");
  rows.forEach((li, i) => li.style.setProperty("--i", i));
  if (!S.reduce && "IntersectionObserver" in window) {
    const r = ul.getBoundingClientRect();
    if (r.top > innerHeight * .92) {
      ul.classList.add("pre");
      const io = new IntersectionObserver(es => {
        if (es[0].isIntersecting) { ul.classList.remove("pre"); ul.classList.add("in"); io.disconnect(); }
      }, { threshold: .08 });
      io.observe(ul);
    }
  }

  if (mode === "still") { demoK.textContent = "пример"; }
  showPreset(0);
  lastKey = "dif|rare,sharp";
})();
