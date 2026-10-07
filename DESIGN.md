# Seminar 2 page — design direction

Subject: comparative analysis of VAE, GAN and diffusion models (KazNU, "Generative AI", seminar 2).
Reader: the lecturer and the group. Single job: in five minutes, show how the three families differ in mechanism and when to pick which.

## Hero idea
One target, three samplers. The same seeded Gaussian noise is fed to three canvases; each turns it into a 2D Swiss roll the way its model family does:
- VAE: one step, lands on the roll but smeared (blur from the Gaussian decoder + KL pressure).
- GAN: one step, crisp, but only 2–3 arcs of the roll (mode collapse; the arcs hop between loops).
- Diffusion: ~50 visible denoising steps (t = 1000 → 0), ends crisp and covers the whole roll.
The faint grey roll is p_data. Speed, quality and diversity are visible without a caption: the trilemma.

## Palette (dark-first, light variant)
| token | dark | light | role |
|---|---|---|---|
| --bg | #0C0F14 | #F1F2EE | background (cool ink / cool paper) |
| --fg | #E7EAEE | #12161B | text, action (buttons/links: fg + underline) |
| --muted | #98A3B3 | #586270 | secondary text |
| --line | 10% fg | 12% fg | hairlines |
| --vae | #74A7FF | #2459C9 | data: VAE |
| --gan | #FF6A4D | #CC3F22 | data: GAN |
| --dif | #F4C24D | #9A6F00 | data: diffusion |
Model hues are data only (never decoration). p_data is fg at low alpha.

## Type
- Display: Unbounded 400–600, tight leading 1.02, tracking −0.025em. Cyrillic only (no Greek: formulas never go in display).
- Text: Literata, 18 px, line length ≤ 66ch. Scientific-journal voice.
- Mono: JetBrains Mono for labels, step counters, formulas (has Greek, ∇, ‖, 𝔼, 𝒩; lacks ≈ ≤ ∑ √ — avoid them).
fonts-check: Cyrillic passes for all three.

## Sections (each its own form)
1. Hero — three live samplers + headline.
2. Mechanism — tabbed pipeline stage (encoder → z → decoder / G vs D / forward + reverse noise) with one key formula each; diffusion tab has a t-slider that scrubs noise.
3. Trilemma — triangle (quality / diversity / speed), each model is the edge it owns; criteria table below it.
4. Timeline 2013–2024 — horizontal track, dots coloured by family.
5. Which to pick — task → model decision list.
6. Conclusions + references (numbered, mono indices) + author line.

## Motion and sound
- One orchestrated moment: the hero sampling loop. Elsewhere: small eased reveals, nothing bounces.
- Coded SFX (Web Audio, no files), off by default, toggled from the top bar: VAE soft filtered swoosh, GAN two detuned oscillators snapping, diffusion hiss that narrows into a pure tone step by step, reset = reverse whoosh. Master gain 0.18.
- prefers-reduced-motion: the hero shows the final sampled state, still.
- Film grain overlay (noise is the subject), hairlines instead of boxes.

## 3D
No. The subject is distributions in 2D; a canvas shows the real mechanism, 3D would be decoration.
