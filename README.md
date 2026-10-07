# VAE, GAN, Diffusion — Seminar 2

Interactive comparative analysis of three families of generative models: VAE, GAN and diffusion models.
Seminar 2 of the course "Generative AI: Technologies and Applications", al-Farabi Kazakh National University, 2026.

**Live page:** https://dybys.me/genai-seminar-2/

What is inside:
- three samplers turning the same Gaussian noise into a 2D Swiss roll (VAE blur, GAN mode collapse, diffusion steps);
- a VAE latent-space playground with a KL-weight (β) slider;
- a 1D GAN trained live in the browser, showing mode hopping;
- a DDIM sampler with an exact optimal denoiser and a step-count trade-off;
- the generative learning trilemma, a comparison table, a 2013–2024 timeline, a model chooser and conclusions;
- synthesized sound effects (Web Audio), off by default.

The page is a single static file (`index.html`). Sources live in `src/` (one module per section) and are assembled with:

```sh
python3 build.py
```

Author: Alikhan Amirkhan.
