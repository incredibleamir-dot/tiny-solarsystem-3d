# Credits

## Textures

The planet, moon and Sun maps are the free equirectangular textures from
[Solar System Scope](https://www.solarsystemscope.com/textures/), licensed
under the **CC BY 4.0** attribution license:

- `2k_sun.jpg`, `2k_mercury.jpg`, `2k_venus_surface.jpg`,
  `2k_earth_daymap.jpg`, `2k_earth_nightmap.jpg`, `2k_mars.jpg`,
  `2k_jupiter.jpg`, `2k_saturn.jpg`, `2k_saturn_ring_alpha.png`,
  `2k_uranus.jpg`, `2k_neptune.jpg`, `2k_moon.jpg`
- `2k_ceres_fictional.jpg`, `2k_eris_fictional.jpg` (fictional maps — Ceres
  and Eris have not been mapped in detail)

**Pluto** is the NASA/JPL LORRI global mosaic (`2k_pluto.png`), extracted from
the NASA 3D resources model at
[https://solarsystem.nasa.gov/gltf_embed/2357/](https://solarsystem.nasa.gov/gltf_embed/2357/).

## Orbital model

Planet and Moon positions are computed by the vendored `solarsystem` package,
based on Paul Schlyter's
*[How to compute planetary positions](https://stjarnhimlen.se/comp/ppcomp.html)*
and published by **Ioannis Nasios** under the **MIT license**
(Copyright (c) 2020, Ioannis Nasios).

> If you use the solarsystem library in published work, please cite:
>
> ```text
> @misc{nasios2026solarsystemvalidatedlightweightpython,
>       title={Solarsystem: A Validated Lightweight Python Package for
>              Planetary Positions and Solar-Lunar Event Calculations},
>       author={Ioannis Nasios},
>       year={2026},
>       eprint={2606.27055},
>       archivePrefix={arXiv},
>       primaryClass={astro-ph.EP},
>       url={https://arxiv.org/abs/2606.27055},
> }
> ```

## Inspiration & architecture

- The **3D scene architecture** (ModernGL textured spheres, orbit camera,
  additive corona, starfield) is derived from
  [moon-watch-3d](https://github.com/incredibleamir-dot/moon-watch-3d) —
  itself a descendant of this project's 2D sibling.
- The **kid-friendly design**, HUD, panel and taskbar come from the original
  2D [tiny-solarsystem](https://github.com/incredibleamir-dot/tiny-solarsystem)
  explorer.

Coded with love by Amir for Aiman.
