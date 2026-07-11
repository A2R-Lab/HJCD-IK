# Where to drop project media

The landing page (`docs/landing/index.html`) and the docs reference media by **exact filename**. Drop files
at the paths below and they are picked up with **no code changes** — each currently shows a placeholder.

## Source-of-truth rule

Every public number/table/plot must match the **camera-ready paper** exactly. The result *tables* are
already transcribed from the paper (landing + `docs/.../benchmarks/results.rst`). The benchmark plots we
generated locally (`benchmark/results/`) are **gitignored run-reference only** — never publish them.

## Landing-page media → `docs/landing/static/`

| Put the file here | What it is | Notes |
|---|---|---|
| `docs/landing/static/videos/teaser.mp4` | Hero/teaser video (Franka hardware demo) | Or give me a **YouTube link** and I'll embed it instead (preferred if the file is large — keeps it out of git; otherwise we'll use git-LFS). ≤~30 s, ≥720p. |
| `docs/landing/static/images/method.png` | Method / pipeline figure (paper Fig. 3) | High-res PNG/SVG. |
| `docs/landing/static/images/teaser.png` | Social-preview / hero still (og:image) | Optional; a still from the video or Fig. 1. |
| `docs/landing/static/images/a2r_lab.png` | A2R Lab logo | **Already in place** (reused from GLASS). Replace if you have an HJCD-specific logo. |
| `docs/landing/static/images/favicon.ico` | Favicon | **Already in place.** |

## Optional paper figures for the docs → `docs/source/_static/paper/`

| Put the file here | What it is |
|---|---|
| `docs/source/_static/paper/*.png` | Pareto / DoF-study / qualitative figures (paper Figs. 1, 4–6), if you want them embedded on the docs Results page alongside the tables. Tell me the filenames and I'll wire the `.. figure::` directives. |

## Video hosting

Raw `.mp4` bloats git history. Preference order: **YouTube/Vimeo embed** → **git-LFS** → raw mp4 in
`static/videos/`. Tell me which when you drop it.
