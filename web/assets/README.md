# `web/assets/` — media for the landing page

Drop project media here; the landing page (`web/landing/`) references these files. **Until the real
assets land, the landing uses placeholders** — we do **not** substitute our local benchmark plots into
any public page.

## Source-of-truth rule (important)

Every public-facing number, table, and plot — on the landing page **and** in the docs — must come from
and **match the camera-ready paper exactly**. The benchmark plots/tables produced on our machines
(`benchmark/results/`) are **local run-reference only** and are gitignored; they are never published.

## What to provide

| File / item | Used for | Notes |
|---|---|---|
| `paper.pdf` (camera-ready) | **Authoritative** source for all public figures/tables | Result + method figures are extracted from here; tables transcribed verbatim. Also used to refresh authors / abstract / BibTeX. |
| `teaser.mp4` **or** a YouTube link | Hero / teaser video | Prefer a YouTube embed or git-LFS if the file is large (avoid a heavy binary in git — see below). ≤~30 s, ≥720p. |
| `method.(png\|svg)` | "Approach / Method" figure(s) | High-res; pipeline / warp-layout figure. Provide separately if not cleanly extractable from the PDF. |
| `results/*.png` | "Results" section figures | Extracted from the camera-ready paper (Pareto / scaling / tables). |
| `logo.(png\|svg)`, `favicon.ico` | Branding (optional) | Falls back to the A2R-Lab logo. |
| hardware / qualitative photos (optional) | Extra results / demo imagery | |

## Video hosting

A raw `.mp4` bloats the git history. Preferred order: **YouTube/Vimeo embed** → **git-LFS** → raw mp4.
Tell me which you want when you drop the asset and I'll wire it accordingly.

## Layout

```
web/assets/
  paper.pdf
  teaser.mp4            (or a youtube link in this README)
  method.png
  results/             figures extracted from the camera-ready paper
  logo.png  favicon.ico
```
