#!/usr/bin/env bash
# Build the full HJCD-IK public website into ./_site :
#   _site/            the static landing page (web/landing/) at the site root
#   _site/docs/       the Sphinx docs (Doxygen + Breathe)
#
# This is the SINGLE source of truth for assembling the site — the gh-pages CI
# workflow runs this exact script, so the deployed site always matches a local
# build (no drift). On every push to `main`, CI regenerates and redeploys
# automatically; run this locally only to preview.
#
#   ./scripts/build_site.sh        # build into ./_site
#   xdg-open _site/index.html      # preview the landing ( _site/docs/ for the docs )
#
# Requires the docs toolchain on PATH (Sphinx + Breathe + pydata theme) and
# `doxygen`. `./scripts/setup/setup_dev.sh` installs both into .venv — activate it
# first (`source .venv/bin/activate`) when building locally.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

echo "[site] (1/2) building docs (doxygen + sphinx) ..."
make -C docs all

echo "[site] (2/2) assembling _site (landing at /, docs under /docs/) ..."
rm -rf _site && mkdir -p _site
cp -r web/landing/. _site/
cp -r docs/build/html _site/docs
touch _site/.nojekyll            # serve Sphinx _static/_sources verbatim (no Jekyll)

echo "[site] done -> $ROOT/_site   (open _site/index.html)"
