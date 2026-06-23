# IKFlow assets (Tables I + IV)

Co-author's IKFlow model registry + pretrained weights for the paper's IKFlow baseline.

- `model_descriptions.yaml` — the co-author's IKFlow model registry. The Panda entry used by the
  benchmark is **`panda__full__lp191_5.25m`** (12 nodes, latent dim 7, `lyric-puddle-191` @ 5.25M steps).
  This is committed; it is *merged into* the installed `ikflow` package's own descriptions at runtime by
  `benchmark/baseline_ikflow.py` so the key resolves with the correct hyper-parameters (the stock `ikflow`
  ships a *different* `panda_full_tpm`: latent dim 9, `lucky-pond-7642` weights — the wrong architecture
  for these weights).
- `weights/` — the pretrained `.pkl` (≈200 MB, **gitignored**). `baseline_ikflow.py` stages whatever
  `.pkl` lives here into ikflow's weight cache (`~/.cache/ikflow/models/`) under the filename ikflow
  derives from the model URL, so loading is fully offline (the GCS bucket 403s here).

## Staging

`benchmark/baseline_ikflow.py` does this automatically on load (default `--weights-dir` = this dir, override
with `$IKFLOW_WEIGHTS_DIR`). To stage by hand for a different model, copy the `.pkl` here and it will be
picked up by URL basename.

Fetch IKFlow weights are not here yet (the public download 403s) — request them from the co-author; once
present, the same staging path handles them (registry already lists the Fetch keys).
