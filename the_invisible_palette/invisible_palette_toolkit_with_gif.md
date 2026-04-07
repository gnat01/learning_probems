# Invisible Palette Toolkit with GIFs

This document covers the GIF-enabled companion script:

```bash
python invisible_palette_toolkit_with_gif.py
```

It follows the same Bayesian setup and inference philosophy as the base toolkit, but adds:

- per-round posterior frame generation,
- combined round-summary panels,
- optional animated GIF export.

The core mathematical model is the same as in `invisible_palette_toolkit.md`:

1. `distinct_only`
2. `full_uniform`
3. `full_dirichlet`

---

## 1. Shared modeling setup

Everything about:

- the hidden urn construction,
- the meaning of `C`,
- the three likelihoods,
- the role of `--alpha`,
- the meaning of `occ`,
- the audit record in `run_info.txt`,

is the same as in:

`invisible_palette_toolkit.md`

This file focuses on what is added by the GIF-oriented script.

---

## 2. Additional outputs

In addition to the static outputs from the base script, the GIF script writes per-round frame directories:

- `posterior_frames_combined/`
- `posterior_frames_distinct_only/`
- `posterior_frames_full_uniform/`
- `posterior_frames_full_dirichlet/`

Each round gets:

- one combined panel summarizing all three posteriors plus run metadata,
- one single-mode posterior plot for each inference mode.

If `--make-gif` is supplied and a supported backend is available, the script also writes:

- `posterior_evolution_combined.gif`
- `posterior_evolution_distinct_only.gif`
- `posterior_evolution_full_uniform.gif`
- `posterior_evolution_full_dirichlet.gif`

If GIF creation is not available, the frame directories are still written.

---

## 3. GIF-specific CLI flags

The GIF script supports all flags from `invisible_palette_toolkit.py`, plus:

- `--make-gif`
  Build animated GIFs from the generated frame sequences.

- `--gif-fps`
  Frames per second for the output GIFs.

Example:

```bash
python invisible_palette_toolkit_with_gif.py \
  --m 8 \
  --count-mode uniform \
  --min-count 1 \
  --max-count 10 \
  --batch-size 8 \
  --rounds 20 \
  --seed 0 \
  --c-max 20 \
  --alpha 0.5 \
  --prior-type uniform \
  --prior-lam 0.2 \
  --make-gif \
  --gif-fps 2.0 \
  --outdir results_with_gif
```

If you do not want GIFs, just omit `--make-gif`. The script will still generate the static plots and per-round frames.

---

## 4. GIF backend behavior

The script tries GIF backends in this order:

1. `imageio`
2. Pillow

If neither is available:

- static outputs are still generated,
- frame directories are still generated,
- GIF creation is skipped with a diagnostic message.

The detected backend is recorded in `run_info.txt`.

---

## 5. Performance note

The inference core has been optimized substantially, but the GIF script is primarily a visualization workload once posterior computation is done.

End-to-end runtime can be dominated by:

- static PNG rendering,
- per-round frame generation,
- GIF encoding,
- first-use `matplotlib` font/cache setup.

So it is normal for:

- raw posterior computation to be very fast,
- full `invisible_palette_toolkit_with_gif.py` runs to still take noticeably longer.

If runtime matters more than animation, the base script is the cheaper path:

```bash
python invisible_palette_toolkit.py ...
```

If visualization matters more, keep the GIF script and expect rendering to dominate the total wall clock time.
