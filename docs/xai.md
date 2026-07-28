# Explainable AI (XAI) Toolkit

## Overview

Stock Donkeycar's deep-learning autopilot gives you a steering/throttle number and nothing else — you can't tell *how sure* the model was, or *why* it steered the way it did, without guessing. This fork adds an explainability and uncertainty toolkit on top of the existing pilot, built from four independent pieces:

1. **Confidence** (MC-Dropout) — "do the model's internal sub-networks agree with each other?" A live 0-100% signal.
2. **Novelty / out-of-distribution (OOD) detection** — "does this camera frame look like anything the model was trained on?" Also live, 0-100%.
3. **TTA stability** — "if I nudge the lighting slightly, does the model's answer wobble?" Also live, 0-100%.
4. **Confidence-aware throttle scaling** — an optional part that slows the car down (and can bring it to a full stop) when any of the three signals above indicates trouble, so uncertainty isn't just a number nobody acts on.

Alongside the live signals, there's an **offline analysis viewer**: point it at a recorded drive and a model, and it produces a frame-by-frame, browser-based playback showing *where* the model was looking and *why* it was or wasn't confident, using four different visual attribution techniques (Grad-CAM, Grad-CAM++, Integrated Gradients, and pixel-gradient saliency).

All four live signals and the offline viewer share one dependency: a **calibration file** (`<model>.calib.json`) that turns each raw internal number into a 0-100% score that's meaningful for *your* specific model and *your* specific training data. This is also the toolkit's connection point to this fork's other feature: if your model was trained with any [`AUGMENTATIONS`](augmentations.md) turned on, calibration needs to know about that too, or the novelty signal will misfire — covered in detail in [§2.8](#28-how-augmentations-feed-into-calibration).

This document's Section 1 is a usage walkthrough: what to turn on, in what order, and what you'll see. Section 2 is a technical explanation of how each signal actually works.

---

## 1. Usage

### 1.1 The pipeline, at a glance

```
record data  →  train  →  calibrate  →  drive live (dashboard)
                                     ↘  analyse offline (viewer)
```

Calibration sits in the middle because every downstream piece — the live dashboard percentages, throttle scaling, and the offline viewer's confidence graph — depends on it existing.

### 1.2 Recording and training

No changes here — record a tub and train exactly as stock Donkeycar describes (`donkey train`). The XAI toolkit only reads the model file afterwards; it doesn't change how training itself works. One thing to know: MC-Dropout confidence (and, by extension, most of this toolkit) currently only supports the `linear` model type, since it relies on the `Dropout` layers already present in that architecture staying active at inference time — see [§2.2](#22-dropout-and-monte-carlo-dropout-confidence) for why.

**Recording an autopilot drive (for offline review, not for training).** By default Donkeycar only records frames while you're driving in `'user'` (manual) mode — switching into autopilot stops recording, even if the recording toggle is on. If you want to feed an actual autonomous run into this toolkit's offline viewer (§1.6), for example to see what the model was looking at right before a near-miss, set the stock Donkeycar option in `myconfig.py`:

```python
RECORD_DURING_AI = True
```

Autopilot frames recorded this way are automatically written to a separate tub — a sibling folder named `<your_tub>_autopilot` next to your regular tub — instead of being mixed into your training data. Your manual tub's name and location never change. This happens automatically as soon as `RECORD_DURING_AI` is on; there's nothing else to configure, and no risk of an autonomous run accidentally ending up in a `donkey train` run, since that command only ever looks at the tub path(s) you explicitly give it.

### 1.3 Calibration

Calibration replays a tub of "known good" driving through the model once and records how the raw signals behave on data the model actually learned from — this becomes the yardstick every live percentage is measured against. **Nothing lights up on the dashboard until this has run once** for a given model.

**Option A — automatic, at the end of training.** In `myconfig.py`:

```python
XAI_CONFIDENCE_AUTO_CALIBRATE = True
```

This adds one extra replay pass over your training tubs right after `donkey train` finishes, and saves `<model>.calib.json` next to the model automatically. Off by default because it adds time to every training run.

**Option B — manual, from the command line** (needed if you trained before turning on auto-calibrate, or want to recalibrate later):

```bash
python -m donkeycar.parts.mc_calibrate --tub data/mytub --model models/mypilot.h5
```

```
usage: mc_calibrate [-h] --tub TUB [TUB ...] --model MODEL [--config CONFIG]
                     [--limit LIMIT] [--passes PASSES] [--alpha ALPHA]
                     [--out OUT] [--augment | --no-augment]

  --tub TUB [TUB ...]  one or more tub paths to replay
  --model MODEL        path to the .h5 model
  --config CONFIG      path to config.py (defaults to ./config.py, falling
                        back to the bundled complete template)
  --limit LIMIT        only use the first N frames
  --passes PASSES      override XAI_CONFIDENCE_PASSES
  --alpha ALPHA        override XAI_CONFIDENCE_ALPHA
  --out OUT            output calibration path (default <model>.calib.json)
  --augment            widen the novelty baselines with augmented frames
  --no-augment         fit the novelty baselines on clean frames only
```

Point `--tub` at the **same data the model was actually trained on** — calibration is meant to answer "what does normal look like to this model," and calibrating against a different drive (e.g. a test/inference tub) silently defeats the whole signal. The offline GUI launcher (§1.6) tries to auto-detect this for you from Donkeycar's own training record; the CLI here has no such lookup, so it's on you to point `--tub` correctly.

One calibration run covers **all three signals at once** — confidence, novelty and TTA stability are always all written into the file, whether or not you currently have them switched on for driving. So turning `XAI_TTA_ENABLED` (or any other signal) on later just works; you don't need to recalibrate to "add" it.

**When to recalibrate:**
- After retraining or swapping the model (new weights mean the old numbers no longer describe it).
- After changing `AUGMENTATIONS` in a meaningful way (see [§2.8](#28-how-augmentations-feed-into-calibration)).

**On image size:** calibration and offline analysis both read the input size directly from the model file, so a model trained at a different resolution than your current config still works — you'll just see a warning saying which size was used. If that warning looks wrong, point `--config` at the `config.py` the model was actually trained with.

### 1.4 Driving live with the signals on

Each signal is an independent switch in `myconfig.py` — turn on only what you want:

```python
XAI_CONFIDENCE_ENABLED = True   # MC-Dropout confidence
XAI_NOVELTY_ENABLED = True      # feature-space novelty / OOD
XAI_TTA_ENABLED = True          # test-time-augmentation stability
```

Once enabled and calibrated, driving with the web dashboard open shows a panel per signal:

| Panel heading (as shown) | Meaning | Green (safe) | Amber | Red |
|---|---|---|---|---|
| "Do the model's 'sub-networks' agree?" | Confidence | ≥ 65% | 25-65% | < 25% |
| "Does this look familiar?" | Novelty (OOD) | ≤ 25% | 25-65% | > 65% |
| "Is the answer stable?" | TTA stability | ≥ 65% | 25-65% | < 25% |

(Novelty's color scale is intentionally flipped from the other two — a *high* number there means "unfamiliar," which is bad, whereas a *high* number for confidence/stability is good.) Each panel has a "What does this mean? ▾" toggle with a plain-English explanation, and the confidence panel is explicit that it's *"[r]elative to this model's own baseline, not a calibrated probability"* — see [§2.3](#23-calibration-mechanics) for what that means concretely.

**If a signal is enabled but not yet calibrated**, its panel simply never appears — instead, on connecting to the dashboard you'll see a dismissible amber banner: *"Not calibrated: the following won't show a percentage until calibrated,"* listing exactly which signal(s) and which model file. Go run calibration (§1.3) and reconnect.

**Pi / performance tuning:** each signal has its own `XAI_*_INTERVAL` (seconds; `0.0` = every frame, no rate-limiting at all). Out of the box these are already relaxed rather than `0.0` — confidence and novelty default to `0.3` (~3 updates/sec), TTA to `0.5` (~2 updates/sec) since it's the priciest signal (it runs `XAI_TTA_SAMPLES` extra forward passes per update). Confidence never costs steering responsiveness either way — the car still steers on a cheap single-pass inference every frame regardless of its interval, only the confidence *number* updates less often. Novelty and TTA are pure observers, so a held frame between updates costs zero forward passes. Raise any of these further if a Raspberry Pi is still struggling to keep up with `DRIVE_LOOP_HZ` — TTA is the first knob to turn. See [§1.5](#15-confidence-aware-throttle-scaling) for how a longer interval trades off against throttle-scaling reaction time.

### 1.5 Confidence-aware throttle scaling

An optional fourth switch that actually *acts* on the three signals above instead of just displaying them:

```python
XAI_THROTTLE_SCALING_ENABLED = True
```

With this on, whichever of confidence/novelty/stability is enabled gets combined: if any signal crosses its "reduced" threshold, throttle is scaled down (linearly, down to `XAI_THROTTLE_MIN_SCALE`, default `0.4` = never below 40% of the commanded throttle from a single signal); if any signal is past its "critical" threshold continuously for `XAI_THROTTLE_STOP_DURATION` seconds (default `1.0`), throttle is forced to zero. It only ever touches throttle — steering is never modified. Turning this on without enabling at least one of the three signals just logs a warning and does nothing (there's nothing to scale on).

**Reaction-time tradeoff with the interval knobs (§1.4):** novelty and TTA are only ever as fresh as their `XAI_*_INTERVAL` — a sudden problem won't be seen until the next scheduled update, up to that many seconds later, and `XAI_THROTTLE_STOP_DURATION` then still needs to see it stay critical continuously before forcing a stop. At the defaults, that's roughly up to `0.3 + 1.0 = 1.3s` for novelty-triggered or `0.5 + 1.0 = 1.5s` for TTA-triggered stops, worst case. That's fine if you're using these signals for dashboard/monitoring, but if you're relying on `XAI_THROTTLE_SCALING_ENABLED` as a real-time safety net, consider lowering `XAI_NOVELTY_INTERVAL`/`XAI_TTA_INTERVAL` back toward `0.0` so the car reacts faster, at the cost of more compute per frame.

### 1.6 Offline analysis — the GUI launcher (recommended)

The easiest way to review a drive is the browser-based launcher, which wraps the CLI tool below so you never need to remember its flags. From your car's directory (`donkey createcar` installs this script for you):

```bash
python run_gradcam_analysis.py
```

This starts a local web server and prints a URL to open. You'll see a **"Run Explainability Analysis"** form:

| Field | What to put |
|---|---|
| Tub path | e.g. `data/mytub` (autocompletes from tubs found under `data/`) |
| Model path | e.g. `models/mypilot.h5` (autocompletes from `models/`) |
| Frames to analyse | **top-K most uncertain** (default, analyse a fixed number of the hardest frames), **percentile ≥** (analyse the worst N% of the drive), or **all frames** (only for short drives) |
| Limit / range of frames considered | optional — restrict to e.g. `500` or `1000-2000` instead of the whole tub |
| Auto-calibrate this model first if it has no calibration yet | checked by default — saves a manual `mc_calibrate` step |
| Calibration tub (training data) | auto-fills from Donkeycar's own training record for that model if it can find one; type over it if you want to force a specific tub |
| Export every camera frame | check this if you want the output folder to be self-contained (e.g. to copy off a remote GPU box) — otherwise only the analysed frames' images are saved |

Click **Run Analysis** and a progress readout tracks the current stage (variance → novelty → Grad-CAM). If auto-calibration had to fall back to using the analysis tub itself (because no training-tub record could be found), you'll see an amber warning and the page will **not** auto-continue — you have to click **Continue to viewer** explicitly, so a possibly-wrong calibration baseline can't slip by unnoticed. Otherwise it jumps to the viewer automatically when done.

**The viewer**, once open, shows the current frame plus a chosen overlay layer:

- *Where the model was unsure (uncertainty)* — the default view
- *Where the model looked (Grad-CAM)*
- *Where the model looked, sharper (Grad-CAM++)*
- *Have I seen this before? (novelty)*
- *Exact pixels that mattered (saliency)*
- *Exact pixels that mattered, cleaner (integrated gradients)*
- *Just the camera*

A short plain-language explanation for whichever layer is selected is always shown underneath the dropdown — no separate "advanced mode" to find it in. Playback controls: Space/arrow keys or the play button at 2/5/10 fps, a scrub bar, and click-to-jump on the confidence graph below the image. The graph plots confidence/novelty/TTA-stability as separate, independently toggleable lines (checkboxes above the graph — whichever signals you actually calibrated/enabled), with the frames that got a full image analysis marked along the top edge. A **"What do confidence / novelty / stability / variance mean?"** link expands a glossary covering all of these terms in plain language, always available with one click.

### 1.7 Offline analysis — CLI (advanced / scripted use)

Equivalent to the launcher's form, useful for automation or a headless box:

```bash
python -m donkeycar.parts.gradcam_uncertainty --tub data/mytub --model models/mypilot.h5 --top-k 50
```

```
usage: gradcam_uncertainty [-h] --tub TUB --model MODEL [--config CONFIG]
                            [--out OUT] [--passes PASSES] [--top-k TOP_K]
                            [--percentile PERCENTILE] [--ig-steps IG_STEPS]
                            [--all] [--limit LIMIT] [--start START]
                            [--export-frames]

  --out OUT             output dir (default <tub>/gradcam_analysis)
  --passes PASSES       MC-Dropout passes (default cfg or 15)
  --top-k TOP_K         analyse the K most uncertain frames (default)
  --percentile PERCENTILE
                        instead analyse frames >= this variance percentile
  --ig-steps IG_STEPS   Integrated Gradients Riemann steps (default cfg
                        XAI_IG_STEPS or 32)
  --all                 analyse every frame (short drives only)
  --limit LIMIT         only consider N records
  --start START         skip this many records before considering any frames
  --export-frames       copy every camera frame into the output dir too
```

Then view the result (with or without the launcher's form):

```bash
python -m donkeycar.parts.uncertainty_viewer --analysis data/mytub/gradcam_analysis
```

### 1.8 Config reference

<details>
<summary>Confidence (MC-Dropout)</summary>

```python
XAI_CONFIDENCE_ENABLED = False           # master on/off
XAI_CONFIDENCE_PASSES = 15               # stochastic forward passes per frame
XAI_CONFIDENCE_ALPHA = 0.2               # EMA smoothing of the variance
XAI_CONFIDENCE_INTERVAL = 0.3            # seconds between updates (0 = every frame, no rate-limiting)
XAI_CONFIDENCE_AUTO_CALIBRATE = False    # calibrate automatically after donkey train
XAI_CONFIDENCE_CALIBRATE_LIMIT = None    # cap frames used for auto-calibration
```

</details>

<details>
<summary>Novelty / OOD detection</summary>

```python
XAI_NOVELTY_ENABLED = False
XAI_NOVELTY_ALPHA = 0.2
XAI_NOVELTY_ENCODER = 'mobilenet_v2'     # 'mobilenet_v2' | 'mobilenet'
XAI_NOVELTY_ENCODER_INPUT = 128          # square input size fed to the encoder
XAI_NOVELTY_ENCODER_ALPHA = 1.0          # encoder width multiplier (e.g. 0.35 on a Pi)
XAI_NOVELTY_INTERVAL = 0.3                # seconds between updates (0 = every frame, no rate-limiting)
```

</details>

<details>
<summary>TTA stability</summary>

```python
XAI_TTA_ENABLED = False
XAI_TTA_SAMPLES = 8       # M, augmented copies per frame
XAI_TTA_ALPHA = 0.2
XAI_TTA_INTERVAL = 0.5    # seconds between updates (0 = every frame, no rate-limiting)
XAI_TTA_STRENGTH = 0.2    # photometric jitter strength (0 = no augmentation)
```

</details>

<details>
<summary>Throttle scaling</summary>

```python
XAI_THROTTLE_SCALING_ENABLED = False
XAI_CONFIDENCE_REDUCED_THRESHOLD = 65.0
XAI_CONFIDENCE_CRITICAL_THRESHOLD = 25.0
XAI_NOVELTY_REDUCED_THRESHOLD = 25.0
XAI_NOVELTY_CRITICAL_THRESHOLD = 65.0
XAI_TTA_REDUCED_THRESHOLD = 65.0
XAI_TTA_CRITICAL_THRESHOLD = 25.0
XAI_THROTTLE_MIN_SCALE = 0.4
XAI_THROTTLE_STOP_DURATION = 1.0   # see the reaction-time note in §1.5
```

</details>

<details>
<summary>Offline analysis / calibration extras</summary>

```python
XAI_IG_STEPS = 32                        # Integrated Gradients Riemann steps
XAI_CALIBRATE_WITH_AUGMENTATIONS = True  # see augmentations.md + §2.8
XAI_CALIBRATE_AUG_PASSES = 2
XAI_CALIBRATE_AUG_MAX_SAMPLES = 1500
```

</details>

### 1.9 Things to watch out for

- **MC-Dropout needs the Keras `.h5` model, not a `.tflite` export** — TFLite conversion bakes dropout out, so there's nothing left to disagree.
- **`linear` model type only**, for the same reason.
- **A calibrated confidence % is not a probability.** It's a percentile rank relative to your own training data's variance distribution — see [§2.3](#23-calibration-mechanics) before treating it as "the model is X% likely to be right."
- **Novelty saturates at the extremes.** Wildly out-of-distribution input (camera covered, solid color, pointed at a wall) reads as ~98%, same as any other very-unfamiliar scene — it tells you *that* something is unfamiliar, not *how* unfamiliar in fine detail.
- Recalibration triggers are listed in §1.3 — it's easy to forget after retraining or changing augmentations, and the dashboard's "uncalibrated" banner is there specifically to catch that. The banner checks each signal's *data*, not just that a calibration file exists, so a calibration written by an older version will still correctly report which signal is missing.
- **`RECORD_DURING_AI`** (§1.2) writes autopilot frames to a separate `<tub>_autopilot` folder automatically — nothing further to configure, and nothing to remember before training.

---

## 2. Technical deep dive

### 2.1 Why explainability and uncertainty matter here

A CNN driving policy is a black box mapping camera pixels to a steering angle; by default it gives you the same confident-looking single number whether the road ahead is a textbook straightaway or something the model has never seen. For a small robot that can hit a wall or a person, being able to ask *"how sure are you"* and *"why did you decide that"* is the difference between a system you can trust incrementally and one you can only evaluate by watching it crash. Everything in this toolkit is one of two things: a way of estimating **uncertainty** (confidence, novelty, TTA stability — "should we trust this frame's answer?"), or a way of producing **visual attribution** (Grad-CAM, Grad-CAM++, Integrated Gradients, saliency — "what part of the image caused this answer?").

### 2.2 Dropout and Monte Carlo Dropout (confidence)

**Dropout**, as used during normal training, randomly zeroes out a fraction of a layer's neurons on every training step (Donkeycar's default `linear` architecture has `Dropout(0.2)` after several convolutional and dense layers — see [`donkeycar/parts/keras.py`](../donkeycar/parts/keras.py)). This forces the network to not rely too heavily on any single neuron (since it might be zeroed out next step), which is a well-known technique for reducing overfitting. Normally, once training finishes, dropout is **switched off** for inference — every neuron participates, and you get one deterministic prediction.

**Monte Carlo Dropout** (Gal & Ghahramani, 2016) is the trick of *leaving dropout switched on at inference time* and running the same input through the network multiple times. Since a different random set of neurons is zeroed out on each pass, each pass is effectively asking a slightly different, randomly-thinned "sub-network" for its opinion. If the input is something the model is confident about, most sub-networks should agree on roughly the same steering angle regardless of which neurons got dropped — the underlying feature is redundantly represented. If the input is ambiguous or unfamiliar, different sub-networks latch onto different (possibly spurious) cues and disagree more. **Confidence in this toolkit is literally the inverse of that disagreement**: run N stochastic passes, measure the variance of the resulting steering angle across them, and treat high variance as low confidence.

Implementation detail worth knowing: running N passes *sequentially* is too slow for a 20 Hz drive loop (roughly 165 ms for 15 passes on a desktop in early testing). Instead, the same image is **replicated N times into one batch** and passed through the model in a single batched call — the network still applies an independent random dropout mask to each element of the batch, so this is mathematically the same N independent stochastic passes, just computed in parallel (roughly 29 ms for the same 15 passes). See [`donkeycar/parts/mc_dropout.py`](../donkeycar/parts/mc_dropout.py).

The raw variance is smoothed frame-to-frame with an exponential moving average (`XAI_CONFIDENCE_ALPHA`) so the displayed number doesn't flicker, then converted into the 0-100% shown on the dashboard via calibration (§2.3).

### 2.3 Calibration mechanics

A raw MC-Dropout variance number (or a raw Mahalanobis distance, for novelty) is meaningless on its own — its scale depends on this specific model's weights, this specific dataset, even the random dropout masks drawn. Calibration ([`mc_calibrate.py`](../donkeycar/parts/mc_calibrate.py)) solves this by replaying a known tub through the model once, collecting the raw variance for every frame, and computing percentiles (e.g. the 50th/80th/97th) of that distribution. A **monotonic piecewise-linear map** is then built from "raw variance value" to "0-100% score," anchored at those percentiles, and saved to `<model>.calib.json`. At drive time, a new raw variance is looked up against this same map to produce the displayed percentage.

The important consequence: **a confidence score is a statement about where this frame's variance falls relative to your own training data's variance distribution — not a calibrated probability of correctness.** This is also why the dashboard explicitly disclaims it as "relative to this model's own baseline." It also means the percentile anchors are inherently **relative**, not absolute: by construction, roughly 20% of any tub replayed against an 80th-percentile-derived threshold — even the original training data itself — will read above that threshold. A "low confidence" reading doesn't necessarily mean something is objectively rare; it means it's rarer than most of what this model saw during calibration.

Novelty and TTA stability are calibrated the same way, with their own percentile anchors, just walking in the opposite direction (ascending for novelty — higher raw distance is *worse* — vs. descending for confidence and TTA stability, where higher is *better*).

### 2.4 Feature-space novelty / out-of-distribution detection

The goal here is different from confidence: instead of "do sub-networks of *this* model agree," it's "does this frame look anything like what the model was trained on at all" — catching genuine out-of-distribution (OOD) input like the car leaving the track entirely.

The underlying statistic is **Mahalanobis distance**: instead of plain Euclidean distance from a "typical" feature vector, it's a distance that accounts for how much each dimension is *expected* to vary — a value 3 standard deviations away from the mean on a dimension that's normally almost constant is treated as far more surprising than the same absolute distance on a dimension that's normally noisy. Concretely: fit a Gaussian to a set of feature vectors extracted from calibration frames (mean + per-dimension variance — a **diagonal covariance** approximation, i.e. we don't model correlations between dimensions, which is far cheaper to compute and invert than a full covariance matrix), then for a new frame's feature vector, compute how many (variance-normalized) standard deviations away it falls.

**The interesting part of this feature is *which* features that distance is measured in — and getting it wrong the first time was a genuine, real finding.** The first implementation measured distance in the driving model's own penultimate layer (`dense_2`, 50-dimensional). This didn't work: a model trained *only* to predict steering angle has every incentive to discard anything not needed for that task — texture, color, semantic content — so grass, carpet, and even the correct dark hallway frame it was never trained on. Measuring against real training data, grass scored as *more familiar* than a legitimate on-track frame in some cases (a "grass is 90%+ confident" result that was genuinely reproduced, not a hypothetical). The fix: extract features from a **generic, frozen ImageNet-pretrained encoder** (MobileNetV2 by default, `XAI_NOVELTY_ENCODER`) instead of the task-specific steering model. A network trained on ImageNet's thousand object categories has to keep general visual information (texture, color, shape) around, so grass and track genuinely land in different regions of *that* feature space — the same math, applied to a feature space that hasn't collapsed away the information needed to tell them apart, correctly separated grass/carpet (scoring as clearly unfamiliar) from track frames.

The live detector ([`donkeycar/parts/novelty.py`](../donkeycar/parts/novelty.py)) runs one extra deterministic forward pass through this encoder per update, computes the Mahalanobis distance against the calibrated Gaussian, smooths it with an EMA, and maps it to 0-100% the same way as confidence (just ascending instead of descending). The offline viewer's per-pixel novelty overlay works slightly differently and more cheaply (no extra encoder pass): it reuses the driving model's own conv features spatially, per grid location — noted as a known limitation in §2.9, since it's still measuring in the (task-collapsed) driving model's feature space rather than the fixed encoder's.

### 2.5 Test-Time Augmentation (TTA) stability

A third, independent question: *if the same real-world scene were lit very slightly differently, would the model give a different answer?* TTA stability answers this directly rather than by proxy: it takes the current frame, generates `XAI_TTA_SAMPLES` copies with small random **photometric-only** jitter (brightness/contrast/gamma/noise — deliberately *not* geometric changes like crop or rotation, since those would actually change what the correct steering angle is), batches them through the model in one deterministic forward pass (dropout off, unlike MC-Dropout), and measures the variance of the resulting steering angle across that batch. High variance means the model's answer is fragile to lighting noise that shouldn't matter; low variance means it's robust to it.

This is a genuinely different signal from MC-Dropout confidence, not a duplicate of it — MC-Dropout asks "do many random internal sub-networks of this one exact image agree," while TTA asks "does the same sub-network agree with itself across slightly different lightings of the same scene." Measuring both against the same real footage showed a correlation of only about 0.19 between their raw variances — confirming they pick up on different kinds of trouble.

### 2.6 Confidence-aware throttle scaling

[`ThrottleScaler`](../donkeycar/parts/mc_dropout.py) takes whichever of the three 0-100% signals are enabled and, for each one independently, computes a *scale factor* between the configured min-scale and `1.0`: full throttle above the signal's "reduced" threshold, linearly interpolated down to `XAI_THROTTLE_MIN_SCALE` between "reduced" and "critical," and held at the minimum below "critical." Confidence and TTA stability are *descending* (high value = good, so the scale shrinks as the value drops); novelty is *ascending* (high value = bad, so the scale shrinks as the value rises) — both use the same underlying interpolation, just mirrored.

The **final throttle scale is the minimum across every enabled signal's scale** — i.e. whichever signal currently looks worst wins, a direct implementation of "slow down if *any* signal indicates trouble," rather than averaging signals together (which could let one bad signal be diluted by two fine ones). If any signal has been continuously in its critical range for `XAI_THROTTLE_STOP_DURATION` seconds, throttle is forced to exactly zero; the timer is shared across signals, so a car that goes from "novelty-critical" straight into "confidence-critical" without recovering in between still counts as one continuous critical episode, not two resets. Steering is never touched by this part.

### 2.7 Visual attribution: four ways to ask "why"

The offline viewer computes four different attribution maps per analysed frame, because they answer subtly different questions and none of them is uniquely "correct":

**Grad-CAM** — computes the gradient of the (summed) steering output with respect to the activations of a late convolutional layer (`conv2d_5`), uses those gradients to weight each channel of that layer's feature map, sums the weighted channels, and applies a ReLU (keeping only the parts that positively influenced the output). Because it operates on a late-stage conv layer, the result is naturally coarse — a small grid (e.g. 8×13) upsampled back to image size — showing *roughly where* in the image mattered, not exact pixels. In this toolkit, Grad-CAM is combined with MC-Dropout: it's computed once per dropout sub-network in the same batch, and the *mean* map across sub-networks becomes the "attention" overlay while the *pixel-wise variance* across those same maps becomes the "uncertainty" overlay — literally, "where did the model look" and "where did different sub-networks disagree about where to look."

**Grad-CAM++** — a refinement of Grad-CAM's channel weighting that accounts for a feature appearing in multiple locations at once (plain Grad-CAM's simple gradient-averaging weight can under-represent that case). The textbook derivation assumes a softmax classification output; since this is a regression output (a single steering value), this toolkit uses a practical approximation of the pixel-wise weighting derived from first- and second-order gradient terms (`alpha = g² / (2g² + (ΣA)g³)`) rather than the classification-specific formula. In practice it tends to produce sharper, more spatially precise blobs than plain Grad-CAM.

**Integrated Gradients (IG)** — instead of looking at gradients at the actual input alone (which can saturate — a very confidently-classified pixel can have a near-zero local gradient even though it clearly mattered), IG integrates the gradient along a straight-line path from a neutral **baseline** (a black image) to the real input, in `XAI_IG_STEPS` discrete steps (a Riemann-sum approximation of the integral), and multiplies the averaged path gradient by `(input - baseline)`. This satisfies a *completeness* property that plain gradients don't: the attributions sum up to exactly the difference between the model's output on the real image and on the baseline. Because it operates directly on input pixels rather than a coarse conv layer, its maps are full resolution.

**Vanilla gradient saliency** — the simplest and cheapest: one backward pass, the gradient of the output directly with respect to every input pixel. (One implementation wrinkle: the model's final activation is temporarily linearized, since a saturating output activation would otherwise clip gradients right where they matter most.) Like IG, it's full-resolution, but with no baseline/integration step, so it's noisier and more speckled.

**Why look at all four:** the two conv-layer-based methods (Grad-CAM, Grad-CAM++) give smooth, coarse, easy-to-read localization; the two pixel-gradient methods (saliency, IG) give precise-but-noisier, exact per-pixel attribution. On a real test frame containing a pedestrian, Grad-CAM/++ showed two smooth blobs on the ground near the pedestrian, while saliency showed many small, scattered points across the pedestrian, nearby pillars, *and* background architecture — a genuinely different, complementary answer, not a redundant one. Relying on only one technique risks mistaking "coarse but clean" for "the whole picture," or "precise but noisy" for "definitely irrelevant."

### 2.8 How augmentations feed into calibration

This is the connection point with [augmentations.md](augmentations.md). Calibration's job is to define "what does normal training data look like" — which becomes the novelty detector's baseline. If a model is trained with, say, `AUGMENTATIONS = ['SHADOW', 'GAMMA']` (so it has learned to drive correctly through shadows and dim lighting), but calibration is only ever shown **clean, un-augmented** frames, the novelty baseline never learns that shadows/dim lighting are "normal" — so the very conditions the model was made robust to would get flagged as unfamiliar and throttle down live, exactly backwards from the intent.

This was measured directly, not just reasoned about: fitting the novelty baseline on clean frames only, then scoring augmented-but-otherwise-normal frames against it, **79% crossed the "reduced" novelty threshold (vs. 38% for clean frames) and 50% crossed "critical" (vs. 10%)**. A model trained to handle shadows would have been throttled down by its own safety feature the moment it encountered one.

**The fix**, in [`mc_calibrate.py`](../donkeycar/parts/mc_calibrate.py): when `AUGMENTATIONS` is non-empty and `XAI_CALIBRATE_WITH_AUGMENTATIONS` is on (the default), calibration pushes a strided subset of frames (capped by `XAI_CALIBRATE_AUG_MAX_SAMPLES`) through the *same* `ImageAugmentation` pipeline used in training, `XAI_CALIBRATE_AUG_PASSES` times each (re-randomized every pass), and pools those augmented features into the novelty baseline alongside the clean ones. The resulting `calib.json` carries a small `"augmentation"` provenance block recording whether this happened and with what settings, so it's inspectable after the fact.

**This is scoped to novelty only** — not confidence or TTA. Both of those carry an exponential moving average over a *time-ordered* sequence of frames; splicing randomly-augmented frames into that sequence would corrupt the smoothing (a sudden synthetic shadow appearing between two real consecutive frames isn't something either signal is designed to interpret). Novelty's calibration, by contrast, is a plain unordered percentile fit over a feature distribution, so mixing in augmented samples is safe and (per the measurement above) necessary.

### 2.9 Known limitations

- **The offline viewer's per-location novelty overlay is still task-collapsed.** Unlike the live novelty detector (fixed in §2.4 to use a generic ImageNet encoder), the offline spatial map reuses the driving model's own convolutional features for per-pixel positioning, since the generic encoder doesn't have a matching spatial grid to overlay against. It still shows *something* useful, but inherits the original task-collapsed weakness at a per-location level. Lower priority than the live fix since it's a secondary visualization, not the number that drives throttle scaling.
- **`TRANSFORMATIONS` (e.g. `CROP`) are not currently replayed during calibration**, even though they always apply at both training and inference (see [augmentations.md §2.4](augmentations.md#24-known-limitations)). If your config sets `TRANSFORMATIONS`, calibration baselines may be measured on frames shaped differently from what the live model actually receives.
- **MC-Dropout's calibration percentiles are unseeded.** Calibrating the same model against the same tub twice will produce slightly different p50/p80/p97 anchors each time (a real, measured difference, not a bug) — this is expected statistical noise from the random dropout masks, so don't expect bit-identical thresholds across reruns.
- **A generic ImageNet encoder for novelty runs on CPU today.** A Hailo-NPU-accelerated backend is a planned follow-up (to match the rest of the pipeline running on-device on hardware that has one), but isn't implemented in this toolkit yet — `CPUEncoderExtractor` ([`donkeycar/parts/ood.py`](../donkeycar/parts/ood.py)) is currently the only backend.
- **Novelty scores saturate rather than scale gracefully at the extreme end** (§1.9) — useful for a binary "is this track or not" read, less useful for graded ranking between different kinds of out-of-distribution input.
