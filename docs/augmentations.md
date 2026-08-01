# Augmentations Expansion

## Overview

Donkeycar already supported a couple of training-time image augmentations (`BRIGHTNESS`, `BLUR`). This fork expands that list with six more, aimed specifically at the lighting conditions a small outdoor/indoor track actually produces, plus two aimed at reducing reliance on color/brightness entirely:

| Augmentation | Simulates |
|---|---|
| `GAMMA`     | Non-linear glare (too bright) or dusk/night (too dark) |
| `NOISE`     | Sensor grain from a cheap camera in low light |
| `SHADOW`    | Partial shadows (trees, buildings) crossing the track |
| `SUNLIGHT`  | A patch of harsh direct sun next to shaded track, as seen around midday |
| `GRAYSCALE` | Drops color/hue for some training images, so the model can't lean on color that shifts across lighting |
| `HIGHPASS`  | Suppresses flat color/brightness, emphasizing edges and structure (lane lines, track boundaries) instead |

All of them are **training-only** — none of this touches what the camera feeds the model while it's actually driving. The idea is simple: if your training data was all recorded on one overcast afternoon, but you plan to race at noon or at dusk, augmentation lets the model *see* those lighting conditions during training without you having to physically go and re-record hours of data in every possible weather condition.

This document covers the augmentations feature end-to-end: Section 1 is a usage walkthrough (what to type into `myconfig.py`, how to preview the effect before committing to a training run, nothing else). Section 2 explains, technically, what each augmentation is actually doing to the pixels and why. If you're also using this fork's [Explainable AI toolkit](xai.md), read that document's technical section on calibration — it explains why enabling augmentations means you should recalibrate.

---

## 1. Usage

### 1.1 Where this lives

Everything in this document is configured in one file: your car's `myconfig.py` (the file `donkey createcar` generates for you, sitting next to `manage.py`). You never need to touch any Python code to use augmentations — just uncomment/add a few lines. The defaults and full documentation for every option live in [`donkeycar/templates/cfg_complete.py`](../donkeycar/templates/cfg_complete.py) under the `AUGMENTATIONS` and `TRANSFORMATIONS` headings — `myconfig.py` only needs to *override* the handful of values you actually want to change.

### 1.2 Two different lists — don't mix them up

Donkeycar's config has two separate settings that both sound like "change the image," and it matters which one you use:

- **`AUGMENTATIONS`** — applied **only during training**, and only *randomly* (each one has its own probability). This is what this document is about. The live camera feed while driving is never touched.
- **`TRANSFORMATIONS`** — applied **both during training and while actually driving**, always, in the order listed (e.g. `CROP` to mask out the top of the image, `RESIZE`, colour-space conversion). These already existed in Donkeycar and this fork didn't change them — they're mentioned here only so you don't confuse the two.

If you want the model to become *robust to* a lighting condition, use `AUGMENTATIONS`. If you want to permanently *change what the model sees* (crop out the sky, resize, convert colour space), that's `TRANSFORMATIONS` and is unrelated to this feature.

### 1.3 Turning on augmentations

In `myconfig.py`:

```python
AUGMENTATIONS = ['GAMMA', 'NOISE', 'SHADOW']
```

That's it — nothing else is required to enable them. Each entry in the list is applied, in the order listed, with its own independent probability (so most training images get zero or one effect, not all of them stacked). The order matters a little: local lighting effects (`SUNLIGHT`, `SHADOW`) are meant to run on clean pixels first, then global tone changes (`GAMMA`, `BRIGHTNESS`), then `NOISE`, then `BLUR` last (to simulate optical softening of the final image) — the ready-made profiles below already follow this order.

### 1.4 The eight available augmentations

| Add to `AUGMENTATIONS` | Key config (in `myconfig.py`) | Default | What it does |
|---|---|---|---|
| `'BRIGHTNESS'` | `AUG_BRIGHTNESS_RANGE` | `0.2` | Randomly brightens/darkens the whole image by up to ±20%. `AUG_CONTRAST_RANGE` (optional) varies contrast independently — if unset it just reuses the brightness range. |
| `'BLUR'` | `AUG_BLUR_RANGE` | `(0, 3)` | Gaussian blur, kernel size drawn from this range. |
| `'GAMMA'` | `AUG_GAMMA_RANGE`, `AUG_GAMMA_PROBABILITY` | `(80, 120)`, `0.5` | Non-linear brightness curve. Below 100 = brighter/glare, above 100 = darker/dusk. A single range covers both directions. |
| `'NOISE'` | `AUG_NOISE_STD_RANGE`, `AUG_NOISE_MEAN_RANGE`, `AUG_NOISE_PROBABILITY` | `(0.05, 0.15)`, `(0.0, 0.0)`, `0.3` | Adds Gaussian sensor grain, most visible in low light. |
| `'SHADOW'` | `AUG_SHADOW_PROBABILITY`, `AUG_SHADOW_DARKNESS_RANGE`, `AUG_SHADOW_COUNT_RANGE`, `AUG_SHADOW_DIMENSION`, `AUG_SHADOW_ROI`, `AUG_SHADOW_BLUR_KSIZE` | `0.3`, `(0.4, 0.7)`, `(1, 2)`, `5`, `(0.0, 0.3, 1.0, 1.0)`, `21` | Draws 1-2 randomly shaped partial shadows onto the lower part of the image. |
| `'SUNLIGHT'` | `AUG_SUNLIGHT_PROBABILITY`, `AUG_SUNLIGHT_STRENGTH_RANGE`, `AUG_SUNLIGHT_COVERAGE_RANGE`, `AUG_SUNLIGHT_REGION_COUNT_RANGE`, `AUG_SUNLIGHT_BLUR_KERNEL_RANGE`, `AUG_SUNLIGHT_ROAD_REGION_START` | `0.3`, `(1.15, 1.8)`, `(0.15, 0.55)`, `(1, 2)`, `(11, 41)`, `0.25` | Brightens one or more localized regions (unlike `BRIGHTNESS`, which changes the *whole* image at once). |
| `'GRAYSCALE'` | `AUG_GRAYSCALE_PROBABILITY`, `AUG_GRAYSCALE_METHOD` | `0.5`, `'weighted_average'` | Converts an image to grayscale, then replicates it back to 3 channels (input shape stays the same). Training-only — see 1.9 for the permanent/live-driving equivalent. |
| `'HIGHPASS'` | `AUG_HIGHPASS_PROBABILITY`, `AUG_HIGHPASS_BLUR_SIGMA_RANGE`, `AUG_HIGHPASS_STRENGTH_RANGE`, `AUG_HIGHPASS_BLEND_RANGE` | `0.5`, `(3.0, 8.0)`, `(0.7, 1.3)`, `(0.0, 0.15)` | Subtracts a blurred copy of the image from itself, so flat color/brightness collapses toward gray while edges and texture stand out. |

You only need to override a config value if you want something different from the default — most people can get away with just setting `AUGMENTATIONS` and leaving everything else alone.

### 1.5 Ready-made profiles

Rather than guessing values, copy one of these straight into `myconfig.py` as a starting point (all from [`cfg_complete.py`](../donkeycar/templates/cfg_complete.py)):

```python
# Mostly indoor, controlled lighting
AUGMENTATIONS = ['BRIGHTNESS', 'GAMMA']

# Outdoor track, variable natural light
AUGMENTATIONS = ['BRIGHTNESS', 'SHADOW', 'BLUR']

# Evening / low-light driving
AUGMENTATIONS = ['GAMMA', 'NOISE', 'SHADOW']

# Midday, sharp sun/shade boundaries on the track
AUGMENTATIONS = ['SUNLIGHT', 'SHADOW', 'GAMMA', 'BRIGHTNESS', 'NOISE', 'BLUR']

# One model trained on combined day/midday/night data — a good default
# starting point if you're not sure which conditions you'll race in
AUGMENTATIONS = ['BRIGHTNESS', 'BLUR', 'SHADOW', 'GAMMA', 'NOISE']
AUG_GAMMA_RANGE = (60, 160)  # widen to cover stronger night/glare

# Leans further into color-invariance than all_conditions: on top of the
# lighting-change augmentations, GRAYSCALE and HIGHPASS regularly strip
# color and/or flat brightness so the model is pushed toward relying on
# contrast/edges/structure rather than color or absolute brightness, both
# of which vary a lot across times of day. Try this if all_conditions
# still seems to rely too much on color-specific cues.
AUGMENTATIONS = ['GRAYSCALE', 'HIGHPASS', 'GAMMA', 'NOISE', 'SHADOW']
AUG_GAMMA_RANGE = (60, 160)
AUG_GRAYSCALE_PROBABILITY = 0.5
AUG_HIGHPASS_PROBABILITY = 0.5
```

### 1.6 Previewing augmentations before you train

Training runs take time, so don't tune these blind. `tools/preview_augmentations.py` loads a few real images from one of your tubs, applies each augmentation (forced to 100% probability so the effect is always visible), and saves a side-by-side comparison grid as a PNG — no training run needed, nothing in your tub is modified.

```bash
python tools/preview_augmentations.py --data data/mytub --output augmentation_previews --num-samples 6
```

- `--data` — a tub folder (containing an `images/` subfolder), or a plain folder of `.jpg`/`.png` files.
- `--output` — where the preview PNGs are written (default `augmentation_previews`).
- `--num-samples` — how many source images to preview (default 6).
- `--seed` — change this to preview a different random sample / different random augmentation draw; reuse the same seed to reproduce a preview exactly.
- `--clean-output` — deletes old `preview_*.png` files in `--output` first, so the folder only ever shows the latest run (refuses to run if `--output` would point at your tub, so it can never delete real data).

Each generated image shows eight panels: Original, Brightness, Gamma, Shadow, Noise, Grayscale, High Pass, and an "All Conditions" combined view (which itself still only stacks the original five — see the `contrast_focus` profile above for a version that also includes Grayscale/High Pass) — useful for judging whether stacking every effect at once still leaves the lane markings visible, before you spend an hour training on it.

### 1.7 Training

No separate step is needed — `donkey train` (or your training script) automatically picks up whatever `AUGMENTATIONS` is set to and applies it during training only, via the existing training pipeline. Leave `AUGMENTATIONS = []` (the default) to train with no augmentation at all, exactly as stock Donkeycar behaves.

### 1.8 Config reference

<details>
<summary>Full list of AUG_* config keys and defaults</summary>

```python
AUGMENTATIONS = []                          # e.g. ['BRIGHTNESS', 'GAMMA']

AUG_BRIGHTNESS_RANGE = 0.2
# AUG_CONTRAST_RANGE = 0.2                  # optional, defaults to brightness range
AUG_BLUR_RANGE = (0, 3)

AUG_GAMMA_RANGE = (80, 120)
AUG_GAMMA_PROBABILITY = 0.5

AUG_NOISE_PROBABILITY = 0.3
AUG_NOISE_STD_RANGE = (0.05, 0.15)
AUG_NOISE_MEAN_RANGE = (0.0, 0.0)

AUG_SHADOW_PROBABILITY = 0.3
AUG_SHADOW_DARKNESS_RANGE = (0.4, 0.7)
AUG_SHADOW_COUNT_RANGE = (1, 2)
AUG_SHADOW_DIMENSION = 5
AUG_SHADOW_ROI = (0.0, 0.3, 1.0, 1.0)       # (x_min, y_min, x_max, y_max) as fractions
AUG_SHADOW_BLUR_KSIZE = 21

AUG_SUNLIGHT_PROBABILITY = 0.3
AUG_SUNLIGHT_STRENGTH_RANGE = (1.15, 1.8)
AUG_SUNLIGHT_COVERAGE_RANGE = (0.15, 0.55)
AUG_SUNLIGHT_REGION_COUNT_RANGE = (1, 2)
AUG_SUNLIGHT_BLUR_KERNEL_RANGE = (11, 41)
AUG_SUNLIGHT_ROAD_REGION_START = 0.25

AUG_GRAYSCALE_PROBABILITY = 0.5
AUG_GRAYSCALE_METHOD = 'weighted_average'   # or 'from_lab'/'desaturation'/'average'/'max'/'pca'

AUG_HIGHPASS_PROBABILITY = 0.5
AUG_HIGHPASS_BLUR_SIGMA_RANGE = (3.0, 8.0)
AUG_HIGHPASS_STRENGTH_RANGE = (0.7, 1.3)
AUG_HIGHPASS_BLEND_RANGE = (0.0, 0.15)       # 0 = pure high-pass ("edges on gray")
```

</details>

### 1.9 Things to watch out for

- **Augmentations never affect live driving.** If your car behaves differently on-track than in the preview tool, augmentation isn't the cause — check `TRANSFORMATIONS` instead.
- **Probabilities are per-image, per-effect**, so `AUGMENTATIONS = ['SHADOW', 'GAMMA']` does not mean every image gets both; most images get zero or one.
- **If you also use this fork's XAI toolkit** (confidence / novelty / stability signals — see [xai.md](xai.md)), changing `AUGMENTATIONS` after a model is already calibrated means you should recalibrate. The short version: the novelty/OOD detector's "what does normal look like" baseline is built from your training data, and by default it now automatically includes augmented frames too — but only if you recalibrate after changing the config. Full explanation in [xai.md](xai.md)'s technical section.
- **Overdoing it hurts training**, not just realism — very dark shadows, very high noise, or stacking every effect on every image can make the lane genuinely hard to see even for a human. Use the preview tool (1.6) before committing to a full run.
- **`GRAYSCALE`/`HIGHPASS` are training-only, same as every other augmentation here** — the live camera feed stays full color. If you want the car to *always* drive on grayscale or edge-only input (a stronger, permanent version of the same idea), that's a `TRANSFORMATIONS` setting instead (e.g. `['RGB2GRAY', 'GRAY2RGB']` for grayscale, or the existing `CANNY` transformation for a more aggressive always-on edge-only option) — see [xai.md](xai.md) for how transformations are configured and verified.

---

## 2. Technical deep dive

### 2.1 What is "data augmentation," and why bother?

A neural network only ever learns what it's shown. If every training image was recorded on one cloudy afternoon, the model has no evidence that "the lane markings are still the lane markings" when a cloud moves and the whole scene suddenly gets brighter, or when a tree's shadow falls across the track. Without ever having seen that variation, the model has no reason to have learned to ignore it — it might latch onto brightness or a particular contrast pattern as if it were a meaningful feature, because in the training set it happened to be constant.

Data augmentation manufactures that variation synthetically: take a real, correctly-labelled image (the steering angle label is still correct — we never move the lane, just change the lighting) and perturb its pixels in a way that mimics a real-world condition you didn't happen to record. The network then sees many differently-lit versions of the same driving scene during training and is forced to find features that are invariant to lighting — which is usually much closer to what actually indicates "where's the lane" (edges, geometry, relative position) than to brightness itself.

Everything below is implemented with [Albumentations](https://albumentations.ai/), an image augmentation library, orchestrated by [`donkeycar/pipeline/augmentations.py`](../donkeycar/pipeline/augmentations.py)'s `ImageAugmentation` class, which reads your `AUGMENTATIONS` list and builds an `albumentations.Compose` pipeline from it — that pipeline is run once per training image, each epoch, so the *same* source image is augmented differently on each pass through the dataset.

### 2.2 Each augmentation, explained

**`BRIGHTNESS`** (`RandomBrightnessContrast`) — the simplest case: scales every pixel's value up or down by a random factor within `AUG_BRIGHTNESS_RANGE`, and independently varies contrast within `AUG_CONTRAST_RANGE`. This is a *global*, *linear* change — every pixel shifts by the same amount regardless of its original value.

**`GAMMA`** (`RandomGamma`) — also global, but *non-linear*. A camera sensor's response to light is not linear in the way a `BRIGHTNESS` shift assumes; a gamma curve raises each (normalized) pixel value to a power: `output = input ^ (100 / gamma_percent)`. A `gamma_limit` below 100 raises pixel values overall (brightens, closer to what overexposure/glare looks like), and above 100 darkens (closer to dusk/night), because the curve compresses shadows or highlights non-uniformly rather than shifting everything by a flat offset. That's why the docs describe it as "simulates glare / low light" rather than just "another brightness control" — it changes the *shape* of the brightness response, not just its average.

**`NOISE`** (`GaussNoise`) — adds independent Gaussian-distributed random noise to every pixel, with standard deviation drawn from `AUG_NOISE_STD_RANGE` (expressed as a fraction of the maximum pixel value). This approximates *sensor/shot noise*: real camera sensors get noisier in low light because the same photon-counting error becomes a larger fraction of a dimmer signal, and cheap camera modules exhibit this more than expensive ones.

**`BLUR`** (`GaussianBlur`) — convolves the image with a Gaussian blurring kernel (size from `AUG_BLUR_RANGE`), simulating a slightly out-of-focus lens or motion blur.

**`SHADOW`** (`RandomShadow`, a custom transform added in this fork, [`augmentations.py:18`](../donkeycar/pipeline/augmentations.py)) — this one is worth walking through because of a deliberate implementation choice: it converts the image to the **HLS** color space (Hue, Lightness, Saturation) instead of directly darkening RGB pixels. It then generates 1-2 random, irregular polygons (`AUG_SHADOW_DIMENSION` vertices each, inside the region defined by `AUG_SHADOW_ROI`), blurs their edges (`AUG_SHADOW_BLUR_KSIZE`) so they fade in rather than looking like a cardboard cutout, and multiplies *only the Lightness channel* inside those polygons by `(1 - darkness)`. Hue and Saturation are left completely alone. Why this matters: if you instead just multiplied RGB values down, you'd also shift the apparent *color* of whatever's in shadow, which is not what a real shadow does — real shadows reduce brightness while roughly preserving hue. Since the lane markings' color is one of the main things the model can use to find the road, preserving hue under a fake shadow keeps the augmentation physically honest.

**`SUNLIGHT`** (`RandomLocalSunlight`, also new in this fork, [`augmentations.py:90`](../donkeycar/pipeline/augmentations.py)) — the mirror image of `SHADOW`: same HLS-lightness-only trick, but *boosting* lightness inside 1-2 randomly generated regions instead of reducing it. The key difference from a plain `BRIGHTNESS` augmentation is that `BRIGHTNESS` changes the *entire* image by one factor, while `SUNLIGHT` is local — it can leave half the frame untouched and only brighten a patch, an ellipse, or a directional gradient band (three region shapes are randomly chosen: `polygon`, `ellipse`, `gradient`), which is what real midday sun/shade boundaries on a track actually look like. The region is biased toward the lower part of the frame (`AUG_SUNLIGHT_ROAD_REGION_START`) since that's where the road is, and its edge is blurred by a random kernel size so the sun/shade transition ranges from a soft glow to a near-hard boundary, matching how real transitions vary.

**`GRAYSCALE`** (`ToGray`, from Albumentations) — converts a training image to grayscale using `AUG_GRAYSCALE_METHOD` (default `'weighted_average'`, i.e. `0.299R + 0.587G + 0.114B` — a perceptual approximation of how the human eye weights each channel), then replicates that single channel back out to 3 channels so the model's input shape never changes. The idea is straightforward: hue and saturation are exactly the pixel values that shift the most across different times of day (white balance, sodium vs. LED lighting, direct sun vs. shade), so an image that only ever sees color-consistent training data can end up quietly using color as a shortcut for "where's the lane" — a shortcut that breaks the moment the lighting looks different. Training on some grayscale samples removes that shortcut outright, pushing the model toward luminance and contrast instead.

**`HIGHPASS`** (`RandomHighPass`, a custom transform added in this fork, [`augmentations.py`](../donkeycar/pipeline/augmentations.py)) — a genuine high-pass filter: blur the image (Gaussian, sigma from `AUG_HIGHPASS_BLUR_SIGMA_RANGE`) to get a "low-frequency" version representing smooth color and broad brightness, then subtract that from the original and re-center the result at mid-gray. What's left is dominated by edges and texture — lane lines, track boundaries — since those are the *high*-frequency content a blur removes and a subtraction reveals. At `AUG_HIGHPASS_BLEND_RANGE` near `0`, the result looks like edges floating on a flat gray background, since smooth regions collapse toward the same mid-gray value regardless of whether they started out bright, dark, or brightly colored; a higher blend value keeps this a milder, sharpening-like effect that still shows some of the original color/brightness. This is a different, complementary strategy to `GRAYSCALE` — grayscale removes color while keeping the original brightness/shading intact, high-pass removes *both* color and flat brightness, keeping only structure.

### 2.3 Augmentations vs. Transformations, technically

The distinction in section 1.2 is intentional and inherited from stock Donkeycar, not something this fork changed — but it's worth understanding *why* it exists. `AUGMENTATIONS` exist purely to widen the training distribution; applying them at inference would make no sense (there's no "correct steering label" being generated live, and you'd just be degrading the live camera image for no benefit). `TRANSFORMATIONS` (`CROP`, `RESIZE`, colour-space conversion, etc.) instead change what *shape*/*representation* of image the model expects as input, which must be identical between training and inference or the trained weights are looking at a different kind of input than they were trained on. Augmentations are randomized per-call; transformations are deterministic and always applied, in the configured order.

### 2.4 Known limitations

- **`TRANSFORMATIONS` are handled by the XAI toolkit, but they change what its numbers mean.** Calibration and offline analysis now replay tub frames through your configured `TRANSFORMATIONS` before feeding the model, so baselines match what the model actually sees while driving (see [xai.md §2.9](xai.md#29-how-transformations-affect-the-signals)). The consequence is that a calibration is only valid for the transformation config it was built with — change `TRANSFORMATIONS` and you must recalibrate. The dashboard warns you if it detects this.
- **`SHADOW`/`SUNLIGHT` are skipped (no-op) on non-RGB images** (e.g. if you set `IMAGE_DEPTH = 1` for grayscale) since the HLS conversion they rely on needs 3 channels.
- The five newer augmentations were tuned and previewed against one physical track's camera and lighting; the numeric defaults (darkness ranges, noise levels, etc.) are reasonable starting points, not universal constants — always preview (1.6) against your own footage before trusting a profile.
