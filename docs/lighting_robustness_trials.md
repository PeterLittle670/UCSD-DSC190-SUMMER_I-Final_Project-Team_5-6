# Lighting-Robustness Trials

Data was recorded on one afternoon/evening session: darker light, no hard
shadows, smooth even lighting. The goal is a model that still drives when
raced at a different time of day. These trials test augmentation profiles
and an illumination-invariant input representation (`LANE_ISOLATE`) against
each other.

Full background: [augmentations.md](augmentations.md) for what each
augmentation does technically, [`donkeycar/parts/cv.py`](../donkeycar/parts/cv.py)
`ImgLaneIsolate` for the transform's implementation.

## Fixed for every trial

Set these regardless of which row of the table you're running, so the only
thing that varies between trials is the augmentation/transformation choice:

```python
IMAGE_W = 192
IMAGE_H = 108
POST_TRANSFORMATIONS = ['CROP']   # + 'LANE_ISOLATE' for trials 5-6, see below
ROI_CROP_TOP = 55
ROI_CROP_BOTTOM = 0
ROI_CROP_LEFT = 0
ROI_CROP_RIGHT = 0
```

`IMAGE_W`/`IMAGE_H` must match your camera's actual resolution (192x108 for
the OAK-D rig this was tuned on) — a mismatch gets silently resized by
`donkeycar/utils.py`, distorting the aspect ratio the model trains on.

`ROI_CROP_TOP = 55` masks the top 51% of the frame (buildings, windows,
sky — all change completely between morning and evening). It's deeper than
a plain-augmentation setup would need on its own, because `LANE_ISOLATE`
(trials 5-6) amplifies thin bright background structures — wall rails,
window frames — almost as strongly as lane tape, so it needs more headroom
above the horizon. Kept at 55 for every trial so crop depth isn't a
confound.

If your rig is the OAK-D camera, also set (see "OAK-D colour order" below):

```python
LANE_ISOLATE_COLOR_ORDER = 'bgr'
LANE_ISOLATE_CHROMA_ANGLE = 90.0
LANE_ISOLATE_CHROMA_SCALE = 5.0
```

## The trial table

| # | Name | `AUGMENTATIONS` | `POST_TRANSFORMATIONS` | Question it answers |
|---|---|---|---|---|
| 0 | `baseline` | `[]` | `['CROP']` | Reference — no augmentation, no illumination-invariant representation |
| 1 | `global_tone` | `['BRIGHTNESS', 'GAMMA']` | `['CROP']` | Does cheap, global photometric augmentation alone close the gap? |
| 2 | `local_light` | `['SUNLIGHT', 'SHADOW']` | `['CROP']` | Does simulating local sun/shade boundaries — the thing our data has none of — matter on its own? |
| 3 | `full_photometric` | `['SUNLIGHT', 'SHADOW', 'BRIGHTNESS', 'GAMMA', 'NOISE']` | `['CROP']` | Expected best of the augmentation-only arm |
| 4 | `all_six` | `['SUNLIGHT', 'SHADOW', 'BRIGHTNESS', 'GAMMA', 'NOISE', 'BLUR']` | `['CROP']` | Upper bound / checks whether stacking everything over-augments |
| 5 | `lane_isolate` | `[]` | `['CROP', 'LANE_ISOLATE']` | Can an illumination-invariant representation replace augmentation entirely? |
| 6 | `lane_isolate_aug` | `['SHADOW', 'GAMMA']` | `['CROP', 'LANE_ISOLATE']` | Are representation and augmentation complementary, or redundant? |

**Ordering within `AUGMENTATIONS` matters** — list local lighting effects
first (`SUNLIGHT`, `SHADOW`), then global tone (`BRIGHTNESS`, `GAMMA`), then
`NOISE`, then `BLUR` last. The table above already follows this order;
match it if you add or remove anything.

The comparison that actually decides the approach is **3 vs 5 vs 6**:
simulate the lighting variation, remove it via the transform, or both.
Runs 0–2 exist so that if 3 wins, you know *which part* of it did the work.

### Reading the results

- Trials 5 and 6 may score **worse** on afternoon validation loss than 3,
  and still be the better choice. `LANE_ISOLATE` throws away information
  the model could legitimately use within one lighting condition, so it's
  expected to fit the training distribution slightly worse. Judge 5 and 6
  on a different-time-of-day tub, not on validation loss against the
  afternoon data.
- If 6 scores about the same as 5, that's an informative result (the
  transform already normalizes away what the augmentations would have
  simulated), not a failed trial.
- **A same-lighting validation split cannot rank these models.** All seven
  will fit the afternoon data about equally well; the entire point is
  out-of-distribution behavior. You need at least an unlabeled tub — ideally
  a short RC-driven one — recorded at a different time of day before any of
  these numbers mean something.

## Recommended parameter ranges

Only listed where a default is worth widening for this experiment. Anything
not listed here can be left at the `cfg_complete.py` default.

### Augmentations

| Setting | Default | Recommended for trials | Why |
|---|---|---|---|
| `AUG_GAMMA_RANGE` | `(80, 120)` | `(60, 160)` | Default doesn't reach midday glare or full dusk. Use for trials 1, 3, 4, 6. |
| `AUG_BRIGHTNESS_RANGE` | `0.2` | `0.2`–`0.35` | Wider if trial 3/4 still underperforms after widening gamma. |
| `AUG_SHADOW_DARKNESS_RANGE` | `(0.4, 0.7)` | leave default | Already reasonable; preview before changing. |
| `AUG_SUNLIGHT_STRENGTH_RANGE` | `(1.15, 1.8)` | leave default | Same. |
| `AUG_NOISE_STD_RANGE` | `(0.005, 0.03)` | leave default | Low-light sensor grain; fine as-is. |

Preview any change with `tools/preview_augmentations.py` before committing
to a full training run — it's much cheaper than discovering the shadows are
too dark after an hour of training.

### `LANE_ISOLATE`

| Setting | Default | Notes |
|---|---|---|
| `LANE_ISOLATE_KERNEL` | `9` | Width filter in pixels — structures thinner than this are treated as markings. Scale with `IMAGE_W`; **not yet swept**, worth tuning if trial 5/6 pavement texture is noisy (see below). |
| `LANE_ISOLATE_GAIN` | `3.0` | Border-tape channel strength. |
| `LANE_ISOLATE_CHROMA_SCALE` | `5.0` | Centre-line channel strength. Tuned from a clipping sweep: `8.0` (an earlier default) clipped 35% of tape pixels; `5.0` keeps ~44x tape/road separation with only ~5% clipping. Don't raise without re-checking clipping. |
| `LANE_ISOLATE_LC_SIGMA` | `7.0` | Blur radius for the local-contrast channel. |
| `LANE_ISOLATE_BG_SIGMA` | `3.0` | Blur radius for the tape channel's background estimate. |
| `LANE_ISOLATE_COLOR_ORDER` | `'rgb'` | **Must be set correctly or the centre-line channel silently goes blank or floods with noise — no error is raised.** See below. |
| `LANE_ISOLATE_CHROMA_ANGLE` | `90.0` | Hue of the centre markings in the CIELAB a*/b* plane, in degrees: `90` = yellow, `-90` = blue, `180` = green, `0` = magenta/red, `-135` = cyan. Measure your own tape's hue if it differs. |

**Known limitation, untuned:** pavement texture (gravel grain) partially
passes the width filter and adds noise to the tape channel. Increasing
`LANE_ISOLATE_KERNEL` should suppress it at the cost of dropping far-distance
markings — this tradeoff hasn't been swept yet.

**Known limitation, no fix available:** if the raw frame clips to white
(direct sun, harsh synthetic "midday" test), all `LANE_ISOLATE` channels
collapse along with it. No transform recovers genuinely clipped pixels —
this is a camera-exposure problem, not a software one.

## OAK-D colour order

If your rig uses the **OAK-D** camera: its `getCvFrame()` call returns BGR
(OpenCV convention), and nothing in the camera or tub-writing code converts
it before it's saved — so recorded JPEGs hold BGR pixel data inside an
RGB-labelled file. This is harmless for every other augmentation/transform
in this codebase (they're channel-order-symmetric), and harmless for the
model itself (training and driving see the same convention). It only
matters for `LANE_ISOLATE`'s centre-line channel, which reads actual hue.

Confirmed on `data_baseline`: the centre tape is really **yellow**
(`a*=-1, b*=+33` in true colour), which reads as cyan if you assume RGB.

```python
LANE_ISOLATE_COLOR_ORDER = 'bgr'
LANE_ISOLATE_CHROMA_ANGLE = 90.0   # yellow
```

If you're on a **PiCamera** rig instead, use the defaults
(`LANE_ISOLATE_COLOR_ORDER = 'rgb'`) — PiCamera's `BGR888` format setting is
a libcamera naming quirk that actually returns RGB.

If in doubt, preview a frame: get the wrong order and the centre-line
channel doesn't error, it either goes blank or floods with noise depending
on which way it's wrong — always visually confirm before training.

## Quick reference: full override block per trial

Paste the relevant block into `myconfig.py`, on top of the "Fixed for every
trial" settings above.

```python
# --- 0: baseline ---
AUGMENTATIONS = []

# --- 1: global_tone ---
AUGMENTATIONS = ['BRIGHTNESS', 'GAMMA']
AUG_GAMMA_RANGE = (60, 160)

# --- 2: local_light ---
AUGMENTATIONS = ['SUNLIGHT', 'SHADOW']

# --- 3: full_photometric ---
AUGMENTATIONS = ['SUNLIGHT', 'SHADOW', 'BRIGHTNESS', 'GAMMA', 'NOISE']
AUG_GAMMA_RANGE = (60, 160)

# --- 4: all_six ---
AUGMENTATIONS = ['SUNLIGHT', 'SHADOW', 'BRIGHTNESS', 'GAMMA', 'NOISE', 'BLUR']
AUG_GAMMA_RANGE = (60, 160)

# --- 5: lane_isolate ---
AUGMENTATIONS = []
POST_TRANSFORMATIONS = ['CROP', 'LANE_ISOLATE']

# --- 6: lane_isolate_aug ---
AUGMENTATIONS = ['SHADOW', 'GAMMA']
AUG_GAMMA_RANGE = (60, 160)
POST_TRANSFORMATIONS = ['CROP', 'LANE_ISOLATE']
```

Remember: whatever `POST_TRANSFORMATIONS` (and `LANE_ISOLATE_*` /
`ROI_CROP_*`) you train with must be identical in the `myconfig.py` on the
car — these are transformations, not augmentations, and they run at
inference time too. A mismatch doesn't error, it just steers badly.
