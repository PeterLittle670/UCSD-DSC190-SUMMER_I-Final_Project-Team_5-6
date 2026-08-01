# DonkeyCar Robustness & Explainable AI Toolkit
DSC190 SSI Final Project
Team #5+6, Summer 2026

## Team Members
Yash Tandon
Peter Little
Kaitlyn Tam
Jacey Chow

![Team #5+6 DonkeyCar](docs/DSC190-Robot.png)

## Abstract
This project extends the open-source [DonkeyCar](https://github.com/autorope/donkeycar) autonomous racing library along two independent tracks. **Track 1** improves the robustness of DonkeyCar's default steering CNN by expanding its training-time data augmentation pipeline (brightness, blur, gamma, noise, shadow, sunlight) and adding post-training image transformations (crop, lane-isolate), so a model trained under one lighting condition can still drive reliably at other times of day. **Track 2** builds a suite of explainability and uncertainty tools that let a user look inside the CNN's "black box" — live confidence, novelty/out-of-distribution, and prediction-stability signals surfaced through the web dashboard, automatic throttle reduction when the model is unsure, and an offline Grad-CAM/saliency viewer for diagnosing exactly where and why the model failed.

---

## Track 1: Improve DonkeyCar CNN by Data Augmentation

### What We Promised
* Collect midday driving data
* Implement several new training-time augmentations
* Have the model work throughout the day under different lighting conditions, despite being trained on data collected at only one time of day

### What We Have Done
* Collected midday data in addition to the existing dataset
* Implemented six new augmentations: brightness, blur, sunlight, shadow, gamma, and noise
* Implemented post-training image transformations: crop, and a new `LANE_ISOLATE` transform for contrasting lane detection

### Demos
Side-by-side track runs comparing the augmented model against the baseline model. The baseline model failed twice — losing the lane boundary once on a tight turn and once when the background scenery changed — while the augmentation-trained model completed the same course cleanly. Augmentation samples (brightness, gamma, shadow, noise, and all conditions combined) were also generated and visually verified against a sample night-time frame to confirm each effect looked realistic before training on it.

### What Didn't Work → Our Approach
* **Initial trials:** the model failed to drive under direct sunlight and at night, and struggled to steer correctly through drastic lighting changes.
  → Added a sunlight augmentation, which fixed daytime/afternoon driving. Night-time driving remained unreliable.
* **After testing the first batch of models:** we found the model was relying too heavily on background scenery rather than the lane markings themselves, and that shadow/noise augmentation alone weren't enough to fix this.
  → Cropped and resized training images by 50% before training to force the model to focus on the lanes instead of the background.

### If We Had More Time
* Find a better way to make the model rely less on background and focus more on the lanes
* Fine-tune the brightness augmentation range to normalize brightness across different driving conditions
* Tune the values of the other augmentations
* Research additional augmentations that could help with lighting-contrast issues

### Timeline
| Dates (July 2026) | Task |
|---|---|
| 7/16 – 7/18 | Create augmentations |
| 7/17 – 7/21 | Train and record augmentation combinations |
| 7/22 – 7/23 | Create more augmentations |
| 7/23 – 7/25 | Train more and record performance |
| 7/26 – 7/27 | Implement post-training transformations |
| 7/27 – 7/29 | Train transformation + augmentation combos |
| 7/29 – 7/30 | Prepare presentation |

### Documentation
* [Augmentations guide](docs/augmentations.md) — usage walkthrough and technical explanation of every augmentation and transformation
* [Lighting robustness trials](docs/lighting_robustness_trials.md) — trial notes behind the "what didn't work" findings above

---

## Track 2: Expansion of DonkeyCar via Explainable AI

### What We Promised

**Must have**
* Main goal: build a suite of tools letting users look inside the CNN "black box" to help find points of failure
* Real-time explainable AI indicators for the CNN's predictions: uncertainty detection via Monte Carlo Dropout, and novelty (out-of-distribution) detection
* Integration with the existing DonkeyCar web viewer
* Automatic throttle reduction in low-confidence situations
* Offline visualization tools showing which parts of an image drive uncertainty and novelty: Grad-CAM saliency maps of attention disagreement across Monte Carlo Dropout passes, and a heatmap viewer for out-of-distribution spatial features

**Nice to have**
* Expansion to other explainable AI metrics: saliency, counterfactual tracking, and other state-of-the-art uncertainty/novelty methods
* Mechanistic interpretability (e.g., sparse autoencoders) to understand learned image features
* An LED or other physical component to trigger when uncertainty is high
* An AI hat running object detection with LLM-generated narration under uncertainty (e.g. "There's a person in front of me, I'm unsure of what I should do")

### What We Have Done
* **Main goal:** built a full toolkit exposing the CNN's internals — confidence, novelty, and prediction stability — all live and inspectable
* **Uncertainty detection (Monte Carlo Dropout):** N stochastic forward passes batched in parallel; variance is converted into a calibrated 0–100% confidence score
* **Novelty / out-of-distribution detection:** Mahalanobis distance computed over a frozen, general-purpose ImageNet encoder (switched to this after discovering the task-trained model's own features had collapsed away the information needed to tell grass from track)
* **Web viewer integration:** live dashboard panels per signal, color-coded green/amber/red, with an "uncalibrated" warning banner and plain-English explanations for each metric
* **Automatic throttle reduction:** a throttle scaler reduces (or fully stops) throttle when any enabled signal crosses its threshold, while leaving steering untouched
* **Offline visualization tools:** Grad-CAM saliency maps showing attention disagreement across Monte Carlo Dropout passes (mean map = attention, pixel-wise variance across passes = disagreement), plus a heatmap viewer for out-of-distribution spatial features

**Expansion to other XAI metrics**
* ✅ Saliency (vanilla gradient-based)
* ❌ Counterfactual tracking — not implemented
* ✅ Other state-of-the-art methods — Grad-CAM++, Integrated Gradients, and a bonus third live signal: Test-Time Augmentation (TTA) stability
* ❌ Mechanistic interpretability / sparse autoencoders — not implemented
* ❌ LED or other physical uncertainty trigger — not implemented
* ❌ AI-hat object detection + LLM narration — not implemented

**Beyond the original scope**
* **Calibration system** tying all signals to a per-model, per-dataset baseline (`<model>.calib.json`)
* **Extremely customizable** — settings for every feature auto-generated under `myconfig.py`
* **Augmentation-aware calibration** — fixed a real bug where shadow/lighting-robust models were falsely flagged as "novel" on the exact conditions they were trained to handle (measured false-flag rate improvement: 79% → 38%)
* **GUI launcher** for offline analysis that auto-detects the training tub, tracks analysis progress, and auto-continues into the viewer
* **`RECORD_DURING_AI` support** — autopilot runs are auto-saved to a separate tub for offline review without polluting training data
* **Detailed documentation** covering usage of every XAI capability plus a technical deep-dive into the underlying concepts

### Demos
The web dashboard shows three live signals per frame — "Do the model's sub-networks agree?" (confidence), "Does this look familiar?" (novelty), and "Is the answer stable?" (TTA stability) — each with a plain-English explanation on click. Demonstrated across day and night driving, and in front of unfamiliar objects (a person, a cone) to show novelty spiking correctly. The offline viewer was shown stepping through a recorded drive frame-by-frame with Grad-CAM overlays highlighting where the model's attention landed (and disagreed) on the road, on a pedestrian, and on track-edge pillars — plus a "salient pixels" layer showing the exact pixels that most affected the steering prediction. The offline analysis GUI launcher was also demoed end-to-end: pick a tub and model, auto-calibrate if needed, and get dropped straight into the viewer.

### Problems Faced (and Solved)
* **Power brownout issues:** running camera capture, CNN inference, and every XAI indicator in real time at 20Hz was more than the Pi could handle.
  → Reduced camera resolution and added customizable settings (interval, number of passes) for each XAI indicator so per-signal compute cost can be tuned to the hardware.
* **Novelty detection ignoring scene appearance:** the steering model's own internal features ignored scene appearance, so different environments all looked "normal" and reported low novelty.
  → Switched to a general-purpose image encoder so the novelty detector compares richer visual features that preserve scene appearance.
* **User-friendly tool experience:** running the offline analysis originally required typing a long command with many parameters before even reaching the GUI.
  → Added auto-generation of calibration files after training and an interactive, detailed GUI for entering parameters before analysis, so a single simple command launches everything.

### Unsolved Problems
* If augmentations are applied during training, the calibration step doesn't replay the exact same data, since augmentations are applied randomly and calibration is run separately.
* The metrics are not literal percentages — they're relative scores based on the variance distribution of the training data — which makes it hard to say what counts as a "good" vs. "bad" score in absolute terms.
* Reading and interpreting Grad-CAM maps still depends on the user being able to recognize patterns of uncertainty, novelty, and saliency.

### Extensions
* **Real-time object detection with the Hailo AI-Hat+:** when novelty stays high for several consecutive frames, run Grad-CAM on one of those frames and feed the resulting heatmap into an object-detection or vision-language model on the AI hat, potentially narrating findings out loud (e.g. "I recognize a `<object>` in front of me, I'm not sure what to do").
* **XAI model comparison dashboard:** a side-by-side table/graph view comparing confidence, novelty, TTA stability, and saliency across multiple trained models, to identify which model is most robust under which conditions.
* **Salience source mapping:** apply mechanistic interpretability (linear probes, sparse autoencoders) to see what concepts the CNN's top filters actually encode — e.g. testing whether track-edge transitions or turn direction are linearly decodable, and pulling the highest-activating image patches for each top filter/feature.

### Documentation
* [Explainable AI (XAI) toolkit guide](docs/xai.md) — usage walkthrough (recording, calibration, live dashboard, offline viewer) and a technical deep-dive into how each signal works

---

## Final Project Videos / Presentation

[![Presentation slides](docs/slides_thumbnail.png)](https://drive.google.com/file/d/YOUR_FILE_ID/view?usp=sharing)

---

## Acknowledgements
README.md Format, reference to [winter-2024-final-project-team-7](https://github.com/UCSD-ECEMAE-148/winter-2024-final-project-team-7)

Thank you to Professor Silberman and TAs Evan Chou and Jose Castillo for facilitating this course!

---

## Contacts

* Yash Tandon - [ytandon@ucsd.edu](mailto:ytandon@ucsd.edu) | [LinkedIn](https://linkedin.com/in/yashtandon05)
* Peter Little -
* Kaitlyn Tam - 
* Jacey Chow -
