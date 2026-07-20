"""
Vanilla-gradient saliency: how much would each *pixel* of the input image
change the model's steering(+throttle) output if perturbed, via the gradient
of the summed output(s) w.r.t. the raw input image.

This is a different technique from Grad-CAM
(``donkeycar.parts.gradcam_uncertainty``), which operates on the last
convolutional layer's coarse feature grid and is smoothed by channel-pooling
into a small number of spatial cells. Vanilla-gradient saliency computes
gradients directly against full-resolution input pixels instead, giving a
finer but visibly noisier, more spatially-diffuse view of "what matters to
the prediction" -- a complementary lens, not a replacement. In practice
Grad-CAM tends to produce a small number of clean, localized hotspots, while
vanilla-gradient saliency tends to highlight many pixels weakly across a
wider area (see the "Caveats" note below).

This module offers two pixel-level attribution methods:
  * ``VanillaGradientSaliency`` -- a single raw input gradient. Cheap, but the
    gradient saturates for deep nets, so the map is speckly/noisy.
  * ``IntegratedGradients`` -- integrates the gradient along a straight path
    from a baseline (black image) to the input. Much cleaner attributions
    (satisfies the sensitivity + implementation-invariance axioms of
    Sundararajan et al., 2017) at the cost of ``steps`` extra passes. Offline
    only -- never put it on the live drive loop.

Used by:
  * ``donkeycar.management.makemovie`` -- the ``--salient`` flag on
    ``donkey makemovie`` burns vanilla saliency into an exported video, frame
    by frame.
  * ``donkeycar.parts.gradcam_uncertainty`` -- offers both vanilla saliency
    and integrated gradients as overlay layers alongside the Grad-CAM
    attention/uncertainty maps and the feature-space novelty map, for the same
    analysed frames.

Caveats:
  * Compared to Grad-CAM, pixel-level gradients are typically noisier and
    less spatially decisive -- treat these as one more data point, not a more
    "correct" answer than the other overlays.
  * Model scope: like the rest of this toolkit, only architectures with
    identifiable output layers (name containing "out", excluding dropout
    layers) are supported -- this covers the default linear and categorical
    architectures.
"""
import os
import tempfile

import numpy as np
import tensorflow as tf
from tensorflow.keras import activations
from tensorflow.keras.models import load_model


def _apply_linear_output_activation(model):
    """
    Force the model's output layer(s) (name contains "out", excluding
    dropout layers) to linear activation, and rebuild the graph via a
    save/reload round-trip -- directly setting ``layer.activation`` does not
    actually rebuild the compute graph. A no-op for architectures whose
    outputs are already linear (e.g. the default linear model); needed for
    architectures with a non-linear output activation (e.g. categorical's
    softmax), so gradients are taken against raw scores, not squashed ones.

    :return: (model, found_any) -- `model` is unchanged if no matching
             output layer was found, and `found_any` is False.
    """
    output_idx = [i for i, layer in enumerate(model.layers)
                 if 'dropout' not in layer.name.lower()
                 and 'out' in layer.name.lower()]
    if not output_idx:
        return model, False

    for i in output_idx:
        model.layers[i].activation = activations.linear

    model_path = os.path.join(tempfile.gettempdir(),
                              next(tempfile._get_candidate_names()) + '.h5')
    try:
        model.save(model_path)
        return load_model(model_path, compile=False), True
    finally:
        os.remove(model_path)


class VanillaGradientSaliency:
    """
    Computes a full-resolution pixel-saliency map: the L2 norm, across
    output channels, of the gradient of each output w.r.t. each input pixel.

    :param model:       a raw Keras model (e.g. ``pilot.interpreter.model``).
    :param categorical: True for categorical/binned outputs (uses each
                        output's argmax score), False (default) for
                        continuous linear outputs (uses the raw score).
    """

    def __init__(self, model, categorical=False):
        self.categorical = categorical
        self.model, self.found_output_layers = \
            _apply_linear_output_activation(model)

    def saliency_map(self, norm_img):
        """
        :param norm_img: float32 image, [0,1], shape (H, W, C)
        :return: (H, W) float map normalised to [0,1]
        """
        img = tf.Variable(norm_img[np.newaxis, ...], dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(img)
            preds = self.model(img, training=False)
            preds = preds if isinstance(preds, (list, tuple)) else [preds]
            if self.categorical:
                pred_list = [p[0][tf.math.argmax(p[0])] for p in preds]
            else:
                pred_list = preds

        grads_sq = 0
        for p in pred_list:
            grads_sq += tf.math.square(tape.gradient(p, img))
        grads = tf.math.sqrt(grads_sq)
        grads = tf.reduce_sum(grads, axis=-1)[0].numpy()   # (H, W)

        gmin, gmax = float(grads.min()), float(grads.max())
        if gmax > gmin:
            grads = (grads - gmin) / (gmax - gmin)
        else:
            grads = np.zeros_like(grads)
        return grads

    def shutdown(self):
        pass


class IntegratedGradients:
    """
    Integrated Gradients (Sundararajan, Taly & Yan, ICML 2017): a cleaner
    pixel-attribution method than :class:`VanillaGradientSaliency`.

    Rather than taking one gradient at the real image (which saturates and
    looks speckly), IG walks a straight path from a *baseline* image (black by
    default) to the real input in ``steps`` increments, takes the gradient at
    each step, and averages them; the average is then multiplied elementwise by
    ``(input - baseline)``::

        IG(x) = (x - baseline) * mean_over_path( dF/dx )

    Averaging along the path cancels much of the per-point gradient noise and
    gives the method its two axioms: *sensitivity* (a pixel that changes the
    output always gets non-zero attribution) and *implementation-invariance*
    (functionally identical models give identical attributions) -- neither of
    which vanilla gradients guarantee.

    Attribution target is the model's first output (steering on the linear
    model), matching Grad-CAM's choice, so this reads as "which pixels drove
    the steering decision". Cost is ``steps`` forward+backward passes per
    frame (all ``steps`` are batched into ONE pass), so this is offline-only --
    never put it on the live drive loop.

    :param model:       a raw Keras model (e.g. ``pilot.interpreter.model``).
    :param categorical: True for categorical/binned outputs (attributes the
                        top class along the path), False (default) for a
                        continuous linear output.
    :param steps:       number of Riemann-sum steps along the baseline->input
                        path (typical 20-50). More steps = smoother, slower.
    :param baseline:    optional (H, W, C) float32 [0,1] baseline image. None
                        (default) uses a black (zeros) baseline.
    """

    def __init__(self, model, categorical=False, steps=32, baseline=None):
        self.categorical = categorical
        self.steps = int(steps)
        self.baseline = baseline
        self.model, self.found_output_layers = \
            _apply_linear_output_activation(model)

    def saliency_map(self, norm_img):
        """
        :param norm_img: float32 image, [0,1], shape (H, W, C)
        :return: (H, W) float map normalised to [0,1]
        """
        x = tf.convert_to_tensor(norm_img[np.newaxis, ...], dtype=tf.float32)
        if self.baseline is None:
            baseline = tf.zeros_like(x)
        else:
            baseline = tf.convert_to_tensor(
                self.baseline[np.newaxis, ...], dtype=tf.float32)

        # Interpolated inputs along the straight path, batched: (steps, H, W, C).
        alphas = tf.reshape(tf.linspace(0.0, 1.0, self.steps), (-1, 1, 1, 1))
        path = baseline + alphas * (x - baseline)

        with tf.GradientTape() as tape:
            tape.watch(path)
            preds = self.model(path, training=False)
            preds = preds if isinstance(preds, (list, tuple)) else [preds]
            target_out = preds[0]                        # steering head
            if self.categorical:
                per_row = tf.reduce_max(target_out, axis=-1)
            else:
                per_row = tf.reshape(target_out, (self.steps, -1))[:, 0]
            # Rows of the path batch are independent, so the gradient of the
            # summed score w.r.t. the path gives each step its own gradient.
            score = tf.reduce_sum(per_row)
        path_grads = tape.gradient(score, path)          # (steps, H, W, C)

        avg_grads = tf.reduce_mean(path_grads, axis=0, keepdims=True)
        ig = (x - baseline) * avg_grads                  # (1, H, W, C)
        ig = tf.reduce_sum(tf.abs(ig), axis=-1)[0].numpy()   # (H, W)

        gmin, gmax = float(ig.min()), float(ig.max())
        if gmax > gmin:
            ig = (ig - gmin) / (gmax - gmin)
        else:
            ig = np.zeros_like(ig)
        return ig

    def shutdown(self):
        pass
