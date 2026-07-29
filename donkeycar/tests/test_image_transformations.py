import json
import os

import numpy as np
import pytest

from donkeycar.config import Config
from donkeycar.parts.cv import ImgCropMask, ImgCropResize
from donkeycar.parts.image_transformations import (
    _pipeline_steps,
    build_preprocessing_metadata,
    check_preprocessing_metadata,
    image_transformer,
    save_preprocessing_metadata,
)


def make_row_marker_image(height=108, width=192):
    """ uint8 RGB image where every pixel in row r has value r + 1 (never
        zero), so a crop can be checked against the exact source rows it
        should have kept. """
    rows = np.arange(1, height + 1, dtype=np.uint8).reshape(height, 1, 1)
    return np.broadcast_to(rows, (height, width, 3)).copy()


class TestImgCropResize:
    def test_output_shape_matches_target(self):
        img = make_row_marker_image(108, 192)
        crop = ImgCropResize(top=54, bottom=0, left=0, right=0,
                             target_width=192, target_height=108)
        out = crop.run(img)
        assert out.shape == (108, 192, 3)

    def test_no_black_masking_band(self):
        img = make_row_marker_image(108, 192)
        # target size equals the cropped region exactly, so no resize
        # interpolation is needed and pixel values must be exact.
        crop = ImgCropResize(top=54, bottom=0, left=0, right=0,
                             target_width=192, target_height=54)
        out = crop.run(img)
        assert out.shape == (54, 192, 3)
        # a real crop never introduces zeroed-out pixels
        assert np.all(out != 0)

    def test_retains_correct_bottom_section(self):
        img = make_row_marker_image(108, 192)
        crop = ImgCropResize(top=54, bottom=0, left=0, right=0,
                             target_width=192, target_height=54)
        out = crop.run(img)
        # row 0 of the output must be row 54 of the source (marker 55),
        # not row 0 of the source (marker 1) and not a masked/black row
        assert out[0, 0, 0] == 55
        assert out[-1, 0, 0] == 108

    def test_preserves_dtype_and_channels(self):
        img = make_row_marker_image(108, 192)
        crop = ImgCropResize(top=54, target_width=192, target_height=54)
        out = crop.run(img)
        assert out.dtype == img.dtype
        assert out.shape[2] == img.shape[2]

    def test_crop_without_target_size_just_crops(self):
        img = make_row_marker_image(108, 192)
        crop = ImgCropResize(top=54, bottom=0, left=0, right=0)
        out = crop.run(img)
        assert out.shape == (54, 192, 3)

    def test_negative_margin_raises(self):
        with pytest.raises(ValueError):
            ImgCropResize(top=-5)

    def test_empty_crop_raises(self):
        img = make_row_marker_image(108, 192)
        # top + bottom >= height -> empty result
        crop = ImgCropResize(top=60, bottom=60, target_width=192,
                             target_height=1)
        with pytest.raises(ValueError):
            crop.run(img)

    def test_differs_from_mask_based_crop(self):
        """ ImgCropMask keeps full dimensions and zeroes the excluded
            region; ImgCropResize must not. """
        img = make_row_marker_image(108, 192)
        masked = ImgCropMask(left=0, top=54, right=0, bottom=0).run(img)
        cropped = ImgCropResize(top=54, target_width=192,
                                target_height=54).run(img)
        assert masked.shape == img.shape
        assert np.any(masked == 0)
        assert cropped.shape != img.shape
        assert np.all(cropped != 0)


class TestImageTransformerFactory:
    def test_crop_config_wires_to_crop_resize(self):
        cfg = Config()
        cfg.IMAGE_W = 192
        cfg.IMAGE_H = 108
        cfg.ROI_CROP_TOP = 54
        cfg.ROI_CROP_BOTTOM = 0
        cfg.ROI_CROP_LEFT = 0
        cfg.ROI_CROP_RIGHT = 0
        transformer = image_transformer('CROP', cfg)
        assert isinstance(transformer, ImgCropResize)

        img = make_row_marker_image(108, 192)
        out = transformer.run(img)
        assert out.shape == (108, 192, 3)


class TestPipelineOrder:
    def test_augmentation_runs_before_crop_in_training(self):
        cfg = Config()
        cfg.TRANSFORMATIONS = []
        cfg.POST_TRANSFORMATIONS = ['CROP']
        cfg.AUGMENTATIONS = ['BRIGHTNESS', 'GAMMA']

        training_steps = _pipeline_steps(cfg, include_augmentations=True)
        assert training_steps == ['BRIGHTNESS', 'GAMMA', 'CROP']
        assert training_steps.index('CROP') > training_steps.index('GAMMA')

    def test_validation_and_driving_never_see_augmentation(self):
        cfg = Config()
        cfg.TRANSFORMATIONS = []
        cfg.POST_TRANSFORMATIONS = ['CROP']
        cfg.AUGMENTATIONS = ['BRIGHTNESS', 'GAMMA']

        steps = _pipeline_steps(cfg, include_augmentations=False)
        assert steps == ['CROP']
        assert 'BRIGHTNESS' not in steps
        assert 'GAMMA' not in steps


class TestPreprocessingMetadata:
    def crop_cfg(self):
        cfg = Config()
        cfg.IMAGE_W = 192
        cfg.IMAGE_H = 108
        cfg.TRANSFORMATIONS = []
        cfg.POST_TRANSFORMATIONS = ['CROP']
        cfg.ROI_CROP_TOP = 54
        cfg.ROI_CROP_BOTTOM = 0
        cfg.ROI_CROP_LEFT = 0
        cfg.ROI_CROP_RIGHT = 0
        return cfg

    def test_build_metadata_reflects_config(self):
        meta = build_preprocessing_metadata(self.crop_cfg())
        assert meta['image_width'] == 192
        assert meta['image_height'] == 108
        assert meta['transformations'] == ['CROP_RESIZE']
        assert meta['roi_crop_top'] == 54
        assert meta['crop_mode'] == 'physical_crop_and_resize'

    def test_save_then_check_matching_config_does_not_raise(self, tmp_path):
        cfg = self.crop_cfg()
        model_path = str(tmp_path / 'mypilot.h5')
        with open(model_path, 'w') as f:
            f.write('placeholder')
        save_preprocessing_metadata(cfg, model_path)

        sidecar = tmp_path / 'mypilot.preprocessing.json'
        assert sidecar.exists()
        with open(sidecar) as f:
            saved = json.load(f)
        assert saved['roi_crop_top'] == 54

        # must not raise when config matches what was saved
        check_preprocessing_metadata(cfg, model_path)

    def test_check_detects_mismatch(self, tmp_path):
        cfg = self.crop_cfg()
        model_path = str(tmp_path / 'mypilot.h5')
        with open(model_path, 'w') as f:
            f.write('placeholder')
        save_preprocessing_metadata(cfg, model_path)

        drifted_cfg = self.crop_cfg()
        drifted_cfg.ROI_CROP_TOP = 30  # simulates a stale/edited car config
        with pytest.raises(RuntimeError, match='roi_crop_top'):
            check_preprocessing_metadata(drifted_cfg, model_path)

    def test_missing_sidecar_only_warns(self, tmp_path):
        cfg = self.crop_cfg()
        model_path = str(tmp_path / 'never_trained.h5')
        # no sidecar file written at all
        check_preprocessing_metadata(cfg, model_path)  # must not raise
