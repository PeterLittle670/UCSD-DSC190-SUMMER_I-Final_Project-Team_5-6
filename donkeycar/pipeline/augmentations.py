import random
import albumentations.core.transforms_interface
import logging
import albumentations as A
import cv2
import numpy as np
from albumentations import GaussianBlur, RandomGamma, GaussNoise
from albumentations.augmentations import RandomBrightnessContrast
from albumentations.core.transforms_interface import ImageOnlyTransform


from donkeycar.config import Config


logger = logging.getLogger(__name__)


class RandomShadow(ImageOnlyTransform):
    """ Adds one or more randomly shaped, randomly placed shadows to an
        image, so the model sees the kind of partial, irregular shadows
        (trees, buildings, etc.) that cross the lane at different times of
        day. Only the lightness is reduced (via the HLS color space), so hue
        and saturation - and hence the road / lane colors - are preserved,
        rather than the whole image being darkened. """

    def __init__(self,
                num_shadows_range=(1, 2),
                darkness_range=(0.4, 0.7),
                shadow_dimension=5,
                shadow_roi=(0.0, 0.3, 1.0, 1.0),
                blur_ksize=21,
                p=0.5):
        super().__init__(p=p)
        self.num_shadows_range = num_shadows_range
        self.darkness_range = darkness_range
        self.shadow_dimension = shadow_dimension
        self.shadow_roi = shadow_roi
        self.blur_ksize = blur_ksize

    def apply(self, img, **params):
        # Compatibility guard: shadow simulation reduces lightness via the
        # HLS color space, which requires a 3-channel (RGB/BGR) image.
        # Skip (no-op) on grayscale or other channel counts (e.g. if a
        # config sets IMAGE_DEPTH = 1) instead of letting cv2 raise.
        if img.ndim != 3 or img.shape[2] != 3:
            return img

        height, width = img.shape[:2]
        x_min, y_min, x_max, y_max = self.shadow_roi
        roi_x_min, roi_x_max = int(x_min * width), int(x_max * width)
        roi_y_min, roi_y_max = int(y_min * height), int(y_max * height)

        # Build a single 0..1 mask that can hold several shadow shapes.
        # Each shape is a random, irregular polygon (not a rectangle), so
        # shadows look like real cast shadows rather than a hard box.
        mask = np.zeros((height, width), dtype=np.float32)
        num_shadows = random.randint(*self.num_shadows_range)
        for _ in range(num_shadows):
            vertices = np.array(
                [[random.randint(roi_x_min, roi_x_max),
                 random.randint(roi_y_min, roi_y_max)]
                 for _ in range(self.shadow_dimension)], dtype=np.int32)
            darkness = random.uniform(*self.darkness_range)
            polygon_mask = np.zeros((height, width), dtype=np.float32)
            cv2.fillPoly(polygon_mask, [vertices], 1.0)
            mask = np.maximum(mask, polygon_mask * darkness)

        # Blur the mask edges so the shadow fades into the image instead of
        # appearing as a hard-edged cutout.
        if self.blur_ksize and self.blur_ksize > 1:
            k = self.blur_ksize | 1  # cv2 requires an odd kernel size
            mask = cv2.GaussianBlur(mask, (k, k), 0)

        # Reduce lightness only. cv2's RGB<->HLS and BGR<->HLS conversions
        # produce the same L channel (it only depends on per-pixel max/min
        # across channels), and we convert back with the matching HLS2RGB
        # code, so this is correct whether img is in RGB or BGR order.
        orig_dtype = img.dtype
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float32)
        hls[:, :, 1] *= (1.0 - mask)
        hls = np.clip(hls, 0, 255).astype(np.uint8)
        shadowed = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)
        return shadowed.astype(orig_dtype)

    def get_transform_init_args_names(self):
        return ('num_shadows_range', 'darkness_range', 'shadow_dimension',
                'shadow_roi', 'blur_ksize')


class RandomLocalSunlight(ImageOnlyTransform):
    """ Simulates strong sunlight hitting only part of the image, e.g. half
        the road in direct sun while the rest stays shaded, a diagonal
        sunbeam crossing the track, or dappled light through trees. This is
        different from a plain brightness augmentation (RandomBrightnessContrast),
        which scales the *whole* image uniformly - real midday lighting on a
        track is highly localized, with a sharp or softened boundary between
        sun and shade, so the model needs to see that too. As with
        RandomShadow, only the lightness (HLS 'L' channel) inside the random
        mask is boosted, so hue/saturation - and hence lane and road colors -
        are not distorted. """

    def __init__(self,
                sunlight_strength_range=(1.15, 1.8),
                coverage_range=(0.15, 0.55),
                num_regions_range=(1, 2),
                blur_kernel_range=(11, 41),
                road_region_start=0.25,
                p=0.3):
        super().__init__(p=p)
        self.sunlight_strength_range = sunlight_strength_range
        self.coverage_range = coverage_range
        self.num_regions_range = num_regions_range
        self.blur_kernel_range = blur_kernel_range
        self.road_region_start = road_region_start

    def apply(self, img, **params):
        # Compatibility guard: sunlight simulation boosts lightness via the
        # HLS color space, which requires a 3-channel (RGB/BGR) image.
        # Skip (no-op) on grayscale or other channel counts (e.g. if a
        # config sets IMAGE_DEPTH = 1) instead of letting cv2 raise.
        if img.ndim != 3 or img.shape[2] != 3:
            return img

        height, width = img.shape[:2]
        total_area = float(height * width)

        # Bias sunlit regions toward the lower part of the frame (the road),
        # since that's what the model needs to learn to drive through. Every
        # so often let a region start higher up too (e.g. dappled sunlight
        # coming through trees onto the track ahead) for variety.
        roi_y_min = int(self.road_region_start * height)
        if random.random() < 0.2:
            roi_y_min = random.randint(0, roi_y_min)

        num_regions = random.randint(*self.num_regions_range)
        coverage = random.uniform(*self.coverage_range)

        # Build a single gain map (how much to multiply lightness by, minus
        # 1) that can hold several sunlit shapes. Regions are combined with
        # max rather than summed, so overlapping regions don't stack into
        # unrealistically blown-out whites.
        gain = np.zeros((height, width), dtype=np.float32)
        for _ in range(num_regions):
            region_area = (coverage / num_regions) * total_area
            strength = random.uniform(*self.sunlight_strength_range)
            shape = random.choice(('polygon', 'ellipse', 'gradient'))
            region_mask = self._make_region_mask(shape, height, width,
                                                  roi_y_min, region_area)
            gain = np.maximum(gain, region_mask * (strength - 1.0))

        # Soften (or occasionally keep sharp) the sun/shade boundary, since
        # real transitions range from crisp to gradually diffused.
        k_min, k_max = self.blur_kernel_range
        k = random.randint(k_min, k_max) | 1  # cv2 requires an odd kernel
        if k > 1:
            gain = cv2.GaussianBlur(gain, (k, k), 0)

        # Boost lightness only. cv2's RGB<->HLS and BGR<->HLS conversions
        # produce the same L channel (it only depends on per-pixel max/min
        # across channels), and we convert back with the matching HLS2RGB
        # code, so this is correct whether img is in RGB or BGR order (see
        # RandomShadow above).
        orig_dtype = img.dtype
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float32)
        hls[:, :, 1] *= (1.0 + gain)
        hls = np.clip(hls, 0, 255).astype(np.uint8)
        sunlit = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)
        return np.ascontiguousarray(sunlit.astype(orig_dtype))

    @staticmethod
    def _make_region_mask(shape, height, width, roi_y_min, area):
        """ Builds a single 0..1 mask (full strength inside the sunlit
            region) of roughly the requested pixel area, using one of three
            shapes so sunlight can look like a blob, an elongated/rotated
            patch, or a directional beam crossing the frame. """
        mask = np.zeros((height, width), dtype=np.float32)
        roi_y_min = min(roi_y_min, height - 1)

        if shape == 'ellipse':
            # A rotated ellipse: axis ratio and angle give the sunlit patch
            # an elongated, directional look.
            aspect = random.uniform(0.5, 2.0)
            b = max(1, int(np.sqrt(area / (np.pi * aspect))))
            a = max(1, int(b * aspect))
            cx = random.randint(0, width - 1)
            cy = random.randint(roi_y_min, height - 1)
            angle = random.uniform(0, 180)
            cv2.ellipse(mask, (cx, cy), (a, b), angle, 0, 360, 1.0, -1)

        elif shape == 'gradient':
            # A directional band (e.g. half the road lit, half in shade, or
            # a diagonal sunbeam) rather than a compact blob.
            angle = random.uniform(0, 180)
            theta = np.deg2rad(angle)
            yy, xx = np.mgrid[0:height, 0:width]
            proj = xx * np.cos(theta) + yy * np.sin(theta)
            proj_min, proj_max = proj.min(), proj.max()
            proj_norm = (proj - proj_min) / max(proj_max - proj_min, 1e-6)
            coverage_frac = np.clip(area / (width * height), 0.05, 0.95)
            edge = 1.0 - coverage_frac
            ramp = 0.15
            mask = np.clip((proj_norm - edge) / ramp + 1.0, 0.0, 1.0) \
                .astype(np.float32)
            # Keep most of the gradient below the sky line.
            mask[:roi_y_min, :] *= 0.3

        else:  # 'polygon': an irregular, randomly shaped bright patch
            side = max(4, int(np.sqrt(area)))
            cx = random.randint(0, width - 1)
            cy = random.randint(roi_y_min, height - 1)
            num_vertices = random.randint(5, 8)
            vertices = []
            for i in range(num_vertices):
                a_ = 2 * np.pi * i / num_vertices + random.uniform(-0.3, 0.3)
                r = side * random.uniform(0.5, 1.0)
                vx = int(np.clip(cx + r * np.cos(a_), 0, width - 1))
                vy = int(np.clip(cy + r * np.sin(a_), 0, height - 1))
                vertices.append([vx, vy])
            cv2.fillPoly(mask, [np.array(vertices, dtype=np.int32)], 1.0)

        return mask

    def get_transform_init_args_names(self):
        return ('sunlight_strength_range', 'coverage_range',
                'num_regions_range', 'blur_kernel_range',
                'road_region_start')


class ImageAugmentation:
    def __init__(self, cfg, key, prob=0.5):
        aug_list = getattr(cfg, key, [])
        augmentations = [ImageAugmentation.create(a, cfg, prob)
                         for a in aug_list]
        self.augmentations = A.Compose(augmentations)

    @classmethod
    def create(cls, aug_type: str, config: Config, prob) -> \
            albumentations.core.transforms_interface.BasicTransform:
        """ Augmentation factory. Cropping and trapezoidal mask are
            transformations which should be applied in training, validation
            and inference. Multiply, Blur and similar are augmentations
            which should be used only in training. """

        if aug_type == 'BRIGHTNESS':
            b_limit = getattr(config, 'AUG_BRIGHTNESS_RANGE', 0.2)
            # AUG_CONTRAST_RANGE is optional and defaults to the brightness
            # range, so existing configs that only set AUG_BRIGHTNESS_RANGE
            # keep their exact previous behaviour (brightness and contrast
            # limits equal).
            c_limit = getattr(config, 'AUG_CONTRAST_RANGE', b_limit)
            logger.info(f'Creating augmentation {aug_type} '
                       f'brightness={b_limit} contrast={c_limit}')
            return RandomBrightnessContrast(brightness_limit=b_limit,
                                            contrast_limit=c_limit,
                                            p=prob)

        elif aug_type == 'BLUR':
            b_range = getattr(config, 'AUG_BLUR_RANGE', 3)
            logger.info(f'Creating augmentation {aug_type} {b_range}')
            return GaussianBlur(sigma_limit=b_range, blur_limit=(13, 13),
                                p=prob)

        elif aug_type == 'GAMMA':
            # Nonlinear brightness change. gamma_limit is a percentage
            # around 100: values below 100 brighten the image, values above
            # 100 darken it, so a single range like (60, 160) simulates
            # both glare/overexposure and dim/nighttime conditions without
            # needing separate augmentations for each direction.
            gamma_limit = getattr(config, 'AUG_GAMMA_RANGE', (80, 120))
            gamma_prob = getattr(config, 'AUG_GAMMA_PROBABILITY', prob)
            logger.info(f'Creating augmentation {aug_type} {gamma_limit} '
                       f'p={gamma_prob}')
            return RandomGamma(gamma_limit=gamma_limit, p=gamma_prob)

        elif aug_type == 'NOISE':
            # Simulates sensor grain, most noticeable in low-light /
            # nighttime footage. std_range/mean_range are fractions of the
            # image's max pixel value (e.g. 0.1 ~= 10% of 255 for uint8
            # images).
            std_range = getattr(config, 'AUG_NOISE_STD_RANGE', (0.005, 0.03))
            mean_range = getattr(config, 'AUG_NOISE_MEAN_RANGE', (0.0, 0.0))
            noise_prob = getattr(config, 'AUG_NOISE_PROBABILITY', prob)
            logger.info(f'Creating augmentation {aug_type} std={std_range} '
                       f'p={noise_prob}')

            try:
                # Albumentations 2.x
                return GaussNoise(std_range=std_range,
                                  mean_range=mean_range,
                                  p=noise_prob)
            except TypeError:
                # Albumentations 1.x used in the DSMLP GPU env
                var_limit = tuple((s * 255.0) ** 2 for s in std_range)
                mean = sum(mean_range) / 2.0 * 255.0

                return GaussNoise(var_limit=var_limit,
                                  mean=mean,
                                  p=noise_prob)

        elif aug_type == 'SHADOW':
            shadow_prob = getattr(config, 'AUG_SHADOW_PROBABILITY', prob)
            num_shadows_range = getattr(config, 'AUG_SHADOW_COUNT_RANGE',
                                        (1, 2))
            darkness_range = getattr(config, 'AUG_SHADOW_DARKNESS_RANGE',
                                     (0.4, 0.7))
            shadow_dimension = getattr(config, 'AUG_SHADOW_DIMENSION', 5)
            shadow_roi = getattr(config, 'AUG_SHADOW_ROI',
                                 (0.0, 0.3, 1.0, 1.0))
            blur_ksize = getattr(config, 'AUG_SHADOW_BLUR_KSIZE', 21)
            logger.info(f'Creating augmentation {aug_type} '
                       f'darkness={darkness_range} p={shadow_prob}')
            return RandomShadow(num_shadows_range=num_shadows_range,
                                darkness_range=darkness_range,
                                shadow_dimension=shadow_dimension,
                                shadow_roi=shadow_roi,
                                blur_ksize=blur_ksize,
                                p=shadow_prob)

        elif aug_type == 'SUNLIGHT':
            sunlight_prob = getattr(config, 'AUG_SUNLIGHT_PROBABILITY', prob)
            strength_range = getattr(config, 'AUG_SUNLIGHT_STRENGTH_RANGE',
                                     (1.15, 1.8))
            coverage_range = getattr(config, 'AUG_SUNLIGHT_COVERAGE_RANGE',
                                     (0.15, 0.55))
            region_range = getattr(config, 'AUG_SUNLIGHT_REGION_COUNT_RANGE',
                                   (1, 2))
            blur_kernel_range = getattr(config,
                                        'AUG_SUNLIGHT_BLUR_KERNEL_RANGE',
                                        (11, 41))
            road_region_start = getattr(config,
                                        'AUG_SUNLIGHT_ROAD_REGION_START',
                                        0.25)
            logger.info(f'Creating augmentation {aug_type} '
                       f'strength={strength_range} p={sunlight_prob}')
            return RandomLocalSunlight(sunlight_strength_range=strength_range,
                                       coverage_range=coverage_range,
                                       num_regions_range=region_range,
                                       blur_kernel_range=blur_kernel_range,
                                       road_region_start=road_region_start,
                                       p=sunlight_prob)


    # Parts interface
    def run(self, img_arr):
        if len(self.augmentations) == 0:
            return img_arr
        aug_img_arr = self.augmentations(image=img_arr)["image"]
        return aug_img_arr

