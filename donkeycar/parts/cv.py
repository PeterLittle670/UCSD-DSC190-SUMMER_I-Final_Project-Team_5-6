import time
import cv2
import numpy as np
import logging

from donkeycar.parts.camera import CameraError

logger = logging.getLogger(__name__)


def image_shape(image):
    if image is None:
        return None
    if 2 == len(image.shape):
        height, width = image.shape
        return height, width, 1
    return image.shape


class ImgGreyscale:

    def run(self, img_arr):
        if img_arr is None:
            return None

        try:
            return cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
        except:
            logger.error("Unable to convert RGB image to greyscale")
            return None

    def shutdown(self):
        pass


class ImgGRAY2RGB:
    def run(self, img_arr):
        if img_arr is None:
            return None

        try:
            return cv2.cvtColor(img_arr, cv2.COLOR_GRAY2RGB)
        except:
            logger.error(F"Unable to convert greyscale image of shape {img_arr.shape} to RGB")
            return None

    def shutdown(self):
        pass


class ImgGRAY2BGR:
    def run(self, img_arr):
        if img_arr is None:
            return None

        try:
            return cv2.cvtColor(img_arr, cv2.COLOR_GRAY2BGR)
        except:
            logger.error(F"Unable to convert greyscale image of shape {img_arr.shape} to RGB")
            return None

    def shutdown(self):
        pass


class ImgRGB2GRAY(ImgGreyscale):
    pass


class ImgBGR2GRAY:

    def run(self, img_arr):
        if img_arr is None:
            return None

        try:
            return cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
        except:
            logger.error("Unable to convert BGR image to greyscale")
            return None

    def shutdown(self):
        pass


class ImgHSV2GRAY:

    def run(self, img_arr):
        if img_arr is None:
            return None

        try:
            return cv2.cvtColor(img_arr, cv2.COLOR_HSV2GRAY)
        except:
            logger.error("Unable to convert HSV image to greyscale")
            return None

    def shutdown(self):
        pass


class ImgWriter:

    def __init__(self, filename):
        self.filename = filename

    def run(self, img_arr):
        cv2.imwrite(self.filename, img_arr)

    def shutdown(self):
        pass


class ImgBGR2RGB:

    def run(self, img_arr):
        if img_arr is None:
            return None

        try:
            return cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        except:
            logger.error("Unable to convert BGR image to RGB")
            return None

    def shutdown(self):
        pass


class ImgRGB2BGR:

    def run(self, img_arr):
        if img_arr is None:
            return None

        try:
            return cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
        except:
            logger.error("Unable to convert RGB image to BRG")
            return None

    def shutdown(self):
        pass


class ImgHSV2RGB:

    def run(self, img_arr):
        if img_arr is None:
            return None

        try:
            return cv2.cvtColor(img_arr, cv2.COLOR_HSV2RGB)
        except:
            logger.error("Unable to convert HSV image to RGB")
            return None

    def shutdown(self):
        pass


class ImgRGB2HSV:

    def run(self, img_arr):
        if img_arr is None:
            return None

        try:
            return cv2.cvtColor(img_arr, cv2.COLOR_RGB2HSV)
        except:
            logger.error("Unable to convert RGB image to HSV")
            return None

    def shutdown(self):
        pass


class ImgHSV2BGR:

    def run(self, img_arr):
        if img_arr is None:
            return None

        try:
            return cv2.cvtColor(img_arr, cv2.COLOR_HSV2BGR)
        except:
            logger.error("Unable to convert HSV image to BGR")
            return None

    def shutdown(self):
        pass


class ImgBGR2HSV:

    def run(self, img_arr):
        if img_arr is None:
            return None

        try:
            return cv2.cvtColor(img_arr, cv2.COLOR_BGR2HSV)
        except:
            logger.error("Unable to convert BGR image to HSV")
            return None

    def shutdown(self):
        pass


class ImageScale:

    def __init__(self, scale, scale_height=None):
        if scale is None or scale <= 0:
            raise ValueError("ImageScale: scale must be > 0")
        if scale_height is not None and scale_height <= 0:
            raise ValueError("ImageScale: scale_height must be > 0")
        self.scale = scale
        self.scale_height = scale_height if scale_height is not None else scale

    def run(self, img_arr):
        if img_arr is None:
            return None

        try:
            return cv2.resize(img_arr, (0,0), fx=self.scale, fy=self.scale_height)
        except:
            logger.error("Unable to scale image")
            return None

    def shutdown(self):
        pass


class ImageResize:
    def __init__(self, width:int, height:int) -> None:
        if width is None or width <= 0:
            raise ValueError("ImageResize: width must be > 0")
        if height is None or height <= 0:
            raise ValueError("ImageResize: height must be > 0")
        self.width = width
        self.height = height

    def run(self, img_arr):
        if img_arr is None:
            return None

        try:
            return cv2.resize(img_arr, (self.width, self.height))
        except:
            logger.error("Unable to resize image")
            return None

    def shutdown(self):
        pass


class ImageRotateBound:
    '''
    credit:
    https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
    '''

    def __init__(self, rot_deg):
        self.rot_deg = rot_deg

    def run(self, image):
        if image is None:
            return None

        # grab the dimensions of the image and then determine the
        # center
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
    
        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), -self.rot_deg, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
    
        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
    
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
    
        # perform the actual rotation and return the image
        return cv2.warpAffine(image, M, (nW, nH))

    def shutdown(self):
        pass


class ImgCanny:

    def __init__(self, low_threshold=60, high_threshold=110, aperture_size=3, l2gradient=False):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.aperture_size = aperture_size   # 3, 5 or 7
        self.l2gradient = l2gradient

    def run(self, img_arr):
        if img_arr is None:
            return None

        try:
            return cv2.Canny(img_arr,
                             self.low_threshold,
                             self.high_threshold,
                             apertureSize=self.aperture_size,
                             L2gradient=self.l2gradient)
        except:
            logger.error("Unable to apply canny edge detection to image.")
            return None

    def shutdown(self):
        pass


class ImgLaneIsolate:
    """
    Reduce a track image to the lane markings on it, in a way that is
    largely unaffected by how bright the scene is.

    The problem this solves: a model trained on footage from one time of day
    can latch onto absolute pixel brightness, and then fails when the light
    changes. Rather than asking the network to learn that invariance from
    augmented examples, this throws the illumination away before the network
    ever sees the image.

    The key idea is that a change in lighting is *multiplicative* - the whole
    frame gets scaled by some gain - so dividing by a local estimate of the
    background cancels it, while subtracting (as an edge filter or high-pass
    does) does not. Measured on data_baseline with a synthetic -55% relight,
    mean drift of the output was 2.6% of range for this transform versus
    10.0% for a high-pass filter and 30.7% for raw pixels.

    Produces a 3 channel uint8 image the same size as the input, so
    IMAGE_DEPTH and the model input shape are unchanged:

      0: border tape     - a ratio top-hat. Greyscale opening with an
                           elliptical kernel erases anything thinner than
                           the kernel, which is an estimate of "the road
                           without its markings"; the ratio against it keeps
                           only marking-width bright structures.
      1: centre line     - the CIELAB chroma of the frame projected onto the
                           hue of the centre markings, which isolates tape
                           that sits at roughly the same lightness as the
                           pavement and so is nearly invisible to channel 0.
      2: local contrast  - the same ratio trick at a coarser scale, kept as
                           general scene context.

    Channel 1 depends on the colour of the markings AND on the channel order
    of the incoming array, so both are configurable - see chroma_angle and
    color_order. Get either wrong and channel 1 silently goes blank rather
    than raising, so preview it before training on it.

    Note this is a TRANSFORMATION, not an augmentation: it must be applied
    identically during training and while driving.
    """

    def __init__(self, kernel=9, gain=3.0, chroma_scale=8.0, lc_sigma=7.0,
                 bg_sigma=3.0, color_order='rgb', chroma_angle=90.0):
        self.kernel = kernel
        self.gain = gain
        self.chroma_scale = chroma_scale
        self.lc_sigma = lc_sigma
        self.bg_sigma = bg_sigma
        self.color_order = str(color_order).lower()
        self.chroma_angle = chroma_angle
        theta = np.deg2rad(chroma_angle)
        self.chroma_axis = (float(np.cos(theta)), float(np.sin(theta)))
        self.structuring_element = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel, kernel))

    def run(self, img_arr):
        if img_arr is None:
            return None

        # The a*/b* channels depend on which of R/B comes first, so unlike
        # the lightness-only transforms elsewhere in this codebase the
        # colour order genuinely matters here and cannot be ignored.
        if img_arr.ndim != 3 or img_arr.shape[2] != 3:
            logger.error("ImgLaneIsolate requires a 3 channel image; "
                         "returning image unchanged.")
            return img_arr

        try:
            code = (cv2.COLOR_BGR2LAB if self.color_order == 'bgr'
                    else cv2.COLOR_RGB2LAB)
            lab = cv2.cvtColor(img_arr, code).astype(np.float32)
            lightness = lab[:, :, 0]
            a = lab[:, :, 1] - 128.0
            b = lab[:, :, 2] - 128.0
            eps = 1e-3

            # --- channel 0: bright marking-width structures, as a fraction
            # of their own local background so that scene gain cancels.
            background = cv2.morphologyEx(lightness, cv2.MORPH_OPEN,
                                          self.structuring_element)
            background = cv2.GaussianBlur(background, (0, 0), self.bg_sigma)
            tape = (lightness - background) / (background + eps)
            tape = np.clip(tape * self.gain * 255.0, 0, 255)

            # --- channel 1: centre line, by projecting the (a*, b*) chroma
            # vector onto the marking's own hue direction and keeping the
            # positive lobe. Scaled flat rather than normalised by lightness
            # - the tape on this track is pale (projection ~30 against a
            # pavement median of ~0), so dividing by lightness would push it
            # below the noise floor.
            axis_a, axis_b = self.chroma_axis
            chroma = a * axis_a + b * axis_b
            chroma = np.clip(chroma, 0, None)
            chroma = np.clip(chroma * self.chroma_scale, 0, 255)

            # --- channel 2: coarse local contrast, centred at mid grey.
            coarse = cv2.GaussianBlur(lightness, (0, 0), self.lc_sigma)
            local_contrast = (lightness - coarse) / (coarse + eps)
            local_contrast = np.clip(local_contrast * 510.0 + 128.0, 0, 255)

            return np.dstack([tape, chroma, local_contrast]).astype(np.uint8)
        except Exception:
            logger.error("Unable to apply lane isolation to image.")
            return None

    def shutdown(self):
        pass


class ImgGaussianBlur:

    def __init__(self, kernel_size=5, kernel_y=None):
        self.kernel_size = (kernel_size, kernel_y if kernel_y is not None else kernel_size)
        
    def run(self, img_arr):
        if img_arr is None:
            return None

        try:
            return cv2.GaussianBlur(img_arr,
                                    self.kernel_size, 
                                    0)
        except:
            logger.error("Unable to apply gaussian blur to image.")
            return None

    def shutdown(self):
        pass


class ImgSimpleBlur:

    def __init__(self, kernel_size=5, kernel_y=None):
        self.kernel_size = (kernel_size, kernel_y if kernel_y is not None else kernel_size)
        
    def run(self, img_arr):
        if img_arr is None:
            return None

        try:
            return cv2.blur(img_arr, self.kernel_size)
        except:
            logger.error("Unable to apply simple blur to image.")
            return None

    def shutdown(self):
        pass


class ImgTrapezoidalMask:
    def __init__(self, left, right, bottom_left, bottom_right, top, bottom, fill=[255,255,255]) -> None:
        """
        Apply a trapezoidal mask to an image, keeping image in
        the trapezoid and turns everything else the fill color
        """
        self.bottom_left = bottom_left
        self.bottom_right = bottom_right
        self.top_left = left
        self.top_right = right
        self.top = top
        self.bottom = bottom
        self.fill = fill
        self.masks = {}

    def run(self, image):
        """
        Apply trapezoidal mask
        # # # # # # # # # # # # #
        # xxxxxxxxxxxxxxxxxxxxxxx
        # xxxx ul     ur xxxxxxxx min_y
        # xxx             xxxxxxx
        # xx               xxxxxx
        # x                 xxxxx
        # ll                lr xx max_y
        """
        transformed = None
        if image is not None:
            mask = None
            key = str(image.shape)
            if self.masks.get(key) is None:
                mask = np.zeros(image.shape, dtype=np.int32)
                points = [
                    [self.top_left, self.top],
                    [self.top_right, self.top],
                    [self.bottom_right, self.bottom],
                    [self.bottom_left, self.bottom]
                ]
                cv2.fillConvexPoly(mask,
                                    np.array(points, dtype=np.int32),
                                    self.fill)
                mask = np.asarray(mask, dtype='bool')
                self.masks[key] = mask

            mask = self.masks[key]
            transformed = np.multiply(image, mask)

        return transformed
    
    def shutdown(self):
        self.masks = {}  # free cached masks


class ImgTrapezoidalEdgeMask:
    def __init__(self, upper_left, upper_right, lower_left, lower_right, top, bottom, fill=[255,255,255]) -> None:
        """
        Apply a trapezoidal mask to an image, where bounds are
        relative the the edge of the image, conserving the 
        image pixels within the trapezoid and masking everything 
        other pixels with the fill color
        """
        self.lower_left = lower_left
        self.lower_right = lower_right
        self.upper_left = upper_left
        self.upper_right = upper_right
        self.top = top
        self.bottom = bottom
        self.fill = fill
        self.masks = {}

    def run(self, image):
        """
        Apply trapezoidal mask
        # # # # # # # # # # # # #
        # xxxxxxxxxxxxxxxxxxxxxxx
        # xxxx ul     ur xxxxxxxx min_y
        # xxx             xxxxxxx
        # xx               xxxxxx
        # x                 xxxxx
        # ll                lr xx max_y
        """
        transformed = None
        if image is not None:
            mask = None
            key = str(image.shape)
            if self.masks.get(key) is None:
                height, width, depth = image_shape(image)
                mask = np.zeros(image.shape, dtype=np.int32)
                points = [
                    [self.upper_left, self.top],
                    [width - self.upper_right, self.top],
                    [width - self.lower_right, height - self.bottom],
                    [self.lower_left, height - self.bottom]
                ]
                cv2.fillConvexPoly(mask,
                                    np.array(points, dtype=np.int32),
                                    self.fill)
                mask = np.asarray(mask, dtype='bool')
                self.masks[key] = mask

            mask = self.masks[key]
            transformed = np.multiply(image, mask)

        return transformed
    
    def shutdown(self):
        self.masks = {}  # free cached masks


class ImgCropMask:
    def __init__(self, left=0, top=0, right=0, bottom=0, fill=[255, 255, 255]) -> None:
        """
        Apply a mask to top and/or bottom of image.
        """
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom
        self.fill = fill
        self.masks = {}

    def run(self, image):
        """
        Apply border mask
        # # # # # # # # # # # # #
        # xxxxxxxxxxxxxxxxxxxxx #
        # xxxxxxxxxxxxxxxxxxxxx #
        # xx                 xx # top
        # xx                 xx #
        # xx                 xx #
        # xxxxxxxxxxxxxxxxxxxxx # (height - bottom)
        # xxxxxxxxxxxxxxxxxxxxx #
        # # # # # # # # # # # # #
          left                width - right
        """
        transformed = None
        if image is not None:
            mask = None
            key = str(image.shape)
            if self.masks.get(key) is None:
                height, width, depth = image_shape(image)
                top = self.top if self.top is not None else 0
                bottom = (height - self.bottom) if self.bottom is not None else height
                left = self.left if self.left is not None else 0
                right = (width - self.right) if self.right is not None else width
                mask = np.zeros(image.shape, dtype=np.int32)
                points = [
                    [left, top],
                    [right, top],
                    [right, bottom],
                    [left, bottom]
                ]
                cv2.fillConvexPoly(mask,
                                    np.array(points, dtype=np.int32),
                                    self.fill)
                mask = np.asarray(mask, dtype='bool')
                self.masks[key] = mask

            mask = self.masks[key]
            transformed = np.multiply(image, mask)

        return transformed

    def shutdown(self):
        self.masks = {}  # free cached masks


class ImgCropResize:
    """
    Apply a true crop to top/bottom/left/right of an image, then resize the
    remaining region back to (target_width, target_height).

    Unlike ImgCropMask, which keeps the original image dimensions and paints
    the excluded border with a fill color, this physically removes the
    excluded rows/columns - there is no masked-out region, no fill color and
    no hard black/image boundary in the output. The output is always exactly
    (target_height, target_width, depth) when a target size is given.
    """

    def __init__(self, left=0, top=0, right=0, bottom=0,
                target_width=None, target_height=None) -> None:
        self.left = left or 0
        self.top = top or 0
        self.right = right or 0
        self.bottom = bottom or 0
        if self.left < 0 or self.top < 0 or self.right < 0 or self.bottom < 0:
            raise ValueError(
                f"ImgCropResize: crop margins must be >= 0, got "
                f"left={self.left}, top={self.top}, right={self.right}, "
                f"bottom={self.bottom}")
        self.target_width = target_width
        self.target_height = target_height
        self._resize = None
        if target_width is not None and target_height is not None:
            # Reuse the resize convention already used elsewhere in the
            # codebase rather than a second, inconsistent implementation.
            self._resize = ImageResize(target_width, target_height)

    def run(self, image):
        if image is None:
            return None

        height, width, depth = image_shape(image)
        top, bottom = self.top, height - self.bottom
        left, right = self.left, width - self.right

        if bottom <= top or right <= left:
            raise ValueError(
                f"ImgCropResize: crop values produce an empty image for "
                f"input shape {image.shape} (left={self.left}, "
                f"top={self.top}, right={self.right}, bottom={self.bottom} "
                f"-> rows [{top}:{bottom}], cols [{left}:{right}])")

        cropped = image[top:bottom, left:right]
        if self._resize is None:
            return cropped
        return self._resize.run(cropped)

    def shutdown(self):
        pass


class ArrowKeyboardControls:
    '''
    kind of sucky control, only one press active at a time. 
    good enough for a little testing.
    requires that you have an CvImageView open and it has focus.
    '''
    def __init__(self):
        self.left = 2424832
        self.right = 2555904
        self.up = 2490368
        self.down = 2621440
        self.codes = [self.left, self.right, self.down, self.up]
        self.vec = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def run(self):
        code = cv2.waitKeyEx(delay=100)
        for iCode, keyCode in enumerate(self.codes):
            if keyCode == code:
                return self.vec[iCode]
        return 0., 0.


class Pipeline:
    def __init__(self, steps):
        self.steps = steps
    
    def run(self, val):
        for step in self.steps:
            f = step['f']
            args = step['args']
            kwargs = step['kwargs']
            
            val = f(val, *args, **kwargs)
        return val


class CvImgFromFile(object):
    def __init__(self, file_path, image_w=None, image_h=None, image_d=None, copy=False):
        """
        Part to load image from file and output as RGB image
        """
        if file_path is None:
            raise ValueError("CvImage passed empty file_path")

        image = cv2.imread(file_path)
        if image is None:
            raise ValueError(f"CvImage file_path did not resolve to a readable image file: {file_path}")
        
        #
        # resize if there are overrides
        #
        height, width, depth = image_shape(image)
        if (image_h is not None and image_h != height) or (image_w is not None and image_w != width):
            if image_h is not None:
                height = image_h
            if image_w is not None:
                width = image_w
            image = cv2.resize(image, (width, height))

        #
        # change color depth if there are overrides.
        # by default a color image will be loaded as BGR,
        # so make color image into RGB.
        #
        if image_d is not None and image_d != depth:
            if 1 == image_d:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif 3 == image_d:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif 3 == depth:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.image = image
        self.copy = copy

    def run(self):
        if self.copy:
            return self.image.copy()
        return self.image


class CvCam(object):
    def __init__(self, image_w=160, image_h=120, image_d=3, iCam=0, warming_secs=5):
        self.width = image_w
        self.height = image_h
        self.depth = image_d

        self.frame = None
        self.cap = cv2.VideoCapture(iCam)

        # warm up until we get a frame or we timeout
        if self.cap is not None:
            # self.cap.set(3, image_w)
            # self.cap.set(4, image_h)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_h)
            logger.info('CvCam opened...')
            warming_time = time.time() + warming_secs  # quick after 5 seconds
            while self.frame is None and time.time() < warming_time:
                logger.info("...warming camera")
                self.run()
                time.sleep(0.2)

            if self.frame is None:
                raise CameraError("Unable to start CvCam.")
        else:
            raise CameraError("Unable to open CvCam.")

        self.running = True
        logger.info("CvCam ready.")

    def poll(self):
        if self.cap.isOpened():
            _, self.frame = self.cap.read()
            if self.frame is not None:
                width, height = self.frame.shape[:2]
                if width != self.width or height != self.height:
                    self.frame = cv2.resize(self.frame, (self.width, self.height))

    def update(self):
        '''
        poll the camera for a frame
        '''
        while self.running:
            self.poll()

    def run_threaded(self):
        return self.frame

    def run(self):
        self.poll()
        return self.frame

    def shutdown(self):
        self.running = False
        time.sleep(0.2)
        self.cap.release()


class CvImageView(object):

    def run(self, image):
        if image is None:
            return

        try:
            cv2.imshow('frame', image)
            cv2.waitKey(1)
        except:
            logger.error("Unable to open image window.")

    def shutdown(self):
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    import sys
    
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--camera", type=int, default=0,
                        help = "index of camera if using multiple cameras")
    parser.add_argument("-wd", "--width", type=int, default=160,
                        help = "width of image to capture")
    parser.add_argument("-ht", "--height", type=int, default=120,
                        help = "height of image to capture")
    parser.add_argument("-f", "--file", type=str,
                        help = "path to image file to user rather that a camera")
    parser.add_argument("-a", "--aug", required=True, type=str.upper, 
                        choices=['CROP', 'TRAPEZE',
                                 "RGB2HSV", "HSV2RGB", "RGB2BGR", "BGR2RGB", "BGR2HSV", "HSV2BRG",
                                 "RGB2GREY", "BGR2GREY", "HSV2GREY",
                                 "CANNY",
                                 "BLUR", "GBLUR",
                                 "RESIZE", "SCALE"],
                        help = "augmentation to apply")
    parser.add_argument("-l", "--left", type=int, default=0,
                        help="top left horizontal pixel index, defaults to zero")
    parser.add_argument("-lb", "--left-bottom", type=int, default=None,
                    help="bottom, left horizontal pixel index, defaults to zero")
    parser.add_argument("-r", "--right", type=int, default=None,
                        help="top, right horizontal pixel index, defaults to image width")
    parser.add_argument("-rb", "--right-bottom", type=int, default=None,
                    help="bottom, right horizontal pixel index, defaults to image width")
    parser.add_argument("-t", "--top", type=int, default=0,
                        help="top vertical pixel index, defaults to 0")
    parser.add_argument("-b", "--bottom", type=int, default=None,
                    help="bottom vertical pixel index, defaults to image height")
    parser.add_argument("-cl", "--canny-low", type=int, default=60,
                        help="Canny edge detection low threshold value of intensity gradient.")
    parser.add_argument("-ch", "--canny-high", type=int, default=110,
                        help="Canny edge detection high threshold value of intensity gradient.")
    parser.add_argument("-ca", "--canny-aperture", type=int, choices=[3, 5, 7], default=3,
                        help="Canny edge detect aperture in pixels")
    parser.add_argument("-gk", "--guassian-kernel", type=int, choices=[3, 5, 7, 9], default=3,
                        help="Guassian blue kernel size in pixels")
    parser.add_argument("-gky", "--guassian-kernel-y", type=int, choices=[3, 5, 7, 9],
                        help="Guassian blue kernel y size in pixels, defaults to a square kernel")
    parser.add_argument("-bk", "--blur-kernel", type=int, choices=[3, 5, 7, 9], default=3,
                        help="Guassian blue kernel size in pixels")
    parser.add_argument("-bky", "--blur-kernel-y", type=int, choices=[3, 5, 7, 9],
                        help="Simple blur kernel y size in pixels, defaults to a square kernel")
    parser.add_argument("-sw", "--scale", type=float,
                        help = "scale factor for image width")
    parser.add_argument("-sh", "--scale-height", type=float, 
                        help = "scale factor for image height.  Defaults to scale")

    #
    # setup augmentations
    #
    transformations = {}

    # Read arguments from command line
    args = parser.parse_args()
    
    image_source = None
    help = []
    if args.file is None:
        if args.camera < 0:
            help.append("-c/--camera must be >= 0")
        if args.width is None or args.width < 160:
            help.append("-wd/--width must be >= 160")
        if args.height is None or args.height < 120:
            help.append("-ht/--height must be >= 120")

    if "SCALE" == args.aug:
        if args.scale is None or args.scale <= 0:
            help.append("-sw/--scale must be > 0")
        elif args.scale_height is not None and args.scale_height <= 0:
            help.append("-sh/--scale height must be > 0")

    if "RESIZE" == args.aug:
        if args.width is None or args.width < 160:
            help.append("-wd/--width must be >= 160")
        if args.height is None or args.height < 120:
            help.append("-ht/--height must be >= 120")


    if len(help) > 0:
        parser.print_help()
        for h in help:
            print("  " + h)
        sys.exit(1)

    #
    # load file OR setup camera
    #
    cap = None
    width = None
    height = None
    depth = 3
    if args.file is not None:
        image_source = CvImgFromFile(args.file, image_w=args.width, image_h=args.height, copy=True)
        height, width, depth = image_shape(image_source.run())
    else:
        width = args.width
        height = args.height
        image_source = CvCam(image_w=width, image_h=height, iCam=args.camera)

    transformer = None
    transformation = args.aug

    #
    # masking tranformations
    #
    if "TRAPEZE" == transformation or "TRAPEZE_EDGE" == transformation or "CROP" == transformation: 
        #
        # masking transformations
        #
        if "TRAPEZE" == transformation:
            transformer = ImgTrapezoidalMask(
                args.left if args.left is not None else 0,
                args.right if args.right is not None else width,
                args.left_bottom if args.left_bottom is not None else 0,
                args.right_bottom if args.right_bottom is not None else width,
                args.top if args.top is not None else 0,
                args.bottom if args.bottom is not None else height
            )
        elif "TRAPEZE_EDGE" == transformation:
            transformer = ImgTrapezoidalEdgeMask(
                args.left if args.left is not None else 0,
                args.right if args.right is not None else width,
                args.left_bottom if args.left_bottom is not None else 0,
                args.right_bottom if args.right_bottom is not None else width,
                args.top if args.top is not None else 0,
                args.bottom if args.bottom is not None else height
            )
        else:
            transformer = ImgCropMask(
                args.left if args.left is not None else 0, 
                args.top if args.top is not None else 0, 
                args.right if args.right is not None else 0, 
                args.bottom if args.bottom is not None else 0)
    #
    # color space transformations
    #
    elif "RGB2BGR" == transformation:
        transformer = ImgRGB2BGR()
    elif "BGR2RGB" == transformation:
        transformer = ImgBGR2RGB()
    elif "RGB2HSV" == transformation:
        transformer = ImgRGB2HSV()
    elif "HSV2RGB" == transformation:
        transformer = ImgHSV2RGB()
    elif "BGR2HSV" == transformation:
        transformer = ImgBGR2HSV()
    elif "HSV2BGR" == transformation:
        transformer = ImgHSV2BGR()
    elif "RGB2GREY" == transformation:
        transformer = ImgRGB2GRAY()
    elif "BGR2GREY" == transformation:
        transformer = ImgBGR2GRAY()
    elif "HSV2GREY" == transformation:
        transformer = ImgHSV2GRAY()
    elif "CANNY" == transformation:
        # canny edge detection
        transformer = ImgCanny(args.canny_low, args.canny_high, args.canny_aperture)
    # 
    # blur transformations
    #
    elif "GBLUR" == transformation:
        transformer = ImgGaussianBlur(args.guassian_kernel, args.guassian_kernel_y)
    elif "BLUR" == transformation:
        transformer = ImgSimpleBlur(args.blur_kernel, args.blur_kernel_y)
    # 
    # resize transformations
    #
    elif "RESIZE" == transformation:
        transformer = ImageResize(args.width, args.height)
    elif "SCALE" == transformation:
        transformer = ImageScale(args.scale, args.scale_height)
    else:
        print("-a/--aug is not a valid augmentation")
        exit()

    # Creating a window for later use
    window_name = 'hsv_range_picker'
    cv2.namedWindow(window_name)

    while(1):

        frame = image_source.run()

        #
        # apply the augmentation
        #
        transformed_image = transformer.run(frame)

        #
        # show augmented image
        #
        cv2.imshow(window_name, transformed_image)

        k = cv2.waitKey(5) & 0xFF
        if k == ord('q') or k == ord('Q'):  # 'Q' or 'q'
            break
    
    if cap is not None:
        cap.release()

    cv2.destroyAllWindows()
