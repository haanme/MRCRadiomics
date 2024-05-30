import numpy as np
from dipy.align.reslice import reslice
import cv2

try:
    xrange
except NameError:
    xrange = range

"""
Resolve non-zero subregion in the image
@param img: mask image
@return x_lo, x_hi, y_lo, y_hi for x and y bounds
"""


def find_bounded_subregion3D2D(img):
    if len(img.shape) > 2:
        x_lo = 0
        for x in range(img.shape[0]):
            if np.max(img[x, :, :]) > 0:
                x_lo = x - 1
                if (x_lo < 0):
                    x_lo = 0
                break
        x_hi = img.shape[0] - 1
        for x in range(img.shape[0] - 1, -1, -1):
            if np.max(img[x, :, :]) > 0:
                x_hi = x + 1
            if (x_hi > img.shape[0] - 1):
                x_hi = img.shape[0] - 1
            break
        y_lo = 0
        for y in range(img.shape[1]):
            if np.max(img[:, y, :]) > 0:
                y_lo = y - 1
                if (y_lo < 0):
                    y_lo = 0
                break
        y_hi = img.shape[1] - 1
        for y in range(img.shape[1] - 1, -1, -1):
            if np.max(img[:, y, :]) > 0:
                y_hi = y + 1
                if (y_hi > img.shape[1] - 1):
                    y_hi = img.shape[1] - 1
                break
    else:
        x_lo = 0
        for x in range(img.shape[0]):
            if np.max(img[x, :]) > 0:
                x_lo = x - 1
                if (x_lo < 0):
                    x_lo = 0
                break
        x_hi = img.shape[0] - 1
        for x in range(img.shape[0] - 1, -1, -1):
            if np.max(img[x, :]) > 0:
                x_hi = x + 1
            if (x_hi > img.shape[0] - 1):
                x_hi = img.shape[0] - 1
            break
        y_lo = 0
        for y in range(img.shape[1]):
            if np.max(img[:, y]) > 0:
                y_lo = y - 1
                if (y_lo < 0):
                    y_lo = 0
                break
        y_hi = img.shape[1] - 1
        for y in range(img.shape[1] - 1, -1, -1):
            if np.max(img[:, y]) > 0:
                y_hi = y + 1
                if (y_hi > img.shape[1] - 1):
                    y_hi = img.shape[1] - 1
                break
    return x_lo, x_hi, y_lo, y_hi


"""
Resolve non-zero subregion in the image

@param img: 3D image data
@returns: [x_lo, x_hi, y_lo, y_hi, z_lo, z_hi] low and high bounds for bounding box
"""


def find_bounded_subregion3D(img):
    x_lo = 0
    for x in range(img.shape[0]):
        if np.max(img[x, :, :]) > 0:
            x_lo = x
            break
    x_hi = img.shape[0]-1
    for x in range(img.shape[0] - 1, -1, -1):
        if np.max(img[x, :, :]) > 0:
            x_hi = x
            break
    y_lo = 0
    for y in range(img.shape[1]):
        if np.max(img[:, y, :]) > 0:
            y_lo = y
            break
    y_hi = img.shape[1]-1
    for y in range(img.shape[1] - 1, -1, -1):
        if np.max(img[:, y, :]) > 0:
            y_hi = y
            break
    z_lo = 0
    for z in range(img.shape[2]):
        if np.max(img[:, :, z]) > 0:
            z_lo = z
            break
    z_hi = img.shape[2]-1
    for z in range(img.shape[2] - 1, -1, -1):
        if np.max(img[:, :, z]) > 0:
            z_hi = z
            break
    return x_lo, x_hi, y_lo, y_hi, z_lo, z_hi


"""
Resolve non-zero subregion in the image
@param slice2D: slice for finding subregio
"""


def find_bounded_subregion2D(slice2D):
    x_lo = 0
    for x in range(slice2D.shape[0]):
        if np.max(slice2D[x, :]) > 0:
            x_lo = x
            break
    x_hi = slice2D.shape[0] - 1
    for x in range(slice2D.shape[0] - 1, -1, -1):
        if np.max(slice2D[x, :]) > 0:
            x_hi = x
            break
    y_lo = 0
    for y in range(slice2D.shape[1]):
        if np.max(slice2D[:, y]) > 0:
            y_lo = y
            break
    y_hi = slice2D.shape[1] - 1
    for y in range(slice2D.shape[1] - 1, -1, -1):
        if np.max(slice2D[:, y]) > 0:
            y_hi = y
            break
    return x_lo, x_hi, y_lo, y_hi


"""
Apply 2D sliding window to image data
@param image: image whre window is applied
@param stepSize: step size to x and y directions, in pixels
@param windowSize: x and y size of window
@return x_lo, x_hi, y_lo, y_hi for x and y bounds
"""


def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in xrange(0, image.shape[0], stepSize):
        for x in xrange(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


"""
3D sliding window help function

@param image: Intensity image
@param mask: Masking window
@param stepSize: step size in voxels
@param windowSize: window size in voxels
@returns: yield for sliding window processing
"""


def sliding_window3D(image, mask, stepSize, windowSize):
    # slide a window across the image
    for z in range(0, image.shape[2], stepSize):
        if np.max(mask[:, :, z]) == 0:
            continue
        for y in range(0, image.shape[1], stepSize):
            if np.max(mask[:, y, z]) == 0:
                continue
            for x in range(0, image.shape[0], stepSize):
                # yield the current window
                yield x, y, z, image[x:x + windowSize[0], y:y + windowSize[1], z:z + windowSize[2]]
        print(('%d/%d Laws' % (z, image.shape[2])))


"""
Resolve non-zero subregion in the image

@param data: data to be resliced
@param orig_resolution: original data resolution in mm
@param new_resolution: new data resolution in mm
@param int_order: interpolation order
@returns: resliced image data, 4x4 matrix of resliced data
"""


def reslice_array(data, orig_resolution, new_resolution, int_order):
    zooms = orig_resolution
    new_zooms = (new_resolution[0], new_resolution[1], new_resolution[2])
    affine = np.eye(4)
    affine[0, 0] = orig_resolution[0]
    affine[1, 1] = orig_resolution[1]
    affine[2, 2] = orig_resolution[2]
    print(data.shape)
    print(zooms)
    print(new_zooms)
    data2, affine2 = reslice(data, affine, zooms, new_zooms, order=int_order)
    data3 = np.zeros((data2.shape[1], data2.shape[0], data2.shape[2]))
    for zi in range(data3.shape[2]):
        data3[:, :, zi] = np.rot90(data2[:, :, zi], k=3)
    return data3, affine2


"""
Creates cv2 image from 2D numpy array
"""
def make_cv2_slice2D(slice2D):
    # re-scale to 0..255
    slice2D -= np.min(slice2D)
    if (not (np.max(slice2D) == 0)):
        slice2D = (slice2D / np.max(slice2D)) * 255.0
    cvimg = np.transpose(cv2.resize(slice2D.astype(np.uint8), (slice2D.shape[0], slice2D.shape[1])))
    return cvimg