#!/usr/bin/env python

import os
import copy
from dipy.align.reslice import reslice
from scipy.signal import correlate2d
import scipy.ndimage
import skimage
from scipy import stats
import cv2
from skimage import measure
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.measure import regionprops
from scipy.stats import iqr
from skimage.filters import rank
from skimage.filters import scharr
from skimage.feature import peak_local_max
from skimage.filters import frangi, hessian
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import UnivariateSpline
import pywt
import visualizations

try:
    from skimage import filters
except ImportError:
    from skimage import filter as filters
try:
    xrange
except NameError:
    xrange = range

# Directory where result data are located
experiment_dir = ''
stat_funs = []
seg36_base_segnames = ['2a', '2p', '13asr', '13asl', '1a', '1ap', '1p', '7a', '7ap', '7p', '8a', '8p']
seg36_mid_segnames = ['4a', '4p', '14asr', '14asl', '3a', '3ap', '3p', '9a', '9ap', '9p', '10a', '10p']
seg36_apex_segnames = ['6a', '6p', '5a', '5ap', '5p', '15asr', '15asl', '11a', '11ap', '11p', '12a', '12p']
seg36_all_segnames = seg36_base_segnames + seg36_mid_segnames + seg36_apex_segnames


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def make_cv2_slice2D(slice2D):
    # re-scale to 0..255
    slice2D -= np.min(slice2D)
    if (not (np.max(slice2D) == 0)):
        slice2D = (slice2D / np.max(slice2D)) * 255.0
    cvimg = np.transpose(cv2.resize(slice2D.astype(np.uint8), (slice2D.shape[0], slice2D.shape[1])))
    return cvimg


# Resolve non-zero subregion in the image
def find_bounded_subregion2DBG(BG_roi, offset):
    x_lo = 0
    for x in range(BG_roi.shape[0]):
        if np.max(BG_roi[x, :]) > 0:
            x_lo = x - offset
            break
    x_hi = BG_roi.shape[0] - 1
    for x in range(BG_roi.shape[0] - 1, -1, -1):
        if np.max(BG_roi[x, :]) > 0:
            x_hi = x + offset
            break
    y_lo = 0
    for y in range(BG_roi.shape[1]):
        if np.max(BG_roi[:, y]) > 0:
            y_lo = y - offset
            break
    y_hi = BG_roi.shape[1] - 1
    for y in range(BG_roi.shape[1] - 1, -1, -1):
        if np.max(BG_roi[:, y]) > 0:
            y_hi = y + offset
            break
    return x_lo, x_hi, y_lo, y_hi


# Resolve non-zero subregion in the image
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


# Resolve non-zero subregion in the image
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


def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in xrange(0, image.shape[0], stepSize):
        for x in xrange(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


def reslice_array(data, orig_resolution, new_resolution, int_order):
    if len(data.shape) > 2:
        zooms = orig_resolution
        new_zooms = (new_resolution[0], new_resolution[1], new_resolution[2])
        affine = np.eye(4)
        affine[0, 0] = orig_resolution[0]
        affine[1, 1] = orig_resolution[1]
        affine[2, 2] = orig_resolution[2]
        data2, affine2 = reslice(data, affine, zooms, new_zooms, order=int_order)
        data3 = np.zeros((data2.shape[1], data2.shape[0], data2.shape[2]))
        for zi in range(data3.shape[2]):
            data3[:, :, zi] = np.rot90(data2[:, :, zi], k=3)
    else:
        zooms = orig_resolution
        new_zooms = (new_resolution[0], new_resolution[1], new_resolution[2])
        affine = np.eye(4)
        affine[0, 0] = orig_resolution[0]
        affine[1, 1] = orig_resolution[1]
        affine[2, 2] = orig_resolution[2]

        data_temp = np.zeros([data.shape[0], data.shape[1], 3])
        data_temp[:, :, 0] = data[:, :]
        data_temp[:, :, 1] = data[:, :]
        data_temp[:, :, 2] = data[:, :]
        data2, affine2 = reslice(data_temp, affine, zooms, new_zooms, order=int_order)
        data2 = data2[:, :, 1]
        data3 = np.zeros((data2.shape[1], data2.shape[0]))
        data3[:, :] = np.rot90(data2[:, :])

    return data3, affine2


def zernike_2D_slice(slicedata, s, n, m):
    from pyzernikemoment import Zernikemoment
    output = np.zeros_like(slicedata)
    mid = int(np.floor(s / 2.0))
    for (x, y, window) in sliding_window(slicedata, 1, (s, s)):
        if np.min(window) == np.max(window):
            continue
        val = filters.threshold_otsu(window)
        window2 = np.zeros_like(window)
        window2[window >= val] = 1
        Z, A, Phi = Zernikemoment(window2, n, m)
        xmid = x + mid
        ymid = y + mid
        if xmid >= output.shape[0]:
            xmid = output.shape[0] - 1
        if ymid >= output.shape[1]:
            ymid = output.shape[1] - 1
        output[xmid, ymid] = A
    return output


def zernike_2D(data, params, roidata):
    """
    [1] A. Tahmasbi, F. Saki, S. B. Shokouhi,
        Classification of Benign and Malignant Masses Based on Zernike Moments,
        Comput. Biol. Med., vol. 41, no. 8, pp. 726-735, 2011.

    [2] F. Saki, A. Tahmasbi, H. Soltanian-Zadeh, S. B. Shokouhi,
        Fast opposite weight learning rules with application in breast cancer
        diagnosis, Comput. Biol. Med., vol. 43, no. 1, pp. 32-41, 2013.
    """
    # Size of patch
    s = params[0]
    # n = The order of Zernike moment (scalar)
    n = params[1]
    # m = The repetition number of Zernike moment (scalar)
    m = params[2]

    # print('data.shape:' + str(data.shape))
    outdata = np.zeros(data.shape)
    for slice_i in range(data.shape[2]):
        if np.max(roidata[:, :, slice_i]) == 0:
            # print('Skipped [outside ROI] ' + str(slice_i+1) + '/' + str(data.shape[2]))    
            continue
        if len(data.shape) > 3:
            for t in range(data.shape[3]):
                slicedata = data[:, :, slice_i, t]
                Z_amplitude = zernike_2D_slice(slicedata, s, n, m)
                outdata[:, :, slice_i, t] = Z_amplitude.transpose()
        else:
            slicedata = data[:, :, slice_i]
            Z_amplitude = zernike_2D_slice(slicedata, s, n, m)
            outdata[:, :, slice_i] = Z_amplitude.transpose()
        # print('Filtered ' + str(slice_i+1) + '/' + str(data.shape[2]))
    # print(np.max(outdata))
    outdata = np.divide(outdata, np.max(outdata))
    # print('outdata.shape:' + str(outdata.shape))
    return [outdata]


def casefun_3D_2D_Zernike(LESIONDATAr, LESIONr, BG_roi, resolution, s, n, m):
    if np.max(LESIONr[0]) == 0:
        return float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float(
            'nan'), float('nan')
    Zdata = zernike_2D(LESIONDATAr, [s, n, m], LESIONr[0])
    ROIdata = Zdata[0][LESIONr[0] > 0]

    mean = np.mean(ROIdata)
    median = np.median(ROIdata)
    p25 = np.percentile(ROIdata, 25)
    p75 = np.percentile(ROIdata, 75)
    skewness = scipy.stats.skew(ROIdata)
    kurtosis = scipy.stats.kurtosis(ROIdata)
    SD = np.std(ROIdata)
    rng = np.max(ROIdata) - np.min(ROIdata)
    if not mean == 0:
        CV = SD / mean
    else:
        CV = 0.0
    return mean, median, p25, p75, skewness, kurtosis, SD, rng, CV


casefun_3D_2D_Zernike_9_8_8_names = (
'Z9_8_8_mean', 'Z9_8_8_median', 'Z9_8_8_25percentile', 'Z9_8_8_75percentile', 'Z9_8_8_skewness', 'Z9_8_8_kurtosis',
'Z9_8_8_SD', 'Z9_8_8_range', 'Z9_8_8_CV')


def casefun_3D_2D_Zernike_9_8_8(LESIONDATAr, LESIONr, BG_roi, resolution):
    return casefun_3D_2D_Zernike(LESIONDATAr, LESIONr, BG_roi, resolution, 9, 8, 8)


casefun_3D_2D_Zernike_15_5_5_names = (
'Z15_5_5_mean', 'Z15_5_5_median', 'Z15_5_5_25percentile', 'Z15_5_5_75percentile', 'Z15_5_5_skewness',
'Z15_5_5_kurtosis', 'Z15_5_5_SD', 'Z15_5_5_range', 'Z15_5_5_CV')


def casefun_3D_2D_Zernike_15_5_5(LESIONDATAr, LESIONr, BG_roi, resolution):
    return casefun_3D_2D_Zernike(LESIONDATAr, LESIONr, BG_roi, resolution, 15, 5, 5)


casefun_3D_2D_Zernike_15_6_6_names = (
'Z15_6_6_mean', 'Z15_6_6_median', 'Z15_6_6_25percentile', 'Z15_6_6_75percentile', 'Z15_6_6_skewness',
'Z15_6_6_kurtosis', 'Z15_6_6_SD', 'Z15_6_6_range', 'Z15_6_6_CV')


def casefun_3D_2D_Zernike_15_6_6(LESIONDATAr, LESIONr, BG_roi, resolution):
    return casefun_3D_2D_Zernike(LESIONDATAr, LESIONr, BG_roi, resolution, 15, 6, 6)


casefun_3D_2D_Zernike_17_6_6_names = (
'Z17_6_6_mean', 'Z17_6_6_median', 'Z17_6_6_25percentile', 'Z17_6_6_75percentile', 'Z17_6_6_skewness',
'Z17_6_6_kurtosis', 'Z17_6_6_SD', 'Z17_6_6_range', 'Z17_6_6_CV')


def casefun_3D_2D_Zernike_17_6_6(LESIONDATAr, LESIONr, BG_roi, resolution):
    return casefun_3D_2D_Zernike(LESIONDATAr, LESIONr, BG_roi, resolution, 17, 6, 6)


casefun_3D_2D_Zernike_19_6_6_names = (
'Z19_6_6_mean', 'Z19_6_6_median', 'Z19_6_6_25percentile', 'Z19_6_6_75percentile', 'Z19_6_6_skewness',
'Z19_6_6_kurtosis', 'Z19_6_6_SD', 'Z19_6_6_range', 'Z19_6_6_CV')


def casefun_3D_2D_Zernike_19_6_6(LESIONDATAr, LESIONr, BG_roi, resolution):
    return casefun_3D_2D_Zernike(LESIONDATAr, LESIONr, BG_roi, resolution, 19, 6, 6)


casefun_3D_2D_Zernike_21_8_8_names = (
'Z21_8_8_mean', 'Z21_8_8_median', 'Z21_8_8_25percentile', 'Z21_8_8_75percentile', 'Z21_8_8_skewness',
'Z21_8_8_kurtosis', 'Z21_8_8_SD', 'Z21_8_8_range', 'Z21_8_8_CV')


def casefun_3D_2D_Zernike_21_8_8(LESIONDATAr, LESIONr, BG_roi, resolution):
    return casefun_3D_2D_Zernike(LESIONDATAr, LESIONr, BG_roi, resolution, 21, 8, 8)


casefun_3D_2D_Zernike_25_12_12_names = (
'Z25_12_12_mean', 'Z25_12_12_median', 'Z25_12_12_25percentile', 'Z25_12_12_75percentile', 'Z25_12_12_skewness',
'Z25_12_12_kurtosis', 'Z25_12_12_SD', 'Z25_12_12_range', 'Z25_12_12_CV')


def casefun_3D_2D_Zernike_25_12_12(LESIONDATAr, LESIONr, BG_roi, resolution):
    return casefun_3D_2D_Zernike(LESIONDATAr, LESIONr, BG_roi, resolution, 25, 12, 12)


def wavelet_2D_slice4(slicedata, waveletname):
    s = 16
    output = np.zeros([slicedata.shape[0], slicedata.shape[1], 11])
    mid = int(np.floor(s / 2.0))
    for (x, y, window) in sliding_window(slicedata, 1, (s, s)):
        if np.min(window) == np.max(window):
            continue
        coeffs = pywt.wavedec2(window, waveletname, mode='periodization', level=4)
        xmid = x + mid
        ymid = y + mid
        if xmid >= slicedata.shape[0]:
            xmid = slicedata.shape[0] - 1
        if ymid >= slicedata.shape[1]:
            ymid = slicedata.shape[1] - 1
        output[xmid, ymid, 0] = coeffs[0][0][0]
        output[xmid, ymid, 1] = np.mean(np.abs(coeffs[1]))
        output[xmid, ymid, 2] = np.mean(np.abs(coeffs[2]))
        output[xmid, ymid, 3] = np.mean(np.abs(coeffs[3]))
        output[xmid, ymid, 4] = np.mean(np.abs(coeffs[4]))
        output[xmid, ymid, 5] = np.mean(np.abs(coeffs[4][0][0]))
        output[xmid, ymid, 5] = np.mean(np.abs(coeffs[4][0][0]))
        output[xmid, ymid, 7] = np.mean(np.abs(coeffs[4][1][0]))
        output[xmid, ymid, 9] = np.mean(np.abs(coeffs[4][2][0]))
        if len(coeffs[4][0]) > 1:
            output[xmid, ymid, 6] = np.mean(np.abs(coeffs[4][0][1]))
            output[xmid, ymid, 6] = np.mean(np.abs(coeffs[4][0][1]))
            output[xmid, ymid, 8] = np.mean(np.abs(coeffs[4][1][1]))
            output[xmid, ymid, 10] = np.mean(np.abs(coeffs[4][2][1]))
    return output


def wavelet_2D(data, roidata, waveletname):
    if len(data.shape) > 2:
        # print('data.shape:' + str(data.shape))
        outdata = np.zeros([data.shape[0], data.shape[1], data.shape[2], 11])
        for slice_i in range(data.shape[2]):
            if np.max(roidata[:, :, slice_i]) == 0:
                # print('Skipped [outside ROI] ' + str(slice_i+1) + '/' + str(data.shape[2]))    
                continue
            slicedata = data[:, :, slice_i]
            output = wavelet_2D_slice4(slicedata, waveletname)
            outdata[:, :, slice_i, :] = output
            # print('Filtered ' + str(slice_i+1) + '/' + str(data.shape[2]))
        # print('outdata.shape:' + str(outdata.shape))
    else:
        outdata = np.zeros([data.shape[0], data.shape[1], 1, 11])
        if not (np.max(roidata[:, :]) == 0):
            slicedata = data[:, :]
            output = wavelet_2D_slice4(slicedata, waveletname)
            outdata[:, :, 0, :] = output

    return outdata


def casefun_3D_2D_Wavelet(LESIONDATAr, LESIONr, BG_roi, resolution, params):
    # resolution factor affecting laws feature sampling ratio
    # 1: original resolution
    # <1: upsampling
    # >1: downsampling
    res_f = params[1]

    x_lo, x_hi, y_lo, y_hi = find_bounded_subregion2DBG(BG_roi, 10)

    if len(LESIONDATAr.shape) > 2:
        slices = LESIONDATAr.shape[2]
        LESIONDATArs = LESIONDATAr[x_lo:x_hi, y_lo:y_hi, :]
        LESIONrs_temp = LESIONr[0][x_lo:x_hi, y_lo:y_hi, :]
        BG_rois_temp = BG_roi[x_lo:x_hi, y_lo:y_hi, :]
    else:
        slices = 1
        LESIONDATArs = LESIONDATAr[x_lo:x_hi, y_lo:y_hi]
        LESIONrs_temp = LESIONr[0][x_lo:x_hi, y_lo:y_hi]
        BG_rois_temp = BG_roi[x_lo:x_hi, y_lo:y_hi]

    # Create masks and output data to desired resolution, intensity data is resliced later for non-zero only
    min_res = np.max(resolution)
    new_res = [min_res * res_f, min_res * res_f, min_res * res_f]
    LESIONrs_temp, affineLESIONrs_temp = reslice_array(LESIONrs_temp, resolution, new_res, 0)
    BG_rois_temp, affineBG_rois_temp = reslice_array(BG_rois_temp, resolution, new_res, 0)
    LESIONDATArs, affineLESIONDATArs = reslice_array(LESIONDATArs, resolution, new_res, 1)

    if np.max(LESIONr[0]) == 0:
        return float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float(
            'nan'), float('nan')

    outdata = wavelet_2D(LESIONDATAr, LESIONrs_temp, params[0])
    results = []
    for c_i in range(outdata.shape[3]):
        cframe = outdata[:, :, :, c_i]
        ROIdata = cframe[LESIONr[0] > 0]
        BGdata = cframe[BG_roi > 0]

        median = np.median(ROIdata)
        skewness = scipy.stats.skew(ROIdata)
        rng = np.max(ROIdata) - np.min(ROIdata)
        BGmedian = np.median(BGdata)
        BGskewness = scipy.stats.skew(BGdata)
        BG_roing = np.max(BGdata) - np.min(BGdata)
        BG_roimedian = abs(median) / ((median + BGmedian) / 2.0)
        BG_roiskewness = abs(skewness) / ((skewness + BGskewness) / 2.0)
        BG_roirng = abs(rng) / ((rng + BG_roing) / 2.0)
        results.append(median)
        results.append(skewness)
        results.append(rng)
        results.append(BGmedian)
        results.append(BGskewness)
        results.append(BG_roing)
        results.append(BG_roimedian)
        results.append(BG_roiskewness)
        results.append(BG_roirng)
    return results


casefun_3D_2D_Wavelet_names = ('median', 'skewness', 'range')
casefun_3D_2D_Wavelet_cnames = (
'L0c1', 'L1avg1', 'L1avg2', 'L1avg3', 'L1avg4', 'L2avg1', 'L2avg2', 'L2avg3', 'L2avg4', 'L2avg5', 'L2avg6')


def casefun_3D_2D_Wavelet_names_generator(params):
    names = []
    for cname in casefun_3D_2D_Wavelet_cnames:
        for name in casefun_3D_2D_Wavelet_names:
            names.append('UTUW%s_f%3.2f_%s_%s' % (params[0], params[1], name, cname))
        for name in casefun_3D_2D_Wavelet_names:
            names.append('UTUW%s_f%3.2f_BG%s_%s' % (params[0], params[1], name, cname))
        for name in casefun_3D_2D_Wavelet_names:
            names.append('UTUW%s_f%3.2f_BG_roi%s_%s' % (params[0], params[1], name, cname))
    return names


def calculate_2D_contour_measures(contours, resolution):
    """
    Hu moments 1-8
    Contour_arclength
    Contour_area
    Contour_convexitydefect_depth
    Contour_convexitydefect_area
    Contour_convexitydefect_length
    """
    if len(contours[1]) == 0:
        print('len(contours[1]) == 0')
        return None
    points = np.squeeze(np.array(contours[1][0]))
    if (len(points.shape) < 2):
        print('len(points.shape) < 2')
        return None
    if (points.shape[0] < 3):
        print('points.shape[0] < 3')
        return None
    contour = []
    for p in points:
        contour.append((p[0], p[1]))
    contour = np.asarray(contour)

    Contour_arclength = cv2.arcLength(contour, closed=True)
    Contour_area = cv2.contourArea(contour)
    ch = cv2.convexHull(contour, returnPoints=False)
    defects = cv2.convexityDefects(contour, ch)
    if defects is None:
        Contour_convexitydefect_depths = []
        Contour_convexitydefect_areas = []
        Contour_convexitydefect_lengths = []
    else:
        no_defects = len(defects)
        defects = np.squeeze(defects)
        if no_defects == 1:
            defects = [defects]
        Contour_convexitydefect_depths = []
        Contour_convexitydefect_areas = []
        Contour_convexitydefect_lengths = []
        for defect in defects:
            Contour_convexitydefect_depths.append(defect[3] * ((resolution[0] + resolution[1]) / 2.0))
            Ax = contour[defect[0]][0] * resolution[0]
            Ay = contour[defect[0]][1] * resolution[1]
            Bx = contour[defect[1]][0] * resolution[0]
            By = contour[defect[1]][1] * resolution[1]
            Cx = contour[defect[2]][0] * resolution[0]
            Cy = contour[defect[2]][1] * resolution[1]
            darea = abs((Ax * (Bx - Cy) + Bx * (Cy - Ay) + Cx * (Ay - By)) / 2.0)
            dlength = np.sqrt(np.power(Ax - Bx, 2.0) + np.power(Ay - By, 2.0))
            Contour_convexitydefect_areas.append(darea)
            Contour_convexitydefect_lengths.append(dlength)
    m = cv2.moments(contours[0])
    # Third order component
    # J. Flusser: "On the Independence of Rotation Moment Invariants", Pattern Recognition, vol. 33, pp. 1405-1410, 2000.
    Hu_invariant8 = m['m11'] * (np.power(m['m30'] + m['m12'], 2.0) - np.power(m['m03'] + m['m21'], 2.0)) - (
                m['m20'] - m['m02']) * (m['m30'] - m['m12']) * (m['m03'] - m['m21'])
    Humoments = cv2.HuMoments(m)
    return {'Humoments': Humoments, 'Hu_invariant8': Hu_invariant8,
            'Contour_arclength': Contour_arclength, 'Contour_area': Contour_area,
            'Contour_convexitydefect_areas': Contour_convexitydefect_areas,
            'Contour_convexitydefect_lengths': Contour_convexitydefect_lengths,
            'Contour_convexitydefect_depths': Contour_convexitydefect_depths}


"""
Hu, M.K., 1962. Visual pattern recognition by moment invariants. IRE transactions on information theory, 8(2), pp.179-187.
"""
contour2D_names = []
contour2D_names.append('Hu_moment_invariants_1_SD_per_mean')
contour2D_names.append('Hu_moment_invariants_2_SD_per_mean')
contour2D_names.append('Hu_moment_invariants_3_SD_per_mean')
contour2D_names.append('Hu_moment_invariants_4_SD_per_mean')
contour2D_names.append('Hu_moment_invariants_5_SD_per_mean')
contour2D_names.append('Hu_moment_invariants_6_SD_per_mean')
contour2D_names.append('Hu_moment_invariants_7_SD_per_mean')
contour2D_names.append('Hu_moment_invariants_8_SD_per_mean')
contour2D_names.append('2D_contour_arclengths_mean')
contour2D_names.append('2D_contour_arclengths_median')
contour2D_names.append('2D_contour_arclengths_SD')
contour2D_names.append('2D_contour_arclengths_IQR')
contour2D_names.append('2D_contour_areas_mean')
contour2D_names.append('2D_contour_areas_median')
contour2D_names.append('2D_contour_areas_SD')
contour2D_names.append('2D_contour_areas_IQR')
contour2D_names.append('2D_number_of_contour_convexity_defects')
contour2D_names.append('2D_contour_convexity_defect_areas_mean')
contour2D_names.append('2D_contour_convexity_defect_areas_median')
contour2D_names.append('2D_contour_convexity_defect_areas_SD')
contour2D_names.append('2D_contour_convexity_defect_areas_IQR')
contour2D_names.append('2D_contour_convexity_defect_lengths_mean')
contour2D_names.append('2D_contour_convexity_defect_lengths_median')
contour2D_names.append('2D_contour_convexity_defect_lengths_SD')
contour2D_names.append('2D_contour_convexity_defect_lengths_IQR')
contour2D_names.append('2D_contour_convexity_defect_depths_mean')
contour2D_names.append('2D_contour_convexity_defect_depths_median')
contour2D_names.append('2D_contour_convexity_defect_depths_SD')
contour2D_names.append('2D_contour_convexity_defect_depths_IQR')
contour2D_names.append('2D_mean_contour_convexity_defect_lengths_proportional_to_arclength')
contour2D_names.append('2D_median_contour_convexity_defect_lengths_proportional_to_arclength')
contour2D_names.append('2D_SD_contour_convexity_defect_lengths_proportional_to_arclength')
contour2D_names.append('2D_IQR_contour_convexity_defect_lengths_proportional_to_arclength')
contour2D_names.append('2D_mean_contour_convexity_defect_depths_proportional_to_area')
contour2D_names.append('2D_median_contour_convexity_defect_depths_proportional_to_area')
contour2D_names.append('2D_SD_contour_convexity_defect_depths_proportional_to_area')
contour2D_names.append('2D_IQR_contour_convexity_defect_depths_proportional_to_area')


def subfun_3D_2D_Hu_moments(LESIONDATAr, LESIONr, labelimage, resolution, params):
    print((LESIONDATAr.shape, LESIONr.shape, labelimage.shape))
    # Resolve which directions to be calculated
    Contour_measures = []
    if labelimage.shape[1] == labelimage.shape[2]:
        resolution2D = [resolution[1], resolution[2]]
        for x in range(labelimage.shape[0]):
            if len(np.unique(labelimage[x, :, :])) < 2:
                continue
            cvimg = make_cv2_slice2D(labelimage[x, :, :]).copy()
            contours = cv2.findContours(cvimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_measures = calculate_2D_contour_measures(contours, resolution2D)
            if contour_measures is not None:
                Contour_measures.append(contour_measures)
    if labelimage.shape[0] == labelimage.shape[2]:
        resolution2D = [resolution[0], resolution[2]]
        for y in range(labelimage.shape[1]):
            if len(np.unique(labelimage[:, y, :])) < 2:
                continue
            cvimg = make_cv2_slice2D(labelimage[:, y, :]).copy()
            contours = cv2.findContours(cvimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_measures = calculate_2D_contour_measures(contours, resolution2D)
            if contour_measures is not None:
                Contour_measures.append(contour_measures)
    if labelimage.shape[0] == labelimage.shape[1]:
        resolution2D = [resolution[0], resolution[1]]
        for z in range(labelimage.shape[2]):
            if len(np.unique(labelimage[:, :, z])) < 2:
                continue
            cvimg = make_cv2_slice2D(labelimage[:, :, z]).copy()
            contours = cv2.findContours(cvimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_measures = calculate_2D_contour_measures(contours, resolution2D)
            if contour_measures is not None:
                if 'write_visualization' in params[-1]:
                    LESIONDATAr_cvimg = make_cv2_slice2D(LESIONDATAr[:, :, z]).copy()
                    LESIONr_cvimg = make_cv2_slice2D(LESIONr[:, :, z]).copy()
                    basename = params[-1]['name'] + '_2D_curvature_' + str(params[:-1]).replace(' ',
                                                                                                '_') + '_slice' + str(z)
                    visualizations.write_slice2D(LESIONDATAr_cvimg,
                                                 params[-1]['write_visualization'] + os.sep + basename + '_data.tiff')
                    visualizations.write_slice2D_ROI(LESIONDATAr_cvimg, LESIONr_cvimg, params[-1][
                        'write_visualization'] + os.sep + basename + '_lesion.tiff', 0.4)
                    visualizations.write_slice2D_polygon(LESIONDATAr_cvimg, np.squeeze(np.array(contours[1][0])),
                                                         params[-1][
                                                             'write_visualization'] + os.sep + basename + '_contour.tiff')
                Contour_measures.append(contour_measures)

    subfun_3D_2D_Hu_moment_suffixes = ['Hu_Inv_1', 'Hu_Inv_1', 'Hu_Inv_1', 'Hu_Inv_1', 'Hu_Inv_1', 'Hu_Inv_1',
                                       'Hu_Inv_1']
    r = []
    for i in range(16):
        r.append([])
    for measures in Contour_measures:
        # Hu moments 1-8
        r[0].append(measures['Humoments'][0])
        r[1].append(measures['Humoments'][1])
        r[2].append(measures['Humoments'][2])
        r[3].append(measures['Humoments'][3])
        r[4].append(measures['Humoments'][4])
        r[5].append(measures['Humoments'][5])
        r[6].append(measures['Humoments'][6])
        r[7].append(measures['Hu_invariant8'])
        # Contour arclength
        r[8].append(measures['Contour_arclength'])
        # Contour area
        r[9].append(measures['Contour_area'])
        # Number of convexity defects
        if len(measures['Contour_convexitydefect_areas']) > 0:
            r[10].append(len(measures['Contour_convexitydefect_areas']))
            # Convexity defect areas
            r[11] = r[11] + measures['Contour_convexitydefect_areas']
            # Convexity defect lengths
            r[12] = r[12] + measures['Contour_convexitydefect_lengths']
            # Convexity defect depths
            r[13] = r[13] + measures['Contour_convexitydefect_depths']
            # Convexity defect length proportion to arclength
            r[14].append(np.sum(measures['Contour_convexitydefect_lengths']) / measures['Contour_arclength'])
            # Convexity defect depth proportion to area
            r[15].append(np.mean(measures['Contour_convexitydefect_depths']) / measures['Contour_area'])
    if len(Contour_measures) > 0:
        r1 = np.std(r[0]) / np.mean(r[0])
        r2 = np.std(r[1]) / np.mean(r[1])
        r3 = np.std(r[2]) / np.mean(r[2])
        r4 = np.std(r[3]) / np.mean(r[3])
        r5 = np.std(r[4]) / np.mean(r[4])
        r6 = np.std(r[5]) / np.mean(r[5])
        r7 = np.std(r[6]) / np.mean(r[6])
        r8 = np.std(r[7]) / np.mean(r[7])
        r10 = np.mean(r[8])
        r11 = np.median(r[8])
        r12 = np.std(r[8])
        r13 = iqr(r[8])
        r20 = np.mean(r[9])
        r21 = np.median(r[9])
        r22 = np.std(r[9])
        r23 = iqr(r[9])
    else:
        r1 = 0
        r2 = 0
        r3 = 0
        r4 = 0
        r5 = 0
        r6 = 0
        r7 = 0
        r8 = 0
        r10 = 0
        r11 = 0
        r12 = 0
        r13 = 0
        r20 = 0
        r21 = 0
        r22 = 0
        r23 = 0
    if len(Contour_measures) > 0:
        if len(r[10]) > 0:
            r30 = np.mean(r[10])
        else:
            r30 = 0
        if len(r[11]) > 0:
            r31 = np.mean(r[11])
            r32 = np.median(r[11])
            r33 = np.std(r[11])
            r34 = iqr(r[11])
        else:
            r31 = 0
            r32 = 0
            r33 = 0
            r34 = 0
        if len(r[12]) > 0:
            r40 = np.mean(r[12])
            r41 = np.median(r[12])
            r42 = np.std(r[12])
            r43 = iqr(r[12])
        else:
            r40 = 0
            r41 = 0
            r42 = 0
            r43 = 0
        if len(r[13]) > 0:
            r50 = np.mean(r[13])
            r51 = np.median(r[13])
            r52 = np.std(r[13])
            r53 = iqr(r[13])
        else:
            r50 = 0
            r51 = 0
            r52 = 0
            r53 = 0
        if len(r[14]) > 0:
            r60 = np.mean(r[14])
            r61 = np.median(r[14])
            r62 = np.std(r[14])
            r63 = iqr(r[14])
        else:
            r60 = 0
            r61 = 0
            r62 = 0
            r63 = 0
    else:
        r30 = 0
        r31 = 0
        r32 = 0
        r33 = 0
        r34 = 0
        r40 = 0
        r41 = 0
        r42 = 0
        r43 = 0
        r50 = 0
        r51 = 0
        r52 = 0
        r53 = 0
        r60 = 0
        r61 = 0
        r62 = 0
        r63 = 0
    if len(Contour_measures) > 0 and len(r[15]) > 0:
        print(r[15])
        r70 = np.mean(r[15])
        r71 = np.median(r[15])
        r72 = np.std(r[15])
        r73 = iqr(r[15])
    else:
        r70 = 0
        r71 = 0
        r72 = 0
        r73 = 0
    return r1, r2, r3, r4, r5, r6, r7, r8, r10, r11, r12, r13, r20, r21, r22, r23, r30, r31, r32, r33, r34, r40, r41, r42, r43, r50, r51, r52, r53, r60, r61, r62, r63, r70, r71, r72, r73


"""
Hu, M.K., 1962. Visual pattern recognition by moment invariants. IRE transactions on information theory, 8(2), pp.179-187.
"""
casefun_3D_2D_Hu_moments_rawintensity_names = []
for name in contour2D_names:
    casefun_3D_2D_Hu_moments_rawintensity_names.append('2D_curvature_raw_intensity_' + name)


def casefun_3D_2D_Hu_moments_rawintensity(LESIONDATAr, LESIONr, BG_roi, resolution, params):
    if np.max(LESIONDATAr) == 0:
        return [float('nan') for x in casefun_3D_2D_Hu_moments_rawintensity_names]
    labelimage = copy.deepcopy(LESIONDATAr)
    labelimage[LESIONr[0] == 0] = 0
    return subfun_3D_2D_Hu_moments(copy.deepcopy(LESIONDATAr), copy.deepcopy(LESIONr[0]), labelimage, resolution,
                                   ['raw'] + params)


casefun_3D_2D_Hu_moments_gradient_names = []
for name in contour2D_names:
    casefun_3D_2D_Hu_moments_gradient_names.append('2D_curvature_gradient_' + name)


def casefun_3D_2D_Hu_moments_gradient(LESIONDATAr, LESIONr, BG_roi, resolution, params):
    if np.max(LESIONDATAr) == 0:
        return [float('nan') for x in casefun_3D_2D_Hu_moments_gradient_names]
    x_lo, x_hi, y_lo, y_hi = find_bounded_subregion3D2D(LESIONDATAr)
    LESIONDATArs = LESIONDATAr[x_lo:x_hi, y_lo:y_hi, :]
    LESIONrs = LESIONr[0][x_lo:x_hi, y_lo:y_hi, :]
    labelimage = np.zeros_like(LESIONDATArs)
    for slice_i in range(LESIONDATArs.shape[2]):
        img_slice = LESIONDATArs[:, :, slice_i]
        cvimg = cv2.resize(img_slice.astype(np.float), (img_slice.shape[0], img_slice.shape[1]))
        laplacian = cv2.Laplacian(cvimg, cv2.CV_64F)
        laplacian[LESIONrs[:, :, slice_i] == 0] = 0
        labelimage[:, :, slice_i] = laplacian
    return subfun_3D_2D_Hu_moments(copy.deepcopy(LESIONDATArs), copy.deepcopy(LESIONrs), labelimage, resolution,
                                   ['gradient'] + params)


casefun_3D_2D_Hu_moments_2bins_names = []
for name in contour2D_names:
    casefun_3D_2D_Hu_moments_2bins_names.append('2D_curvature_2_bins_histogram_' + name)
casefun_3D_2D_Hu_moments_3bins_names = []
for name in contour2D_names:
    casefun_3D_2D_Hu_moments_3bins_names.append('2D_curvature_3_bins_histogram_' + name)
casefun_3D_2D_Hu_moments_4bins_names = []
for name in contour2D_names:
    casefun_3D_2D_Hu_moments_4bins_names.append('2D_curvature_4_bins_histogram_' + name)


def casefun_3D_2D_Hu_moments_bins(LESIONDATAr, LESIONr, BG_roi, resolution, params):
    no_bins = params[0]
    if np.max(LESIONDATAr) == 0:
        return [float('nan') for x in casefun_3D_2D_Hu_moments_2bins_names]
    x_lo, x_hi, y_lo, y_hi = find_bounded_subregion3D2D(LESIONDATAr)
    ROIdata = LESIONDATAr[LESIONr[0] > 0]
    hist, bin_edges = np.histogram(ROIdata, bins=no_bins)
    LESIONDATArs = LESIONDATAr[x_lo:x_hi, y_lo:y_hi, :]
    LESIONrs = LESIONr[0][x_lo:x_hi, y_lo:y_hi, :]
    labelimage = np.zeros_like(LESIONDATArs)
    for bin_edge_i in range(1, len(bin_edges)):
        labelimage = np.where(LESIONDATArs < bin_edges[bin_edge_i], bin_edge_i, labelimage)
    labelimage[LESIONrs == 0] = 0
    return subfun_3D_2D_Hu_moments(copy.deepcopy(LESIONDATArs), copy.deepcopy(LESIONrs), labelimage, resolution,
                                   ['bins'] + params)


subfun_3D_2D_local_binary_pattern_names = ['mean', 'median', 'p25', 'p75', 'skewness', 'kurtosis', 'SD', 'IQR',
                                           'meanBG', 'medianBG', 'Cmedian', 'Cmean', 'CNR']


def subfun_3D_2D_local_binary_pattern(LESIONDATAr, LESIONr, BG_roi, params):
    """
    [R387388]	Multiresolution Gray-Scale and Rotation Invariant Texture Classification with Local Binary Patterns. Timo Ojala, Matti Pietikainen, Topi Maenpaa. http://www.rafbis.it/biplab15/images/stories/docenti/Danielriccio/Articoliriferimento/LBP.pdf, 2002.
    [R388388]	(1, 2) Face recognition with local binary patterns. Timo Ahonen, Abdenour Hadid, Matti Pietikainen, http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.214.6851, 2004.
    """
    # print('LBP')
    # Number of circularly symmetric neighbour set points (quantization of the angular space)
    angles = params[0]
    # Radius of circle (spatial resolution of the operator)
    radius = params[1]
    outdata = np.zeros_like(LESIONDATAr)
    # print('LBP')
    for slice_i in range(LESIONDATAr.shape[2]):
        slicedata = LESIONDATAr[:, :, slice_i]
        # print(len(slicedata[LESIONr[0][:, :, slice_i] > 0]))
        # print(len(np.unique(slicedata[LESIONr[0][:, :, slice_i] > 0])))
        lpb = local_binary_pattern(slicedata, angles, radius, method='uniform')
        outdata[:, :, slice_i] = lpb
    ROIdata = outdata[LESIONr[0] > 0]
    if (len(ROIdata)) == 0:
        return float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float(
            'nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan')
    BGdata = outdata[BG_roi > 0]
    mean = np.mean(ROIdata)
    median = np.median(ROIdata)
    meanBG = np.mean(BGdata)
    medianBG = np.median(BGdata)
    SD = np.std(ROIdata)
    SDBG = np.std(ROIdata)
    p25 = np.percentile(ROIdata, 25)
    p75 = np.percentile(ROIdata, 75)
    skewness = scipy.stats.skew(ROIdata)
    kurtosis = scipy.stats.kurtosis(ROIdata)
    SD = np.std(ROIdata)
    IQR = np.max(ROIdata) - np.min(ROIdata)
    if not mean == 0:
        CV = SD / mean
    else:
        CV = 0.0
    Cmedian = median - medianBG
    Cmean = mean - meanBG
    CNR = abs(Cmean) / ((SD + SDBG) / 2.0)
    return mean, median, p25, p75, skewness, kurtosis, SD, IQR, meanBG, medianBG, Cmedian, Cmean, CNR


casefun_3D_2D_local_binary_pattern_41_names = []
for name in subfun_3D_2D_local_binary_pattern_names:
    casefun_3D_2D_local_binary_pattern_41_names.append('LPB41_' + name)


def casefun_3D_2D_local_binary_pattern_41(LESIONDATAr, LESIONr, BG_roi, resolution):
    return subfun_3D_2D_local_binary_pattern(LESIONDATAr, LESIONr, BG_roi, [4, 1])


casefun_3D_2D_local_binary_pattern_81_names = []
for name in subfun_3D_2D_local_binary_pattern_names:
    casefun_3D_2D_local_binary_pattern_81_names.append('LPB81_' + name)


def casefun_3D_2D_local_binary_pattern_81(LESIONDATAr, LESIONr, BG_roi, resolution):
    return subfun_3D_2D_local_binary_pattern(LESIONDATAr, LESIONr, BG_roi, [8, 1])


casefun_3D_2D_local_binary_pattern_42_names = []
for name in subfun_3D_2D_local_binary_pattern_names:
    casefun_3D_2D_local_binary_pattern_42_names.append('LPB42_' + name)


def casefun_3D_2D_local_binary_pattern_42(LESIONDATAr, LESIONr, BG_roi, resolution):
    return subfun_3D_2D_local_binary_pattern(LESIONDATAr, LESIONr, BG_roi, [4, 2])


casefun_3D_2D_local_binary_pattern_82_names = []
for name in subfun_3D_2D_local_binary_pattern_names:
    casefun_3D_2D_local_binary_pattern_82_names.append('LPB82_' + name)


def casefun_3D_2D_local_binary_pattern_82(LESIONDATAr, LESIONr, BG_roi, resolution):
    return subfun_3D_2D_local_binary_pattern(LESIONDATAr, LESIONr, BG_roi, [8, 2])


casefun_3D_2D_local_binary_pattern_43_names = []
for name in subfun_3D_2D_local_binary_pattern_names:
    casefun_3D_2D_local_binary_pattern_43_names.append('LPB43_' + name)


def casefun_3D_2D_local_binary_pattern_43(LESIONDATAr, LESIONr, BG_roi, resolution):
    return subfun_3D_2D_local_binary_pattern(LESIONDATAr, LESIONr, BG_roi, [4, 3])


casefun_3D_2D_local_binary_pattern_83_names = []
for name in subfun_3D_2D_local_binary_pattern_names:
    casefun_3D_2D_local_binary_pattern_83_names.append('LPB83_' + name)


def casefun_3D_2D_local_binary_pattern_83(LESIONDATAr, LESIONr, BG_roi, resolution):
    return subfun_3D_2D_local_binary_pattern(LESIONDATAr, LESIONr, BG_roi, [8, 3])


subfun_3D_2D_gabor_filter_names = ['mean', 'median', 'p25', 'p75', 'skewness', 'kurtosis', 'SD', 'IQR', 'meanBG',
                                   'medianBG', 'Cmedian', 'Cmean', 'CNR']


def subfun_3D_2D_gabor_filter(data, LESIONr, BG_roi, params):
    """
    2D Gabor filter using skimage python package
    params[0] frequency of filter at index 0
    params[1] directions
    params[2] kernel size in SD units
    :return: real component of filtered data
    """
    outdata = np.zeros_like(data)
    d = params[1]
    radians = [x * np.pi / d * 0.5 for x in range(d + 1)]
    if len(data.shape) > 2:
        slices = data.shape[2]
    else:
        slices = 1
    for slice_i in range(slices):
        if len(data.shape) > 3:
            for t in range(data.shape[3]):
                slicedata = data[:, :, slice_i, t]
                filt_reals = np.zeros([data.shape[0], data.shape[1], len(radians)])
                for r_i in range(len(radians)):
                    filt_real, filt_imag = skimage.filters.gabor(slicedata, frequency=params[0], theta=radians[r_i],
                                                                 n_stds=params[2])
                    filt_reals[:, :, r_i] = filt_real
                outdata[:, :, slice_i, t] = np.mean(filt_reals, 2)
        else:
            if slices == 1:
                slicedata = data[:, :]
            else:
                slicedata = data[:, :, slice_i]
            filt_reals = np.zeros([data.shape[0], data.shape[1], len(radians)])
            for r_i in range(len(radians)):
                filt_real, filt_imag = skimage.filters.gabor(slicedata, frequency=params[0], theta=radians[r_i],
                                                             n_stds=params[2])
                filt_reals[:, :, r_i] = filt_real
            if slices == 1:
                outdata[:, :] = np.mean(filt_reals, 2)
            else:
                outdata[:, :, slice_i] = np.mean(filt_reals, 2)
            if (type(params) == list) and (len(params) > 1) and (not type(params[-1]) == int) and (
                    'write_visualization' in params[-1]):
                if not params[0] == 1.0:
                    continue
                if not params[1] == 2:
                    continue
                if not params[2] == 2:
                    continue
                LESIONDATAr_cvimg = make_cv2_slice2D(slicedata).copy()
                LESIONr_cvimg = make_cv2_slice2D(LESIONr[0][:, :, slice_i]).copy()
                BG_roi_cvimg = make_cv2_slice2D(BG_roi[:, :, slice_i]).copy()
                outdata_cvimg = make_cv2_slice2D(outdata[:, :, slice_i]).copy()
                if (np.max(LESIONr_cvimg) == 0):
                    continue
                basename = params[-1]['name'] + '_Gabor_' + str(params[:-1]).replace(' ', '_') + '_slice' + str(slice_i)
                visualizations.write_slice2D(LESIONDATAr_cvimg,
                                             params[-1]['write_visualization'] + os.sep + basename + '_data.tiff')
                visualizations.write_slice2D_ROI(LESIONDATAr_cvimg, LESIONr_cvimg,
                                                 params[-1]['write_visualization'] + os.sep + basename + '_lesion.tiff',
                                                 0.4)
                visualizations.write_slice2D_ROI_color(LESIONDATAr_cvimg, LESIONr_cvimg, outdata_cvimg, params[-1][
                    'write_visualization'] + os.sep + basename + '_overlay.tiff', 0.8)
                visualizations.write_slice2D_ROI_color(LESIONDATAr_cvimg, BG_roi_cvimg, outdata_cvimg, params[-1][
                    'write_visualization'] + os.sep + basename + '_BGoverlay.tiff', 0.7)

    ROIdata = outdata[LESIONr[0] > 0]
    BGdata = outdata[BG_roi > 0]
    mean = np.mean(ROIdata)
    median = np.median(ROIdata)
    meanBG = np.mean(BGdata)
    medianBG = np.median(BGdata)
    SD = np.std(ROIdata)
    SDBG = np.std(ROIdata)
    p25 = np.percentile(ROIdata, 25)
    p75 = np.percentile(ROIdata, 75)
    skewness = scipy.stats.skew(ROIdata)
    kurtosis = scipy.stats.kurtosis(ROIdata)
    SD = np.std(ROIdata)
    IQR = np.max(ROIdata) - np.min(ROIdata)
    if not mean == 0:
        CV = SD / mean
    else:
        CV = 0.0
    Cmedian = median - medianBG
    Cmean = mean - meanBG
    CNR = abs(Cmean) / ((SD + SDBG) / 2.0)
    return mean, median, p25, p75, skewness, kurtosis, SD, IQR, meanBG, medianBG, Cmedian, Cmean, CNR


def casefun_3D_2D_gabor_filter_name_generator(params):
    names = []
    for name in subfun_3D_2D_gabor_filter_names:
        names.append('UTUGabor_f%3.2f_d%d_k%d_%s' % (params[0], params[1], params[2], name))
    return names


def casefun_3D_2D_gabor_filter(LESIONDATAr, LESIONr, BG_roi, resolution, params):
    return subfun_3D_2D_gabor_filter(LESIONDATAr, LESIONr, BG_roi, params)


# Generic 2D filters applied to 3D
def subfun_3D_2D_stat1(LESIONDATAr, LESIONr, BG_roi, fun2D, params):
    # Number of circularly symmetric neighbour set points (quantization of the angular space)
    angles = params[0]
    # Radius of circle (spatial resolution of the operator)
    radius = params[1]
    outdata = np.zeros_like(LESIONDATAr)
    for slice_i in range(LESIONDATAr.shape[2]):
        slicedata = LESIONDATAr[:, :, slice_i]
        outdata[:, :, slice_i] = fun2D(slicedata, params)
    ROIdata = outdata[LESIONr[0] > 0]
    BGdata = outdata[BG_roi > 0]
    mean = np.mean(ROIdata)
    median = np.median(ROIdata)
    meanBG = np.mean(BGdata)
    medianBG = np.median(BGdata)
    SD = np.std(ROIdata)
    SDBG = np.std(ROIdata)
    p25 = np.percentile(ROIdata, 25)
    p75 = np.percentile(ROIdata, 75)
    skewness = scipy.stats.skew(ROIdata)
    kurtosis = scipy.stats.kurtosis(ROIdata)
    SD = np.std(ROIdata)
    IQR = np.max(ROIdata) - np.min(ROIdata)
    if not mean == 0:
        CV = SD / mean
    else:
        CV = 0.0
    Cmedian = median - medianBG
    Cmean = mean - meanBG
    CNR = abs(Cmean) / ((SD + SDBG) / 2.0)
    return mean, median, p25, p75, skewness, kurtosis, SD, IQR, meanBG, medianBG, Cmedian, Cmean, CNR


casefun_3D_2D_stat1_names = ['mean', 'median', 'p25', 'p75', 'skewness', 'kurtosis', 'SD', 'IQR', 'meanBG', 'medianBG',
                             'Cmedian', 'Cmean', 'CNR']


def casefun_3D_2D_stat1_names_generator(param_strs):
    names = []
    for name in casefun_3D_2D_stat1_names:
        suffix = ''
        for param in param_strs:
            suffix += ('_%s' % param)
        names.append('UTU3D2D%s_%s' % (suffix, name))
    return names


"""
Harris corner detection
Harris, C. and Stephens, M., 1988, August. A combined corner and edge detector. In Alvey vision conference (Vol. 15, No. 50, pp. 10-5244).
blockSize - Neighborhood size (see the details on cornerEigenValsAndVecs()).
ksize - Aperture parameter for the Sobel() operator.
k - Harris detector free parameter.
"""
subfun_3D_2D_Harris_names = (
'No_corners_ROI', 'No_corners_BG', 'No_corners_ratio', 'Corner_density_primary', 'Corner_density_secondary',
'Corner_density_mean', 'Corner_density_ratio', 'Corner_density_ratio_overall')


def casefun_3D_2D_Harris_name_generator(params):
    names = []
    for name in subfun_3D_2D_Harris_names:
        names.append('UTUHarris_b%d_ks%d_k%3.2f_%s' % (params[0], params[1], params[2], name))
    return names


def casefun_3D_2D_Harris(LESIONDATAr, LESIONr, BG_roi, resolution, params):
    blockSize = int(np.round(params[0] / np.mean([resolution[0], resolution[1]])))
    # print('Harris effective block size:' + str(blockSize))
    ksize = params[1]
    k = params[2]
    No_corners_ROI = 0
    No_corners_BG = 0
    Densities_ROI_primary = []
    Densities_ROI_secondary = []
    Densities_ROI_mean = []
    Densities_BG_mean = []
    Densities_ROI_ratio = []
    if (len(LESIONDATAr.shape) > 2):
        slices = LESIONDATAr.shape[2]
    else:
        slices = 1
    for slice_i in range(slices):
        if (slices == 1):
            LS = LESIONr[0][:, :]
            BG = BG_roi[:, :]
        else:
            LS = LESIONr[0][:, :, slice_i]
            BG = BG_roi[:, :, slice_i]
        if (np.max(LS) == 0 and np.max(BG) == 0):
            continue
        if (slices == 1):
            cvimg = make_cv2_slice2D(LESIONDATAr[:, :])
        else:
            cvimg = make_cv2_slice2D(LESIONDATAr[:, :, slice_i])
        edgemap = abs(cv2.cornerHarris(cvimg, blockSize, ksize, k))
        ROIdata = copy.deepcopy(edgemap)
        ROIdata[LS == 0] = 0
        locs_ROI = peak_local_max(ROIdata, min_distance=1, threshold_abs=0.0)
        No_corners_ROI += len(locs_ROI)
        BGdata = copy.deepcopy(edgemap)
        BGdata[BG == 0] = 0
        BGdata[LS > 0] = 0
        locs_BG = peak_local_max(BGdata, min_distance=1)
        No_corners_BG += len(locs_BG)
        if locs_ROI.shape[0] > 3:
            ROI_density = 1
        else:
            ROI_density = 0
        if locs_BG.shape[0] > 3:
            BG_density = 1
        else:
            BG_density = 0
        if ROI_density == 1:
            x = []
            y = []
            for p in locs_ROI:
                x.append(p[0])
                y.append(p[1])
            try:
                kernel = stats.gaussian_kde(np.vstack([x, y]))
                kernel = kernel.covariance * kernel.factor
                covs = sorted([kernel[0, 0], kernel[1, 1]], reverse=True)
                ROI_axis_a = covs[0]
                ROI_axis_b = covs[1]
                ROI_axis_loc = (kernel[1, 0], kernel[0, 1])
            except:
                ROI_density = 0
        if BG_density == 1:
            x = []
            y = []
            for p in locs_BG:
                x.append(p[0])
                y.append(p[1])
            try:
                kernel = stats.gaussian_kde(np.vstack([x, y]))
                kernel = kernel.covariance * kernel.factor
                covs = sorted([kernel[0, 0], kernel[1, 1]], reverse=True)
                BG_axis_a = covs[0]
                BG_axis_b = covs[1]
                BG_axis_loc = (kernel[1, 0], kernel[0, 1])
            except:
                BG_density = 0
        if BG_density == 1 and ROI_density == 1:
            Densities_ROI_primary.append(ROI_axis_a)
            Densities_ROI_secondary.append(ROI_axis_b)
            Densities_ROI_mean.append(np.mean([ROI_axis_a, ROI_axis_b]))
            Densities_BG_mean.append(np.mean([BG_axis_a, BG_axis_b]))
            Densities_ROI_ratio.append(np.mean([ROI_axis_a, ROI_axis_b]) / np.mean([BG_axis_a, BG_axis_b]))
        elif ROI_density == 1:
            Densities_ROI_primary.append(ROI_axis_a)
            Densities_ROI_secondary.append(ROI_axis_b)
            Densities_ROI_mean.append(np.mean([ROI_axis_a, ROI_axis_b]))
            Densities_BG_mean.append(0)
            Densities_ROI_ratio.append(1)
        elif BG_density == 1:
            Densities_BG_mean.append(np.mean([BG_axis_a, BG_axis_b]))
        else:
            Densities_ROI_primary.append(0)
            Densities_ROI_secondary.append(0)
            Densities_ROI_mean.append(0)
            Densities_BG_mean.append(0)
            Densities_ROI_ratio.append(0)
    if No_corners_BG > 0:
        No_corners_ratio = float(No_corners_ROI) / float(No_corners_BG)
    else:
        No_corners_ratio = 0

    if len(Densities_ROI_primary) == 0:
        Corner_density_primary = 0
    else:
        Corner_density_primary = np.mean(Densities_ROI_primary)

    if len(Densities_ROI_secondary) == 0:
        Corner_density_secondary = 0
    else:
        Corner_density_secondary = np.mean(Densities_ROI_secondary)

    if len(Densities_ROI_mean) == 0:
        Corner_density_mean = 0
        Corner_density_ratio_overall = 0
    else:
        Corner_density_mean = np.mean(Densities_ROI_mean)
        Corner_density_ratio_overall = Corner_density_mean / (Corner_density_mean + np.mean(Densities_BG_mean))

    if len(Densities_ROI_ratio) == 0:
        Corner_density_ratio = 0
    else:
        Corner_density_ratio = np.mean(Densities_ROI_ratio)

    return No_corners_ROI, No_corners_BG, No_corners_ratio, Corner_density_primary, Corner_density_secondary, Corner_density_mean, Corner_density_ratio, Corner_density_ratio_overall


"""
Harris corner detection
Harris, C. and Stephens, M., 1988, August. A combined corner and edge detector. In Alvey vision conference (Vol. 15, No. 50, pp. 10-5244).
blockSize - Neighborhood size (see the details on cornerEigenValsAndVecs()).
ksize - Aperture parameter for the Sobel() operator.
k - Harris detector free parameter.
"""
subfun_3D_2D_Harris_names_BG = (
'No_corners_ROI', 'Corner_density_primary', 'Corner_density_secondary', 'Corner_density_mean')


def casefun_3D_2D_Harris_name_generator_BG(params):
    names = []
    for name in subfun_3D_2D_Harris_names_BG:
        names.append('BGUTUHarris_b%d_ks%d_k%3.2f_%s' % (params[0], params[1], params[2], name))
    return names


def casefun_3D_2D_Harris_BG(LESIONDATAr, LESIONr, BG_roi, resolution, params):
    blockSize = int(np.round(params[0] / np.mean([resolution[0], resolution[1]])))
    # print('Harris effective block size:' + str(blockSize))
    ksize = params[1]
    k = params[2]
    No_corners_ROI = 0
    Densities_ROI_primary = []
    Densities_ROI_secondary = []
    Densities_ROI_mean = []
    Densities_BG_mean = []
    Densities_ROI_ratio = []
    if len(LESIONDATAr.shape) > 2:
        slices = LESIONDATAr.shape[2]
    else:
        slices = 1
    for slice_i in range(slices):
        if slices == 1:
            if (np.max(BG_roi[:, :]) == 0):
                continue
            cvimg = make_cv2_slice2D(LESIONDATAr[:, :])
        else:
            if (np.max(BG_roi[:, :, slice_i]) == 0):
                continue
            cvimg = make_cv2_slice2D(LESIONDATAr[:, :, slice_i])
        edgemap = abs(cv2.cornerHarris(cvimg, blockSize, ksize, k))
        ROIdata = copy.deepcopy(edgemap)
        if slices == 1:
            ROIdata[BG_roi[:, :] == 0] = 0
        else:
            ROIdata[BG_roi[:, :, slice_i] == 0] = 0
        locs_ROI = peak_local_max(ROIdata, min_distance=1, threshold_abs=0.0)
        No_corners_ROI += len(locs_ROI)
        if locs_ROI.shape[0] > 3:
            ROI_density = 1
        else:
            ROI_density = 0
        if ROI_density == 1:
            x = []
            y = []
            for p in locs_ROI:
                x.append(p[0])
                y.append(p[1])
            try:
                kernel = stats.gaussian_kde(np.vstack([x, y]))
                kernel = kernel.covariance * kernel.factor
                covs = sorted([kernel[0, 0], kernel[1, 1]], reverse=True)
                ROI_axis_a = covs[0]
                ROI_axis_b = covs[1]
            except:
                ROI_density = 0
        if ROI_density == 1:
            Densities_ROI_primary.append(ROI_axis_a)
            Densities_ROI_secondary.append(ROI_axis_b)
            Densities_ROI_mean.append(np.mean([ROI_axis_a, ROI_axis_b]))
            Densities_BG_mean.append(0)
            Densities_ROI_ratio.append(1)
        else:
            Densities_ROI_primary.append(0)
            Densities_ROI_secondary.append(0)
            Densities_ROI_mean.append(0)
            Densities_BG_mean.append(0)
            Densities_ROI_ratio.append(0)
    if len(Densities_ROI_primary) == 0:
        Corner_density_primary = 0
    else:
        Corner_density_primary = np.mean(Densities_ROI_primary)

    if len(Densities_ROI_secondary) == 0:
        Corner_density_secondary = 0
    else:
        Corner_density_secondary = np.mean(Densities_ROI_secondary)

    if len(Densities_ROI_mean) == 0:
        Corner_density_mean = 0
    else:
        Corner_density_mean = np.mean(Densities_ROI_mean)

    return No_corners_ROI, Corner_density_primary, Corner_density_secondary, Corner_density_mean


"""
Shi-Tomasi corner detection
Shi, J. and Tomasi, C., 1993. Good features to track. Cornell University.
maxCorners Maximum number of corners to return. If there are more corners than are found, the strongest of them is returned.
qualityLevel Parameter characterizing the minimal accepted quality of image corners. The parameter value is multiplied by the best corner quality measure, which is the minimal eigenvalue (see cornerMinEigenVal() ) or the Harris function response (see cornerHarris() ). The corners with the quality measure less than the product are rejected. For example, if the best corner has the quality measure = 1500, and the qualityLevel=0.01 , then all the corners with the quality measure less than 15 are rejected.
minDistance Minimum possible Euclidean distance between the returned corners.
"""
subfun_3D_2D_ShiTomasi_names = (
'No_corners_ROI', 'No_corners_BG', 'No_corners_ratio', 'Corner_density_primary', 'Corner_density_secondary',
'Corner_density_mean', 'Corner_density_ratio', 'Corner_density_ratio_overall')


def casefun_3D_2D_ShiTomasi_name_generator(params):
    names = []
    for name in subfun_3D_2D_Harris_names:
        names.append('UTUShiTomasi_b%d_ks%4.3f_k%3.2f_%s' % (params[0], params[1], params[2], name))
    return names


def casefun_3D_2D_ShiTomasi(LESIONDATAr, LESIONr, BG_roi, resolution, params):
    print('casefun_3D_2D_ShiTomasi')
    maxCorners = params[0]
    qualityLevel = params[1]
    minDistance = params[2] / np.mean([resolution[0], resolution[1]])
    No_corners_ROI = 0
    No_corners_BG = 0
    Densities_ROI_primary = []
    Densities_ROI_secondary = []
    Densities_ROI_mean = []
    Densities_BG_mean = []
    Densities_ROI_ratio = []
    if len(LESIONDATAr.shape) > 2:
        slices = LESIONDATAr.shape[2]
    else:
        slices = 1
    for slice_i in range(slices):
        if slices == 1:
            if (np.max(LESIONr[0][:, :]) == 0 and np.max(BG_roi[:, :]) == 0):
                continue
            cvimg = make_cv2_slice2D(LESIONDATAr[:, :])
            cvROImask = make_cv2_slice2D(LESIONr[0][:, :])
        else:
            if (np.max(LESIONr[0][:, :, slice_i]) == 0 and np.max(BG_roi[:, :, slice_i]) == 0):
                continue
            cvimg = make_cv2_slice2D(LESIONDATAr[:, :, slice_i])
            cvROImask = make_cv2_slice2D(LESIONr[0][:, :, slice_i])
        locs_ROI = cv2.goodFeaturesToTrack(cvimg, maxCorners, qualityLevel, minDistance, mask=cvROImask)
        locs_ROI = np.squeeze(locs_ROI)
        if locs_ROI is None or len(locs_ROI.shape) == 0:
            ROIx = []
            ROIy = []
        elif len(locs_ROI.shape) == 1:
            ROIx = [locs_ROI[0]]
            ROIy = [locs_ROI[1]]
        else:
            ROIx = []
            ROIy = []
            for p in locs_ROI:
                ROIx.append(p[0])
                ROIy.append(p[1])
        No_corners_ROI += len(ROIx)
        if slices == 1:
            sliceBGdata = copy.deepcopy(BG_roi[:, :])
            sliceBGdata[LESIONr[0][:, :] > 0] = 0
        else:
            sliceBGdata = copy.deepcopy(BG_roi[:, :, slice_i])
            sliceBGdata[LESIONr[0][:, :, slice_i] > 0] = 0
        cvBGmask = make_cv2_slice2D(sliceBGdata)
        locs_BG = cv2.goodFeaturesToTrack(cvimg, maxCorners, qualityLevel, minDistance, mask=cvBGmask)
        locs_BG = np.squeeze(locs_BG)
        if locs_BG is None or len(locs_BG.shape) == 0:
            BGx = []
            BGy = []
        elif len(locs_BG.shape) == 1:
            BGx = [locs_BG[0]]
            BGy = [locs_BG[1]]
        else:
            BGx = []
            BGy = []
            for p in locs_BG:
                BGx.append(p[0])
                BGy.append(p[1])
        No_corners_BG += len(BGx)
        if len(ROIx) > 3 and len(np.unique(ROIx)) > 1 and len(np.unique(ROIy)) > 1:
            kernel = stats.gaussian_kde(np.vstack([ROIx, ROIy]))
            kernel = kernel.covariance * kernel.factor
            covs = sorted([kernel[0, 0], kernel[1, 1]], reverse=True)
            ROI_axis_a = covs[0]
            ROI_axis_b = covs[1]
            ROI_axis_loc = (kernel[1, 0], kernel[0, 1])
        if len(BGx) > 3 and len(np.unique(BGx)) > 1 and len(np.unique(BGy)) > 1:
            kernel = stats.gaussian_kde(np.vstack([BGx, BGy]))
            kernel = kernel.covariance * kernel.factor
            covs = sorted([kernel[0, 0], kernel[1, 1]], reverse=True)
            BG_axis_a = covs[0]
            BG_axis_b = covs[1]
            BG_axis_loc = (kernel[1, 0], kernel[0, 1])
        if len(BGx) > 3 and len(ROIx) > 3 and len(np.unique(ROIx)) > 1 and len(np.unique(ROIy)) > 1 and len(
                np.unique(BGx)) > 1 and len(np.unique(BGy)) > 1:
            Densities_ROI_primary.append(ROI_axis_a)
            Densities_ROI_secondary.append(ROI_axis_b)
            Densities_ROI_mean.append(np.mean([ROI_axis_a, ROI_axis_b]))
            Densities_BG_mean.append(np.mean([BG_axis_a, BG_axis_b]))
            Densities_ROI_ratio.append(np.mean([ROI_axis_a, ROI_axis_b]) / np.mean([BG_axis_a, BG_axis_b]))
        elif len(ROIx) > 3 and len(np.unique(ROIx)) > 1 and len(np.unique(ROIy)) > 1:
            Densities_ROI_primary.append(ROI_axis_a)
            Densities_ROI_secondary.append(ROI_axis_b)
            Densities_ROI_mean.append(np.mean([ROI_axis_a, ROI_axis_b]))
            Densities_BG_mean.append(0)
            Densities_ROI_ratio.append(1)
        elif len(BGx) > 3 and len(np.unique(BGx)) > 1 and len(np.unique(BGy)) > 1:
            Densities_BG_mean.append(np.mean([BG_axis_a, BG_axis_b]))
        else:
            Densities_ROI_primary.append(0)
            Densities_ROI_secondary.append(0)
            Densities_ROI_mean.append(0)
            Densities_BG_mean.append(0)
            Densities_ROI_ratio.append(0)
    if No_corners_BG > 0:
        No_corners_ratio = float(No_corners_ROI) / float(No_corners_BG)
    else:
        No_corners_ratio = 0

    if No_corners_BG > 0:
        No_corners_ratio = float(No_corners_ROI) / float(No_corners_BG)
    else:
        No_corners_ratio = 0

    if len(Densities_ROI_primary) == 0:
        Corner_density_primary = 0
    else:
        Corner_density_primary = np.mean(Densities_ROI_primary)

    if len(Densities_ROI_secondary) == 0:
        Corner_density_secondary = 0
    else:
        Corner_density_secondary = np.mean(Densities_ROI_secondary)

    if len(Densities_ROI_mean) == 0:
        Corner_density_mean = 0
        Corner_density_ratio_overall = 0
    else:
        Corner_density_mean = np.mean(Densities_ROI_mean)
        Corner_density_ratio_overall = Corner_density_mean / (Corner_density_mean + np.mean(Densities_BG_mean))

    if len(Densities_ROI_ratio) == 0:
        Corner_density_ratio = 0
    else:
        Corner_density_ratio = np.mean(Densities_ROI_ratio)

    return No_corners_ROI, No_corners_BG, No_corners_ratio, Corner_density_primary, Corner_density_secondary, Corner_density_mean, Corner_density_ratio, Corner_density_ratio_overall


"""
Collection of edge detectors

frangi(image)
A. Frangi, W. Niessen, K. Vincken, and M. Viergever. "Multiscale vessel enhancement filtering," In LNCS, vol. 1496, pages 130-137, Germany, 1998. Springer-Verlag.
hessian(image)
Choon-Ching Ng, Moi Hoon Yap, Nicholas Costen and Baihua Li, "Automatic Wrinkle Detection using Hybrid Hessian Filter".

hyst = filters.apply_hysteresis_threshold(edges, low, high)
edge_scharr = scharr(img)
B. Jaehne, H. Scharr, and S. Koerkel. Principles of filter design. In Handbook of Computer Vision and Applications. Academic Press, 1999.
"""
subfun_3D_2D_objectprops_names = ('Area_mean_mm2', 'Area_median_mm2', 'Area_SD_mm2', 'Area_IQR_mm2', 'Rel_area',
                                  'Ecc_mean', 'Ecc_median', 'Ecc_SD', 'Ecc_IQR', 'Rel_ecc',
                                  'Ax1len_mean_mm', 'Ax1len_median_mm', 'Ax1len_SD_mm', 'Ax1len_IQR_mm', 'Rel_Ax1len',
                                  'Ax2len_mean_mm', 'Ax2len_median_mm', 'Ax2len_SD_mm', 'Ax2len_IQR_mm', 'Rel_Ax2len',
                                  'Int_mean', 'Int_median', 'Int_SD', 'Int_IQR', 'Rel_Int',
                                  'Ori_SD', 'Ori_IQR', 'Rel_Ori',
                                  'Per_mean_mm', 'Per_median_mm', 'Per_SD_mm', 'Per_IQR_mm', 'Rel_Per',
                                  'Den_mean', 'Den_median', 'Rel_Den', 'N_objs', 'Rel_objs')


def casefun_3D_2D_objectprops_name_generator(filtername, param_strs):
    names = []
    for name in subfun_3D_2D_objectprops_names:
        if len(param_strs) > 0:
            suffix = ''
            for param in param_strs:
                suffix += ('_%s' % param_strs)
            names.append('UTU3D2D%s_%s_objprops_%s' % (filtername, suffix, name))
        else:
            names.append('UTU3D2D%s_objprops_%s' % (filtername, name))
    return names


subfun_3D_2D_objectprops_names_BG = ('Area_mean_mm2', 'Area_median_mm2', 'Area_SD_mm2', 'Area_IQR_mm2',
                                     'Ecc_mean', 'Ecc_median', 'Ecc_SD', 'Ecc_IQR',
                                     'Ax1len_mean_mm', 'Ax1len_median_mm', 'Ax1len_SD_mm', 'Ax1len_IQR_mm',
                                     'Ax2len_mean_mm', 'Ax2len_median_mm', 'Ax2len_SD_mm', 'Ax2len_IQR_mm',
                                     'Int_mean', 'Int_median', 'Int_SD', 'Int_IQR',
                                     'Ori_SD', 'Ori_IQR',
                                     'Per_mean_mm', 'Per_median_mm', 'Per_SD_mm', 'Per_IQR_mm',
                                     'Den_mean', 'Den_median', 'N_objs')


def casefun_3D_2D_objectprops_name_generator_BG(filtername, param_strs):
    names = []
    for name in subfun_3D_2D_objectprops_names_BG:
        if len(param_strs) > 0:
            suffix = ''
            for param in param_strs:
                suffix += ('_%s' % param_strs)
            names.append('BGUTU3D2D%s_%s_objprops_%s' % (filtername, suffix, name))
        else:
            names.append('BGUTU3D2D%s_objprops_%s' % (filtername, name))
    return names


def casefun_3D_2D_objectprops(LESIONDATAr, LESIONr, BG_roi, resolution, fun2D, params):
    area = []
    eccentricity = []
    major_axis_length = []
    mean_intensity = []
    minor_axis_length = []
    orientation = []
    perimeter = []
    density = []
    BGarea = []
    BGeccentricity = []
    BGmajor_axis_length = []
    BGmean_intensity = []
    BGminor_axis_length = []
    BGorientation = []
    BGperimeter = []
    BGdensity = []
    blobs = 0
    BGblobs = 0
    if len(LESIONDATAr.shape) > 2:
        slices = LESIONDATAr.shape[2]
    else:
        slices = 1
    for slice_i in range(slices):
        if slices == 1:
            if (np.max(LESIONr[0][:, :]) == 0 or np.max(BG_roi[:, :]) == 0):
                continue
            slice2Ddata = LESIONDATAr[:, :]
        else:
            if (np.max(LESIONr[0][:, :, slice_i]) == 0 or np.max(BG_roi[:, :, slice_i]) == 0):
                continue
            slice2Ddata = LESIONDATAr[:, :, slice_i]
        x_lo, x_hi, y_lo, y_hi = find_bounded_subregion2D(slice2Ddata)
        slice2Ddata = slice2Ddata[x_lo:x_hi, y_lo:y_hi]
        if slices == 1:
            slice2D_ROI = LESIONr[0][x_lo:x_hi, y_lo:y_hi]
            slice2D_BG = BG_roi[x_lo:x_hi, y_lo:y_hi]
        else:
            slice2D_ROI = LESIONr[0][x_lo:x_hi, y_lo:y_hi, slice_i]
            slice2D_BG = BG_roi[x_lo:x_hi, y_lo:y_hi, slice_i]

        # Resize to 1x1 mm space
        cvimg = cv2.resize(slice2Ddata, None, fx=resolution[0], fy=resolution[1], interpolation=cv2.INTER_LANCZOS4)
        cvBG = cv2.resize(slice2D_BG, None, fx=resolution[0], fy=resolution[1], interpolation=cv2.INTER_NEAREST)
        cvROI = cv2.resize(slice2D_ROI, None, fx=resolution[0], fy=resolution[1], interpolation=cv2.INTER_NEAREST)
        slice2D = fun2D(cvimg, params)

        if (type(params) == list) and (len(params) > 1) and (not type(params[-1]) == int) and (
                'write_visualization' in params[-1]):
            LESIONDATAr_cvimg = make_cv2_slice2D(LESIONDATAr[:, :, slice_i]).copy()
            LESIONr_cvimg = make_cv2_slice2D(LESIONr[:, :, slice_i]).copy()
            basename = params[-1]['name'] + '_2D_curvature_' + str(params[:-1]).replace(' ', '_')
            visualizations.write_slice2D(cvimg, params[-1]['write_visualization'] + os.sep + basename + '_data.tiff')
            visualizations.write_slice2D_ROI(LESIONDATAr_cvimg, LESIONr_cvimg,
                                             params[-1]['write_visualization'] + os.sep + basename + '_lesion.tiff',
                                             0.4)

        sliceROI = copy.deepcopy(slice2D)
        sliceROI[cvROI == 0] = 0
        sliceBG = copy.deepcopy(slice2D)
        sliceBG[cvBG == 0] = 0
        sliceBG[cvROI > 0] = 0

        # label non-zero regions
        all_labels_ROI = measure.label(sliceROI)
        all_labels_BG = measure.label(sliceBG)
        # calculate region properties
        regions_ROI = regionprops(all_labels_ROI, intensity_image=cvimg, cache=True)
        regions_BG = regionprops(all_labels_BG, intensity_image=cvimg, cache=True)
        centroid_x = []
        centroid_y = []
        blobs += len(regions_ROI)
        for region in regions_ROI:
            area.append(region.area)
            centroid_x.append(region.centroid[0])
            centroid_y.append(region.centroid[1])
            eccentricity.append(region.eccentricity)
            major_axis_length.append(region.major_axis_length)
            mean_intensity.append(region.mean_intensity)
            minor_axis_length.append(region.minor_axis_length)
            orientation.append(region.orientation)
            perimeter.append(region.perimeter)
        if len(centroid_x) > 3:
            kernel = stats.gaussian_kde(np.vstack([centroid_x, centroid_y]))
            kernel = kernel.covariance * kernel.factor
            density.append(np.mean([kernel[0, 0], kernel[1, 1]]))

        centroid_x = []
        centroid_y = []
        BGblobs += len(regions_BG)
        for region in regions_BG:
            BGarea.append(region.area)
            centroid_x.append(region.centroid[0])
            centroid_y.append(region.centroid[1])
            BGeccentricity.append(region.eccentricity)
            BGmajor_axis_length.append(region.major_axis_length)
            BGmean_intensity.append(region.mean_intensity)
            BGminor_axis_length.append(region.minor_axis_length)
            BGorientation.append(region.orientation)
            BGperimeter.append(region.perimeter)
        if len(centroid_x) > 3 and (len(np.unique(centroid_x)) > 1) and (len(np.unique(centroid_y)) > 1):
            kernel = stats.gaussian_kde(np.vstack([centroid_x, centroid_y]))
            kernel = kernel.covariance * kernel.factor
            BGdensity.append(np.mean([kernel[0, 0], kernel[1, 1]]))

    ret = []
    if len(area) > 0:
        meanarea = np.mean(area)
        ret.append(meanarea)
        ret.append(np.median(area))
        ret.append(np.std(area))
        ret.append(iqr(area))
    else:
        meanarea = 0.0
        ret.append(meanarea)
        ret.append(0.0)
        ret.append(float('NaN'))
        ret.append(float('NaN'))
    if meanarea == 0:
        ret.append(0)
    else:
        if len(BGarea) > 0:
            ret.append(meanarea / (meanarea + np.mean(BGarea)))
        else:
            ret.append(1.0)
    if (len(eccentricity) == 0):
        meaneccentricity = 0
        ret.append(0)
        ret.append(0)
        ret.append(0)
        ret.append(0)
    else:
        meaneccentricity = np.mean(eccentricity)
        ret.append(meaneccentricity)
        ret.append(np.median(eccentricity))
        ret.append(np.std(eccentricity))
        ret.append(iqr(eccentricity))
    if meaneccentricity == 0:
        ret.append(0)
    else:
        if len(BGeccentricity) > 0:
            ret.append(meaneccentricity / (meaneccentricity + np.mean(BGeccentricity)))
        else:
            ret.append(1)
    if (len(major_axis_length) == 0):
        meanmajor_axis_length = 0
        ret.append(meanmajor_axis_length)
        ret.append(0)
        ret.append(0)
        ret.append(0)
    else:
        meanmajor_axis_length = np.mean(major_axis_length)
        ret.append(meanmajor_axis_length)
        ret.append(np.median(major_axis_length))
        ret.append(np.std(major_axis_length))
        ret.append(iqr(major_axis_length))
    if meanmajor_axis_length == 0:
        ret.append(0)
    else:
        if len(BGmajor_axis_length) > 0:
            ret.append(meanmajor_axis_length / (meanmajor_axis_length + np.mean(BGmajor_axis_length)))
        else:
            ret.append(1)
    if (len(minor_axis_length) == 0):
        meanminor_axis_length = 0
        ret.append(meanminor_axis_length)
        ret.append(0)
        ret.append(0)
        ret.append(0)
    else:
        meanminor_axis_length = np.mean(minor_axis_length)
        ret.append(meanminor_axis_length)
        ret.append(np.median(minor_axis_length))
        ret.append(np.std(minor_axis_length))
        ret.append(iqr(minor_axis_length))
    if meanminor_axis_length == 0:
        ret.append(0)
    else:
        if len(BGminor_axis_length) > 0:
            ret.append(meanminor_axis_length / (meanminor_axis_length + np.mean(BGminor_axis_length)))
        else:
            ret.append(1)
    if (len(mean_intensity) == 0):
        mean_mean_intensity = 0
        ret.append(mean_mean_intensity)
        ret.append(0)
        ret.append(0)
        ret.append(0)
    else:
        mean_mean_intensity = np.mean(mean_intensity)
        ret.append(mean_mean_intensity)
        ret.append(np.median(mean_intensity))
        ret.append(np.std(mean_intensity))
        ret.append(iqr(mean_intensity))
    if mean_mean_intensity == 0:
        ret.append(0)
    else:
        if len(BGmean_intensity) > 0:
            ret.append(mean_mean_intensity / (mean_mean_intensity + np.mean(BGmean_intensity)))
        else:
            ret.append(1)
    if (len(orientation) == 0):
        ret.append(0)
        ret.append(0)
        meanorientation = 0
        ret.append(0)
    else:
        ret.append(np.std(orientation))
        ret.append(iqr(orientation))
        meanorientation = np.mean(orientation)
        if len(BGorientation) > 0:
            ret.append(meanorientation / (meanorientation + np.mean(BGorientation)))
        else:
            ret.append(1)
    if (len(perimeter) == 0):
        meanperimeter = 0
        ret.append(0)
        ret.append(0)
        ret.append(0)
        ret.append(0)
    else:
        meanperimeter = np.mean(perimeter)
        ret.append(meanperimeter)
        ret.append(np.median(perimeter))
        ret.append(np.std(perimeter))
        ret.append(iqr(perimeter))
    if meanperimeter == 0:
        ret.append(0)
    else:
        if len(BGperimeter) > 0:
            ret.append(meanperimeter / (meanperimeter + np.mean(BGperimeter)))
        else:
            ret.append(1)
    if len(density) == 0:
        ret.append(0)
        ret.append(0)
        ret.append(0)
    else:
        # print(density)
        meandensity = np.mean(density)
        # print(meandensity)
        ret.append(meandensity)
        ret.append(np.median(density))
        # print(np.median(density))
        if len(BGdensity) > 0:
            ret.append(meandensity / (meandensity + np.mean(BGdensity)))
        else:
            ret.append(1)
    if blobs == 0:
        ret.append(0)
        ret.append(0)
    else:
        ret.append(blobs)
        ret.append(blobs / (blobs + BGblobs))
    return ret


def casefun_3D_2D_objectprops_BG(LESIONDATAr, LESIONr, BG_roi, resolution, fun2D, params):
    area = []
    eccentricity = []
    major_axis_length = []
    mean_intensity = []
    minor_axis_length = []
    orientation = []
    perimeter = []
    density = []
    blobs = 0
    if len(LESIONDATAr.shape) > 2:
        slices = LESIONDATAr.shape[2]
    else:
        slices = 1
    for slice_i in range(slices):
        if slices == 1:
            if (np.max(BG_roi[:, :]) == 0):
                continue
            slice2Ddata = LESIONDATAr[:, :]
        else:
            if (np.max(BG_roi[:, :, slice_i]) == 0):
                continue
            slice2Ddata = LESIONDATAr[:, :, slice_i]
        x_lo, x_hi, y_lo, y_hi = find_bounded_subregion2D(slice2Ddata)
        slice2Ddata = slice2Ddata[x_lo:x_hi, y_lo:y_hi]
        if slices == 1:
            slice2D_ROI = BG_roi[x_lo:x_hi, y_lo:y_hi]
        else:
            slice2D_ROI = BG_roi[x_lo:x_hi, y_lo:y_hi, slice_i]
        # Resize to 1x1 mm space
        cvimg = cv2.resize(slice2Ddata, None, fx=resolution[0], fy=resolution[1], interpolation=cv2.INTER_LANCZOS4)
        cvROI = cv2.resize(slice2D_ROI, None, fx=resolution[0], fy=resolution[1], interpolation=cv2.INTER_NEAREST)
        slice2D = fun2D(cvimg, params)

        sliceROI = copy.deepcopy(slice2D)
        sliceROI[cvROI == 0] = 0

        # label non-zero regions
        all_labels_ROI = measure.label(sliceROI)
        # calculate region properties
        regions_ROI = regionprops(all_labels_ROI, intensity_image=cvimg, cache=True)
        centroid_x = []
        centroid_y = []
        blobs += len(regions_ROI)
        for region in regions_ROI:
            area.append(region.area)
            centroid_x.append(region.centroid[0])
            centroid_y.append(region.centroid[1])
            eccentricity.append(region.eccentricity)
            major_axis_length.append(region.major_axis_length)
            mean_intensity.append(region.mean_intensity)
            minor_axis_length.append(region.minor_axis_length)
            orientation.append(region.orientation)
            perimeter.append(region.perimeter)
        if len(centroid_x) > 3:
            kernel = stats.gaussian_kde(np.vstack([centroid_x, centroid_y]))
            kernel = kernel.covariance * kernel.factor
            density.append(np.mean([kernel[0, 0], kernel[1, 1]]))

    ret = []
    if (blobs == 0):
        for ret_i in range(27):
            ret.append(0)
    else:
        meanarea = np.mean(area)
        ret.append(meanarea)
        ret.append(np.median(area))
        ret.append(np.std(area))
        ret.append(iqr(area))

        meaneccentricity = np.mean(eccentricity)
        ret.append(meaneccentricity)
        ret.append(np.median(eccentricity))
        ret.append(np.std(eccentricity))
        ret.append(iqr(eccentricity))

        meanmajor_axis_length = np.mean(major_axis_length)
        ret.append(meanmajor_axis_length)
        ret.append(np.median(major_axis_length))
        ret.append(np.std(major_axis_length))
        ret.append(iqr(major_axis_length))

        meanminor_axis_length = np.mean(minor_axis_length)
        ret.append(meanminor_axis_length)
        ret.append(np.median(minor_axis_length))
        ret.append(np.std(minor_axis_length))
        ret.append(iqr(minor_axis_length))

        mean_mean_intensity = np.mean(mean_intensity)
        ret.append(mean_mean_intensity)
        ret.append(np.median(mean_intensity))
        ret.append(np.std(mean_intensity))
        ret.append(iqr(mean_intensity))

        ret.append(np.std(orientation))
        ret.append(iqr(orientation))

        meanperimeter = np.mean(perimeter)
        ret.append(meanperimeter)
        ret.append(np.median(perimeter))
        ret.append(np.std(perimeter))
        ret.append(iqr(perimeter))

    if len(density) == 0:
        ret.append(0)
        ret.append(0)
    else:
        meandensity = np.mean(density)
        ret.append(meandensity)
        ret.append(np.median(density))

    if blobs == 0:
        ret.append(0)
    else:
        ret.append(blobs)

    return ret


casefun_3D_2D_Frangi_objectprops_names = casefun_3D_2D_objectprops_name_generator('Frangi', [])


def subfun_Frangi(slice2D, params):
    # Create binary frangi
    edge_frangi = frangi(slice2D)
    val = filters.threshold_otsu(edge_frangi)
    frangi_bin = np.zeros_like(edge_frangi)
    frangi_bin[edge_frangi >= val] = 1
    return frangi_bin


def casefun_3D_2D_Frangi_objectprops(LESIONDATAr, LESIONr, BG_roi, resolution, params):
    print('casefun_3D_2D_Frangi_objectprops')
    return casefun_3D_2D_objectprops(LESIONDATAr, LESIONr, BG_roi, resolution, subfun_Frangi, params)


def casefun_3D_2D_Hessian_objectprops_name_generator(params):
    names = []
    for name in subfun_3D_2D_objectprops_names:
        names.append('UTU3D2DHessian_%4.3f_%2.1f_objprops_%s' % (params[0], params[1], name))
    return names


def casefun_3D_2D_Hessian_objectprops_name_generator_BG(params):
    names = []
    for name in subfun_3D_2D_objectprops_names_BG:
        names.append('BGUTU3D2DHessian_%4.3f_%2.1f_objprops_%s' % (params[0], params[1], name))
    return names


def subfun_Hessian(slice2D, params):
    # Create binary edge image
    edge_hessian = hessian(slice2D, beta=params[0], gamma=params[1])
    val = filters.threshold_otsu(edge_hessian)
    hessian_bin = np.zeros_like(edge_hessian)
    hessian_bin[edge_hessian >= val] = 1
    return hessian_bin


def casefun_3D_2D_Hessian_objectprops(LESIONDATAr, LESIONr, BG_roi, resolution, params):
    print('casefun_3D_2D_Hessian_objectprops')
    return casefun_3D_2D_objectprops(LESIONDATAr, LESIONr, BG_roi, resolution, subfun_Hessian, params)


def casefun_3D_2D_Hessian_objectprops_BG(LESIONDATAr, LESIONr, BG_roi, resolution, params):
    return casefun_3D_2D_objectprops_BG(LESIONDATAr, LESIONr, BG_roi, resolution, subfun_Hessian, params)


casefun_3D_2D_Scharr_objectprops_names = casefun_3D_2D_objectprops_name_generator('Scharr', [])


def subfun_Scharr(slice2D, params):
    # Create binary edge image
    edge_scharr = scharr(slice2D)
    scharr_bin = np.zeros_like(edge_scharr)
    scharr_bin[edge_scharr > 0] = 1
    return scharr_bin


def casefun_3D_2D_Scharr_objectprops(LESIONDATAr, LESIONr, BG_roi, resolution, params):
    print('casefun_3D_2D_Scharr_objectprops')
    return casefun_3D_2D_objectprops(LESIONDATAr, LESIONr, BG_roi, resolution, subfun_Scharr, params)


"""
Laws texture features
K. Laws "Textured Image Segmentation", Ph.D. Dissertation, University of Southern California, January 1980
5th vector
A. Meyer-Base, "Pattern Recognition for Medical Imaging", Academic Press, 2004.
"""
casefun_3D_2D_Laws_names = ['mean_ROI', 'median_ROI', 'SD_ROI', 'IQR_ROI', 'skewnessROI', 'kurtosisROI', 'p25ROI',
                            'p75ROI', 'rel']
laws_names = ['1_L5E5_E5L5', '2_L5R5_R5L5', '3_E5S5_S5E5', '4_S5S5', '5_R5R5', '6_L5S5_S5E5', '7_E5E5', '8_E5R5_R5E5',
              '9_S5R5_R5S5']


def casefun_3D_2D_Laws_names_generator(params):
    names = []
    for name_i in range(len(laws_names)):
        for name in casefun_3D_2D_Laws_names:
            names.append('UTU3D2DLaws%s_%s_f%2.1f' % (laws_names[name_i], name, params[0]))
    return names


# Define the 1D kernels
L5 = np.array([1, 4, 6, 4, 1])  # level
E5 = np.array([-1, -2, 0, 2, 1])  # edge
S5 = np.array([-1, 0, 2, 0, -1])  # spot
W5 = np.array([-1, 2, 0, -2, 1])  # waves
R5 = np.array([1, -4, 6, -4, 1])  # ripples
# Generate 2D kernels
L5L5 = np.outer(L5, L5)
L5E5 = np.outer(L5, E5)
L5S5 = np.outer(L5, S5)
L5R5 = np.outer(L5, R5)
L5W5 = np.outer(L5, W5)
E5L5 = np.outer(E5, L5)
E5E5 = np.outer(E5, E5)
E5S5 = np.outer(E5, S5)
E5R5 = np.outer(E5, R5)
E5W5 = np.outer(E5, W5)
S5L5 = np.outer(S5, L5)
S5E5 = np.outer(S5, E5)
S5S5 = np.outer(S5, S5)
S5R5 = np.outer(S5, R5)
S5W5 = np.outer(S5, W5)
R5L5 = np.outer(R5, L5)
R5E5 = np.outer(R5, E5)
R5S5 = np.outer(R5, S5)
R5R5 = np.outer(R5, R5)
R5W5 = np.outer(R5, W5)


def append_Laws_results(outdata, LESIONrs, BG_rois, ret):
    ROIdata = outdata[LESIONrs > 0]
    BGdata = outdata[BG_rois > 0]
    mean1 = np.mean(ROIdata)
    ret.append(np.mean(ROIdata))
    median1 = np.median(ROIdata)
    ret.append(np.median(ROIdata))
    std1 = np.std(ROIdata)
    ret.append(std1)
    iqr1 = iqr(ROIdata)
    ret.append(iqr(ROIdata))
    ret.append(scipy.stats.skew(ROIdata))
    ret.append(scipy.stats.kurtosis(ROIdata))
    ret.append(np.percentile(ROIdata, 25))
    ret.append(np.percentile(ROIdata, 75))
    if (mean1 == 0):
        ret.append(0)
    else:
        ret.append(mean1 / (mean1 + np.mean(BGdata)))
    return ret


def casefun_3D_2D_Laws(LESIONDATAr, LESIONr, BG_roi, resolution, params):
    # resolution factor affecting laws feature sampling ratio
    # 1: original resolution
    # <1: upsampling
    # >1: downsampling
    res_f = params[0]

    if np.max(LESIONDATAr) == 0:
        return [float('nan') for x in casefun_3D_2D_Laws_names_generator(params)]

    x_lo, x_hi, y_lo, y_hi = find_bounded_subregion3D2D(LESIONDATAr)
    # print('bounded_subregion3D2D:' + str((x_lo, x_hi, y_lo, y_hi)))
    if len(LESIONDATAr.shape) > 2:
        slices = LESIONDATAr.shape[2]
        LESIONDATArs = LESIONDATAr[x_lo:x_hi, y_lo:y_hi, :]
        LESIONrs_temp = LESIONr[0][x_lo:x_hi, y_lo:y_hi, :]
        BG_rois_temp = BG_roi[x_lo:x_hi, y_lo:y_hi, :]
        # Create masks and output data to desired resolution, intensity data is resliced later for non-zero only
        slice2Ddata = LESIONDATArs[:, :, 0]
    else:
        slices = 1
        LESIONDATArs = LESIONDATAr[x_lo:x_hi, y_lo:y_hi]
        LESIONrs_temp = LESIONr[0][x_lo:x_hi, y_lo:y_hi]
        BG_rois_temp = BG_roi[x_lo:x_hi, y_lo:y_hi]
        # Create masks and output data to desired resolution, intensity data is resliced later for non-zero only
        slice2Ddata = LESIONDATArs[:, :]
    cvimg = cv2.resize(slice2Ddata, None, fx=resolution[0] * res_f, fy=resolution[1] * res_f,
                       interpolation=cv2.INTER_NEAREST)
    if slices == 1:
        LESIONrs = np.zeros([cvimg.shape[0], cvimg.shape[1]])
    else:
        LESIONrs = np.zeros([cvimg.shape[0], cvimg.shape[1], LESIONDATAr.shape[2]])
    BG_rois = np.zeros_like(LESIONrs)
    accepted_slices = 0
    for slice_i in range(slices):
        if slices == 1:
            if (np.max(LESIONrs_temp[:, :]) == 0 and np.max(BG_rois_temp[:, :]) == 0):
                continue
        else:
            if (np.max(LESIONrs_temp[:, :, slice_i]) == 0 and np.max(BG_rois_temp[:, :, slice_i]) == 0):
                continue
        accepted_slices += 1
        if slices == 1:
            slice2Ddata = LESIONrs_temp[:, :]
        else:
            slice2Ddata = LESIONrs_temp[:, :, slice_i]
        LESIONcvimg = cv2.resize(slice2Ddata, None, fx=resolution[0] * res_f, fy=resolution[1] * res_f,
                                 interpolation=cv2.INTER_NEAREST)
        if slices == 1:
            LESIONrs[:, :] = LESIONcvimg
            slice2Ddata = BG_rois_temp[:, :]
        else:
            LESIONrs[:, :, slice_i] = LESIONcvimg
            slice2Ddata = BG_rois_temp[:, :, slice_i]
        BGcvimg = cv2.resize(slice2Ddata, None, fx=resolution[0] * res_f, fy=resolution[1] * res_f,
                             interpolation=cv2.INTER_NEAREST)
        if slices == 1:
            BG_rois[:, :] = BGcvimg
        else:
            BG_rois[:, :, slice_i] = BGcvimg
    outdata_1 = np.zeros_like(LESIONrs)
    outdata_2 = np.zeros_like(LESIONrs)
    outdata_3 = np.zeros_like(LESIONrs)
    outdata_4 = np.zeros_like(LESIONrs)
    outdata_5 = np.zeros_like(LESIONrs)
    outdata_6 = np.zeros_like(LESIONrs)
    outdata_7 = np.zeros_like(LESIONrs)
    outdata_8 = np.zeros_like(LESIONrs)
    outdata_9 = np.zeros_like(LESIONrs)
    # print(str(accepted_slices) + ' accepted slices')
    if accepted_slices == 0:
        return [float('nan') for x in casefun_3D_2D_Laws_names_generator(params)]

    s = 5
    mid = int(np.floor(s / 2.0))
    for slice_i in range(slices):
        if slices == 1:
            if (np.max(LESIONrs[:, :]) == 0 and np.max(BG_rois[:, :]) == 0):
                continue
            slice2Ddata = LESIONDATArs[:, :]
        else:
            if (np.max(LESIONrs[:, :, slice_i]) == 0 and np.max(BG_rois[:, :, slice_i]) == 0):
                continue
            slice2Ddata = LESIONDATArs[:, :, slice_i]
        cvimg = cv2.resize(slice2Ddata, None, fx=resolution[0] * res_f, fy=resolution[1] * res_f,
                           interpolation=cv2.INTER_LANCZOS4)
        for (x, y, window) in sliding_window(cvimg, 1, (s, s)):
            window = np.subtract(window, np.mean(window))
            w_std = np.std(window)
            if w_std > 0:
                window = np.divide(window, np.std(window))
            fL5E5 = correlate2d(window, L5E5)[2:7, 2:7]
            fE5L5 = correlate2d(window, E5L5)[2:7, 2:7]
            fL5R5 = correlate2d(window, L5R5)[2:7, 2:7]
            fR5L5 = correlate2d(window, R5L5)[2:7, 2:7]
            fE5S5 = correlate2d(window, E5S5)[2:7, 2:7]
            fS5E5 = correlate2d(window, S5E5)[2:7, 2:7]
            fS5S5 = correlate2d(window, S5S5)[2:7, 2:7]
            fR5R5 = correlate2d(window, R5R5)[2:7, 2:7]
            fL5S5 = correlate2d(window, L5S5)[2:7, 2:7]
            fS5L5 = correlate2d(window, S5L5)[2:7, 2:7]
            fE5E5 = correlate2d(window, E5E5)[2:7, 2:7]
            fE5R5 = correlate2d(window, E5R5)[2:7, 2:7]
            fR5E5 = correlate2d(window, R5E5)[2:7, 2:7]
            fS5R5 = correlate2d(window, S5R5)[2:7, 2:7]
            fR5S5 = correlate2d(window, R5S5)[2:7, 2:7]
            # Truncate to 9 by removing redundant information
            Laws_1 = np.sum(np.divide(np.add(fL5E5, fE5L5), 2.0))
            Laws_2 = np.sum(np.divide(np.add(fL5R5, fR5L5), 2.0))
            Laws_3 = np.sum(np.divide(np.add(fE5S5, fS5E5), 2.0))
            Laws_4 = np.sum(fS5S5)
            Laws_5 = np.sum(fR5R5)
            Laws_6 = np.sum(np.divide(np.add(fL5S5, fS5L5), 2.0))
            Laws_7 = np.sum(fE5E5)
            Laws_8 = np.sum(np.divide(np.add(fE5R5, fR5E5), 2.0))
            Laws_9 = np.sum(np.divide(np.add(fS5R5, fR5S5), 2.0))

            xmid = x + mid
            ymid = y + mid
            if xmid >= outdata_1.shape[0]:
                xmid = outdata_1.shape[0] - 1
            if ymid >= outdata_1.shape[1]:
                ymid = outdata_1.shape[1] - 1
            if slices == 1:
                outdata_1[xmid, ymid] = Laws_1
                outdata_2[xmid, ymid] = Laws_2
                outdata_3[xmid, ymid] = Laws_3
                outdata_4[xmid, ymid] = Laws_4
                outdata_5[xmid, ymid] = Laws_5
                outdata_6[xmid, ymid] = Laws_6
                outdata_7[xmid, ymid] = Laws_7
                outdata_8[xmid, ymid] = Laws_8
                outdata_9[xmid, ymid] = Laws_9
            else:
                outdata_1[xmid, ymid, slice_i] = Laws_1
                outdata_2[xmid, ymid, slice_i] = Laws_2
                outdata_3[xmid, ymid, slice_i] = Laws_3
                outdata_4[xmid, ymid, slice_i] = Laws_4
                outdata_5[xmid, ymid, slice_i] = Laws_5
                outdata_6[xmid, ymid, slice_i] = Laws_6
                outdata_7[xmid, ymid, slice_i] = Laws_7
                outdata_8[xmid, ymid, slice_i] = Laws_8
                outdata_9[xmid, ymid, slice_i] = Laws_9
    ret = []
    ret = append_Laws_results(outdata_1, LESIONrs, BG_rois, ret)
    ret = append_Laws_results(outdata_2, LESIONrs, BG_rois, ret)
    ret = append_Laws_results(outdata_3, LESIONrs, BG_rois, ret)
    ret = append_Laws_results(outdata_4, LESIONrs, BG_rois, ret)
    ret = append_Laws_results(outdata_5, LESIONrs, BG_rois, ret)
    ret = append_Laws_results(outdata_6, LESIONrs, BG_rois, ret)
    ret = append_Laws_results(outdata_7, LESIONrs, BG_rois, ret)
    ret = append_Laws_results(outdata_8, LESIONrs, BG_rois, ret)
    ret = append_Laws_results(outdata_9, LESIONrs, BG_rois, ret)
    return ret


"""
Naive signal - background divisions for reference against radiomics features
K. Laws "Textured Image Segmentation", Ph.D. Dissertation, University of Southern California, January 1980
5th vector
A. Meyer-Base, "Pattern Recognition for Medical Imaging", Academic Press, 2004.
"""
casefun_3D_2D_FFT2D_names = ['mean_ROI', 'median_ROI', 'SD_ROI', 'IQR_ROI', 'skewnessROI', 'kurtosisROI', 'p25ROI',
                             'p75ROI', 'rel']
freq_names = ['original', 'fb']


def casefun_3D_2D_FFT2D_names_generator(params):
    res_f = params[0]
    start_FWHM = params[1]
    end_FWHM = params[2]
    step_FWHM = params[3]
    thresholds_FWHM = [float(x) for x in np.linspace(int(start_FWHM), int(end_FWHM), int(step_FWHM))]
    # FWHM = 2*sigma*sqrt(2*ln(2))=2.35*sigma
    # more accurately 2.3548200450309493
    names = []
    for threshold_FWHM in thresholds_FWHM:
        for name in casefun_3D_2D_FFT2D_names:
            names.append('UTU3D2DFFT2D_%s_f%2.1f_FWHM%3.2f_LP' % (name, res_f, threshold_FWHM))
        for name in casefun_3D_2D_FFT2D_names:
            names.append('UTU3D2DFFT2D_%s_f%2.1f_FWHM%3.2f_HP' % (name, res_f, threshold_FWHM))
    return names


def casefun_3D_2D_FFT2D_names_generator_BG(params):
    res_f = params[0]
    start_FWHM = params[1]
    end_FWHM = params[2]
    step_FWHM = params[3]
    thresholds_FWHM = [float(x) for x in np.linspace(int(start_FWHM), int(end_FWHM), int(step_FWHM))]
    # FWHM = 2*sigma*sqrt(2*ln(2))=2.35*sigma
    # more accurately 2.3548200450309493
    names = []
    for threshold_FWHM in thresholds_FWHM:
        for name_i in range(len(casefun_3D_2D_FFT2D_names) - 1):
            name = casefun_3D_2D_FFT2D_names[name_i]
            names.append('BGUTU3D2DFFT2D_%s_f%2.1f_FWHM%3.2f_LP' % (name, res_f, threshold_FWHM))
        for name_i in range(len(casefun_3D_2D_FFT2D_names) - 1):
            name = casefun_3D_2D_FFT2D_names[name_i]
            names.append('BGUTU3D2DFFT2D_%s_f%2.1f_FWHM%3.2f_HP' % (name, res_f, threshold_FWHM))
    return names


def append_FFT2D_results(outdata, LESIONrs, BG_rois, ret):
    ROIdata = outdata['data_lo'][LESIONrs > 0]
    BGdata = outdata['data_lo'][BG_rois > 0]
    mean1 = np.mean(ROIdata)
    ret.append(np.mean(ROIdata))
    median1 = np.median(ROIdata)
    ret.append(np.median(ROIdata))
    std1 = np.std(ROIdata)
    ret.append(std1)
    iqr1 = iqr(ROIdata)
    ret.append(iqr(ROIdata))
    ret.append(scipy.stats.skew(ROIdata))
    ret.append(scipy.stats.kurtosis(ROIdata))
    ret.append(np.percentile(ROIdata, 25))
    ret.append(np.percentile(ROIdata, 75))
    if (mean1 == 0):
        ret.append(0)
    else:
        ret.append(mean1 / (mean1 + np.mean(BGdata)))
    ROIdata = outdata['data_hi'][LESIONrs > 0]
    BGdata = outdata['data_hi'][BG_rois > 0]
    mean1 = np.mean(ROIdata)
    ret.append(np.mean(ROIdata))
    median1 = np.median(ROIdata)
    ret.append(np.median(ROIdata))
    std1 = np.std(ROIdata)
    ret.append(std1)
    iqr1 = iqr(ROIdata)
    ret.append(iqr(ROIdata))
    ret.append(scipy.stats.skew(ROIdata))
    ret.append(scipy.stats.kurtosis(ROIdata))
    ret.append(np.percentile(ROIdata, 25))
    ret.append(np.percentile(ROIdata, 75))
    if (mean1 == 0):
        ret.append(0)
    else:
        ret.append(mean1 / (mean1 + np.mean(BGdata)))
    return ret


def gkern2(kernlen, nsig):
    """Returns a 2D Gaussian kernel array."""

    # create nxn zeros
    inp = np.zeros((kernlen, kernlen))
    # set element at the middle to one, a dirac delta
    inp[kernlen // 2, kernlen // 2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    return gaussian_filter(inp, nsig)


def FWHM(X, Y):
    spline = UnivariateSpline(X, Y - np.max(Y) / 2, s=0)
    r1, r2 = spline.roots()  # find the roots
    return r2 - r1


def casefun_3D_2D_FFT2D(LESIONDATAr, LESIONr, BG_roi, resolution, params):
    if np.max(LESIONDATAr) == 0 or np.max(LESIONr[0]) == 0:
        return [float('nan') for x in casefun_3D_2D_FFT2D_names_generator(params)]

    # res_f: resolution factor affecting laws feature sampling ratio
    # 1: original resolution
    # <1: upsampling
    # >1: downsampling
    # start_FWHM: starting FWHM threshold in mm
    # end_FWHM: starting FWHM threshold in mm
    # step_FWHM: starting FWHM threshold in mm
    res_f = params[0]
    start_FWHM = params[1]
    end_FWHM = params[2]
    step_FWHM = params[3]
    thresholds_FWHM = [float(x) for x in np.linspace(int(start_FWHM), int(end_FWHM), int(step_FWHM))]

    # print(np.max(LESIONDATAr))
    x_lo, x_hi, y_lo, y_hi = find_bounded_subregion3D2D(LESIONDATAr)
    # print((x_lo, x_hi, y_lo, y_hi))
    if len(LESIONDATAr.shape) > 2:
        slices = LESIONDATAr.shape[2]
        LESIONDATArs_temp = LESIONDATAr[x_lo:x_hi, y_lo:y_hi, :]
        LESIONrs_temp = LESIONr[0][x_lo:x_hi, y_lo:y_hi, :]
        BG_rois_temp = BG_roi[x_lo:x_hi, y_lo:y_hi, :]
        # Create masks and output data to desired resolution, starting with 1mm x 1mm resolution
        slice2Ddata = LESIONDATArs_temp[:, :, 0]
    else:
        slices = 1
        LESIONDATArs_temp = LESIONDATAr[x_lo:x_hi, y_lo:y_hi]
        LESIONrs_temp = LESIONr[0][x_lo:x_hi, y_lo:y_hi]
        BG_rois_temp = BG_roi[x_lo:x_hi, y_lo:y_hi]
        # Create masks and output data to desired resolution, starting with 1mm x 1mm resolution
        slice2Ddata = LESIONDATArs_temp[:, :]
    # print(resolution)
    # print(res_f)
    # print(slice2Ddata.shape)
    try:
        cvimg = cv2.resize(slice2Ddata, None, fx=resolution[0] * res_f, fy=resolution[1] * res_f,
                           interpolation=cv2.INTER_NEAREST)
    except:
        print('FWHM estimation failed')
        return [float('nan') for x in casefun_3D_2D_FFT2D_names_generator(params)]
    out_dim = np.max([cvimg.shape[0], cvimg.shape[1]])
    if np.mod(out_dim, 2) == 0:
        out_dim += 1

    # FWHM = 2*sigma*sqrt(2*ln(2))=2.35*sigma
    # more accurately 2.3548200450309493
    try:
        for threshold_FWHM_i in range(len(thresholds_FWHM)):
            kern = gkern2(out_dim, thresholds_FWHM[threshold_FWHM_i] / 2.3548200450309493)
            Y = kern[out_dim // 2, :]
            X = range(len(Y))
            fwhm_est = FWHM(X, Y)
            # print('FWHM[' + str(threshold_FWHM_i+1) + ']:' + str(fwhm_est) + 'mm')
    except:
        print('FWHM estimation failed')
        return [float('nan') for x in casefun_3D_2D_FFT2D_names_generator(params)]

    pad_x = out_dim - cvimg.shape[0]
    pad_y = out_dim - cvimg.shape[1]
    # print((out_dim, pad_x, pad_y))
    if slices > 1:
        LESIONrs = np.zeros([out_dim, out_dim, LESIONDATAr.shape[2]])
    else:
        LESIONrs = np.zeros([out_dim, out_dim, 1])
    BG_rois = np.zeros_like(LESIONrs)
    LESIONDATArs = np.zeros_like(LESIONrs)

    for slice_i in range(slices):
        if slices > 1:
            slice2Ddata = LESIONrs_temp[:, :, slice_i]
        else:
            slice2Ddata = LESIONrs_temp[:, :]
        LESIONcvimg = cv2.resize(slice2Ddata, None, fx=resolution[0] * res_f, fy=resolution[1] * res_f,
                                 interpolation=cv2.INTER_NEAREST)
        # print(LESIONcvimg.shape)
        # print(LESIONrs.shape)
        LESIONrs[pad_x:pad_x + LESIONcvimg.shape[0], pad_y:pad_y + LESIONcvimg.shape[1], slice_i] = LESIONcvimg
        if slices > 1:
            slice2Ddata = BG_rois_temp[:, :, slice_i]
        else:
            slice2Ddata = BG_rois_temp[:, :]
        BGcvimg = cv2.resize(slice2Ddata, None, fx=resolution[0] * res_f, fy=resolution[1] * res_f,
                             interpolation=cv2.INTER_NEAREST)
        BG_rois[pad_x:pad_x + BGcvimg.shape[0], pad_y:pad_y + BGcvimg.shape[1], slice_i] = BGcvimg
        if slices > 1:
            slice2Ddata = LESIONDATArs_temp[:, :, slice_i]
        else:
            slice2Ddata = LESIONDATArs_temp[:, :]
        DATAcvimg = cv2.resize(slice2Ddata, None, fx=resolution[0] * res_f, fy=resolution[1] * res_f,
                               interpolation=cv2.INTER_LANCZOS4)
        LESIONDATArs[pad_x:pad_x + DATAcvimg.shape[0], pad_y:pad_y + DATAcvimg.shape[1], slice_i] = DATAcvimg
    outdata = []
    for threshold_FWHM in thresholds_FWHM:
        outdata.append({'FWHM': threshold_FWHM, 'data_lo': np.zeros_like(LESIONrs), 'data_hi': np.zeros_like(LESIONrs)})

    for slice_i in range(LESIONDATArs.shape[2]):
        if (np.max(LESIONrs[:, :, slice_i]) == 0 and np.max(BG_rois[:, :, slice_i]) == 0):
            continue
        img = LESIONDATArs[:, :, slice_i]
        for threshold_FWHM_i in range(len(thresholds_FWHM)):
            data_lo = gaussian_filter(img, thresholds_FWHM[threshold_FWHM_i])
            data_hi = np.subtract(img, data_lo)
            outdata[threshold_FWHM_i]['data_lo'][:, :, slice_i] = data_lo
            outdata[threshold_FWHM_i]['data_hi'][:, :, slice_i] = data_hi
        # print(('%d/%d FFT2D' % (slice_i, LESIONDATArs.shape[2])))

    ret = []
    for threshold_FWHM_i in range(len(thresholds_FWHM)):
        ret = append_FFT2D_results(outdata[threshold_FWHM_i], LESIONrs, BG_rois, ret)

    return ret


def append_FFT2D_results_BG(outdata, BG_rois, ret):
    ROIdata = outdata['data_lo'][BG_rois > 0]
    mean1 = np.mean(ROIdata)
    ret.append(np.mean(ROIdata))
    median1 = np.median(ROIdata)
    ret.append(np.median(ROIdata))
    std1 = np.std(ROIdata)
    ret.append(std1)
    iqr1 = iqr(ROIdata)
    ret.append(iqr(ROIdata))
    ret.append(scipy.stats.skew(ROIdata))
    ret.append(scipy.stats.kurtosis(ROIdata))
    ret.append(np.percentile(ROIdata, 25))
    ret.append(np.percentile(ROIdata, 75))
    ROIdata = outdata['data_hi'][BG_rois > 0]
    mean1 = np.mean(ROIdata)
    ret.append(np.mean(ROIdata))
    median1 = np.median(ROIdata)
    ret.append(np.median(ROIdata))
    std1 = np.std(ROIdata)
    ret.append(std1)
    iqr1 = iqr(ROIdata)
    ret.append(iqr(ROIdata))
    ret.append(scipy.stats.skew(ROIdata))
    ret.append(scipy.stats.kurtosis(ROIdata))
    ret.append(np.percentile(ROIdata, 25))
    ret.append(np.percentile(ROIdata, 75))
    return ret


def casefun_3D_2D_FFT2D_BG(LESIONDATAr, LESIONr, BG_roi, resolution, params):
    if np.max(LESIONDATAr) == 0:
        return [float('nan') for x in casefun_3D_2D_FFT2D_names_generator(params)]

    # res_f: resolution factor affecting laws feature sampling ratio
    # 1: original resolution
    # <1: upsampling
    # >1: downsampling
    # start_FWHM: starting FWHM threshold in mm
    # end_FWHM: starting FWHM threshold in mm
    # step_FWHM: starting FWHM threshold in mm
    res_f = params[0]
    start_FWHM = params[1]
    end_FWHM = params[2]
    step_FWHM = params[3]
    thresholds_FWHM = [float(x) for x in np.linspace(int(start_FWHM), int(end_FWHM), int(step_FWHM))]

    # print(np.max(LESIONDATAr))
    x_lo, x_hi, y_lo, y_hi = find_bounded_subregion3D2D(LESIONDATAr)
    # print((x_lo, x_hi, y_lo, y_hi))
    LESIONDATArs_temp = LESIONDATAr[x_lo:x_hi, y_lo:y_hi, :]
    BG_rois_temp = BG_roi[x_lo:x_hi, y_lo:y_hi, :]

    # Create masks and output data to desired resolution, starting with 1mm x 1mm resolution
    slice2Ddata = LESIONDATArs_temp[:, :, 0]
    # print(resolution)
    # print(res_f)
    # print(slice2Ddata.shape)
    cvimg = cv2.resize(slice2Ddata, None, fx=resolution[0] * res_f, fy=resolution[1] * res_f,
                       interpolation=cv2.INTER_NEAREST)
    out_dim = np.max([cvimg.shape[0], cvimg.shape[1]])
    if np.mod(out_dim, 2) == 0:
        out_dim += 1

    # FWHM = 2*sigma*sqrt(2*ln(2))=2.35*sigma
    # more accurately 2.3548200450309493
    try:
        for threshold_FWHM_i in range(len(thresholds_FWHM)):
            kern = gkern2(out_dim, thresholds_FWHM[threshold_FWHM_i] / 2.3548200450309493)
            Y = kern[out_dim // 2, :]
            X = range(len(Y))
            fwhm_est = FWHM(X, Y)
            # print('FWHM[' + str(threshold_FWHM_i+1) + ']:' + str(fwhm_est) + 'mm')
    except:
        print('FWHM estimation failed')
        return [float('nan') for x in casefun_3D_2D_FFT2D_names_generator(params)]

    pad_x = out_dim - cvimg.shape[0]
    pad_y = out_dim - cvimg.shape[1]
    # print((out_dim, pad_x, pad_y))
    BG_rois = np.zeros([out_dim, out_dim, LESIONDATAr.shape[2]])
    LESIONDATArs = np.zeros_like(BG_rois)
    for slice_i in range(BG_rois.shape[2]):
        slice2Ddata = BG_rois_temp[:, :, slice_i]
        BGcvimg = cv2.resize(slice2Ddata, None, fx=resolution[0] * res_f, fy=resolution[1] * res_f,
                             interpolation=cv2.INTER_NEAREST)
        BG_rois[pad_x:pad_x + BGcvimg.shape[0], pad_y:pad_y + BGcvimg.shape[1], slice_i] = BGcvimg
        slice2Ddata = LESIONDATArs_temp[:, :, slice_i]
        DATAcvimg = cv2.resize(slice2Ddata, None, fx=resolution[0] * res_f, fy=resolution[1] * res_f,
                               interpolation=cv2.INTER_LANCZOS4)
        LESIONDATArs[pad_x:pad_x + DATAcvimg.shape[0], pad_y:pad_y + DATAcvimg.shape[1], slice_i] = DATAcvimg
    outdata = []
    for threshold_FWHM in thresholds_FWHM:
        outdata.append({'FWHM': threshold_FWHM, 'data_lo': np.zeros_like(BG_rois), 'data_hi': np.zeros_like(BG_rois)})

    for slice_i in range(LESIONDATArs.shape[2]):
        if (np.max(BG_rois[:, :, slice_i]) == 0):
            continue
        img = LESIONDATArs[:, :, slice_i]
        for threshold_FWHM_i in range(len(thresholds_FWHM)):
            data_lo = gaussian_filter(img, thresholds_FWHM[threshold_FWHM_i])
            data_hi = np.subtract(img, data_lo)
            outdata[threshold_FWHM_i]['data_lo'][:, :, slice_i] = data_lo
            outdata[threshold_FWHM_i]['data_hi'][:, :, slice_i] = data_hi
        # print(('%d/%d FFT2D' % (slice_i, LESIONDATArs.shape[2])))

    ret = []
    for threshold_FWHM_i in range(len(thresholds_FWHM)):
        ret = append_FFT2D_results_BG(outdata[threshold_FWHM_i], BG_rois, ret)

    return ret
