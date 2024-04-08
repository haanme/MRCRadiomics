#!/usr/bin/env python

from scipy.stats import iqr
import scipy.ndimage
import cv2
import numpy as np
import dwi.standardize
import textures_2D


def make_cv2_slice2D(slice2D):
    # re-scale to 0..255
    slice2D -= np.min(slice2D)
    slice2D = (slice2D/np.max(slice2D))*255.0
    cvimg = cv2.resize(slice2D.astype(np.uint8), (slice2D.shape[0], slice2D.shape[1]))
    return cvimg


def procfun_N_Kwak(data, roidata, params):
    data_ROI = data[np.where(roidata>0)]
    th1 = np.percentile(data_ROI, 1)
    th99 = np.percentile(data_ROI, 99)
    roidata[np.where(data <= th1)] = 0
    roidata[np.where(data >= th99)] = 0
    data_ROI = data[roidata>0]
    m = np.median(data_ROI)
    sd = np.std(data_ROI)
    d = m + 2 * sd
    if d > 0:
        data_ROI = np.divide(data_ROI, d)
    else:
        data_ROI = np.zeros_like(data_ROI)
    data[roidata > 0] = data_ROI
    return data


# Features for each
casefun_ADCKwak_01_moments_names = ('meanKwak', 'medianKwak', '25percentileKwak', '75percentileKwak', 'skewnessKwak', 'kurtosisKwak', 'SDKwak', 'IQRKwak',
                                    'relWGmeanKwak', 'relWGmedianKwak', 'relWG25percentileKwak', 'relWG75percentileKwak', 'relWGskewnessKwak', 'relWGkurtosisKwak', 'relWGSDKwak', 'relWGIQRKwak')
def casefun_ADCKwak_01_moments(LESIONDATAr, LESIONr, WGr, resolution):
    # ADC normalization by Kwak et al
    WGdata_all = LESIONDATAr[WGr > 0]
    th1 = np.percentile(WGdata_all, 1)
    th99 = np.percentile(WGdata_all, 99)
    WGr_temp = np.zeros_like(WGr)
    WGr_temp[np.where(WGr > 0)] = 1
    WGr_temp[np.where(LESIONDATAr <= th1)] = 0
    WGr_temp[np.where(LESIONDATAr >= th99)] = 0
    WGdata_all = LESIONDATAr[WGr_temp > 0]
    all_median = np.median(WGdata_all)
    all_SD = np.std(WGdata_all)
    d = all_median + 2 * all_SD
    if d > 0:
        LESIONDATAr = np.divide(LESIONDATAr, d)
    else:
        LESIONDATAr = np.zeros_like(LESIONDATAr)
    ROIdata = LESIONDATAr[LESIONr[0] > 0]
    WGr[LESIONr[0] == 1] = 0
    WGdata = LESIONDATAr[WGr > 0]

    mean = np.mean(ROIdata)
    median = np.median(ROIdata)
    p25 = np.percentile(ROIdata, 25)
    p75 = np.percentile(ROIdata, 75)
    skewness = scipy.stats.skew(ROIdata)
    kurtosis = scipy.stats.kurtosis(ROIdata)
    SD = np.std(ROIdata)
    IQrange = iqr(ROIdata)

    wmean = np.mean(WGdata)
    wmedian = np.median(WGdata)
    wp25 = np.percentile(WGdata, 25)
    wp75 = np.percentile(WGdata, 75)
    wskewness = scipy.stats.skew(WGdata)
    wkurtosis = scipy.stats.kurtosis(WGdata)
    wSD = np.std(WGdata)
    wIQrange = iqr(WGdata)

    if(mean == 0):
      WGmean = 0
    else:
      WGmean = mean/(mean+np.mean(WGdata))
    if(median == 0):
      WGmedian = 0
    else:
      WGmedian = median/(median+np.median(WGdata))
    if(p25 == 0):
      WGp25 = 0
    else:
      WGp25 = p25/(p25+np.percentile(WGdata, 25))
    if(p75 == 0):
      WGp75 = 0
    else:
      WGp75 = p75/(p75+np.percentile(WGdata, 75))
    if(skewness == 0):
      WGskewness = 0
    else:
      WGskewness = skewness/(skewness+scipy.stats.skew(WGdata))
    if(kurtosis == 0):
      WGkurtosis = 0
    else:
      WGkurtosis = kurtosis/(kurtosis+scipy.stats.kurtosis(WGdata))
    if(SD == 0):
      WGSD = 0
    else:
      WGSD = SD/(SD+np.std(WGdata))
    if(IQrange == 0):
      WGIQrange = 0
    else:
      WGIQrange = IQrange/(IQrange+iqr(WGdata))

    return mean, median, p25, p75, skewness, kurtosis, SD, IQrange, WGmean, WGmedian, WGp25, WGp75, WGskewness, WGkurtosis, WGSD, WGIQrange



# Features for each
casefun_ADCKwak_01_moments_names_WG = ('WGmeanKwak', 'WGmedianKwak', 'WG25percentileKwak', 'WG75percentileKwak', 'WGskewnessKwak', 'WGkurtosisKwak', 'WGSDKwak', 'WGIQRKwak')
def casefun_ADCKwak_01_moments_WG(LESIONDATAr, LESIONr, WGr, resolution):
    # ADC normalization by Kwak et al
    WGdata_all = LESIONDATAr[WGr > 0]
    th1 = np.percentile(WGdata_all, 1)
    th99 = np.percentile(WGdata_all, 99)
    WGr_temp = np.zeros_like(WGr)
    WGr_temp[np.where(WGr > 0)] = 1
    WGr_temp[np.where(LESIONDATAr <= th1)] = 0
    WGr_temp[np.where(LESIONDATAr >= th99)] = 0
    WGdata_all = LESIONDATAr[WGr_temp > 0]
    all_median = np.median(WGdata_all)
    all_SD = np.std(WGdata_all)
    d = all_median + 2 * all_SD
    if d > 0:
        LESIONDATAr = np.divide(LESIONDATAr, d)
    else:
        LESIONDATAr = np.zeros_like(LESIONDATAr)
    WGdata = LESIONDATAr[WGr > 0]

    wmean = np.mean(WGdata)
    wmedian = np.median(WGdata)
    wp25 = np.percentile(WGdata, 25)
    wp75 = np.percentile(WGdata, 75)
    wskewness = scipy.stats.skew(WGdata)
    wkurtosis = scipy.stats.kurtosis(WGdata)
    wSD = np.std(WGdata)
    wIQrange = iqr(WGdata)

    return wmean, wmedian, wp25, wp75, wskewness, wkurtosis, wSD, wIQrange


"""
Hu, M.K., 1962. Visual pattern recognition by moment invariants. IRE transactions on information theory, 8(2), pp.179-187.
"""
casefun_3D_2D_Hu_moments_rawintensity_names_Nyul = []
for name in textures_2D.contour2D_names:
    casefun_3D_2D_Hu_moments_rawintensity_names_Nyul.append('2D_curvature_Nyul_intensity_' + name)
def casefun_3D_2D_Hu_moments_rawintensity_Nyul(LESIONDATAr, LESIONr, WGr, resolution):
    if np.max(LESIONDATAr) == 0:
        return [float('nan') for x in casefun_3D_2D_Hu_moments_rawintensity_names_Nyul]

    ROIdata = LESIONDATAr[LESIONr[0] > 0]
    WGr[LESIONr[0] == 1] = 0
    WGdata = LESIONDATAr[WGr > 0]
    p, scores = dwi.standardize.landmark_scores(LESIONDATAr, Nyul_d['pc'], Nyul_d['landmarks'], Nyul_d['thresholding'], mask=WGr.astype(np.bool))

    # x_lo, x_hi, y_lo, y_hi, z_lo, z_hi = textures_3D.find_bounded_subregion3D(LESIONDATAr)
    # LESIONDATArs = LESIONDATAr[x_lo:x_hi, y_lo:y_hi, :]
    labelimage = LESIONDATAr
    labelimage[LESIONr[0] == 0] = 0

    # Normalize intensity values
    labelimage = dwi.standardize.transform(labelimage, p, scores, Nyul_d['scale'], Nyul_d['mapped_scores'])

    return textures_2D.subfun_3D_2D_Hu_moments(labelimage, resolution)
    