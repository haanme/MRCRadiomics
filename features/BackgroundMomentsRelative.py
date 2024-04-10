

from Feature import FeatureIndexandBackground
import numpy as np
import scipy.stats

"""
Statistical descriptors of intensity value distributions
"""


class BackgroundMomentsRelative(FeatureIndexandBackground):
    """
    Names for first order statistics of raw intensity using background region.

      relBGmean: Relative mean intensity to background
      relBGmedian: Relative mean intensity to background
      relBG25percentile: : Relative 25% percentile to background
      relBG75percentile: : Relative 75% percentile to background
      relBGskewness: : Relative skewness to background
      relBGkurtosis: : Relative kurtosity to background
      relBGSD: : Relative standard deviation to background
      BGIQR: : Relative iterquatile range to background
    """
    casefun_01_moments_relativeBG_names = (
        'relBGmean', 'relBGmedian', 'relBG25percentile', 'relBG75percentile', 'relBGskewness', 'relBGkurtosis',
        'relBGSD',
        'BGIQR')

    """
    Initialization

    @param name: short name of the feature
    @param params: not used
    """
    def __init__(self, name, params):
        super('BackgroundMomentsRelative', params)


    """
    Executes the feature
    
    @param intensity_images: intensity values images
    @param foreground_mask_images: foreground mask images
    @param background_mask_images: background mask images
    @param resolution: image resolution mm x mm x mm
    @return number of return values matching get_return_value_descriptions
    """
    def fun(self, intensity_images, foreground_mask_images, background_mask_images, resolution, **kwargs):
        ROIdata = intensity_images[foreground_mask_images > 0]
        BGdata = intensity_images[background_mask_images > 0]
        volume = len(BGdata) * (0.001 * resolution[0] * resolution[1] * resolution[2])
        print((resolution, volume))
        mean = np.mean(ROIdata)
        median = np.median(ROIdata)
        print(ROIdata.shape)
        p25 = np.percentile(ROIdata, 25)
        p75 = np.percentile(ROIdata, 75)
        skewness = scipy.stats.skew(ROIdata)
        kurtosis = scipy.stats.kurtosis(ROIdata)
        SD = np.std(ROIdata)
        IQrange = scipy.stats.iqr(ROIdata)
        if mean == 0:
            BGmean = 0
        else:
            BGmean = mean / (mean + np.mean(BGdata))
        if median == 0:
            BGmedian = 0
        else:
            BGmedian = median / (median + np.median(BGdata))
        if p25 == 0:
            BGp25 = 0
        else:
            BGp25 = p25 / (p25 + np.percentile(BGdata, 25))
        if p75 == 0:
            BGp75 = 0
        else:
            BGp75 = p75 / (p75 + np.percentile(BGdata, 75))
        print(skewness)
        if skewness == 0:
            BGskewness = 0
        else:
            BGskewness = skewness / (skewness + scipy.stats.skew(BGdata))
        if kurtosis == 0:
            BGkurtosis = 0
        else:
            BGkurtosis = kurtosis / (kurtosis + scipy.stats.kurtosis(BGdata))
        if SD == 0:
            BGSD = 0
        else:
            BGSD = SD / (SD + np.std(BGdata))
        if IQrange == 0:
            BGIQrange = 0
        else:
            BGIQrange = IQrange / (IQrange + scipy.stats.iqr(BGdata))

        return BGmean, BGmedian, BGp25, BGp75, BGskewness, BGkurtosis, BGSD, BGIQrange

    """
    Returns list of output value short names 
    
    @return list of return value short names, without spaces
    """
    def get_return_value_short_names(self):
        return self.casefun_01_moments_relativeBG_names

    """
    Returns list of strings decsribing boilerplate information about feature, including citations, if nay
    """
    def get_boilerplate(self):
        return ["1st order statistics of raw intensity using background region"]

    """
    Returns number of required intensity images
    """
    def number_of_intensity_images_required(self):
        return 1

    """
    Returns number of required foreground mask images
    """
    def number_of_foreground_mask_images_required(self):
        return 1

    """
    Returns number of required background mask images
    """
    def number_of_background_mask_images_required(self):
        return 1


