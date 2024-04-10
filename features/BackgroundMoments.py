

from Feature import FeatureIndexandBackground
import numpy as np
import scipy.stats

"""
Statistical descriptors of intensity value distributions
"""


class BackgroundMoments(FeatureIndexandBackground):
    """
    Names for statistical moments of intensity values inside whole organ

      wmean: mean of
      wmedian: median of organ
      wp25: 25% percentile of organ
      wp75: 75% percentile of organ
      wskewness: skewness of organ
      wkurtosis: kurtosis of organ
      wSD: standard deviation of organ
      wIQrange: range of organ
      volume: volume of organ
    """
    casefun_01_moments_BG_names = (
        'BGmean', 'BGmedian', 'BG25percentile', 'BG75percentile', 'BGskewness', 'BGkurtosis', 'BGSD', 'BGIQR', 'BGml')

    """
    Initialization

    @param name: short name of the feature
    @param params: not used
    """
    def __init__(self, name, params):
        super('BackgroundMoments', params)


    """
    Executes the feature
    
    @param intensity_images: intensity values images
    @param foreground_mask_images: foreground mask images
    @param background_mask_images: background mask images
    @param resolution: image resolution mm x mm x mm
    @return number of return values matching get_return_value_descriptions
    """
    def fun(self, intensity_images, foreground_mask_images, background_mask_images, resolution, **kwargs):
        BGdata = intensity_images[background_mask_images > 0]
        volume = len(BGdata) * (0.001 * resolution[0] * resolution[1] * resolution[2])
        print((resolution, volume))
        wmean = np.mean(BGdata)
        wmedian = np.median(BGdata)
        wp25 = np.percentile(BGdata, 25)
        wp75 = np.percentile(BGdata, 75)
        wskewness = scipy.stats.skew(BGdata)
        wkurtosis = scipy.stats.kurtosis(BGdata)
        wSD = np.std(BGdata)
        wIQrange = scipy.stats.iqr(BGdata)

        return wmean, wmedian, wp25, wp75, wskewness, wkurtosis, wSD, wIQrange, volume

    """
    Returns list of output value short names 
    
    @return list of return value short names, without spaces
    """
    def get_return_value_short_names(self):
        return self.casefun_01_moments_BG_names

    """
    Returns list of strings decsribing boilerplate information about feature, including citations, if nay
    """
    def get_boilerplate(self):
        return ["1st order statistical descriptors on background region"]

    """
    Returns number of required intensity images
    """
    def number_of_intensity_images_required(self):
        return 1

    """
    Returns number of required foreground mask images
    """
    def number_of_foreground_mask_images_required(self):
        return 0

    """
    Returns number of required background mask images
    """
    def number_of_background_mask_images_required(self):
        return 1


