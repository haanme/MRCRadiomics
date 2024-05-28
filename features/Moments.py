

from Feature import FeatureIndexandBackground
import numpy as np
import scipy.stats

"""
Statistical descriptors of intensity value distributions
"""


class Moments(FeatureIndexandBackground):

    """
    Names for statistical moments of intensity values inside lesion

      mean: mean of lesion
      median: median of lesion
      p25: 25% percentile of lesion
      p75: 75% percentile of lesion
      skewness: skewness of lesion
      kurtosis: kurtosis of lesion
      SD: standard deviation of lesion
      rng: range of lesion
      volume: volume of lesion
      CV: coefficient of variation for lesion
    """
    casefun_01_moments_names = ('mean', 'median', '25percentile', '75percentile', 'skewness', 'kurtosis', 'SD', 'range', 'ml', 'CV')

    """
    Initialization

    """
    def __init__(self):
        super(Moments, self).__init__('Moments', None)


    """
    Executes the feature
    
    @param intensity_images: intensity values images
    @param foreground_mask_images: foreground mask images
    @param background_mask_images: background mask images
    @param resolution: image resolution mm x mm x mm
    @return number of return values matching get_return_value_descriptions
    """
    def fun(self, intensity_images, foreground_mask_images, background_mask_images, resolution, **kwargs):
        if type(foreground_mask_images) is list:
            foreground_mask_images = foreground_mask_images[0]
        ROIdata = intensity_images[foreground_mask_images > 0]
        mean = np.mean(ROIdata)
        median = np.median(ROIdata)
        p25 = np.percentile(ROIdata, 25)
        p75 = np.percentile(ROIdata, 75)
        skewness = scipy.stats.skew(ROIdata)
        kurtosis = scipy.stats.kurtosis(ROIdata)
        SD = np.std(ROIdata)
        rng = np.max(ROIdata) - np.min(ROIdata)
        # Volume as cubic mm to cubic centimeters, which is mL
        volume = len(ROIdata) * (0.001 * resolution[0] * resolution[1] * resolution[2])
        print((resolution, volume))
        if not mean == 0:
            CV = SD / mean
        else:
            CV = 0.0
        return mean, median, p25, p75, skewness, kurtosis, SD, rng, volume, CV

    """
    Returns list of output value short names 
    
    @return list of return value short names, without spaces
    """
    def get_return_value_short_names(self):
        return self.casefun_01_moments_names

    """
    Returns list of input value descriptions 

    @return list of stings, or None
    """

    def get_input_descriptions(self):
        return None

    """
    Returns list of strings decsribing boilerplate information about feature, including citations, if any
    """

    @staticmethod
    def get_boilerplate():
        return ["1st order statistical descriptors"]

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
        return 0


