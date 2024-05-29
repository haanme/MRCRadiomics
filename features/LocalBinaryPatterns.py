

from Feature import FeatureIndexandBackground
import numpy as np
from skimage.feature import local_binary_pattern
import scipy.stats



"""
Radiomic feature base class for methods using one index lesion and background mask
"""


class LocalBinaryPatterns(FeatureIndexandBackground):

    """
    Names of return values
    """
    subfun_3D_2D_local_binary_pattern_names = ['mean', 'median', 'p25', 'p75', 'skewness', 'kurtosis', 'SD', 'IQR',
                                               'meanBG', 'medianBG', 'Cmedian', 'Cmean', 'CNR']

    """
    Initialization

    @param name: name of the feature class, without spaces
    @param params: 
    1) Number of circularly symmetric neighbour set points (quantization of the angular space)
    2) Radius of circle (spatial resolution of the operator)
    """

    def __init__(self, params):
        super(LocalBinaryPatterns, self).__init__('LocalBinaryPatterns', params)


    """
    Executes the feature

    @param intensity_images: intensity values images
    @param foreground_mask_images: foreground mask images
    @param background_mask_images: background mask images
    @param resolution: image resolution mm x mm x mm
    @return number of return values matching get_return_value_descriptions
    """

    def fun(self, intensity_images, foreground_mask_images, background_mask_images, resolution, **kwargs):
        if type(intensity_images) == list:
            intensity_images = intensity_images[0]
        if type(foreground_mask_images) == list:
            foreground_mask_images = foreground_mask_images[0]
        if type(background_mask_images) == list:
            background_mask_images = background_mask_images[0]

        # print('LBP')
        # Number of circularly symmetric neighbour set points (quantization of the angular space)
        angles = self.params[0]
        # Radius of circle (spatial resolution of the operator)
        radius = self.params[1]
        outdata = np.zeros_like(intensity_images)
        # print('LBP')
        for slice_i in range(intensity_images.shape[2]):
            slicedata = intensity_images[:, :, slice_i]
            # print(len(slicedata[LESIONr[0][:, :, slice_i] > 0]))
            # print(len(np.unique(slicedata[LESIONr[0][:, :, slice_i] > 0])))
            lpb = local_binary_pattern(slicedata, angles, radius, method='uniform')
            outdata[:, :, slice_i] = lpb
        ROIdata = outdata[foreground_mask_images > 0]
        if (len(ROIdata)) == 0:
            return float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float(
                'nan'), float(
                'nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan')
        BGdata = outdata[background_mask_images > 0]
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

    """
    Returns list of output value short names 

    @return list of return value short names, without spaces
    """

    def get_return_value_short_names(self):
        casefun_3D_2D_local_binary_pattern_names = []
        for name in self.subfun_3D_2D_local_binary_pattern_names:
            casefun_3D_2D_local_binary_pattern_names.append('LPB' + str(self.params[0]) + str(self.params[1]) + '_' + name)
        return casefun_3D_2D_local_binary_pattern_names

    """
    Returns list of input value descriptions 

    @return list of stings, or None
    """

    def get_input_descriptions(self):
        return ["Number of circularly symmetric neighbour set points (quantization of the angular space)",
                "Radius of circle (spatial resolution of the operator)"]

    """
    Returns list of strings decsribing boilerplate information about feature, including citations, if nay
    """

    @staticmethod
    def get_boilerplate():
        return ["[R387388]	Multiresolution Gray-Scale and Rotation Invariant Texture Classification with Local Binary Patterns. Timo Ojala, Matti Pietikainen, Topi Maenpaa. http://www.rafbis.it/biplab15/images/stories/docenti/Danielriccio/Articoliriferimento/LBP.pdf, 2002.",
                "[R388388]	(1, 2) Face recognition with local binary patterns. Timo Ahonen, Abdenour Hadid, Matti Pietikainen, http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.214.6851, 2004."]
        pass


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
