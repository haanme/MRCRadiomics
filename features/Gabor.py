

from Feature import FeatureIndexandBackground
import numpy as np
import os
import scipy.stats
import skimage.filters.gabor
from visualizations import visualizations
import utils

"""
2D Gabor filter using skimage python package
"""


class Gabor(FeatureIndexandBackground):

    """
    Names of the calcualted statistics
    """
    subfun_3D_2D_gabor_filter_names = ['mean', 'median', 'p25', 'p75', 'skewness', 'kurtosis', 'SD', 'IQR', 'meanBG',
                                       'medianBG', 'Cmedian', 'Cmean', 'CNR']

    """
    Initialization

    @param name: short name of the feature
    @param params: parameter list for the feature instance
    1) Number of circularly symmetric neighbour set points (quantization of the angular space)
    2) Radius of circle (spatial resolution of the operator)
    3) kernel size in SD units
    """
    def __init__(self, name, params):
        super('Gabor', params)


    """
    Executes the feature
    
    @param intensity_images: intensity values images
    @param foreground_mask_images: foreground mask images
    @param background_mask_images: background mask images
    @param resolution: image resolution mm x mm x mm
    @return number of return values matching get_return_value_descriptions
    """
    def fun(self, intensity_images, foreground_mask_images, background_mask_images, resolution, **kwargs):
        """
        params[0] frequency of filter at index 0
        params[1] directions
        params[2] kernel size in SD units
        :return: real component of filtered data
        """
        outdata = np.zeros_like(intensity_images)
        d = self.params[1]
        radians = [x * np.pi / d * 0.5 for x in range(d + 1)]
        if len(intensity_images.shape) > 2:
            slices = intensity_images.shape[2]
        else:
            slices = 1
        for slice_i in range(slices):
            if len(intensity_images.shape) > 3:
                for t in range(intensity_images.shape[3]):
                    slicedata = intensity_images[:, :, slice_i, t]
                    filt_reals = np.zeros([intensity_images.shape[0], intensity_images.shape[1], len(radians)])
                    for r_i in range(len(radians)):
                        filt_real, filt_imag = skimage.filters.gabor(slicedata, frequency=self.params[0], theta=radians[r_i],
                                                                     n_stds=self.params[2])
                        filt_reals[:, :, r_i] = filt_real
                    outdata[:, :, slice_i, t] = np.mean(filt_reals, 2)
            else:
                if slices == 1:
                    slicedata = intensity_images[:, :]
                else:
                    slicedata = intensity_images[:, :, slice_i]
                filt_reals = np.zeros([intensity_images.shape[0], intensity_images.shape[1], len(radians)])
                for r_i in range(len(radians)):
                    filt_real, filt_imag = skimage.filters.gabor(slicedata, frequency=self.params[0], theta=radians[r_i],
                                                                 n_stds=self.params[2])
                    filt_reals[:, :, r_i] = filt_real
                if slices == 1:
                    outdata[:, :] = np.mean(filt_reals, 2)
                else:
                    outdata[:, :, slice_i] = np.mean(filt_reals, 2)
                if (type(self.params) == list) and (len(self.params) > 1) and (not type(self.params[-1]) == int) and (
                        'write_visualization' in self.params[-1]):
                    if not self.params[0] == 1.0:
                        continue
                    if not self.params[1] == 2:
                        continue
                    if not self.params[2] == 2:
                        continue
                    LESIONDATAr_cvimg = utils.make_cv2_slice2D(slicedata).copy()
                    LESIONr_cvimg = utils.make_cv2_slice2D(foreground_mask_images[:, :, slice_i]).copy()
                    BG_roi_cvimg = utils.make_cv2_slice2D(background_mask_images[:, :, slice_i]).copy()
                    outdata_cvimg = utils.make_cv2_slice2D(outdata[:, :, slice_i]).copy()
                    if (np.max(LESIONr_cvimg) == 0):
                        continue
                    basename = self.params[-1]['name'] + '_Gabor_' + str(self.params[:-1]).replace(' ', '_') + '_slice' + str(
                        slice_i)
                    visualizations.write_slice2D(LESIONDATAr_cvimg,
                                                 self.params[-1]['write_visualization'] + os.sep + basename + '_data.tiff')
                    visualizations.write_slice2D_ROI(LESIONDATAr_cvimg, LESIONr_cvimg,
                                                     self.params[-1][
                                                         'write_visualization'] + os.sep + basename + '_lesion.tiff',
                                                     0.4)
                    visualizations.write_slice2D_ROI_color(LESIONDATAr_cvimg, LESIONr_cvimg, outdata_cvimg, self.params[-1][
                        'write_visualization'] + os.sep + basename + '_overlay.tiff', 0.8)
                    visualizations.write_slice2D_ROI_color(LESIONDATAr_cvimg, BG_roi_cvimg, outdata_cvimg, self.params[-1][
                        'write_visualization'] + os.sep + basename + '_BGoverlay.tiff', 0.7)

        ROIdata = outdata[foreground_mask_images > 0]
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
        names = []
        for name in self.subfun_3D_2D_gabor_filter_names:
            names.append('UTUGabor_f%3.2f_d%d_k%d_%s' % (self.params[0], self.params[1], self.params[2], name))
        return names

    """
    Returns list of strings decsribing boilerplate information about feature, including citations, if nay
    """
    def get_boilerplate(self):
        return ['Gagor filter',
                'Gabor, D.(1946)."Theory of communication".J.Inst.Electr.Eng.93.']

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


