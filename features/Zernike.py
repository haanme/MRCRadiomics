import Feature
import numpy as np
import scipy
import utils

try:
    from skimage import filters
except ImportError:
    from skimage import filter as filters

"""
Class representing Zernike moments features
"""


class Zernike(Feature):
    """
    Initialization

    @param params: parameter list for the feature instance
    1) size of patch
    2) order of zernike moment
    3) repetition number of zernike moment
    """

    def __init__(self, name, params):
        super('Laws2D', params)
        # Size of patch
        self.s = params[0]
        # n = The order of Zernike moment (scalar)
        self.n = params[1]
        # m = The repetition number of Zernike moment (scalar)
        self.m = params[2]

    """
    Zernike moments for a 2D slice
    
    @param slicedata: a 2D slice
    """

    def zernike_2D_slice(self, slicedata):
        from pyzernikemoment import Zernikemoment
        output = np.zeros_like(slicedata)
        mid = int(np.floor(self.s / 2.0))
        for (x, y, window) in utils.sliding_window(slicedata, 1, (self.s, self.s)):
            if np.min(window) == np.max(window):
                continue
            val = filters.threshold_otsu(window)
            window2 = np.zeros_like(window)
            window2[window >= val] = 1
            Z, A, Phi = Zernikemoment(window2, self.n, self.m)
            xmid = x + mid
            ymid = y + mid
            if xmid >= output.shape[0]:
                xmid = output.shape[0] - 1
            if ymid >= output.shape[1]:
                ymid = output.shape[1] - 1
            output[xmid, ymid] = A
        return output

    """
    Calculates zernikes moments for 2D slices
    
    @param data: 3D matrix
    @param roidata: 3D mask for region of interest
    """

    def zernike_2D(self, data, roidata):
        outdata = np.zeros(data.shape)
        for slice_i in range(data.shape[2]):
            if np.max(roidata[:, :, slice_i]) == 0:
                continue
            if len(data.shape) > 3:
                for t in range(data.shape[3]):
                    slicedata = data[:, :, slice_i, t]
                    Z_amplitude = self.zernike_2D_slice(slicedata)
                    outdata[:, :, slice_i, t] = Z_amplitude.transpose()
            else:
                slicedata = data[:, :, slice_i]
                Z_amplitude = self.zernike_2D_slice(slicedata)
                outdata[:, :, slice_i] = Z_amplitude.transpose()
        outdata = np.divide(outdata, np.max(outdata))
        return [outdata]


    """
    Executes the feature

    @param intensity_images: intensity values images
    @param mask_images: mask images
    @param resolution: image resolution mm x mm x mm
    @return number of return values matching get_return_value_descriptions
    """

    def fun(self, intensity_images, mask_images, resolution, **kwargs):
        if np.max(mask_images) == 0:
            return float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float(
                'nan'), float(
                'nan'), float('nan')
        Zdata = self.zernike_2D(intensity_images, mask_images)
        ROIdata = Zdata[0][mask_images > 0]

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

    """
    Returns list of output value short names 

    @return list of return value short names, without spaces
    """

    def get_return_value_short_names(self):
        casefun_names = []
        for statname in ['mean', 'meadian', '25percentile', '75percentile', 'skewness', 'kutosis', 'SD', 'range', 'CV']
            casefun_names.append('Z' + str(self.s) + '_' + str(self.n) + '_' + str(self.m) + '_' + statname)
        return casefun_names

    """
    Returns list of strings decsribing boilerplate information about feature, including citations, if nay
    """

    @staticmethod
    def get_boilerplate(self):
        ret = []
        ret.append(
            "[1] A. Tahmasbi, F. Saki, S. B. Shokouhi, Classification of Benign and Malignant Masses Based on Zernike Moments, Comput. Biol. Med., vol. 41, no. 8, pp. 726-735, 2011.")
        ret.append(
            "[2] F. Saki, A. Tahmasbi, H. Soltanian-Zadeh, S. B. Shokouhi, Fast opposite weight learning rules with application in breast cancer diagnosis, Comput. Biol. Med., vol. 43, no. 1, pp. 32-41, 2013.")
        return ret


    """
    Returns number of required intensity images
    """

    def number_of_intensity_images_required(self):
        return 1


    """
    Returns number of required background mask images
    """

    def number_of_background_mask_images_required(self):
        return 0
