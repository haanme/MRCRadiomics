

from Feature import FeatureIndexandBackground
import numpy as np
import features.Utils
import scipy.stats
import pywt

"""
Radiomic feature base class for methods using one index lesion and background mask
"""


class Wavelet(FeatureIndexandBackground):

    """
    Wavelet names
    """
    casefun_3D_2D_Wavelet_names = ('median', 'skewness', 'range')
    """
    Wavelet coefficient names
    """
    casefun_3D_2D_Wavelet_cnames = (
        'L0c1', 'L1avg1', 'L1avg2', 'L1avg3', 'L1avg4', 'L2avg1', 'L2avg2', 'L2avg3', 'L2avg4', 'L2avg5', 'L2avg6')

    """
    Initialization

    @param params: name of the feature class, without spaces
    1) wavelet type to be passed to pywt.wavedec2
    2) resolution factor affecting laws feature sampling ratio
    # 1: original resolution
    # <1: upsampling
    # >1: downsampling
    """

    def __init__(self, params):
        super(Wavelet, self).__init__('Wavelet', params)


    """
    Wavelet for one slice
    @param slicedata: one slice
    @param waveletname: wavelet type to be taken
    @param levels: composition levels
    """
    def wavelet_2D_slice4(self, slicedata, waveletname, levels):
        s = 16
        output = np.zeros([slicedata.shape[0], slicedata.shape[1], 11])
        mid = int(np.floor(s / 2.0))
        for (x, y, window) in features.Utils.sliding_window(slicedata, 1, (s, s)):
            if np.min(window) == np.max(window):
                continue
            coeffs = pywt.wavedec2(window, waveletname, mode='periodization', level=levels)
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
            if len(coeffs) > 4:
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



    """
    Wavelet for one slice
    @param data: input 3D data
    @param roidata: mask 3D data
    @waveletname: wavelet type
    @levels: composition levels
    """
    def wavelet_2D(self, data, roidata, waveletname, levels):
        if len(data.shape) > 2:
            # print('data.shape:' + str(data.shape))
            outdata = np.zeros([data.shape[0], data.shape[1], data.shape[2], 11])
            for slice_i in range(data.shape[2]):
                if np.max(roidata[:, :, slice_i]) == 0:
                    # print('Skipped [outside ROI] ' + str(slice_i+1) + '/' + str(data.shape[2]))
                    continue
                slicedata = data[:, :, slice_i]
                output = self.wavelet_2D_slice4(slicedata, waveletname, levels)
                outdata[:, :, slice_i, :] = output
                # print('Filtered ' + str(slice_i+1) + '/' + str(data.shape[2]))
            # print('outdata.shape:' + str(outdata.shape))
        else:
            outdata = np.zeros([data.shape[0], data.shape[1], 1, 11])
            if not (np.max(roidata[:, :]) == 0):
                slicedata = data[:, :]
                output = self.wavelet_2D_slice4(slicedata, waveletname, levels)
                outdata[:, :, 0, :] = output

        return outdata


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

        # resolution factor affecting laws feature sampling ratio
        # 1: original resolution
        # <1: upsampling
        # >1: downsampling
        res_f = self.params[1]
        levels = self.params[2]

        x_lo, x_hi, y_lo, y_hi = features.Utils.find_bounded_subregion2D(background_mask_images)

        if len(intensity_images.shape) > 2:
            slices = intensity_images.shape[2]
            LESIONDATArs = intensity_images[x_lo:x_hi, y_lo:y_hi, :]
            LESIONrs_temp = foreground_mask_images[x_lo:x_hi, y_lo:y_hi, :]
            BG_rois_temp = background_mask_images[x_lo:x_hi, y_lo:y_hi, :]
        else:
            slices = 1
            LESIONDATArs = intensity_images[x_lo:x_hi, y_lo:y_hi]
            LESIONrs_temp = foreground_mask_images[x_lo:x_hi, y_lo:y_hi]
            BG_rois_temp = background_mask_images[x_lo:x_hi, y_lo:y_hi]

        # Create masks and output data to desired resolution, intensity data is resliced later for non-zero only
        min_res = np.max(resolution)
        new_res = [min_res * res_f, min_res * res_f, min_res * res_f]
        LESIONrs_temp, affineLESIONrs_temp = features.Utils.reslice_array(LESIONrs_temp, resolution, new_res, 0)
        BG_rois_temp, affineBG_rois_temp = features.Utils.reslice_array(BG_rois_temp, resolution, new_res, 0)
        LESIONDATArs, affineLESIONDATArs = features.Utils.reslice_array(LESIONDATArs, resolution, new_res, 1)

        if np.max(foreground_mask_images) == 0:
            return float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float(
                'nan'), float(
                'nan'), float('nan')

        outdata = self.wavelet_2D(intensity_images, LESIONrs_temp, self.params[0], levels)
        results = []
        for c_i in range(outdata.shape[3]):
            cframe = outdata[:, :, :, c_i]
            ROIdata = cframe[foreground_mask_images > 0]
            BGdata = cframe[background_mask_images > 0]

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

    """
    Returns list of output value short names 

    @return list of return value short names, without spaces
    """

    def get_return_value_short_names(self):
        names = []
        for cname in self.casefun_3D_2D_Wavelet_cnames:
            for name in self.casefun_3D_2D_Wavelet_names:
                names.append('UTUW%s_f%3.2f_%s_%s' % (self.params[0], self.params[1], name, cname))
            for name in self.casefun_3D_2D_Wavelet_names:
                names.append('UTUW%s_f%3.2f_BG%s_%s' % (self.params[0], self.params[1], name, cname))
            for name in self.casefun_3D_2D_Wavelet_names:
                names.append('UTUW%s_f%3.2f_BG_roi%s_%s' % (self.params[0], self.params[1], name, cname))
        return names

    """
    Returns list of strings decsribing boilerplate information about feature, including citations, if any
    """

    @staticmethod
    def get_boilerplate():
        ret = ['2D multilevel decomposition with four levels',
               'Gregory R. Lee, Ralf Gommers, Filip Wasilewski, Kai Wohlfahrt, Aaron O’Leary (2019). PyWavelets: A Python package for wavelet analysis. Journal of Open Source Software, 4(36), 1237, https://doi.org/10.21105/joss.01237.']
        return ret

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

    def get_input_descriptions(self):
        pass

    def number_of_background_mask_images_required(self):
        return 1
