

from Feature import FeatureIndexandBackground
import numpy as np
from scipy.signal import correlate2d
from scipy import stats
import cv2
from scipy.stats import iqr
from features.Utils import find_bounded_subregion3D2D, sliding_window


"""
Laws features
"""


class Laws2D(FeatureIndexandBackground):

    """
    Names of extracted Laws feature statistical descriptors
    """
    casefun_3D_2D_Laws_names = ['mean_ROI', 'median_ROI', 'SD_ROI', 'IQR_ROI', 'skewnessROI', 'kurtosisROI', 'p25ROI', 'p75ROI', 'rel']

    """
    Laws feature matrix names
    """
    laws_names = ['1_L5E5_E5L5', '2_L5R5_R5L5', '3_E5S5_S5E5', '4_S5S5', '5_R5R5', '6_L5S5_S5E5', '7_E5E5', '8_E5R5_R5E5', '9_S5R5_R5S5']

    """
    Initialization

    @param params: parameter list for the feature instance
    1) resolution factor affecting laws feature sampling ratio
      1: original resolution
     <1: upsampling
     >1: downsampling
    """

    def __init__(self, params):
        super(Laws2D, self).__init__('Laws2D', params)
        # Define the 1D kernels
        L5 = np.array([1, 4, 6, 4, 1])  # level
        E5 = np.array([-1, -2, 0, 2, 1])  # edge
        S5 = np.array([-1, 0, 2, 0, -1])  # spot
        W5 = np.array([-1, 2, 0, -2, 1])  # waves
        R5 = np.array([1, -4, 6, -4, 1])  # ripples
        # Generate 2D kernels
        self.L5L5 = np.outer(L5, L5)
        self.L5E5 = np.outer(L5, E5)
        self.L5S5 = np.outer(L5, S5)
        self.L5R5 = np.outer(L5, R5)
        self.L5W5 = np.outer(L5, W5)
        self.E5L5 = np.outer(E5, L5)
        self.E5E5 = np.outer(E5, E5)
        self.E5S5 = np.outer(E5, S5)
        self.E5R5 = np.outer(E5, R5)
        self.E5W5 = np.outer(E5, W5)
        self.S5L5 = np.outer(S5, L5)
        self.S5E5 = np.outer(S5, E5)
        self.S5S5 = np.outer(S5, S5)
        self.S5R5 = np.outer(S5, R5)
        self.S5W5 = np.outer(S5, W5)
        self.R5L5 = np.outer(R5, L5)
        self.R5E5 = np.outer(R5, E5)
        self.R5S5 = np.outer(R5, S5)
        self.R5R5 = np.outer(R5, R5)
        self.R5W5 = np.outer(R5, W5)

    """
    Appends statistics to output array
    @param lmask: lesion mask
    @param bgmask: background mask
    @parasm ret: return array
    @return ret appended with statistics
    """
    def append_Laws_results(self, outdata, lmask, bgmask, ret):
        ROIdata = outdata[lmask > 0]
        BGdata = outdata[bgmask > 0]
        mean1 = np.mean(ROIdata)
        ret.append(np.mean(ROIdata))
        median1 = np.median(ROIdata)
        ret.append(np.median(ROIdata))
        std1 = np.std(ROIdata)
        ret.append(std1)
        iqr1 = iqr(ROIdata)
        ret.append(iqr(ROIdata))
        ret.append(stats.skew(ROIdata))
        ret.append(stats.kurtosis(ROIdata))
        ret.append(np.percentile(ROIdata, 25))
        ret.append(np.percentile(ROIdata, 75))
        if (mean1 == 0):
            ret.append(0)
        else:
            ret.append(mean1 / (mean1 + np.mean(BGdata)))
        return ret


    """
    Executes the feature

    @param intensity_images: intensity values images
    @param foreground_mask_images: foreground mask images
    @param background_mask_images: background mask images
    @param resolution: image resolution mm x mm x mm
    @param params: feature specific additional parameters
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
        res_f = self.params[0]

        if np.max(intensity_images) == 0:
            return [float('nan') for x in self.get_return_value_short_names()]

        x_lo, x_hi, y_lo, y_hi = find_bounded_subregion3D2D(intensity_images)
        # print('bounded_subregion3D2D:' + str((x_lo, x_hi, y_lo, y_hi)))
        if len(intensity_images.shape) > 2:
            slices = intensity_images.shape[2]
            LESIONDATArs = intensity_images[x_lo:x_hi, y_lo:y_hi, :]
            LESIONrs_temp = foreground_mask_images[x_lo:x_hi, y_lo:y_hi, :]
            BG_rois_temp = background_mask_images[x_lo:x_hi, y_lo:y_hi, :]
            # Create masks and output data to desired resolution, intensity data is resliced later for non-zero only
            slice2Ddata = LESIONDATArs[:, :, 0]
        else:
            slices = 1
            LESIONDATArs = intensity_images[x_lo:x_hi, y_lo:y_hi]
            LESIONrs_temp = foreground_mask_images[x_lo:x_hi, y_lo:y_hi]
            BG_rois_temp = background_mask_images[x_lo:x_hi, y_lo:y_hi]
            # Create masks and output data to desired resolution, intensity data is resliced later for non-zero only
            slice2Ddata = LESIONDATArs[:, :]
        cvimg = cv2.resize(slice2Ddata, None, fx=resolution[0] * res_f, fy=resolution[1] * res_f,
                           interpolation=cv2.INTER_NEAREST)
        if slices == 1:
            LESIONrs = np.zeros([cvimg.shape[0], cvimg.shape[1]])
        else:
            LESIONrs = np.zeros([cvimg.shape[0], cvimg.shape[1], intensity_images.shape[2]])
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
            return [float('nan') for x in self.get_return_value_short_names()]

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
                fL5E5 = correlate2d(window, self.L5E5)[2:7, 2:7]
                fE5L5 = correlate2d(window, self.E5L5)[2:7, 2:7]
                fL5R5 = correlate2d(window, self.L5R5)[2:7, 2:7]
                fR5L5 = correlate2d(window, self.R5L5)[2:7, 2:7]
                fE5S5 = correlate2d(window, self.E5S5)[2:7, 2:7]
                fS5E5 = correlate2d(window, self.S5E5)[2:7, 2:7]
                fS5S5 = correlate2d(window, self.S5S5)[2:7, 2:7]
                fR5R5 = correlate2d(window, self.R5R5)[2:7, 2:7]
                fL5S5 = correlate2d(window, self.L5S5)[2:7, 2:7]
                fS5L5 = correlate2d(window, self.S5L5)[2:7, 2:7]
                fE5E5 = correlate2d(window, self.E5E5)[2:7, 2:7]
                fE5R5 = correlate2d(window, self.E5R5)[2:7, 2:7]
                fR5E5 = correlate2d(window, self.R5E5)[2:7, 2:7]
                fS5R5 = correlate2d(window, self.S5R5)[2:7, 2:7]
                fR5S5 = correlate2d(window, self.R5S5)[2:7, 2:7]
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
        ret = self.append_Laws_results(outdata_1, LESIONrs, BG_rois, ret)
        ret = self.append_Laws_results(outdata_2, LESIONrs, BG_rois, ret)
        ret = self.append_Laws_results(outdata_3, LESIONrs, BG_rois, ret)
        ret = self.append_Laws_results(outdata_4, LESIONrs, BG_rois, ret)
        ret = self.append_Laws_results(outdata_5, LESIONrs, BG_rois, ret)
        ret = self.append_Laws_results(outdata_6, LESIONrs, BG_rois, ret)
        ret = self.append_Laws_results(outdata_7, LESIONrs, BG_rois, ret)
        ret = self.append_Laws_results(outdata_8, LESIONrs, BG_rois, ret)
        ret = self.append_Laws_results(outdata_9, LESIONrs, BG_rois, ret)
        return ret

    """
    Returns list of output value short names 

    @return list of return value short names, without spaces
    """

    def get_return_value_short_names(self):
        names = []
        for name_i in range(len(self.laws_names)):
            for name in self.casefun_3D_2D_Laws_names:
                names.append('UTU3D2DLaws%s_%s_f%2.1f' % (self.laws_names[name_i], name, self.params[0]))
        return names

    """
    Returns list of input value descriptions 

    @return list of stings, or None
    """

    def get_input_descriptions(self):
        return ["Resolution factor affecting laws feature sampling ratio 1: original resolution <1: upsampling >1: downsampling"]

    """
    Returns list of strings decsribing boilerplate information about feature, including citations, if nay
    """

    @staticmethod
    def get_boilerplate():
        return ['Laws texture features',
                'K. Laws "Textured Image Segmentation", Ph.D. Dissertation, University of Southern California, January 1980',
                'A. Meyer-Base, "Pattern Recognition for Medical Imaging", Academic Press, 2004.',
                '5th vector']
