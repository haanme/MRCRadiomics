
from Feature import FeatureIndexandBackground
import numpy as np
import scipy.stats
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import UnivariateSpline
import cv2
import utils
import FastFourier2D

"""
Statistical descriptors of intensity value distributions
"""


class FastFourier2D_background(FeatureIndexandBackground):

    """
    Output names
    """
    casefun_3D_2D_FFT2D_names = ['mean_ROI', 'median_ROI', 'SD_ROI', 'IQR_ROI', 'skewnessROI', 'kurtosisROI', 'p25ROI',
                                 'p75ROI', 'rel']
    """
    Initialization

    @param name: short name of the feature
    @param params: not used
    1) 1: original resolution <1: upsampling >1: downsampling
    2) start_FWHM: starting FWHM threshold in mm
    3) end_FWHM: ending FWHM threshold in mm
    4) step_FWHM: setp FWHM threshold in mm
    """
    def __init__(self, name, params):
        super('FastFourier2D_background', params)

    """
    Appends results
    """
    def append_FFT2D_results_BG(self, outdata, BG_rois, ret):
        ROIdata = outdata['data_lo'][BG_rois > 0]
        mean1 = np.mean(ROIdata)
        ret.append(np.mean(ROIdata))
        median1 = np.median(ROIdata)
        ret.append(np.median(ROIdata))
        std1 = np.std(ROIdata)
        ret.append(std1)
        iqr1 = scipy.stats.iqr(ROIdata)
        ret.append(scipy.stats.iqr(ROIdata))
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
        iqr1 = scipy.stats.iqr(ROIdata)
        ret.append(scipy.stats.iqr(ROIdata))
        ret.append(scipy.stats.skew(ROIdata))
        ret.append(scipy.stats.kurtosis(ROIdata))
        ret.append(np.percentile(ROIdata, 25))
        ret.append(np.percentile(ROIdata, 75))
        return ret


    """
    Executes the feature
    
    @param intensity_images: intensity values images
    @param foreground_mask_images: foreground mask images
    @param background_mask_images: background mask images
    @param resolution: image resolution mm x mm x mm
    @return number of return values matching get_return_value_descriptions
    """
    def fun(self, intensity_images, foreground_mask_images, background_mask_images, resolution, **kwargs):
        if np.max(intensity_images) == 0:
            return [float('nan') for x in self.get_return_value_short_names()]

        # res_f: resolution factor affecting laws feature sampling ratio
        # 1: original resolution
        # <1: upsampling
        # >1: downsampling
        # start_FWHM: starting FWHM threshold in mm
        # end_FWHM: starting FWHM threshold in mm
        # step_FWHM: step FWHM threshold in mm
        res_f = self.params[0]
        start_FWHM = self.params[1]
        end_FWHM = self.params[2]
        step_FWHM = self.params[3]
        thresholds_FWHM = [float(x) for x in np.linspace(int(start_FWHM), int(end_FWHM), int(step_FWHM))]

        # print(np.max(LESIONDATAr))
        x_lo, x_hi, y_lo, y_hi = utils.find_bounded_subregion3D2D(intensity_images)
        # print((x_lo, x_hi, y_lo, y_hi))
        LESIONDATArs_temp = intensity_images[x_lo:x_hi, y_lo:y_hi, :]
        BG_rois_temp = background_mask_images[x_lo:x_hi, y_lo:y_hi, :]

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
                kern = FastFourier2D.gkern2(out_dim, thresholds_FWHM[threshold_FWHM_i] / 2.3548200450309493)
                Y = kern[out_dim // 2, :]
                X = range(len(Y))
                fwhm_est = FastFourier2D.FWHM(X, Y)
                # print('FWHM[' + str(threshold_FWHM_i+1) + ']:' + str(fwhm_est) + 'mm')
        except:
            print('FWHM estimation failed')
            return [float('nan') for x in self.get_return_value_short_names()]

        pad_x = out_dim - cvimg.shape[0]
        pad_y = out_dim - cvimg.shape[1]
        # print((out_dim, pad_x, pad_y))
        BG_rois = np.zeros([out_dim, out_dim, intensity_images.shape[2]])
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
            outdata.append(
                {'FWHM': threshold_FWHM, 'data_lo': np.zeros_like(BG_rois), 'data_hi': np.zeros_like(BG_rois)})

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
            ret = self.append_FFT2D_results_BG(outdata[threshold_FWHM_i], BG_rois, ret)

        return ret

    """
    Returns list of output value short names 
    
    @return list of return value short names, without spaces
    """
    def get_return_value_short_names(self):
        res_f = self.params[0]
        start_FWHM = self.params[1]
        end_FWHM = self.params[2]
        step_FWHM = self.params[3]
        thresholds_FWHM = [float(x) for x in np.linspace(int(start_FWHM), int(end_FWHM), int(step_FWHM))]
        # FWHM = 2*sigma*sqrt(2*ln(2))=2.35*sigma
        # more accurately 2.3548200450309493
        names = []
        for threshold_FWHM in thresholds_FWHM:
            for name_i in range(len(self.casefun_3D_2D_FFT2D_names) - 1):
                name = self.casefun_3D_2D_FFT2D_names[name_i]
                names.append('BGUTU3D2DFFT2D_%s_f%2.1f_FWHM%3.2f_LP' % (name, res_f, threshold_FWHM))
            for name_i in range(len(self.casefun_3D_2D_FFT2D_names) - 1):
                name = self.casefun_3D_2D_FFT2D_names[name_i]
                names.append('BGUTU3D2DFFT2D_%s_f%2.1f_FWHM%3.2f_HP' % (name, res_f, threshold_FWHM))
        return names

    """
    Returns list of input value descriptions 

    @return list of stings, or None
    """

    def get_input_descriptions(self):
        return ["Resolution factor affecting laws feature sampling ratio 1: original resolution <1: upsampling >1: downsampling",
                "Starting FWHM threshold in mm",
                "Ending FWHM threshold in mm",
                "Step size FWHM in mm"]


    """
    Returns list of strings decsribing boilerplate information about feature, including citations, if nay
    """
    def get_boilerplate(self):
        return ["Spatially filtered versions of data, high and low band",
                'Merisaari, H, Taimen, P, Shiradkar, R, et al. Repeatability of radiomics and machine learning for DWI: Short - term repeatability study of 112 patients with prostate cancer.Magn Reson Med.2019; 00: 1– 17. https://doi.org/10.1002/mrm.28058']

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
