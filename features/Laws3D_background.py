

from Feature import FeatureIndexandBackground
import os
import io
from scipy.signal import correlate
import csv
import sys
import numpy as np
import scipy.stats
import features.Utils
import os.path as path

numba_found = False
try:
    import numba
    from numba import njit, prange, vectorize, float64
    numba_found = True
except:
    print('Numba not found and related functions not in use')


    """
    Names for 3D Laws features. Contains basic first order statistics to be takesn for each feature map.
    """
    casefun_3D_Laws_names = ['mean_ROI', 'median_ROI', 'SD_ROI', 'IQR_ROI', 'skewnessROI', 'kurtosisROI', 'p25ROI',
                             'p75ROI', 'rel']

    # Read the kernels from file
    def read_3DLaws_kernel(filename):
        kernels = []
        data = np.empty([5, 5, 5])
        y = 0
        z = 0
        kernelcode = 'N/A'
        with open(filename, 'r') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ')
            line_no = 0
            for row in spamreader:
                line_no += 1
                if len(row) == 0:
                    continue
                row = [x for x in row if x]
                # print(str(line_no) + ':' + str(row))
                if not row[0][0] == '-' and not row[0].isdigit():
                    # print(str(line_no) + '[HEADER]:' + str(row))
                    kernels.append({'name': kernelcode, 'data': data})
                    kernelcode = row[0].strip()
                    data = np.empty([5, 5, 5])
                    data_read = 0
                    x = 0
                    y = 0
                    z = 0
                    continue
                # else:
                #    print(str(line_no) + ':' + str(row))
                for x in range(len(row)):
                    data[x, y, z] = float(row[x])
                y += 1
                if y == 5:
                    y = 0
                    z += 1
        kernels.append({'name': kernelcode, 'data': data})

        # Find redundant kernels
        kernels_pruned = []
        for i in range(len(kernels)):
            already_included = False
            for ii in range(len(kernels_pruned)):
                # print((kernels[i]['data'].shape,kernels_pruned[ii]['data'].shape))
                if np.max(abs(np.subtract(kernels[i]['data'], kernels_pruned[ii]['data']))) == 0:
                    print('Excluding redundant kernel ' + kernels[i]['name'] + ' ' + kernels_pruned[ii]['name'])
                    already_included = True
                    break
            if not already_included:
                kernels_pruned.append(kernels[i])
        # print(str(len(kernels_pruned)) + ' pruned 3D Laws kernels out of ' + str(len(kernels)))
        return kernels_pruned


    # Read kernels into memory by default
    Laws3Dkernel = read_3DLaws_kernel(path.dirname(path.abspath(__file__)) + os.sep + 'mask-3d-5.txt')
    # Initialize array formats for use in numba
    laws3D_names = []
    length_Laws3Dkernel = len(Laws3Dkernel)
    Laws3Dkernel_array = np.zeros([length_Laws3Dkernel, 5, 5, 5])
    for kernel_i in range(0, length_Laws3Dkernel):
        Laws3Dkernel_array[kernel_i, :, :, :] = Laws3Dkernel[kernel_i]['data']
    for kernel_i in range(1, length_Laws3Dkernel):
        laws3D_names.append(Laws3Dkernel[kernel_i]['name'])


"""
3D Laws features for background mask, i e with background mask 
"""


class Laws3D_Background(FeatureIndexandBackground):





    """
    Initialization

    @param params: parameter list for the feature instance
    """

    def __init__(self, params):
        super(Laws3D_Background, self).__init__('Laws3DBackground', params)


    """
    Appends Laws feature values to the end of exiting list for background region

    @param outdata: Intensity data
    @param LESIONr: Lesion mask
    @param BGr: Background region mask
    @param ret: List of results
    @returns: List of results, with new data appended
    """

    def append_Laws_results_BG(outdata, LESIONrs, ret):
        ROIdata = outdata[LESIONrs > 0]
        if len(ROIdata) == 0:
            for x in casefun_3D_Laws_names:
                ret.append(float('nan'))
            return ret
        mean1 = np.mean(ROIdata)
        ret.append(np.mean(ROIdata))
        ret.append(np.median(ROIdata))
        std1 = np.std(ROIdata)
        ret.append(std1)
        ret.append(scipy.stats.iqr(ROIdata))
        ret.append(scipy.stats.skew(ROIdata))
        ret.append(scipy.stats.kurtosis(ROIdata))
        ret.append(np.percentile(ROIdata, 25))
        ret.append(np.percentile(ROIdata, 75))
        return ret

    """
    Appends Laws feature values to the end of exiting list

    @param outdata: Intensity data
    @param LESIONr: Lesion mask
    @param BGr: Background region mask
    @param ret: List of results
    @returns: List of results, with new data appended
    """

    def append_Laws_results(self, outdata, LESIONrs, BGrs, ret):
        ROIdata = outdata[LESIONrs > 0]
        BGdata = outdata[BGrs > 0]
        if len(ROIdata) == 0:
            for x in self.casefun_3D_Laws_names:
                ret.append(float('nan'))
            return ret
        mean1 = np.mean(ROIdata)
        ret.append(np.mean(ROIdata))
        ret.append(np.median(ROIdata))
        std1 = np.std(ROIdata)
        ret.append(std1)
        ret.append(scipy.stats.iqr(ROIdata))
        ret.append(scipy.stats.skew(ROIdata))
        ret.append(scipy.stats.kurtosis(ROIdata))
        ret.append(np.percentile(ROIdata, 25))
        ret.append(np.percentile(ROIdata, 75))
        if mean1 == 0:
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

        s = 5
        x_lo, x_hi, y_lo, y_hi, z_lo, z_hi = features.Utils.find_bounded_subregion3D(intensity_images)
        x_lo -= 2
        x_hi += 2
        y_lo -= 2
        y_hi += 2
        if x_lo < 0:
            x_lo = 0
        if y_lo < 0:
            y_lo = 0
        if x_hi >= background_mask_images.shape[0]:
            x_hi = background_mask_images.shape[0] - 1
        if y_hi >= background_mask_images.shape[1]:
            y_hi = background_mask_images.shape[1] - 1

        LESIONDATArs = intensity_images[x_lo:x_hi, y_lo:y_hi, :]
        BGrs_temp = background_mask_images[x_lo:x_hi, y_lo:y_hi, :]

        # Create masks and output data to desired resolution, intensity data is resliced later for non-zero only
        min_res = np.max(resolution)
        new_res = [min_res * res_f, min_res * res_f, min_res * res_f]
        BGrs_temp, affineBGrs_temp = features.Utils.reslice_array(BGrs_temp, resolution, new_res, 0)
        LESIONDATArs, affineLESIONDATArs = features.Utils.reslice_array(LESIONDATArs, resolution, new_res, 1)

        outdatas = []
        for kernel_i in range(1, len(Laws3Dkernel)):
            outdatas.append(np.zeros_like(LESIONDATArs))

        sys.stderr = io.StringIO()
        for (x, y, z, window) in features.Utils.sliding_window3D(LESIONDATArs, BGrs_temp, 1, (s, s, s)):
            window = np.subtract(window, np.mean(window))
            w_std = np.std(window)
            if w_std > 0:
                window = np.divide(window, np.std(window))
            xmid = x
            ymid = y
            zmid = z
            if xmid >= LESIONDATArs.shape[0]:
                continue
            if ymid >= LESIONDATArs.shape[1]:
                continue
            if zmid >= LESIONDATArs.shape[2]:
                continue

            correlates = []
            for kernel_i in range(1, len(Laws3Dkernel)):
                c = correlate(window, Laws3Dkernel[kernel_i]['data'])
                correlates.append(c[2:7, 2:7, 2:7])
            for c_i in range(len(correlates)):
                outdatas[c_i][xmid, ymid, zmid] = np.sum(correlates[c_i])
        sys.stderr = sys.__stderr__
        ret = []
        for kernel_i in range(1, len(Laws3Dkernel)):
            ret = self.append_Laws_results_BG(outdatas[kernel_i - 1], BGrs_temp, ret)
        return ret

    """
    Returns list of output value short names 

    @return list of return value short names, without spaces
    """

    def get_return_value_short_names(self):
        names = []
        for laws3D_names_i in range(len(laws3D_names)):
            for name_i in range(len(casefun_3D_Laws_names) - 1):
                name = casefun_3D_Laws_names[name_i]
                names.append('BGUTU3DLaws%s_%s_f%2.1f' % (laws3D_names[laws3D_names_i], name, params[0]))
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
        return ['3D analogue of laws texture features.',
                'Suzuki, M.T. and Yaginuma, Y., 2007, January.',
                'A solid texture analysis based on three - dimensional convolution kernels.In Videometrics IX(Vol. 6491,p. 64910W).',
                'International Society for Optics and Photonics.',
                'Suzuki, M.T., Yaginuma, Y., Yamada, T.and Shimizu, Y., 2006, December.A shape feature extraction method based on 3D convolution masks.',
                'In Multimedia, 2006. ISM\'06. Eighth IEEE International Symposium on (pp. 837-844). IEEE.']


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
