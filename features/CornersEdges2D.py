from Feature import FeatureIndexandBackground
import copy
from abc import abstractmethod, ABC
import numpy as np
import os
import utils
import cv2
from skimage import measure
import scipy.stats
from skimage.measure import regionprops
from skimage.feature import peak_local_max
from skimage.filters import frangi, hessian, scharr
from visualizations import visualizations

try:
    from skimage import filters
except ImportError:
    from skimage import filter as filters

"""
Base class for corner edge detectors
"""


class CornersEdges2D(FeatureIndexandBackground):
    """
    Initialization

    @param name: short name of the feature
    @param params: parameter list for the feature instance
    """

    def __init__(self, name, params):
        super().__init__('CornersEdges2D', params)

    """
    Executes the feature
    
    @param intensity_images: intensity values images
    @param foreground_mask_images: foreground mask images
    @param background_mask_images: background mask images
    @param resolution: image resolution mm x mm x mm
    @return number of return values matching get_return_value_descriptions
    """

    @abstractmethod
    def fun(self, intensity_images, foreground_mask_images, background_mask_images, resolution, **kwargs):
        pass

    """
    Returns list of output value short names 
    
    @return list of return value short names, without spaces
    """

    @abstractmethod
    def get_return_value_short_names(self):
        pass

    """
    Returns list of strings decsribing boilerplate information about feature, including citations, if nay
    """

    def get_boilerplate(self: list) -> list:
        return ['Corner edge detector properties',
                'Merisaari, H, Taimen, P, Shiradkar, R, et al. Repeatability of radiomics and machine learning for DWI: Short - term repeatability study of 112 patients with prostate cancer.Magn Reson Med.2019; 00: 1– 17. https://doi.org/10.1002/mrm.28058',
                'Merisaari H, Shiradkar R, Toivonen J, Hiremath A, Khorrami M, Perez IM, Pahikkala T, Taimen P, Verho J, Bosträm PJ, Aronen H, Madabhushi A, Jambor I, Repeatability of radiomics features for prostate cancer diffusion weighted imaging obtained using b-values up to 2000 s / mm2, 27th Annual Meeting & Exhibition ISMRM, May 11-16 2019, Montréal, QC, Canada,  # 7461']

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


"""
Harris-Stephens corner edge detectors
"""


class HarrisStephens(CornersEdges2D):
    """
    Output names
    """
    subfun_3D_2D_Harris_names = (
        'No_corners_ROI', 'No_corners_BG', 'No_corners_ratio', 'Corner_density_primary', 'Corner_density_secondary',
        'Corner_density_mean', 'Corner_density_ratio', 'Corner_density_ratio_overall')

    """
    Initialization

    @param name: short name of the feature
    @param params: parameter list for the feature instance
    1) blockSize - Neighborhood size (see the details on cornerEigenValsAndVecs()).
    2) ksize - Aperture parameter for the Sobel() operator.
    3) k - Harris detector free parameter.
    """

    def __init__(self, name, params):
        super().__init__('HarrisStephens', params)

    """
    Executes the feature

    @param intensity_images: intensity values images
    @param foreground_mask_images: foreground mask images
    @param background_mask_images: background mask images
    @param resolution: image resolution mm x mm x mm
    @return number of return values matching get_return_value_descriptions
    """

    def fun(self, intensity_images, foreground_mask_images, background_mask_images, resolution, **kwargs):
        blockSize = int(np.round(self.params[0] / np.mean([resolution[0], resolution[1]])))
        # print('Harris effective block size:' + str(blockSize))
        ksize = self.params[1]
        k = self.params[2]
        No_corners_ROI = 0
        No_corners_BG = 0
        Densities_ROI_primary = []
        Densities_ROI_secondary = []
        Densities_ROI_mean = []
        Densities_BG_mean = []
        Densities_ROI_ratio = []
        if (len(intensity_images.shape) > 2):
            slices = intensity_images.shape[2]
        else:
            slices = 1
        for slice_i in range(slices):
            ROI_axis_a = None
            ROI_axis_b = None
            BG_axis_a = None
            BG_axis_b = None
            if (slices == 1):
                LS = foreground_mask_images[0][:, :]
                BG = background_mask_images[:, :]
            else:
                LS = foreground_mask_images[0][:, :, slice_i]
                BG = background_mask_images[:, :, slice_i]
            if (np.max(LS) == 0 and np.max(BG) == 0):
                continue
            if (slices == 1):
                cvimg = utils.make_cv2_slice2D(intensity_images[:, :])
            else:
                cvimg = utils.make_cv2_slice2D(intensity_images[:, :, slice_i])
            edgemap = abs(cv2.cornerHarris(cvimg, blockSize, ksize, k))
            ROIdata = copy.deepcopy(edgemap)
            ROIdata[LS == 0] = 0
            locs_ROI = peak_local_max(ROIdata, min_distance=1, threshold_abs=0.0)
            No_corners_ROI += len(locs_ROI)
            BGdata = copy.deepcopy(edgemap)
            BGdata[BG == 0] = 0
            BGdata[LS > 0] = 0
            locs_BG = peak_local_max(BGdata, min_distance=1)
            No_corners_BG += len(locs_BG)
            if locs_ROI.shape[0] > 3:
                ROI_density = 1
            else:
                ROI_density = 0
            if locs_BG.shape[0] > 3:
                BG_density = 1
            else:
                BG_density = 0
            if ROI_density == 1:
                x = []
                y = []
                for p in locs_ROI:
                    x.append(p[0])
                    y.append(p[1])
                try:
                    kernel = scipy.stats.gaussian_kde(np.vstack([x, y]))
                    kernel = kernel.covariance * kernel.factor
                    covs = sorted([kernel[0, 0], kernel[1, 1]], reverse=True)
                    ROI_axis_a = covs[0]
                    ROI_axis_b = covs[1]
                    ROI_axis_loc = (kernel[1, 0], kernel[0, 1])
                except:
                    ROI_density = 0
            if BG_density == 1:
                x = []
                y = []
                for p in locs_BG:
                    x.append(p[0])
                    y.append(p[1])
                try:
                    kernel = scipy.stats.gaussian_kde(np.vstack([x, y]))
                    kernel = kernel.covariance * kernel.factor
                    covs = sorted([kernel[0, 0], kernel[1, 1]], reverse=True)
                    BG_axis_a = covs[0]
                    BG_axis_b = covs[1]
                    BG_axis_loc = (kernel[1, 0], kernel[0, 1])
                except:
                    BG_density = 0
            if BG_density == 1 and ROI_density == 1:
                Densities_ROI_primary.append(ROI_axis_a)
                Densities_ROI_secondary.append(ROI_axis_b)
                Densities_ROI_mean.append(np.mean([ROI_axis_a, ROI_axis_b]))
                Densities_BG_mean.append(np.mean([BG_axis_a, BG_axis_b]))
                Densities_ROI_ratio.append(np.mean([ROI_axis_a, ROI_axis_b]) / np.mean([BG_axis_a, BG_axis_b]))
            elif ROI_density == 1:
                Densities_ROI_primary.append(ROI_axis_a)
                Densities_ROI_secondary.append(ROI_axis_b)
                Densities_ROI_mean.append(np.mean([ROI_axis_a, ROI_axis_b]))
                Densities_BG_mean.append(0)
                Densities_ROI_ratio.append(1)
            elif BG_density == 1:
                Densities_BG_mean.append(np.mean([BG_axis_a, BG_axis_b]))
            else:
                Densities_ROI_primary.append(0)
                Densities_ROI_secondary.append(0)
                Densities_ROI_mean.append(0)
                Densities_BG_mean.append(0)
                Densities_ROI_ratio.append(0)
        if No_corners_BG > 0:
            No_corners_ratio = float(No_corners_ROI) / float(No_corners_BG)
        else:
            No_corners_ratio = 0

        if len(Densities_ROI_primary) == 0:
            Corner_density_primary = 0
        else:
            Corner_density_primary = np.mean(Densities_ROI_primary)

        if len(Densities_ROI_secondary) == 0:
            Corner_density_secondary = 0
        else:
            Corner_density_secondary = np.mean(Densities_ROI_secondary)

        if len(Densities_ROI_mean) == 0:
            Corner_density_mean = 0
            Corner_density_ratio_overall = 0
        else:
            Corner_density_mean = np.mean(Densities_ROI_mean)
            Corner_density_ratio_overall = Corner_density_mean / (Corner_density_mean + np.mean(Densities_BG_mean))

        if len(Densities_ROI_ratio) == 0:
            Corner_density_ratio = 0
        else:
            Corner_density_ratio = np.mean(Densities_ROI_ratio)

        return No_corners_ROI, No_corners_BG, No_corners_ratio, Corner_density_primary, Corner_density_secondary, Corner_density_mean, Corner_density_ratio, Corner_density_ratio_overall

    """
    Returns list of output value short names 

    @return list of return value short names, without spaces
    """

    def get_return_value_short_names(self):
        names = []
        for name in self.subfun_3D_2D_Harris_names:
            names.append('UTUHarris_b%d_ks%d_k%3.2f_%s' % (self.params[0], self.params[1], self.params[2], name))
        return names

    """
    Returns list of strings decsribing boilerplate information about feature, including citations, if nay
    """

    def get_boilerplate(self):
        ret = super().get_boilerplate()
        ret.append('Harris-Stephens corner edge detection')
        ret.append(
            'Harris, C. and Stephens, M., 1988, August. A combined corner and edge detector. In Alvey vision conference (Vol. 15, No. 50, pp. 10-5244).')
        return ret


"""
Shi-Tomasi corner detection
"""


class ShiTomasi(CornersEdges2D):
    """
    Output names
    """
    subfun_3D_2D_ShiTomasi_names = (
        'No_corners_ROI', 'No_corners_BG', 'No_corners_ratio', 'Corner_density_primary', 'Corner_density_secondary',
        'Corner_density_mean', 'Corner_density_ratio', 'Corner_density_ratio_overall')

    """
    Initialization

    @param name: short name of the feature
    @param params: parameter list for the feature instance
    1) maxCorners Maximum number of corners to return. If there are more corners than are found, the strongest of them is returned.
    2) qualityLevel Parameter characterizing the minimal accepted quality of image corners. The parameter value is multiplied by the best corner quality measure, which is the minimal eigenvalue (see cornerMinEigenVal() ) or the Harris function response (see cornerHarris() ). The corners with the quality measure less than the product are rejected. For example, if the best corner has the quality measure = 1500, and the qualityLevel=0.01 , then all the corners with the quality measure less than 15 are rejected.
    3) minDistance Minimum possible Euclidean distance between the returned corners.
    """

    def __init__(self, name, params):
        super().__init__('ShiTomasi', params)

    """
    Executes the feature

    @param intensity_images: intensity values images
    @param foreground_mask_images: foreground mask images
    @param background_mask_images: background mask images
    @param resolution: image resolution mm x mm x mm
    @return number of return values matching get_return_value_descriptions
    """

    def fun(self, intensity_images, foreground_mask_images, background_mask_images, resolution, **kwargs):
        maxCorners = self.params[0]
        qualityLevel = self.params[1]
        minDistance = self.params[2] / np.mean([resolution[0], resolution[1]])
        No_corners_ROI = 0
        No_corners_BG = 0
        Densities_ROI_primary = []
        Densities_ROI_secondary = []
        Densities_ROI_mean = []
        Densities_BG_mean = []
        Densities_ROI_ratio = []
        if len(intensity_images.shape) > 2:
            slices = intensity_images.shape[2]
        else:
            slices = 1
        for slice_i in range(slices):
            ROI_axis_a = None
            ROI_axis_b = None
            BG_axis_a = None
            BG_axis_b = None
            if slices == 1:
                if (np.max(foreground_mask_images[:, :]) == 0 and np.max(background_mask_images[:, :]) == 0):
                    continue
                cvimg = utils.make_cv2_slice2D(intensity_images[:, :])
                cvROImask = utils.make_cv2_slice2D(foreground_mask_images[:, :])
            else:
                if (np.max(foreground_mask_images[:, :, slice_i]) == 0 and np.max(
                        background_mask_images[:, :, slice_i]) == 0):
                    continue
                cvimg = utils.make_cv2_slice2D(intensity_images[:, :, slice_i])
                cvROImask = utils.make_cv2_slice2D(foreground_mask_images[:, :, slice_i])
            locs_ROI = cv2.goodFeaturesToTrack(cvimg, maxCorners, qualityLevel, minDistance, mask=cvROImask)
            locs_ROI = np.squeeze(locs_ROI)
            if locs_ROI is None or len(locs_ROI.shape) == 0:
                ROIx = []
                ROIy = []
            elif len(locs_ROI.shape) == 1:
                ROIx = [locs_ROI[0]]
                ROIy = [locs_ROI[1]]
            else:
                ROIx = []
                ROIy = []
                for p in locs_ROI:
                    ROIx.append(p[0])
                    ROIy.append(p[1])
            No_corners_ROI += len(ROIx)
            if slices == 1:
                sliceBGdata = copy.deepcopy(background_mask_images[:, :])
                sliceBGdata[foreground_mask_images[:, :] > 0] = 0
            else:
                sliceBGdata = copy.deepcopy(background_mask_images[:, :, slice_i])
                sliceBGdata[foreground_mask_images[:, :, slice_i] > 0] = 0
            cvBGmask = utils.make_cv2_slice2D(sliceBGdata)
            locs_BG = cv2.goodFeaturesToTrack(cvimg, maxCorners, qualityLevel, minDistance, mask=cvBGmask)
            locs_BG = np.squeeze(locs_BG)
            if locs_BG is None or len(locs_BG.shape) == 0:
                BGx = []
                BGy = []
            elif len(locs_BG.shape) == 1:
                BGx = [locs_BG[0]]
                BGy = [locs_BG[1]]
            else:
                BGx = []
                BGy = []
                for p in locs_BG:
                    BGx.append(p[0])
                    BGy.append(p[1])
            No_corners_BG += len(BGx)
            if len(ROIx) > 3 and len(np.unique(ROIx)) > 1 and len(np.unique(ROIy)) > 1:
                kernel = scipy.stats.gaussian_kde(np.vstack([ROIx, ROIy]))
                kernel = kernel.covariance * kernel.factor
                covs = sorted([kernel[0, 0], kernel[1, 1]], reverse=True)
                ROI_axis_a = covs[0]
                ROI_axis_b = covs[1]
                ROI_axis_loc = (kernel[1, 0], kernel[0, 1])
            if len(BGx) > 3 and len(np.unique(BGx)) > 1 and len(np.unique(BGy)) > 1:
                kernel = scipy.stats.gaussian_kde(np.vstack([BGx, BGy]))
                kernel = kernel.covariance * kernel.factor
                covs = sorted([kernel[0, 0], kernel[1, 1]], reverse=True)
                BG_axis_a = covs[0]
                BG_axis_b = covs[1]
                BG_axis_loc = (kernel[1, 0], kernel[0, 1])
            if len(BGx) > 3 and len(ROIx) > 3 and len(np.unique(ROIx)) > 1 and len(np.unique(ROIy)) > 1 and len(
                    np.unique(BGx)) > 1 and len(np.unique(BGy)) > 1:
                Densities_ROI_primary.append(ROI_axis_a)
                Densities_ROI_secondary.append(ROI_axis_b)
                Densities_ROI_mean.append(np.mean([ROI_axis_a, ROI_axis_b]))
                Densities_BG_mean.append(np.mean([BG_axis_a, BG_axis_b]))
                Densities_ROI_ratio.append(np.mean([ROI_axis_a, ROI_axis_b]) / np.mean([BG_axis_a, BG_axis_b]))
            elif len(ROIx) > 3 and len(np.unique(ROIx)) > 1 and len(np.unique(ROIy)) > 1:
                Densities_ROI_primary.append(ROI_axis_a)
                Densities_ROI_secondary.append(ROI_axis_b)
                Densities_ROI_mean.append(np.mean([ROI_axis_a, ROI_axis_b]))
                Densities_BG_mean.append(0)
                Densities_ROI_ratio.append(1)
            elif len(BGx) > 3 and len(np.unique(BGx)) > 1 and len(np.unique(BGy)) > 1:
                Densities_BG_mean.append(np.mean([BG_axis_a, BG_axis_b]))
            else:
                Densities_ROI_primary.append(0)
                Densities_ROI_secondary.append(0)
                Densities_ROI_mean.append(0)
                Densities_BG_mean.append(0)
                Densities_ROI_ratio.append(0)
        if No_corners_BG > 0:
            No_corners_ratio = float(No_corners_ROI) / float(No_corners_BG)
        else:
            No_corners_ratio = 0

        if No_corners_BG > 0:
            No_corners_ratio = float(No_corners_ROI) / float(No_corners_BG)
        else:
            No_corners_ratio = 0

        if len(Densities_ROI_primary) == 0:
            Corner_density_primary = 0
        else:
            Corner_density_primary = np.mean(Densities_ROI_primary)

        if len(Densities_ROI_secondary) == 0:
            Corner_density_secondary = 0
        else:
            Corner_density_secondary = np.mean(Densities_ROI_secondary)

        if len(Densities_ROI_mean) == 0:
            Corner_density_mean = 0
            Corner_density_ratio_overall = 0
        else:
            Corner_density_mean = np.mean(Densities_ROI_mean)
            Corner_density_ratio_overall = Corner_density_mean / (Corner_density_mean + np.mean(Densities_BG_mean))

        if len(Densities_ROI_ratio) == 0:
            Corner_density_ratio = 0
        else:
            Corner_density_ratio = np.mean(Densities_ROI_ratio)

        return No_corners_ROI, No_corners_BG, No_corners_ratio, Corner_density_primary, Corner_density_secondary, Corner_density_mean, Corner_density_ratio, Corner_density_ratio_overall

    """
    Returns list of output value short names 

    @return list of return value short names, without spaces
    """

    def get_return_value_short_names(self):
        names = []
        for name in self.subfun_3D_2D_ShiTomasi_names:
            names.append('UTUShiTomasi_b%d_ks%4.3f_k%3.2f_%s' % (self.params[0], self.params[1], self.params[2], name))
        return names

    """
    Returns list of strings decsribing boilerplate information about feature, including citations, if nay
    """

    def get_boilerplate(self):
        ret = super().get_boilerplate()
        ret.append('Shi-Tomasi corner detection')
        ret.append('Shi, J. and Tomasi, C., 1993. Good features to track. Cornell University.')
        return ret


"""
Base class for features using object properties functions
"""


class Objprop(CornersEdges2D, ABC):
    """
    Output names
    """
    subfun_3D_2D_objectprops_names = ('Area_mean_mm2', 'Area_median_mm2', 'Area_SD_mm2', 'Area_IQR_mm2', 'Rel_area',
                                      'Ecc_mean', 'Ecc_median', 'Ecc_SD', 'Ecc_IQR', 'Rel_ecc',
                                      'Ax1len_mean_mm', 'Ax1len_median_mm', 'Ax1len_SD_mm', 'Ax1len_IQR_mm',
                                      'Rel_Ax1len',
                                      'Ax2len_mean_mm', 'Ax2len_median_mm', 'Ax2len_SD_mm', 'Ax2len_IQR_mm',
                                      'Rel_Ax2len',
                                      'Int_mean', 'Int_median', 'Int_SD', 'Int_IQR', 'Rel_Int',
                                      'Ori_SD', 'Ori_IQR', 'Rel_Ori',
                                      'Per_mean_mm', 'Per_median_mm', 'Per_SD_mm', 'Per_IQR_mm', 'Rel_Per',
                                      'Den_mean', 'Den_median', 'Rel_Den', 'N_objs', 'Rel_objs')

    """
    Executes the feature

    @param intensity_images: intensity values images
    @param foreground_mask_images: foreground mask images
    @param background_mask_images: background mask images
    @param resolution: image resolution mm x mm x mm
    @param fun2D: function to filter slice
    @param params: function spefici parameters
    @return number of return values matching get_return_value_descriptions
    """

    def casefun_3D_2D_objectprops(self, intensity_images, foreground_mask_images, background_mask_images, resolution,
                                  fun2D, params):
        area = []
        eccentricity = []
        major_axis_length = []
        mean_intensity = []
        minor_axis_length = []
        orientation = []
        perimeter = []
        density = []
        BGarea = []
        BGeccentricity = []
        BGmajor_axis_length = []
        BGmean_intensity = []
        BGminor_axis_length = []
        BGorientation = []
        BGperimeter = []
        BGdensity = []
        blobs = 0
        BGblobs = 0
        if len(intensity_images.shape) > 2:
            slices = intensity_images.shape[2]
        else:
            slices = 1
        for slice_i in range(slices):
            if slices == 1:
                if (np.max(foreground_mask_images[:, :]) == 0 or np.max(background_mask_images[:, :]) == 0):
                    continue
                slice2Ddata = intensity_images[:, :]
            else:
                if (np.max(foreground_mask_images[:, :, slice_i]) == 0 or np.max(
                        background_mask_images[:, :, slice_i]) == 0):
                    continue
                slice2Ddata = intensity_images[:, :, slice_i]
            x_lo, x_hi, y_lo, y_hi = utils.find_bounded_subregion2D(slice2Ddata)
            slice2Ddata = slice2Ddata[x_lo:x_hi, y_lo:y_hi]
            if slices == 1:
                slice2D_ROI = foreground_mask_images[x_lo:x_hi, y_lo:y_hi]
                slice2D_BG = background_mask_images[x_lo:x_hi, y_lo:y_hi]
            else:
                slice2D_ROI = foreground_mask_images[x_lo:x_hi, y_lo:y_hi, slice_i]
                slice2D_BG = background_mask_images[x_lo:x_hi, y_lo:y_hi, slice_i]

            # Resize to 1x1 mm space
            cvimg = cv2.resize(slice2Ddata, None, fx=resolution[0], fy=resolution[1], interpolation=cv2.INTER_LANCZOS4)
            cvBG = cv2.resize(slice2D_BG, None, fx=resolution[0], fy=resolution[1], interpolation=cv2.INTER_NEAREST)
            cvROI = cv2.resize(slice2D_ROI, None, fx=resolution[0], fy=resolution[1], interpolation=cv2.INTER_NEAREST)
            slice2D = fun2D(cvimg, self.params)

            if (type(self.params) == list) and (len(self.params) > 1) and (not type(self.params[-1]) == int) and (
                    'write_visualization' in self.params[-1]):
                LESIONDATAr_cvimg = utils.make_cv2_slice2D(intensity_images[:, :, slice_i]).copy()
                LESIONr_cvimg = utils.make_cv2_slice2D(foreground_mask_images[:, :, slice_i]).copy()
                basename = self.params[-1]['name'] + '_2D_curvature_' + str(self.params[:-1]).replace(' ', '_')
                visualizations.write_slice2D(cvimg,
                                             self.params[-1]['write_visualization'] + os.sep + basename + '_data.tiff')
                visualizations.write_slice2D_ROI(LESIONDATAr_cvimg, LESIONr_cvimg,
                                                 self.params[-1]
                                                 ['write_visualization'] + os.sep + basename + '_lesion.tiff',
                                                 0.4)

            sliceROI = copy.deepcopy(slice2D)
            sliceROI[cvROI == 0] = 0
            sliceBG = copy.deepcopy(slice2D)
            sliceBG[cvBG == 0] = 0
            sliceBG[cvROI > 0] = 0

            # label non-zero regions
            all_labels_ROI = measure.label(sliceROI)
            all_labels_BG = measure.label(sliceBG)
            # calculate region properties
            regions_ROI = regionprops(all_labels_ROI, intensity_image=cvimg, cache=True)
            regions_BG = regionprops(all_labels_BG, intensity_image=cvimg, cache=True)
            centroid_x = []
            centroid_y = []
            blobs += len(regions_ROI)
            for region in regions_ROI:
                area.append(region.area)
                centroid_x.append(region.centroid[0])
                centroid_y.append(region.centroid[1])
                eccentricity.append(region.eccentricity)
                major_axis_length.append(region.major_axis_length)
                mean_intensity.append(region.mean_intensity)
                minor_axis_length.append(region.minor_axis_length)
                orientation.append(region.orientation)
                perimeter.append(region.perimeter)
            if len(centroid_x) > 3:
                kernel = scipy.stats.gaussian_kde(np.vstack([centroid_x, centroid_y]))
                kernel = kernel.covariance * kernel.factor
                density.append(np.mean([kernel[0, 0], kernel[1, 1]]))

            centroid_x = []
            centroid_y = []
            BGblobs += len(regions_BG)
            for region in regions_BG:
                BGarea.append(region.area)
                centroid_x.append(region.centroid[0])
                centroid_y.append(region.centroid[1])
                BGeccentricity.append(region.eccentricity)
                BGmajor_axis_length.append(region.major_axis_length)
                BGmean_intensity.append(region.mean_intensity)
                BGminor_axis_length.append(region.minor_axis_length)
                BGorientation.append(region.orientation)
                BGperimeter.append(region.perimeter)
            if len(centroid_x) > 3 and (len(np.unique(centroid_x)) > 1) and (len(np.unique(centroid_y)) > 1):
                kernel = scipy.stats.gaussian_kde(np.vstack([centroid_x, centroid_y]))
                kernel = kernel.covariance * kernel.factor
                BGdensity.append(np.mean([kernel[0, 0], kernel[1, 1]]))

        ret = []
        if len(area) > 0:
            meanarea = np.mean(area)
            ret.append(meanarea)
            ret.append(np.median(area))
            ret.append(np.std(area))
            ret.append(scipy.stats.iqr(area))
        else:
            meanarea = 0.0
            ret.append(meanarea)
            ret.append(0.0)
            ret.append(float('NaN'))
            ret.append(float('NaN'))
        if meanarea == 0:
            ret.append(0)
        else:
            if len(BGarea) > 0:
                ret.append(meanarea / (meanarea + np.mean(BGarea)))
            else:
                ret.append(1.0)
        if (len(eccentricity) == 0):
            meaneccentricity = 0
            ret.append(0)
            ret.append(0)
            ret.append(0)
            ret.append(0)
        else:
            meaneccentricity = np.mean(eccentricity)
            ret.append(meaneccentricity)
            ret.append(np.median(eccentricity))
            ret.append(np.std(eccentricity))
            ret.append(scipy.stats.iqr(eccentricity))
        if meaneccentricity == 0:
            ret.append(0)
        else:
            if len(BGeccentricity) > 0:
                ret.append(meaneccentricity / (meaneccentricity + np.mean(BGeccentricity)))
            else:
                ret.append(1)
        if (len(major_axis_length) == 0):
            meanmajor_axis_length = 0
            ret.append(meanmajor_axis_length)
            ret.append(0)
            ret.append(0)
            ret.append(0)
        else:
            meanmajor_axis_length = np.mean(major_axis_length)
            ret.append(meanmajor_axis_length)
            ret.append(np.median(major_axis_length))
            ret.append(np.std(major_axis_length))
            ret.append(scipy.stats.iqr(major_axis_length))
        if meanmajor_axis_length == 0:
            ret.append(0)
        else:
            if len(BGmajor_axis_length) > 0:
                ret.append(meanmajor_axis_length / (meanmajor_axis_length + np.mean(BGmajor_axis_length)))
            else:
                ret.append(1)
        if (len(minor_axis_length) == 0):
            meanminor_axis_length = 0
            ret.append(meanminor_axis_length)
            ret.append(0)
            ret.append(0)
            ret.append(0)
        else:
            meanminor_axis_length = np.mean(minor_axis_length)
            ret.append(meanminor_axis_length)
            ret.append(np.median(minor_axis_length))
            ret.append(np.std(minor_axis_length))
            ret.append(scipy.stats.iqr(minor_axis_length))
        if meanminor_axis_length == 0:
            ret.append(0)
        else:
            if len(BGminor_axis_length) > 0:
                ret.append(meanminor_axis_length / (meanminor_axis_length + np.mean(BGminor_axis_length)))
            else:
                ret.append(1)
        if (len(mean_intensity) == 0):
            mean_mean_intensity = 0
            ret.append(mean_mean_intensity)
            ret.append(0)
            ret.append(0)
            ret.append(0)
        else:
            mean_mean_intensity = np.mean(mean_intensity)
            ret.append(mean_mean_intensity)
            ret.append(np.median(mean_intensity))
            ret.append(np.std(mean_intensity))
            ret.append(scipy.stats.iqr(mean_intensity))
        if mean_mean_intensity == 0:
            ret.append(0)
        else:
            if len(BGmean_intensity) > 0:
                ret.append(mean_mean_intensity / (mean_mean_intensity + np.mean(BGmean_intensity)))
            else:
                ret.append(1)
        if (len(orientation) == 0):
            ret.append(0)
            ret.append(0)
            meanorientation = 0
            ret.append(0)
        else:
            ret.append(np.std(orientation))
            ret.append(scipy.stats.iqr(orientation))
            meanorientation = np.mean(orientation)
            if len(BGorientation) > 0:
                ret.append(meanorientation / (meanorientation + np.mean(BGorientation)))
            else:
                ret.append(1)
        if (len(perimeter) == 0):
            meanperimeter = 0
            ret.append(0)
            ret.append(0)
            ret.append(0)
            ret.append(0)
        else:
            meanperimeter = np.mean(perimeter)
            ret.append(meanperimeter)
            ret.append(np.median(perimeter))
            ret.append(np.std(perimeter))
            ret.append(scipy.stats.iqr(perimeter))
        if meanperimeter == 0:
            ret.append(0)
        else:
            if len(BGperimeter) > 0:
                ret.append(meanperimeter / (meanperimeter + np.mean(BGperimeter)))
            else:
                ret.append(1)
        if len(density) == 0:
            ret.append(0)
            ret.append(0)
            ret.append(0)
        else:
            # print(density)
            meandensity = np.mean(density)
            # print(meandensity)
            ret.append(meandensity)
            ret.append(np.median(density))
            # print(np.median(density))
            if len(BGdensity) > 0:
                ret.append(meandensity / (meandensity + np.mean(BGdensity)))
            else:
                ret.append(1)
        if blobs == 0:
            ret.append(0)
            ret.append(0)
        else:
            ret.append(blobs)
            ret.append(blobs / (blobs + BGblobs))
        return ret

    """
    Name generator for output
    @param: filter name
    @param: parameter string values
    """

    def casefun_3D_2D_objectprops_name_generator(self, filtername, param_strs):
        names = []
        for name in self.subfun_3D_2D_objectprops_names:
            if len(param_strs) > 0:
                suffix = ''
                for param in param_strs:
                    suffix += ('_%s' % param_strs)
                names.append('UTU3D2D%s_%s_objprops_%s' % (filtername, suffix, name))
            else:
                names.append('UTU3D2D%s_objprops_%s' % (filtername, name))
        return names

    """
    Initialization

    @param name: short name of the feature
    @param params: parameter list for the feature instance
    """

    def __init__(self, name, params):
        super().__init__('Objprop', params)


"""
Frangi fintered corner detection
"""


class Frangi(Objprop):
    """
    Initialization

    @param name: short name of the feature
    @param params: not in use
    """

    def __init__(self, name, params):
        super().__init__('Frangi', params)

    """
    Function applying Frangi filter
    @param slice2D: slice for which to calculate features
    @param params: not used
    """

    @staticmethod
    def subfun_Frangi(self, slice2D, params):
        # Create binary frangi
        edge_frangi = frangi(slice2D)
        val = filters.threshold_otsu(edge_frangi)
        frangi_bin = np.zeros_like(edge_frangi)
        frangi_bin[edge_frangi >= val] = 1
        return frangi_bin

    """
    Executes the feature

    @param intensity_images: intensity values images
    @param foreground_mask_images: foreground mask images
    @param background_mask_images: background mask images
    @param resolution: image resolution mm x mm x mm
    @return number of return values matching get_return_value_descriptions
    """

    def fun(self, intensity_images, foreground_mask_images, background_mask_images, resolution, **kwargs):
        super().casefun_3D_2D_objectprops(intensity_images, foreground_mask_images, background_mask_images, resolution,
                                          self.subfun_Frangi, self.params)

    """
    Returns list of output value short names 

    @return list of return value short names, without spaces
    """

    def get_return_value_short_names(self):
        return super().casefun_3D_2D_objectprops_name_generator('Frangi', [])

    """
    Returns list of strings decsribing boilerplate information about feature, including citations, if nay
    """

    def get_boilerplate(self):
        ret = super().get_boilerplate()
        ret.append('Object properties of Frangi filtered data')
        ret.append(
            'A.Frangi, W.Niessen, K.Vincken, and M.Viergever. "Multiscale vessel enhancement filtering," In LNCS, vol. 1496, pages 130 - 137, Germany, 1998. Springer - Verlag.')
        return ret


"""
Scharr fintered corner detection
"""


class Scharr(Objprop):
    """
    Initialization

    @param name: short name of the feature
    @param params: not in use
    """

    def __init__(self, name, params):
        super().__init__('Scharr', params)

    """
    Function applying Frangi filter
    @param slice2D: slice for which to calculate features
    @param params: not used
    """

    @staticmethod
    def subfun_Scharr(self, slice2D, params):
        # Create binary edge image
        edge_scharr = scharr(slice2D)
        scharr_bin = np.zeros_like(edge_scharr)
        scharr_bin[edge_scharr > 0] = 1
        return scharr_bin

    """
    Executes the feature

    @param intensity_images: intensity values images
    @param foreground_mask_images: foreground mask images
    @param background_mask_images: background mask images
    @param resolution: image resolution mm x mm x mm
    @return number of return values matching get_return_value_descriptions
    """

    def fun(self, intensity_images, foreground_mask_images, background_mask_images, resolution, **kwargs):
        super().casefun_3D_2D_objectprops(intensity_images, foreground_mask_images, background_mask_images, resolution,
                                          self.subfun_Scharr, self.params)

    """
    Returns list of output value short names 

    @return list of return value short names, without spaces
    """

    def get_return_value_short_names(self):
        return super().casefun_3D_2D_objectprops_name_generator('Scharr', [])

    """
    Returns list of strings decsribing boilerplate information about feature, including citations, if nay
    """

    def get_boilerplate(self):
        ret = super().get_boilerplate()
        ret.append('Object properties of Scharr filtered data')
        ret.append('B. Jaehne, H. Scharr, and S. Koerkel. Principles of filter design. In Handbook of Computer Vision and Applications. Academic Press, 1999.')
        return ret


"""
Hessian fintered corner detection
"""


class Hessian(Objprop):
    """
    Initialization

    @param name: short name of the feature
    @param params: not in use
    """

    def __init__(self, name, params):
        super().__init__('Hessian', params)

    """
    Function applying Hessian filter
    @param slice2D: slice for which to calculate features
    @param params: not used
    """
    @staticmethod
    def subfun_Hessian(self, slice2D, params):
        # Create binary edge image
        edge_hessian = hessian(slice2D, beta=params[0], gamma=params[1])
        val = filters.threshold_otsu(edge_hessian)
        hessian_bin = np.zeros_like(edge_hessian)
        hessian_bin[edge_hessian >= val] = 1
        return hessian_bin

    """
    Executes the feature

    @param intensity_images: intensity values images
    @param foreground_mask_images: foreground mask images
    @param background_mask_images: background mask images
    @param resolution: image resolution mm x mm x mm
    @return number of return values matching get_return_value_descriptions
    """

    def fun(self, intensity_images, foreground_mask_images, background_mask_images, resolution, **kwargs):
        super().casefun_3D_2D_objectprops(intensity_images, foreground_mask_images, background_mask_images, resolution,
                                          self.subfun_Hessian, self.params)

    """
    Returns list of output value short names 

    @return list of return value short names, without spaces
    """

    def get_return_value_short_names(self):
        names = []
        for name in super().subfun_3D_2D_objectprops_names:
            names.append('UTU3D2DHessian_%4.3f_%2.1f_objprops_%s' % (self.params[0], self.params[1], name))
        return names

    """
    Returns list of strings decsribing boilerplate information about feature, including citations, if nay
    """

    def get_boilerplate(self):
        ret = super().get_boilerplate()
        ret.append('Object properties of Hessian filtered data')
        ret.append('Choon-Ching Ng, Moi Hoon Yap, Nicholas Costen and Baihua Li, "Automatic Wrinkle Detection using Hybrid Hessian Filter".')
        return ret
