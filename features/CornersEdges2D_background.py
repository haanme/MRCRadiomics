from Feature import FeatureIndexandBackground
import copy
from abc import abstractmethod, ABC
import numpy as np
import os
import features.Utils
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
Harris-Stephens corner edge detectors with background ROI
"""


class HarrisStephensBackground(FeatureIndexandBackground):
    """
    Output names
    """
    subfun_3D_2D_Harris_names_BG = (
        'No_corners_ROI', 'Corner_density_primary', 'Corner_density_secondary', 'Corner_density_mean')

    """
    Initialization

    @param params: parameter list for the feature instance
    1) blockSize - Neighborhood size (see the details on cornerEigenValsAndVecs()).
    2) ksize - Aperture parameter for the Sobel() operator.
    3) k - Harris detector free parameter.
    """

    def __init__(self, params):
        super().__init__('HarrisStephensBackground', params)

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
                cvimg = Utils.make_cv2_slice2D(intensity_images[:, :])
            else:
                cvimg = Utils.make_cv2_slice2D(intensity_images[:, :, slice_i])
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
        for name in self.subfun_3D_2D_Harris_names_BG:
            names.append('BGUTUHarris_b%d_ks%d_k%3.2f_%s' % (self.params[0], self.params[1], self.params[2], name))
        return names

    """
    Returns list of input value descriptions 

    @return list of stings, or None
    """

    def get_input_descriptions(self):
        return ["Neighborhood size (see the details on cornerEigenValsAndVecs()).",
                "Aperture parameter for the Sobel() operator.",
                "Harris detector free parameter."]

    """
    Returns list of strings decsribing boilerplate information about feature, including citations, if nay
    """

    @staticmethod
    def get_boilerplate():
        ret = []
        ret.append('Harris-Stephens corner edge detection with background ROI')
        ret.append('Harris, C. and Stephens, M., 1988, August. A combined corner and edge detector. In Alvey vision conference (Vol. 15, No. 50, pp. 10-5244).')
        return ret

    """
    Returns number of required foreground mask images
    """
    def number_of_foreground_mask_images_required(self):
        return 0


"""
Hessian fintered corner detection with background ROI
"""


class HessianBackground(FeatureIndexandBackground):
    """
    Output names
    """
    subfun_3D_2D_objectprops_names_BG = ('Area_mean_mm2', 'Area_median_mm2', 'Area_SD_mm2', 'Area_IQR_mm2',
                                         'Ecc_mean', 'Ecc_median', 'Ecc_SD', 'Ecc_IQR',
                                         'Ax1len_mean_mm', 'Ax1len_median_mm', 'Ax1len_SD_mm', 'Ax1len_IQR_mm',
                                         'Ax2len_mean_mm', 'Ax2len_median_mm', 'Ax2len_SD_mm', 'Ax2len_IQR_mm',
                                         'Int_mean', 'Int_median', 'Int_SD', 'Int_IQR',
                                         'Ori_SD', 'Ori_IQR',
                                         'Per_mean_mm', 'Per_median_mm', 'Per_SD_mm', 'Per_IQR_mm',
                                         'Den_mean', 'Den_median', 'N_objs')

    """
    Initialization
    
    @param params:
    1) Frangi correction constant that adjusts the filter’s sensitivity to deviation from a blob-like structure.
    2) Frangi correction constant that adjusts the filter’s sensitivity to areas of high variance/texture/structure.
    """

    def __init__(self, params):
        super().__init__('HessianBackground', params)

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
    Background object properties calcualtion
    @param intensity_images: intensity value images
    @param background_mask_images: background mask
    @param resolution: image resolution in mm
    @param fun2D: function for filtering
    @param params: function parameters
    """

    @staticmethod
    def casefun_3D_2D_objectprops_BG(self, intensity_images, background_mask_images, resolution, fun2D, params):
        area = []
        eccentricity = []
        major_axis_length = []
        mean_intensity = []
        minor_axis_length = []
        orientation = []
        perimeter = []
        density = []
        blobs = 0
        if len(intensity_images.shape) > 2:
            slices = intensity_images.shape[2]
        else:
            slices = 1
        for slice_i in range(slices):
            if slices == 1:
                if (np.max(background_mask_images[:, :]) == 0):
                    continue
                slice2Ddata = intensity_images[:, :]
            else:
                if (np.max(background_mask_images[:, :, slice_i]) == 0):
                    continue
                slice2Ddata = intensity_images[:, :, slice_i]
            x_lo, x_hi, y_lo, y_hi = Utils.find_bounded_subregion2D(slice2Ddata)
            slice2Ddata = slice2Ddata[x_lo:x_hi, y_lo:y_hi]
            if slices == 1:
                slice2D_ROI = background_mask_images[x_lo:x_hi, y_lo:y_hi]
            else:
                slice2D_ROI = background_mask_images[x_lo:x_hi, y_lo:y_hi, slice_i]
            # Resize to 1x1 mm space
            cvimg = cv2.resize(slice2Ddata, None, fx=resolution[0], fy=resolution[1], interpolation=cv2.INTER_LANCZOS4)
            cvROI = cv2.resize(slice2D_ROI, None, fx=resolution[0], fy=resolution[1], interpolation=cv2.INTER_NEAREST)
            slice2D = fun2D(cvimg, params)

            sliceROI = copy.deepcopy(slice2D)
            sliceROI[cvROI == 0] = 0

            # label non-zero regions
            all_labels_ROI = measure.label(sliceROI)
            # calculate region properties
            regions_ROI = regionprops(all_labels_ROI, intensity_image=cvimg, cache=True)
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

        ret = []
        if (blobs == 0):
            for ret_i in range(27):
                ret.append(0)
        else:
            meanarea = np.mean(area)
            ret.append(meanarea)
            ret.append(np.median(area))
            ret.append(np.std(area))
            ret.append(scipy.stats.iqr(area))

            meaneccentricity = np.mean(eccentricity)
            ret.append(meaneccentricity)
            ret.append(np.median(eccentricity))
            ret.append(np.std(eccentricity))
            ret.append(scipy.stats.iqr(eccentricity))

            meanmajor_axis_length = np.mean(major_axis_length)
            ret.append(meanmajor_axis_length)
            ret.append(np.median(major_axis_length))
            ret.append(np.std(major_axis_length))
            ret.append(scipy.stats.iqr(major_axis_length))

            meanminor_axis_length = np.mean(minor_axis_length)
            ret.append(meanminor_axis_length)
            ret.append(np.median(minor_axis_length))
            ret.append(np.std(minor_axis_length))
            ret.append(scipy.stats.iqr(minor_axis_length))

            mean_mean_intensity = np.mean(mean_intensity)
            ret.append(mean_mean_intensity)
            ret.append(np.median(mean_intensity))
            ret.append(np.std(mean_intensity))
            ret.append(scipy.stats.iqr(mean_intensity))

            ret.append(np.std(orientation))
            ret.append(scipy.stats.iqr(orientation))

            meanperimeter = np.mean(perimeter)
            ret.append(meanperimeter)
            ret.append(np.median(perimeter))
            ret.append(np.std(perimeter))
            ret.append(scipy.stats.iqr(perimeter))

        if len(density) == 0:
            ret.append(0)
            ret.append(0)
        else:
            meandensity = np.mean(density)
            ret.append(meandensity)
            ret.append(np.median(density))

        if blobs == 0:
            ret.append(0)
        else:
            ret.append(blobs)

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
        return self.casefun_3D_2D_objectprops_BG(intensity_images, background_mask_images, resolution,
                                                 self.subfun_Hessian, self.params)

    """
    Returns list of output value short names 

    @return list of return value short names, without spaces
    """

    def get_return_value_short_names(self):
        names = []
        for name in self.subfun_3D_2D_objectprops_names_BG:
            names.append('BGUTU3D2DHessian_%4.3f_%2.1f_objprops_%s' % (self.params[0], self.params[1], name))
        return names

    """
    Returns list of input value descriptions 

    @return list of stings, or None
    """

    def get_input_descriptions(self):
        return ["Frangi correction constant that adjusts the filter’s sensitivity to deviation from a blob-like structure.",
                "Frangi correction constant that adjusts the filter’s sensitivity to areas of high variance/texture/structure."]

    """
    Returns list of strings decsribing boilerplate information about feature, including citations, if nay
    """

    @staticmethod
    def get_boilerplate():
        ret = []
        ret.append('Object properties of Hessian filtered data')
        ret.append('Choon-Ching Ng, Moi Hoon Yap, Nicholas Costen and Baihua Li, "Automatic Wrinkle Detection using Hybrid Hessian Filter".')
        return ret

    """
    Returns number of required foreground mask images
    """
    def number_of_foreground_mask_images_required(self):
        return 0
