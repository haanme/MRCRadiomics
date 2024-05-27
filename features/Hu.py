

from Feature import FeatureIndexandBackground
import numpy as np
import scipy.stats
import os
import copy
import cv2
from visualizations import visualizations
import features.Utils

"""
HU moments
"""


class Hu(FeatureIndexandBackground):

    """
    Output value names
    """
    contour2D_names = []
    contour2D_names.append('Hu_moment_invariants_1_SD_per_mean')
    contour2D_names.append('Hu_moment_invariants_2_SD_per_mean')
    contour2D_names.append('Hu_moment_invariants_3_SD_per_mean')
    contour2D_names.append('Hu_moment_invariants_4_SD_per_mean')
    contour2D_names.append('Hu_moment_invariants_5_SD_per_mean')
    contour2D_names.append('Hu_moment_invariants_6_SD_per_mean')
    contour2D_names.append('Hu_moment_invariants_7_SD_per_mean')
    contour2D_names.append('Hu_moment_invariants_8_SD_per_mean')
    contour2D_names.append('2D_contour_arclengths_mean')
    contour2D_names.append('2D_contour_arclengths_median')
    contour2D_names.append('2D_contour_arclengths_SD')
    contour2D_names.append('2D_contour_arclengths_IQR')
    contour2D_names.append('2D_contour_areas_mean')
    contour2D_names.append('2D_contour_areas_median')
    contour2D_names.append('2D_contour_areas_SD')
    contour2D_names.append('2D_contour_areas_IQR')
    contour2D_names.append('2D_number_of_contour_convexity_defects')
    contour2D_names.append('2D_contour_convexity_defect_areas_mean')
    contour2D_names.append('2D_contour_convexity_defect_areas_median')
    contour2D_names.append('2D_contour_convexity_defect_areas_SD')
    contour2D_names.append('2D_contour_convexity_defect_areas_IQR')
    contour2D_names.append('2D_contour_convexity_defect_lengths_mean')
    contour2D_names.append('2D_contour_convexity_defect_lengths_median')
    contour2D_names.append('2D_contour_convexity_defect_lengths_SD')
    contour2D_names.append('2D_contour_convexity_defect_lengths_IQR')
    contour2D_names.append('2D_contour_convexity_defect_depths_mean')
    contour2D_names.append('2D_contour_convexity_defect_depths_median')
    contour2D_names.append('2D_contour_convexity_defect_depths_SD')
    contour2D_names.append('2D_contour_convexity_defect_depths_IQR')
    contour2D_names.append('2D_mean_contour_convexity_defect_lengths_proportional_to_arclength')
    contour2D_names.append('2D_median_contour_convexity_defect_lengths_proportional_to_arclength')
    contour2D_names.append('2D_SD_contour_convexity_defect_lengths_proportional_to_arclength')
    contour2D_names.append('2D_IQR_contour_convexity_defect_lengths_proportional_to_arclength')
    contour2D_names.append('2D_mean_contour_convexity_defect_depths_proportional_to_area')
    contour2D_names.append('2D_median_contour_convexity_defect_depths_proportional_to_area')
    contour2D_names.append('2D_SD_contour_convexity_defect_depths_proportional_to_area')
    contour2D_names.append('2D_IQR_contour_convexity_defect_depths_proportional_to_area')

    """
    Initialization

    @param name: name of the feature class, without spaces
    """

    def __init__(self, params):
        super('Hu', params)


    """
    Hu moments 1-8
    Contour_arclength
    Contour_area
    Contour_convexitydefect_depth
    Contour_convexitydefect_area
    Contour_convexitydefect_length
    
    @param resolution: spatial resolution
    """

    def calculate_2D_contour_measures(self, contours, resolution):

        if len(contours[1]) == 0:
            print('len(contours[1]) == 0')
            return None
        points = np.squeeze(np.array(contours[1][0]))
        if (len(points.shape) < 2):
            print('len(points.shape) < 2')
            return None
        if (points.shape[0] < 3):
            print('points.shape[0] < 3')
            return None
        contour = []
        for p in points:
            contour.append((p[0], p[1]))
        contour = np.asarray(contour)

        Contour_arclength = cv2.arcLength(contour, closed=True)
        Contour_area = cv2.contourArea(contour)
        ch = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, ch)
        if defects is None:
            Contour_convexitydefect_depths = []
            Contour_convexitydefect_areas = []
            Contour_convexitydefect_lengths = []
        else:
            no_defects = len(defects)
            defects = np.squeeze(defects)
            if no_defects == 1:
                defects = [defects]
            Contour_convexitydefect_depths = []
            Contour_convexitydefect_areas = []
            Contour_convexitydefect_lengths = []
            for defect in defects:
                Contour_convexitydefect_depths.append(defect[3] * ((resolution[0] + resolution[1]) / 2.0))
                Ax = contour[defect[0]][0] * resolution[0]
                Ay = contour[defect[0]][1] * resolution[1]
                Bx = contour[defect[1]][0] * resolution[0]
                By = contour[defect[1]][1] * resolution[1]
                Cx = contour[defect[2]][0] * resolution[0]
                Cy = contour[defect[2]][1] * resolution[1]
                darea = abs((Ax * (Bx - Cy) + Bx * (Cy - Ay) + Cx * (Ay - By)) / 2.0)
                dlength = np.sqrt(np.power(Ax - Bx, 2.0) + np.power(Ay - By, 2.0))
                Contour_convexitydefect_areas.append(darea)
                Contour_convexitydefect_lengths.append(dlength)
        m = cv2.moments(contours[0])
        # Third order component
        # J. Flusser: "On the Independence of Rotation Moment Invariants", Pattern Recognition, vol. 33, pp. 1405-1410, 2000.
        Hu_invariant8 = m['m11'] * (np.power(m['m30'] + m['m12'], 2.0) - np.power(m['m03'] + m['m21'], 2.0)) - (
                m['m20'] - m['m02']) * (m['m30'] - m['m12']) * (m['m03'] - m['m21'])
        Humoments = cv2.HuMoments(m)
        return {'Humoments': Humoments, 'Hu_invariant8': Hu_invariant8,
                'Contour_arclength': Contour_arclength, 'Contour_area': Contour_area,
                'Contour_convexitydefect_areas': Contour_convexitydefect_areas,
                'Contour_convexitydefect_lengths': Contour_convexitydefect_lengths,
                'Contour_convexitydefect_depths': Contour_convexitydefect_depths}


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
        labelimage = copy.deepcopy(intensity_images)
        labelimage[foreground_mask_images == 0] = 0

        print((intensity_images.shape, foreground_mask_images.shape, labelimage.shape))
        # Resolve which directions to be calculated
        Contour_measures = []
        if labelimage.shape[1] == labelimage.shape[2]:
            resolution2D = [resolution[1], resolution[2]]
            for x in range(labelimage.shape[0]):
                if len(np.unique(labelimage[x, :, :])) < 2:
                    continue
                cvimg = Utils.make_cv2_slice2D(labelimage[x, :, :]).copy()
                contours = cv2.findContours(cvimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contour_measures = self.calculate_2D_contour_measures(contours, resolution2D)
                if contour_measures is not None:
                    Contour_measures.append(contour_measures)
        if labelimage.shape[0] == labelimage.shape[2]:
            resolution2D = [resolution[0], resolution[2]]
            for y in range(labelimage.shape[1]):
                if len(np.unique(labelimage[:, y, :])) < 2:
                    continue
                cvimg = Utils.make_cv2_slice2D(labelimage[:, y, :]).copy()
                contours = cv2.findContours(cvimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contour_measures = self.calculate_2D_contour_measures(contours, resolution2D)
                if contour_measures is not None:
                    Contour_measures.append(contour_measures)
        if labelimage.shape[0] == labelimage.shape[1]:
            resolution2D = [resolution[0], resolution[1]]
            for z in range(labelimage.shape[2]):
                if len(np.unique(labelimage[:, :, z])) < 2:
                    continue
                cvimg = Utils.make_cv2_slice2D(labelimage[:, :, z]).copy()
                contours = cv2.findContours(cvimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contour_measures = self.calculate_2D_contour_measures(contours, resolution2D)
                if contour_measures is not None:
                    if 'write_visualization' in self.params[-1]:
                        LESIONDATAr_cvimg = Utils.make_cv2_slice2D(intensity_images[:, :, z]).copy()
                        LESIONr_cvimg = Utils.make_cv2_slice2D(foreground_mask_images[:, :, z]).copy()
                        basename = self.params[-1]['name'] + '_2D_curvature_' + str(self.params[:-1]).replace(' ', '_') + '_slice' + str(z)
                        visualizations.write_slice2D(LESIONDATAr_cvimg,
                                                     self.params[-1][
                                                         'write_visualization'] + os.sep + basename + '_data.tiff')
                        visualizations.write_slice2D_ROI(LESIONDATAr_cvimg, LESIONr_cvimg, self.params[-1][
                            'write_visualization'] + os.sep + basename + '_lesion.tiff', 0.4)
                        visualizations.write_slice2D_polygon(LESIONDATAr_cvimg, np.squeeze(np.array(contours[1][0])),
                                                             self.params[-1][
                                                                 'write_visualization'] + os.sep + basename + '_contour.tiff')
                    Contour_measures.append(contour_measures)

        subfun_3D_2D_Hu_moment_suffixes = ['Hu_Inv_1', 'Hu_Inv_1', 'Hu_Inv_1', 'Hu_Inv_1', 'Hu_Inv_1', 'Hu_Inv_1',
                                           'Hu_Inv_1']
        r = []
        for i in range(16):
            r.append([])
        for measures in Contour_measures:
            # Hu moments 1-8
            r[0].append(measures['Humoments'][0])
            r[1].append(measures['Humoments'][1])
            r[2].append(measures['Humoments'][2])
            r[3].append(measures['Humoments'][3])
            r[4].append(measures['Humoments'][4])
            r[5].append(measures['Humoments'][5])
            r[6].append(measures['Humoments'][6])
            r[7].append(measures['Hu_invariant8'])
            # Contour arclength
            r[8].append(measures['Contour_arclength'])
            # Contour area
            r[9].append(measures['Contour_area'])
            # Number of convexity defects
            if len(measures['Contour_convexitydefect_areas']) > 0:
                r[10].append(len(measures['Contour_convexitydefect_areas']))
                # Convexity defect areas
                r[11] = r[11] + measures['Contour_convexitydefect_areas']
                # Convexity defect lengths
                r[12] = r[12] + measures['Contour_convexitydefect_lengths']
                # Convexity defect depths
                r[13] = r[13] + measures['Contour_convexitydefect_depths']
                # Convexity defect length proportion to arclength
                r[14].append(np.sum(measures['Contour_convexitydefect_lengths']) / measures['Contour_arclength'])
                # Convexity defect depth proportion to area
                r[15].append(np.mean(measures['Contour_convexitydefect_depths']) / measures['Contour_area'])
        if len(Contour_measures) > 0:
            r1 = np.std(r[0]) / np.mean(r[0])
            r2 = np.std(r[1]) / np.mean(r[1])
            r3 = np.std(r[2]) / np.mean(r[2])
            r4 = np.std(r[3]) / np.mean(r[3])
            r5 = np.std(r[4]) / np.mean(r[4])
            r6 = np.std(r[5]) / np.mean(r[5])
            r7 = np.std(r[6]) / np.mean(r[6])
            r8 = np.std(r[7]) / np.mean(r[7])
            r10 = np.mean(r[8])
            r11 = np.median(r[8])
            r12 = np.std(r[8])
            r13 = scipy.stats.iqr(r[8])
            r20 = np.mean(r[9])
            r21 = np.median(r[9])
            r22 = np.std(r[9])
            r23 = scipy.stats.iqr(r[9])
        else:
            r1 = 0
            r2 = 0
            r3 = 0
            r4 = 0
            r5 = 0
            r6 = 0
            r7 = 0
            r8 = 0
            r10 = 0
            r11 = 0
            r12 = 0
            r13 = 0
            r20 = 0
            r21 = 0
            r22 = 0
            r23 = 0
        if len(Contour_measures) > 0:
            if len(r[10]) > 0:
                r30 = np.mean(r[10])
            else:
                r30 = 0
            if len(r[11]) > 0:
                r31 = np.mean(r[11])
                r32 = np.median(r[11])
                r33 = np.std(r[11])
                r34 = scipy.stats.iqr(r[11])
            else:
                r31 = 0
                r32 = 0
                r33 = 0
                r34 = 0
            if len(r[12]) > 0:
                r40 = np.mean(r[12])
                r41 = np.median(r[12])
                r42 = np.std(r[12])
                r43 = scipy.stats.iqr(r[12])
            else:
                r40 = 0
                r41 = 0
                r42 = 0
                r43 = 0
            if len(r[13]) > 0:
                r50 = np.mean(r[13])
                r51 = np.median(r[13])
                r52 = np.std(r[13])
                r53 = scipy.stats.iqr(r[13])
            else:
                r50 = 0
                r51 = 0
                r52 = 0
                r53 = 0
            if len(r[14]) > 0:
                r60 = np.mean(r[14])
                r61 = np.median(r[14])
                r62 = np.std(r[14])
                r63 = scipy.stats.iqr(r[14])
            else:
                r60 = 0
                r61 = 0
                r62 = 0
                r63 = 0
        else:
            r30 = 0
            r31 = 0
            r32 = 0
            r33 = 0
            r34 = 0
            r40 = 0
            r41 = 0
            r42 = 0
            r43 = 0
            r50 = 0
            r51 = 0
            r52 = 0
            r53 = 0
            r60 = 0
            r61 = 0
            r62 = 0
            r63 = 0
        if len(Contour_measures) > 0 and len(r[15]) > 0:
            print(r[15])
            r70 = np.mean(r[15])
            r71 = np.median(r[15])
            r72 = np.std(r[15])
            r73 = scipy.stats.iqr(r[15])
        else:
            r70 = 0
            r71 = 0
            r72 = 0
            r73 = 0
        return r1, r2, r3, r4, r5, r6, r7, r8, r10, r11, r12, r13, r20, r21, r22, r23, r30, r31, r32, r33, r34, r40, r41, r42, r43, r50, r51, r52, r53, r60, r61, r62, r63, r70, r71, r72, r73

    """
    Returns list of output value short names 

    @return list of return value short names, without spaces
    """

    def get_return_value_short_names(self):
        names = []
        if str(self.params[0]) == 'raw':
            for name in self.contour2D_names:
                names.append('2D_curvature_raw_intensity_' + name)
        else:
            for name in self.contour2D_names:
                names.append('2D_curvature_' + str(self.params[0]) + '_bins_histogram_' + name)
        return names

    """
    Returns list of input value descriptions 

    @return list of stings, or None
    """

    def get_input_descriptions(self):
        return ["Number of histogram bins",
                "(Optional) Write visualization"]

    """
    Returns list of strings decsribing boilerplate information about feature, including citations, if nay
    """

    @staticmethod
    def get_boilerplate():
        return ['Hu, M.K., 1962. Visual pattern recognition by moment invariants. IRE transactions on information theory, 8(2), pp.179-187.']


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
