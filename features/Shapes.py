

from Feature import FeatureIndexandBackground
from features.Utils import *
import numpy as np
from skimage import measure
import skimage
import trimesh
import scipy


def create_mesh(data, threshold, resolution):
    verts, faces, normals, values = measure.marching_cubes(data, level=threshold, spacing=(resolution[0], resolution[1], resolution[2]))
    min_loc = [data.shape[0], data.shape[1], data.shape[2]]
    max_loc = [0, 0, 0]
    verts_for_ply = []
    faces_for_ply = []
    for vert in verts:
        if vert[0] < min_loc[0]:
            min_loc[0] = vert[0]
        if vert[1] < min_loc[1]:
            min_loc[2] = vert[1]
        if vert[2] < min_loc[2]:
            min_loc[2] = vert[2]
        if vert[0] > max_loc[0]:
            max_loc[0] = vert[0]
        if vert[1] > max_loc[1]:
            max_loc[1] = vert[1]
        if vert[2] > max_loc[2]:
            max_loc[2] = vert[2]
        verts_for_ply.append((vert[0], vert[1], vert[2]))
    for face in faces:
        faces_for_ply.append(([face[0], face[1], face[2]], 255, 255, 255))

    return verts, faces, normals, values



"""
Radiomic feature base class for methods using one index lesion and background mask
"""


class Shapes(FeatureIndexandBackground):

    """
    Colletion of 3D shape feature names.
    Some features use Trimesh package (https://trimsh.org/trimesh.html).

        sarea3D: 3D surface area
        relsarea3D: relative 3D surface area
        tm_area_faces: trimesh package faces area
        tm_relarea_faces: trimesh relative faces area
        mean_angles: mean surface curvature angles
        median_angles: medina surface curvature angles
        SD_angles: standard deviation of suurface survature angles
        distance_mean: mean distance to the Center of Mass
        distance_median: median distance to the Center of Mass
        CSM_mean_curvature: Cohen-Steiner and Morvan mean curvature
        CSM_Gaus_mean_curvature: Gaussian curvature mean angle
        BGdistROI_median: Median distance to background surface
        BGdistROI_SD: Standard deviation fo distances to background surface
        BGdistROI_skewness: Skewness of distances to background surface
        BGdistROI_kurtosis: Kurtosity of distances to background surface
    """
    casefun_3D_shape_names = (
        'sarea3D', 'relsarea3D', 'tm_area_faces', 'tm_relarea_faces', 'mean_angles', 'median_angles', 'SD_angles',
        'distance_mean', 'distance_median', 'CSM_mean_curvature', 'CSM_Gaus_mean_curvature', 'BGdistROI_median',
        'BGdistROI_SD',
        'BGdistROI_skewness', 'BGdistROI_kurtosis')

    """
    Initialization

    @param name: name of the feature class, without spaces
    """

    def __init__(self):
        super('Shapes', None)


    """
    Executes the feature

    @param intensity_images: intensity values images
    @param foreground_mask_images: foreground mask images
    @param background_mask_images: background mask images
    @param resolution: image resolution mm x mm x mm
    @return number of return values matching get_return_value_descriptions
    """

    def fun(self, intensity_images, foreground_mask_images, background_mask_images, resolution, **kwargs):
        ROIdata = intensity_images[foreground_mask_images > 0]
        BGdata = intensity_images[background_mask_images > 0]
        print((len(ROIdata), np.min(ROIdata), np.max(ROIdata)))
        print((len(BGdata), np.min(BGdata), np.max(BGdata)))

        # try:
        verts, faces, normals, values = create_mesh(foreground_mask_images, 0.5, resolution)
        # except:
        #    print('failed to create')
        #    return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        sarea = skimage.measure.mesh_surface_area(verts, faces)
        # try:
        vertsw, facesw, normalsw, valuesw = create_mesh(background_mask_images, 0.5, resolution)
        # except:
        #    print('failed to create BG')
        #    return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        sareaw = skimage.measure.mesh_surface_area(vertsw, facesw)
        tm = trimesh.base.Trimesh(vertices=verts, faces=faces, face_normals=normals)
        tmw = trimesh.base.Trimesh(vertices=vertsw, faces=facesw, face_normals=normalsw)
        angles = trimesh.base.curvature.face_angles_sparse(tm)
        angles = angles.data
        mean_angles = np.mean(angles)
        median_angles = np.median(angles)
        SD_angles = np.std(angles)
        CoM = tm.center_mass
        distances = []
        for v in tm.vertices:
            distances.append(
                np.sqrt(np.power(v[0] - CoM[0], 2.0) + np.power(v[1] - CoM[1], 2.0) + np.power(v[2] - CoM[2], 2.0)))
        distance_mean = np.mean(distances)
        distance_median = np.median(distances)
        CSM_mean_curvature = trimesh.base.curvature.discrete_mean_curvature_measure(tm, [CoM], np.max(distances))
        CSM_Gaus_mean_curvature = trimesh.base.curvature.discrete_gaussian_curvature_measure(tm, [CoM],
                                                                                             np.max(distances))
        # Distance to whole gland
        closest, distancew, triangle_id = trimesh.base.proximity.closest_point(tm, tmw.vertices)
        w1 = np.median(distancew)
        w2 = np.std(distancew)
        w3 = scipy.stats.skew(distancew)
        w4 = scipy.stats.kurtosis(distancew)

        return sarea, sarea / sareaw, np.median(tm.area_faces), np.median(tm.area_faces) / len(
            ROIdata), mean_angles, median_angles, SD_angles, distance_mean, distance_median, CSM_mean_curvature[0], \
               CSM_Gaus_mean_curvature[0], w1, w2, w3, w4

    """
    Returns list of output value short names 

    @return list of return value short names, without spaces
    """

    def get_return_value_short_names(self):
        return self.casefun_3D_shape_names

    """
    Returns list of input value descriptions 

    @return list of stings, or None
    """

    def get_input_descriptions(self):
        return None

    """
    Returns list of strings decsribing boilerplate information about feature, including citations, if any
    """

    @staticmethod
    def get_boilerplate():
        return ["3D Shape faetures",
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
        return 1

    """
    Returns number of required background mask images
    """

    def number_of_background_mask_images_required(self):
        return 1
