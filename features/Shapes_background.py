

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


class Shapes_background(FeatureIndexandBackground):

    """
    Colletion of 3D shape feature names for background shape.
    Some features use Trimesh package (https://trimsh.org/trimesh.html).

        BGsarea3D: background 3D surface area
        BGtm_area_faces: trimesh package faces area for backround
        BGdistance_mean: mean distance to the Center of Mass of background
        BGdistance_median: median distance to the Center of Mass of background
        BGCSM_mean_curvature: Cohen-Steiner and Morvan mean curvature of background surface
        CSM_Gaus_mean_curvature: Gaussian curvature mean angle of background surface
    """
    casefun_3D_shape_names_BG = (
        'BGsarea3D', 'BGtm_area_faces', 'BGdistance_mean', 'BGdistance_median', 'BGCSM_mean_curvature',
        'BGCSM_Gaus_mean_curvature')

    """
    Initialization

    @param name: name of the feature class, without spaces
    """

    def __init__(self):
        super('Shapes_background', None)


    """
    Executes the feature

    @param intensity_images: intensity values images
    @param foreground_mask_images: foreground mask images
    @param background_mask_images: background mask images
    @param resolution: image resolution mm x mm x mm
    @return number of return values matching get_return_value_descriptions
    """

    def fun(self, intensity_images, foreground_mask_images, background_mask_images, resolution, **kwargs):
        BGdata = intensity_images[background_mask_images > 0]
        print((len(BGdata), np.min(BGdata), np.max(BGdata)))

        vertsw, facesw, normalsw, valuesw = create_mesh(background_mask_images, 0.5, resolution)
        sareaw = skimage.measure.mesh_surface_area(vertsw, facesw)
        tmw = trimesh.base.Trimesh(vertices=vertsw, faces=facesw, face_normals=normalsw)
        CoM = tmw.center_mass
        distances = []
        for v in tmw.vertices:
            distances.append(
                np.sqrt(np.power(v[0] - CoM[0], 2.0) + np.power(v[1] - CoM[1], 2.0) + np.power(v[2] - CoM[2], 2.0)))
        distance_mean = np.mean(distances)
        distance_median = np.median(distances)
        CSM_mean_curvature = trimesh.base.curvature.discrete_mean_curvature_measure(tmw, [CoM], np.max(distances))
        CSM_Gaus_mean_curvature = trimesh.base.curvature.discrete_gaussian_curvature_measure(tmw, [CoM],
                                                                                             np.max(distances))
        return sareaw, np.median(tmw.area_faces), distance_mean, distance_median, CSM_mean_curvature[0], \
               CSM_Gaus_mean_curvature[0]

    """
    Returns list of output value short names 

    @return list of return value short names, without spaces
    """

    def get_return_value_short_names(self):
        return self.casefun_3D_shape_names_BG

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
        return 0

    """
    Returns number of required background mask images
    """

    def number_of_background_mask_images_required(self):
        return 1
