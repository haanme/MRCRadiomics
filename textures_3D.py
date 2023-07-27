#!/usr/bin/env python

"""
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

(c) Harri Merisaari 2018-2022
"""

import os
import io
import tempfile
from dipy.align.reslice import reslice
from scipy.signal import correlate
import csv
from scipy import ndimage
from scipy.stats import iqr
import scipy.ndimage
import skimage
import trimesh
import sys
import subprocess
from plyfile import PlyData, PlyElement
from skimage import measure
from skimage.segmentation import active_contour
from scipy.interpolate import UnivariateSpline
from skimage import feature
import copy
from scipy import stats
import numpy as np
import os.path as path

numba_found = False
try:
    import numba
    from numba import njit, prange, vectorize, float64
    numba_found = True
except:
    print('Numba not found and related functions not in use')


"""
Resolve non-zero subregion in the image

@param img: 3D image data
@returns: [x_lo, x_hi, y_lo, y_hi, z_lo, z_hi] low and high bounds for bounding box
"""


def find_bounded_subregion3D(img):
    x_lo = 0
    for x in range(img.shape[0]):
        if np.max(img[x, :, :]) > 0:
            x_lo = x
            break
    x_hi = img.shape[0]-1
    for x in range(img.shape[0] - 1, -1, -1):
        if np.max(img[x, :, :]) > 0:
            x_hi = x
            break
    y_lo = 0
    for y in range(img.shape[1]):
        if np.max(img[:, y, :]) > 0:
            y_lo = y
            break
    y_hi = img.shape[1]-1
    for y in range(img.shape[1] - 1, -1, -1):
        if np.max(img[:, y, :]) > 0:
            y_hi = y
            break
    z_lo = 0
    for z in range(img.shape[2]):
        if np.max(img[:, :, z]) > 0:
            z_lo = z
            break
    z_hi = img.shape[2]-1
    for z in range(img.shape[2] - 1, -1, -1):
        if np.max(img[:, :, z]) > 0:
            z_hi = z
            break
    return x_lo, x_hi, y_lo, y_hi, z_lo, z_hi


"""
Resolve non-zero subregion in the image

@param data: data to be resliced
@param orig_resolution: original data resolution in mm
@param new_resolution: new data resolution in mm
@param int_order: interpolation order
@returns: resliced image data, 4x4 matrix of resliced data
"""


def reslice_array(data, orig_resolution, new_resolution, int_order):
    zooms = orig_resolution
    new_zooms = (new_resolution[0], new_resolution[1], new_resolution[2])
    affine = np.eye(4)
    affine[0, 0] = orig_resolution[0]
    affine[1, 1] = orig_resolution[1]
    affine[2, 2] = orig_resolution[2]
    print(data.shape)
    print(zooms)
    print(new_zooms)
    data2, affine2 = reslice(data, affine, zooms, new_zooms, order=int_order)
    data3 = np.zeros((data2.shape[1], data2.shape[0], data2.shape[2]))
    for zi in range(data3.shape[2]):
        data3[:, :, zi] = np.rot90(data2[:, :, zi], k=3)
    return data3, affine2


"""
Resolve 2D-spline curvature for x/y-coordinates

@param x: array of x-coordinates
@param y: array of y-coordinates
@param error: standard deviation for spline
@returns: curvature coordinates
"""


def curvature_splines(x, y, error=0.1):
    # handle list of complex case
    t = np.arange(x.shape[0])
    std = error * np.ones_like(x)
    if len(x) < 3:
        return [0]
    fx = UnivariateSpline(t, x, k=4, w=1 / np.sqrt(std))
    fy = UnivariateSpline(t, y, k=4, w=1 / np.sqrt(std))
    xd = fx.derivative(1)(t)
    xdd = fx.derivative(2)(t)
    yd = fy.derivative(1)(t)
    ydd = fy.derivative(2)(t)
    curvature = (xd * ydd - yd * xdd) / np.power(xd ** 2 + yd ** 2, 3 / 2)
    return curvature


"""
Levelset coordinates

@param LESIONDATAr: Intensity data
@param LESIONr: Lesion mask
@param BGr: Background region mask
@param resolution: data resolution in mm [x,y,z]
@returns: [median of curvatures,, standard deviation of curvatures]
"""
casefun_levelset_names = ('Levelset_median', 'Levelset_std')


def casefun_levelset(LESIONDATAr, LESIONr, BGr, resolution):
    Curvatures2 = []
    for z in range(LESIONr[0].shape[2]):
        ROI_img = np.squeeze(LESIONr[0][:, :, z])
        img = np.squeeze(LESIONDATAr[:, :, z])
        img[ROI_img == 0] = 0
        if np.max(ROI_img) == 0:
            continue
        no_values = len(ROI_img[ROI_img > 0])
        # print('positive values:' + str(no_values))
        if no_values < 5:
            continue
        init = np.where(ROI_img > 0)
        CoM = (np.mean(init[0]), np.mean(init[1]))
        Dists = []
        for init_i in range(len(init[0])):
            coord = (init[0][init_i], init[1][init_i])
            Dists.append(np.sqrt(np.power(coord[0] - CoM[0], 2) + np.power(coord[1] - CoM[1], 2)))
        R = np.max(Dists)
        init = np.squeeze(np.array(init)).T
        s = np.linspace(0, 2 * np.pi, 100)
        x = CoM[0] + R * np.cos(s)
        y = CoM[1] + R * np.sin(s)
        init = np.array([x, y]).T

        snake = active_contour(img, init, alpha=0.015, beta=20, gamma=0.001)
        c = curvature_splines(snake[:, 0], snake[:, 1])
        for v in c:
            Curvatures2.append(v)

    return np.median(Curvatures2), np.std(Curvatures2)


"""
3D Gray-level co-occurrence matrix (GLCM) texture features
Lee, J., Jain, R., Khalil, K., Griffith, B., Bosca, R., Rao, G. and Rao, A., 2016. Texture feature ratios from relative CBV maps of perfusion MRI are associated with patient survival in glioblastoma. American Journal of Neuroradiology, 37(1), pp.37-43.

@param LESIONDATAr: Intensity data
@param LESIONr: Lesion mask
@param resolution: data resolution in mm [x,y,z]
@returns: 
  contrast: GLCM contrast
  dissimilarity: GLCM dissimilarity
  homogeneity: GLCM homogeneity
  ASM: GLCM Angular Second Moment (ASM)
  energy: GLCM energy
  correlation: GLCM correlation
"""


def get_3D_GLCM(LESIONDATAr, LESIONr, resolution):
    Ng = [(0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]

    c_all = LESIONDATAr[LESIONr > 0]
    c_min = np.min(c_all)
    c_all = np.subtract(LESIONDATAr, c_min)
    c_max = np.max(c_all)
    c_all = np.divide(c_all, c_max)
    c_all = np.round(np.multiply(c_all, 127))
    G = np.zeros([128, 128, 1, len(Ng)])
    # Calculate Haralick across surface
    comparisons = 0
    ROI_coords = np.nonzero(LESIONr > 0)
    ROI_coord = [0, 0, 0]
    for ROI_coord_i in range(len(ROI_coords)):
        ROI_coord[0] = ROI_coords[0][ROI_coord_i]
        ROI_coord[1] = ROI_coords[1][ROI_coord_i]
        ROI_coord[2] = ROI_coords[2][ROI_coord_i]
        # Divide into
        for Ng_coord_i in range(len(Ng)):
            Ng_coord = Ng[Ng_coord_i]
            v = [ROI_coord[0] + Ng_coord[0], ROI_coord[1] + Ng_coord[1], ROI_coord[2] + Ng_coord[2]]
            avg_loc_x = int(round(v[0]))
            avg_loc_y = int(round(v[1]))
            avg_loc_z = int(round(v[2]))
            if avg_loc_x < 0:
                continue
            if avg_loc_y < 0:
                continue
            if avg_loc_z < 0:
                continue
            if avg_loc_x >= c_all.shape[0]:
                continue
            if avg_loc_y >= c_all.shape[1]:
                continue
            if avg_loc_z >= c_all.shape[2]:
                continue
            i = int(c_all[avg_loc_x, avg_loc_y, avg_loc_z])
            j = int(c_all[avg_loc_x, avg_loc_y, avg_loc_z])
            G[i, j, 0, Ng_coord_i] += 1
            comparisons += 1
    P = np.divide(G, comparisons)
    if len(np.unique(P)) == 1:
        contrast = 0.0
        dissimilarity = 0.0
        homogeneity = np.sum(P)
        ASM = np.sum(P) * np.sum(P)
        energy = np.sqrt(ASM)
        correlation = 1
    else:
        contrast = np.mean(skimage.feature.texture.greycoprops(P, prop='contrast'))
        dissimilarity = np.mean(skimage.feature.texture.greycoprops(P, prop='dissimilarity'))
        homogeneity = np.mean(skimage.feature.texture.greycoprops(P, prop='homogeneity'))
        ASM = np.mean(skimage.feature.texture.greycoprops(P, prop='ASM'))
        energy = np.mean(skimage.feature.texture.greycoprops(P, prop='energy'))
        correlation = np.mean(skimage.feature.texture.greycoprops(P, prop='correlation'))
    return contrast, dissimilarity, homogeneity, ASM, energy, correlation


"""
Get 3D-surface features

@param tmv: 3D surface object
@param verts: 3D vertices
@param faces: 3D faces
@param Idata: intenstiy data
@param resolution: data resolution [x,y,z] in mm
@returns:
  contrast: GLCM contrast
  dissimilarity: GLCM dissimilarity
  homogeneity: GLCM homogeneity
  ASM: GLCM ASM
  energy: GLCM energy
  correlation: GLCM correlation
"""


def get_mesh_surface_features(tmw, verts, faces, Idata, resolution):
    CoM = tmw.center_mass
    c_all = []
    c_all_vis = []
    polar_coordinates = []
    vertex_neighbors = []
    for v_i in range(len(tmw.vertices)):
        vert = tmw.vertices[v_i]
        avg_loc_x = int(round(vert[0] / resolution[0]))
        avg_loc_y = int(round(vert[1] / resolution[1]))
        avg_loc_z = int(round(vert[2] / resolution[2]))
        if avg_loc_x < 0 or avg_loc_y < 0 or avg_loc_z < 0:
            c = float('NaN')
        elif avg_loc_x >= Idata.shape[0] or avg_loc_y >= Idata.shape[1] or avg_loc_z >= Idata.shape[2]:
            c = float('NaN')
        else:
            c = Idata[avg_loc_x, avg_loc_y, avg_loc_z]
        x = (CoM[0] - vert[0])
        y = (CoM[1] - vert[1])
        z = (CoM[2] - vert[2])
        r = np.sqrt(np.power(x, 2.0) + np.power(y, 2.0) + np.power(z, 2.0))
        t = np.arccos(z / r)
        s = np.arctan(y / x)
        if not np.isnan(c):
            c_all.append(c)
            c_all_vis.append(v_i)
            vertex_neighbors.append(tmw.vertex_neighbors[v_i])
            polar_coordinates.append((r, t, s))
    c_all = np.array(c_all)
    if len(c_all) == 0:
        return 0, 0, 0, 0, 0, 0
    if len(np.unique(c_all)) < 2:
        return 0, 0, 0, 0, 0, 0
    c_min = np.min(c_all)
    c_max = np.max(c_all) - c_min
    print((c_all, c_min, c_max))
    c_all = [int(x) for x in np.round(np.multiply(np.divide(np.subtract(c_all, c_min), c_max), 127))]
    G = np.zeros([128, 128, 1, 10])
    # Calculate Haralick across surface
    comparisons = 0
    for v_i in range(len(vertex_neighbors)):
        Ng_verts = vertex_neighbors[v_i]
        v_t = polar_coordinates[v_i][1]
        v_s = polar_coordinates[v_i][2]
        # Divide into
        for Ng_i in range(len(Ng_verts)):
            Ng = Ng_verts[Ng_i]
            if Ng not in c_all:
                continue
            i = c_all[Ng]
            j = c_all[v_i]
            t = polar_coordinates[v_i][1]
            s = polar_coordinates[v_i][2]
            angle = int(round((np.arctan2(v_t - t, v_s - s) + np.pi) / (2 * np.pi) * 9))
            G[i, j, 0, angle] += 1
            comparisons += 1
    P = np.divide(G, comparisons)
    if len(np.unique(P)) == 1:
        contrast = 0.0
        dissimilarity = 0.0
        homogeneity = np.sum(P)
        ASM = np.sum(P) * np.sum(P)
        energy = np.sqrt(ASM)
        correlation = 1
    else:
        contrast = np.mean(skimage.feature.texture.greycoprops(P, prop='contrast'))
        dissimilarity = np.mean(skimage.feature.texture.greycoprops(P, prop='dissimilarity'))
        homogeneity = np.mean(skimage.feature.texture.greycoprops(P, prop='homogeneity'))
        ASM = np.mean(skimage.feature.texture.greycoprops(P, prop='ASM'))
        energy = np.mean(skimage.feature.texture.greycoprops(P, prop='energy'))
        correlation = np.mean(skimage.feature.texture.greycoprops(P, prop='correlation'))

    return contrast, dissimilarity, homogeneity, ASM, energy, correlation


"""
Names for 3D Gray Level Co-occurrence Matrix features. 

All features are median estimates
  contrastS: surface GLCM contrast
  dissimilarityS: surface GLCM dissimilarity
  homogeneityS: surface GLCM homogeneity
  ASMS: surface GLCM ASM
  energyS: surface GLCM energy
  correlationS: surface GLCM correlation
  contrastV: volume GLCM contrast
  dissimilarityV: volume GLCM dissimilarity
  homogeneityV: volume GLCM homogeneity
  ASMV: volume GLCM ASM
  energyV: volume GLCM energy
  correlationV: volume GLCM correlation
"""
casefun_3D_GLCM_names = (
'median_contrastS', 'median_dissimilarityS', 'median_homogeneityS', 'median_ASMS', 'median_energyS',
'median_correlationS', 'median_contrastV', 'median_dissimilarityV', 'median_homogeneityV', 'median_ASMV',
'median_energyV', 'median_correlationV')

"""
Get 3D Gray Level Co-occurrence Matrix features

@param LESIONDATAr: Intensity data
@param LESIONr: Lesion mask
@param BGr: background mask
@param resolution: data resolution in mm [x,y,z]
@returns: Please see casefun_3D_GLCM_names
"""


def casefun_3D_GLCM(LESIONDATAr, LESIONr, BGr, resolution):
    ROIdata = LESIONDATAr[LESIONr[0] > 0]
    verts_all, faces_all = create_mesh_smooth(LESIONDATAr, LESIONr[1], 0.5, resolution, 'L1')
    if len(verts_all) == 0:
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    tm = trimesh.base.Trimesh(vertices=verts_all, faces=faces_all)
    contrastS_all, dissimilarityS_all, homogeneityS_all, ASMS_all, energyS_all, correlationS_all = get_mesh_surface_features(
        tm, verts_all, faces_all, LESIONDATAr, resolution)
    contrastV_all, dissimilarityV_all, homogeneityV_all, ASMV_all, energyV_all, correlationV_all = get_3D_GLCM(
        LESIONDATAr, LESIONr[1], resolution)
    for LESION_i in range(2, len(LESIONr)):
        verts, faces = create_mesh_smooth(LESIONDATAr, LESIONr[LESION_i], 1.0, resolution, 'L' + str(LESION_i - 1))
        tm = trimesh.base.Trimesh(vertices=verts, faces=faces)
        contrastS, dissimilarityS, homogeneityS, ASMS, energyS, correlationS = get_mesh_surface_features(tm, verts,
                                                                                                         faces,
                                                                                                         LESIONDATAr,
                                                                                                         resolution)
        contrastV, dissimilarityV, homogeneityV, ASMV, energyV, correlationV = get_3D_GLCM(LESIONDATAr,
                                                                                           LESIONr[LESION_i],
                                                                                           resolution)
        if LESION_i == 2:
            contrastS_all = [contrastS_all] + contrastS
            dissimilarityS_all = [dissimilarityS_all] + dissimilarityS
            homogeneityS_all = [homogeneityS_all] + homogeneityS
            ASMS_all = [ASMS_all] + ASMS
            energyS_all = [energyS_all] + energyS
            correlationS_all = [correlationS_all] + correlationS
            contrastV_all = [contrastV_all] + contrastV
            dissimilarityV_all = [dissimilarityV_all] + dissimilarityV
            homogeneityV_all = [homogeneityV_all] + homogeneityV
            ASMV_all = [ASMV_all] + ASMV
            energyV_all = [energyV_all] + energyV
            correlationV_all = [correlationV_all] + correlationV
        else:
            contrastS_all = contrastS_all + contrastS
            dissimilarityS_all = dissimilarityS_all + dissimilarityS
            homogeneityS_all = homogeneityS_all + homogeneityS
            ASMS_all = ASMS_all + ASMS
            energyS_all = energyS_all + energyS
            correlationS_all = correlationS_all + correlationS
            contrastV_all = contrastV_all + contrastV
            dissimilarityV_all = dissimilarityV_all + dissimilarityV
            homogeneityV_all = homogeneityV_all + homogeneityV
            ASMV_all = ASMV_all + ASMV
            energyV_all = energyV_all + energyV
            correlationV_all = correlationV_all + correlationV

    contrastS = np.median(contrastS_all)
    dissimilarityS = np.median(dissimilarityS_all)
    homogeneityS = np.median(homogeneityS_all)
    ASMS = np.median(ASMS_all)
    energyS = np.median(energyS_all)
    correlationS = np.median(correlationS_all)
    contrastV = np.median(contrastV_all)
    dissimilarityV = np.median(dissimilarityV_all)
    homogeneityV = np.median(homogeneityV_all)
    ASMV = np.median(ASMV_all)
    energyV = np.median(energyV_all)
    correlationV = np.median(correlationV_all)
    return contrastS, dissimilarityS, homogeneityS, ASMS, energyS, correlationS, contrastV, dissimilarityV, homogeneityV, ASMV, energyV, correlationV


"""
Names for 3D Gray Level Co-occurrence Matrix fro background region

All measurs are median estimates in background region
  contrastS: surface GLCM contrast
  dissimilarityS: surface GLCM dissimilarity
  homogeneityS: surface GLCM homogeneity
  ASMS: surface GLCM ASM
  energyS: surface GLCM energy
  correlationS: surface GLCM correlation
  contrastV: volume GLCM contrast
  dissimilarityV: volume GLCM dissimilarity
  homogeneityV: volume GLCM homogeneity
  ASMV: volume GLCM ASM
  energyV: volume GLCM energy
  correlationV: volume GLCM correlation
"""
casefun_3D_GLCM_names_BG = (
'BGmedian_contrastS', 'BGmedian_dissimilarityS', 'BGmedian_homogeneityS', 'BGmedian_ASMS', 'BGmedian_energyS',
'BGmedian_correlationS', 'BGmedian_contrastV', 'BGmedian_dissimilarityV', 'BGmedian_homogeneityV', 'BGmedian_ASMV',
'BGmedian_energyV', 'BGmedian_correlationV')

"""
Get 3D Gray Level Co-occurrence Matrix fro background region

@param LESIONDATAr: Intensity data
@param LESIONr: not in use
@param BGr: background mask
@param resolution: data resolution in mm [x,y,z]
@returns: Please see casefun_3D_GLCM_names_BG
"""


def casefun_3D_GLCM_BG(LESIONDATAr, LESIONr, BGr, resolution):
    verts_all, faces_all = create_mesh_smooth(LESIONDATAr, BGr, 0.5, resolution, 'BG')
    if len(verts_all) == 0:
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    tm = trimesh.base.Trimesh(vertices=verts_all, faces=faces_all)
    contrastS_all, dissimilarityS_all, homogeneityS_all, ASMS_all, energyS_all, correlationS_all = get_mesh_surface_features(
        tm, verts_all, faces_all, LESIONDATAr, resolution)
    contrastV_all, dissimilarityV_all, homogeneityV_all, ASMV_all, energyV_all, correlationV_all = get_3D_GLCM(
        LESIONDATAr, BGr, resolution)

    contrastS = np.median(contrastS_all)
    dissimilarityS = np.median(dissimilarityS_all)
    homogeneityS = np.median(homogeneityS_all)
    ASMS = np.median(ASMS_all)
    energyS = np.median(energyS_all)
    correlationS = np.median(correlationS_all)
    contrastV = np.median(contrastV_all)
    dissimilarityV = np.median(dissimilarityV_all)
    homogeneityV = np.median(homogeneityV_all)
    ASMV = np.median(ASMV_all)
    energyV = np.median(energyV_all)
    correlationV = np.median(correlationV_all)
    return contrastS, dissimilarityS, homogeneityS, ASMS, energyS, correlationS, contrastV, dissimilarityV, homogeneityV, ASMV, energyV, correlationV


"""
Names for statistical moments of intensity values inside lesion

  mean: mean of lesion
  median: median of lesion
  p25: 25% percentile of lesion
  p75: 75% percentile of lesion
  skewness: skewness of lesion
  kurtosis: kurtosis of lesion
  SD: standard deviation of lesion
  rng: range of lesion
  volume: volume of lesion
  CV: coefficient of variation for lesion
"""
casefun_01_moments_names = (
'mean', 'median', '25percentile', '75percentile', 'skewness', 'kurtosis', 'SD', 'range', 'ml', 'CV')

"""
Statistical moments of intensity values inside lesion

@param LESIONDATAr: Intensity data
@param LESIONr: Lesion mask
@param BGr: background mask
@param resolution: data resolution in mm [x,y,z]
@returns: Please see casefun_01_moments_names
"""


def casefun_01_moments(LESIONDATAr, LESIONr, BGr, resolution):
    ROIdata = LESIONDATAr[LESIONr[0] > 0]
    mean = np.mean(ROIdata)
    median = np.median(ROIdata)
    p25 = np.percentile(ROIdata, 25)
    p75 = np.percentile(ROIdata, 75)
    skewness = scipy.stats.skew(ROIdata)
    kurtosis = scipy.stats.kurtosis(ROIdata)
    SD = np.std(ROIdata)
    rng = np.max(ROIdata) - np.min(ROIdata)
    # Volume as cubic mm to cubic centimeters, which is mL
    volume = len(ROIdata) * (0.001 * resolution[0] * resolution[1] * resolution[2])
    print((resolution, volume))
    if not mean == 0:
        CV = SD / mean
    else:
        CV = 0.0
    return mean, median, p25, p75, skewness, kurtosis, SD, rng, volume, CV


"""
Names for statistical moments of intensity values inside whole organ

  wmean: mean of
  wmedian: median of organ
  wp25: 25% percentile of organ
  wp75: 75% percentile of organ
  wskewness: skewness of organ
  wkurtosis: kurtosis of organ
  wSD: standard deviation of organ
  wIQrange: range of organ
  volume: volume of organ
"""
casefun_01_moments_BG_names = (
'BGmean', 'BGmedian', 'BG25percentile', 'BG75percentile', 'BGskewness', 'BGkurtosis', 'BGSD', 'BGIQR', 'BGml')

"""
Statistical moments of intensity values inside whole organ

@param LESIONDATAr: Intensity data
@param LESIONr: Lesion mask
@param BGr: background mask
@param resolution: data resolution in mm [x,y,z]
@returns: Please see
"""


def casefun_01_moments_BG(LESIONDATAr, LESIONr, BGr, resolution):
    BGdata = LESIONDATAr[BGr > 0]
    volume = len(BGdata) * (0.001 * resolution[0] * resolution[1] * resolution[2])
    print((resolution, volume))
    wmean = np.mean(BGdata)
    wmedian = np.median(BGdata)
    wp25 = np.percentile(BGdata, 25)
    wp75 = np.percentile(BGdata, 75)
    wskewness = scipy.stats.skew(BGdata)
    wkurtosis = scipy.stats.kurtosis(BGdata)
    wSD = np.std(BGdata)
    wIQrange = iqr(BGdata)

    return wmean, wmedian, wp25, wp75, wskewness, wkurtosis, wSD, wIQrange, volume


"""
Names for first order statistics of raw intensity using background region.

  relBGmean: Relative mean intensity to background
  relBGmedian: Relative mean intensity to background
  relBG25percentile: : Relative 25% percentile to background
  relBG75percentile: : Relative 75% percentile to background
  relBGskewness: : Relative skewness to background
  relBGkurtosis: : Relative kurtosity to background
  relBGSD: : Relative standard deviation to background
  BGIQR: : Relative iterquatile range to background
"""
casefun_01_moments_relativeBG_names = (
'relBGmean', 'relBGmedian', 'relBG25percentile', 'relBG75percentile', 'relBGskewness', 'relBGkurtosis', 'relBGSD',
'BGIQR')

"""
First order statistics of raw intensity using background region.

@param LESIONDATAr: Intensity data
@param LESIONr: Lesion mask
@param BGr: background mask
@param resolution: data resolution in mm [x,y,z]
@returns: Please see casefun_01_moments_relativeBG_names
"""


def casefun_01_moments_relativeBG(LESIONDATAr, LESIONr, BGr, resolution):
    ROIdata = LESIONDATAr[LESIONr[0] > 0]
    BGdata = LESIONDATAr[BGr > 0]
    volume = len(BGdata) * (0.001 * resolution[0] * resolution[1] * resolution[2])
    print((resolution, volume))
    mean = np.mean(ROIdata)
    median = np.median(ROIdata)
    print(ROIdata.shape)
    p25 = np.percentile(ROIdata, 25)
    p75 = np.percentile(ROIdata, 75)
    skewness = scipy.stats.skew(ROIdata)
    kurtosis = scipy.stats.kurtosis(ROIdata)
    SD = np.std(ROIdata)
    IQrange = iqr(ROIdata)
    if mean == 0:
        BGmean = 0
    else:
        BGmean = mean / (mean + np.mean(BGdata))
    if median == 0:
        BGmedian = 0
    else:
        BGmedian = median / (median + np.median(BGdata))
    if p25 == 0:
        BGp25 = 0
    else:
        BGp25 = p25 / (p25 + np.percentile(BGdata, 25))
    if p75 == 0:
        BGp75 = 0
    else:
        BGp75 = p75 / (p75 + np.percentile(BGdata, 75))
    print(skewness)
    if skewness == 0:
        BGskewness = 0
    else:
        BGskewness = skewness / (skewness + scipy.stats.skew(BGdata))
    if kurtosis == 0:
        BGkurtosis = 0
    else:
        BGkurtosis = kurtosis / (kurtosis + scipy.stats.kurtosis(BGdata))
    if SD == 0:
        BGSD = 0
    else:
        BGSD = SD / (SD + np.std(BGdata))
    if IQrange == 0:
        BGIQrange = 0
    else:
        BGIQrange = IQrange / (IQrange + iqr(BGdata))

    return BGmean, BGmedian, BGp25, BGp75, BGskewness, BGkurtosis, BGSD, BGIQrange


"""
Data from largest found slice in the lesion

@param LESIONDATAr: Intensity data
@param LESIONr: Lesion mask
@param BGr: background mask
@param resolution: data resolution in mm [x,y,z]
@param amount: not in use
@returns:
  ROIdata: ROI data
  BGdata: background data
"""


def Moment2_fun_largest_slice(LESIONDATAr, LESIONr, BGr, resolution, amount):
    volumes = []
    indexes = []
    max_vol = -1
    max_idx = -1
    for z in range(LESIONr[0].shape[2]):
        LESIONslice = LESIONr[0][:, :, z]
        vol = len(LESIONslice[LESIONslice > 0])
        if vol > 0:
            volumes.append(vol)
            indexes.append(z)
            if vol > max_vol:
                max_vol = vol
                max_idx = z
    eroded = copy.deepcopy(LESIONr[0][:, :, max_idx])
    ROIdata = LESIONDATAr[:, :, max_idx][eroded > 0]
    BGr = BGr[:, :, max_idx]
    BGr[eroded > 0] = 0
    BGdata = LESIONDATAr[:, :, max_idx][BGr > 0]
    return ROIdata, BGdata


"""
Experimental method where data from largest found slice in the lesion, 
using 3x3 region around location with smallest mean intensity in 3x3 
smoothing.

@param LESIONDATAr: Intensity data
@param LESIONr: Lesion mask
@param BGr: background mask
@param resolution: data resolution in mm [x,y,z]
@param amount: not in use
@returns:
  ROIdata: ROI data
  BGdata: background data
"""


def Moment2_fun_largest_slice3x3(LESIONDATAr, LESIONr, BGr, resolution, amount):
    volumes = []
    indexes = []
    max_vol = -1
    max_idx = -1
    print(np.max(LESIONr[0]))
    for z in range(LESIONr[0].shape[2]):
        LESIONslice = LESIONr[0][:, :, z]
        vol = len(LESIONslice[LESIONslice > 0])
        print((vol, z))
        if vol > 0:
            volumes.append(vol)
            indexes.append(z)
            if vol > max_vol:
                max_vol = vol
                max_idx = z
    eroded = copy.deepcopy(LESIONr[0][:, :, max_idx])

    # resolve 3x3 region around minimum or maximum
    NG = []
    for x in [-1, 0, 1]:
        for y in [-1, 0, 1]:
            NG.append((x, y))

    ROIdata = LESIONDATAr[:, :, max_idx][eroded > 0]
    BGdata = LESIONDATAr[:, :, max_idx][BGr[:, :, max_idx] > 0]

    min_val = np.max(ROIdata)
    min_coords = [-1, -1]
    for x in range(eroded.shape[0]):
        for y in range(eroded.shape[1]):
            if eroded[x, y] == 0:
                continue
            NGvals = []
            for NGi in NG:
                NGvals.append(LESIONDATAr[x + NGi[0], y + NGi[1], max_idx])
            NGval_mean = np.mean(NGvals)
            if NGval_mean < min_val:
                min_val = NGval_mean
                min_coords = [x, y]
    eroded = np.zeros_like(eroded)
    for x in [-1, 0, 1]:
        for y in [-1, 0, 1]:
            eroded[min_coords[0] + x, min_coords[1] + y] = 1

    ROIdata = LESIONDATAr[:, :, max_idx][eroded > 0]
    BGr = BGr[:, :, max_idx]
    BGr[eroded > 0] = 0
    BGdata = LESIONDATAr[:, :, max_idx][BGr > 0]
    return ROIdata, BGdata


"""
Experimental method where thresholded region afte kernel density estimation 
is considered

@param LESIONDATAr: Intensity data
@param LESIONr: Lesion mask
@param BGr: background mask
@param resolution: data resolution in mm [x,y,z]
@param amount: coefficient for thresholding of KDE distribution
@returns:
  ROIdata: ROI data
  BGdata: background data
"""


def Moment2_fun_largest_sliceKDE(LESIONDATAr, LESIONr, BGr, resolution, amount):
    volumes = []
    indexes = []
    max_vol = -1
    max_idx = -1
    for z in range(LESIONr[0].shape[2]):
        LESIONslice = LESIONr[0][:, :, z]
        vol = len(LESIONslice[LESIONslice > 0])
        if vol > 0:
            volumes.append(vol)
            indexes.append(z)
            if vol > max_vol:
                max_vol = vol
                max_idx = z
    eroded = copy.deepcopy(LESIONr[0][:, :, max_idx])
    eroded = ndimage.morphology.binary_dilation(eroded, iterations=1).astype("uint8")
    orig_vol = len(eroded[eroded > 0])

    # resolve 5x5 region around minimum or maximum
    centroid_x = []
    centroid_y = []
    ROIdata_z = LESIONDATAr[:, :, max_idx][eroded > 0]
    ROIdata_zmax = np.max(ROIdata_z)
    print(ROIdata_zmax)
    ROIdata_zmin = np.min(ROIdata_z)
    for x in range(eroded.shape[0]):
        for y in range(eroded.shape[1]):
            if eroded[x, y] > 0:
                val = 1 - LESIONDATAr[x, y, max_idx] / ROIdata_zmax
                for v in range(int(val * 1000)):
                    centroid_x.append(x)
                    centroid_y.append(y)
    kernel = stats.gaussian_kde(np.vstack([centroid_x, centroid_y]))
    KDEregion = np.zeros_like(eroded)
    centroid_x = []
    centroid_y = []
    for x in range(eroded.shape[0]):
        for y in range(eroded.shape[1]):
            if eroded[x, y] > 0:
                centroid_x.append(x)
                centroid_y.append(y)
    centroid_xy = np.vstack([centroid_x, centroid_y])
    values = kernel.evaluate(centroid_xy)
    th = np.max(values) * amount
    for c_i in range(len(centroid_xy[0])):
        if values[c_i] > th:
            KDEregion[centroid_xy[0][c_i], centroid_xy[1][c_i]] = 1
    ROIdata = LESIONDATAr[:, :, max_idx][KDEregion > 0]
    BGr_idx = BGr[:, :, max_idx]
    BGr_idx[KDEregion > 0] = 0
    BGdata = LESIONDATAr[:, :, max_idx][BGr_idx > 0]
    return ROIdata, BGdata


"""
Statistical descriptors for background region

    mean
    median
    25percentile
    75percentile
    skewness
    kurtosis
    SD
    IQR
    relBGmean: Relatine mean to background
    relBGmedian: Relatine median to background
    relBG25percentile: Relatine 25% percentile to background
    relBG75percentile: Relatine 75% percentile to background
    relBGskewness: Relatine skewness to background
    relBGkurtosis: Relatine kurtosis to background
    relBGSD: Relatine standard deviation to background
    BGIQR: Relatine interquartile range to background
"""
casefun_01_moments2_BG_names = ('mean', 'median', '25percentile', '75percentile', 'skewness', 'kurtosis', 'SD', 'IQR',
                                'relBGmean', 'relBGmedian', 'relBG25percentile', 'relBG75percentile', 'relBGskewness',
                                'relBGkurtosis', 'relBGSD', 'BGIQR')

"""
Resolve 3D Gray Level Co-occurrence Matrix feature names

@param param: method parameters
@returns: Method names as returned by the corresponding function
"""


def casefun_01_moments2_name_generator(params):
    names = []
    for name in casefun_01_moments2_BG_names:
        names.append('UTUMoments2_%2.1f_%s_%s' % (params[0], params[1], name))
    return names


"""
Smoothing of 3D mesh with external Meshlab tool (http://www.meshlab.net/). 

@param in_file: input meshlab file
@param out_file: input meshlab file
@param binary_path: Meshlab binary location
@param cfg_path: Configuration file location
"""


def Meshlab_smooth(in_file, out_file, binary_path="C:/Program Files/VCG/Meshlab/meshlabserver", cfg_path=" -s ./meshlab_smooth.mlx"):
    # Add input mesh
    command = binary_path + " -i " + in_file
    # Add the filter script
    command += " -s " + cfg_path
    # Add the output filename and output flags
    if os.path.exists(out_file):
        os.remove(out_file)
    command += " -o " + out_file + " -m vn fn"
    # Execute command
    print("Going to execute: " + command)
    try:
        output = subprocess.check_output(command, shell=False)
    except:
        print('Exception in execution of [' + command + ']')
        raise


"""
Get samples along mesh surface

@param verts: Mesh vertices
@param faces: Mesh faces
@param Idata: Intensity data
@param resolution: resolution in mm [x,y,z]
@returns: Intensity values at the center of faces
"""


def get_mesh_surface_samples(verts, faces, Idata, resolution):
    c_all = []
    for face in faces:
        avg_loc_x = int(round(np.mean([verts[face[0]][0], verts[face[1]][0], verts[face[2]][0]]) / resolution[0]))
        avg_loc_y = int(round(np.mean([verts[face[0]][1], verts[face[1]][1], verts[face[2]][1]]) / resolution[1]))
        avg_loc_z = int(round(np.mean([verts[face[0]][2], verts[face[1]][2], verts[face[2]][2]]) / resolution[2]))
        if avg_loc_x < 0 or avg_loc_y < 0 or avg_loc_z < 0:
            c = float('NaN')
        elif avg_loc_x >= Idata.shape[0] or avg_loc_y >= Idata.shape[1] or avg_loc_z >= Idata.shape[2]:
            c = float('NaN')
        else:
            c = Idata[avg_loc_x, avg_loc_y, avg_loc_z]
        c_all.append(c)
    return c_all


"""
Writes ply ASCII mesh definition file

@param verts: Mesh vertices
@param faces: Mesh faces
@param Idata: Intensity data
@param resolution: Data resolution in mm [x,y,z]
@param plyfilename: Output filename
"""


def write_plyfile(verts, faces, Idata, resolution, plyfilename):
    verts_for_ply = []
    faces_for_ply = []
    for vert in verts:
        verts_for_ply.append((vert[0], vert[1], vert[2]))
    max_I = np.amax(Idata)
    c_all = get_mesh_surface_samples(verts, faces, Idata, resolution)
    for face_i in range(len(faces)):
        face = faces[face_i]
        c = c_all[face_i] / max_I * 255
        if not np.isnan(c):
            faces_for_ply.append(([face[0], face[1], face[2]], c, c, c))
        else:
            faces_for_ply.append(([face[0], face[1], face[2]], 0, 0, 0))
    elv = PlyElement.describe(np.array(verts_for_ply, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]), 'vertex')
    elf = PlyElement.describe(
        np.array(faces_for_ply, dtype=[('vertex_indices', 'i4', (3,)), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]),
        'face')
    # Write object with own stream halndling to force flush before function exits
    pdata = PlyData([elv, elf], text=True)
    stream = open(plyfilename, 'wb')
    try:
        stream.write(pdata.header.encode('ascii'))
        stream.write(b'\n')
        for elt in pdata:
            elt._write(stream, pdata.text, pdata.byte_order)
    finally:
        stream.flush()
        stream.close()


"""
Creates 3D mesh from binary segmentation file suign Marching Cubes algorithm. 
This implementation uses skimage.measure .masching_cubes_lewiner function.

@param data: Binary mask data
@param threshold: Intensity threshold value for segmentation
@param resolution: Data resolution in mm [x,y,z]
@returns:
  verts: Mesh vertices
  faces: Mesh faces
  normals: Mesh normal vectors
  values: Maximum value of the datain local region
"""


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

    elv = PlyElement.describe(np.array(verts_for_ply, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]), 'vertex')
    elf = PlyElement.describe(
        np.array(faces_for_ply, dtype=[('vertex_indices', 'i4', (3,)), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]),
        'face')
    plyfilename = next(tempfile._get_candidate_names()) + '.ply'
    PlyData([elv, elf], text=True).write(plyfilename)

    return verts, faces, normals, values


"""
Return closest point to a line

@param a: Point s
@param b: Point b
@param p: Evaluated point
@returns: closest point in line a-b to point p
"""


def closest_point_on_line(a, b, p):
    ap = p - a
    ab = b - a
    result = a + np.dot(ap, ab) / np.dot(ab, ab) * ab
    return result


"""
Creates smoothed mesh for shape feature estimations

@param Idata: Intensity data
@param data: Data for mesh definition
@param threshold: Intensity threshold for mesh definition
@param resolution: data resolution in mm [x,y,z]
@param name: Name for debug use, in local temporary directory
@returns:
    verts: Mesh vertices
    faces: Mesh faces
"""


def create_mesh_smooth(Idata, data, threshold, resolution, name):
    print(resolution)
    verts, faces, normals, values = measure.marching_cubes(data, level=threshold, spacing=(resolution[0], resolution[1], resolution[2]))
    min_loc = [data.shape[0], data.shape[1], data.shape[2]]
    max_loc = [0, 0, 0]
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

    tempname = next(tempfile._get_candidate_names())
    plyfilename = 'temp_raw_' + tempname + '.ply'
    write_plyfile(verts, faces, Idata, resolution, plyfilename)
    plyfilenamep = 'temp_' + tempname + '.ply'

    # Smooth with meshlab
    Meshlab_smooth(plyfilename, plyfilenamep)
    plydata = PlyData.read(plyfilenamep)
    verts = []
    for vert in plydata['vertex'].data:
        verts.append([vert[0], vert[1], vert[2]])
    verts = np.asarray(verts)
    faces = []
    for face in plydata['face'].data['vertex_indices']:
        faces.append(np.asarray(face))
    faces = np.asarray(faces)
    if not os.path.isdir('./temp'):
        os.mkdir('./temp')
    write_plyfile(verts, faces, Idata, resolution, './temp/' + name + 'temp_' + tempname + '_preprocessed.ply')
    if os.path.exists(plyfilename):
        try:
            os.remove(plyfilename)
        except:
            pass
    if os.path.exists(plyfilenamep):
        try:
            os.remove(plyfilenamep)
        except:
            pass
    if os.path.exists('C:/temp/' + name + 'temp_' + tempname + '_preprocessed.ply'):
        try:
            os.remove('C:/temp/' + name + 'temp_' + tempname + '_preprocessed.ply')
        except:
            pass
    return verts, faces


"""
Bresenham's algorithm, subfunction for poitn set normalization.

@param slope: Points in N-D space
@returns: Normalized points
"""


def _bresenhamline_nslope(slope):
    scale = np.amax(np.abs(slope), axis=1).reshape(-1, 1)
    zeroslope = (scale == 0).all(1)
    scale[zeroslope] = np.ones(1)
    normalizedslope = np.array(slope, dtype=np.double) / scale
    normalizedslope[zeroslope] = np.zeros(slope[0].shape)
    return normalizedslope


"""
Bresenham's algorithm in 3-D space, subfunction.

@param start: Starting location
@param end: Ending location
@param max_iter: Maximum allowed iterations
@returns: Point of line in 3-D space
"""


def _bresenhamlines(start, end, max_iter):
    if max_iter == -1:
        max_iter = np.amax(np.amax(np.abs(end - start), axis=1))
    npts, dim = start.shape
    nslope = _bresenhamline_nslope(end - start)

    # steps to iterate on
    stepseq = np.arange(1, max_iter + 1)
    stepmat = np.tile(stepseq, (dim, 1)).T

    # some hacks for broadcasting properly
    bline = start[:, np.newaxis, :] + nslope[:, np.newaxis, :] * stepmat

    # Approximate to nearest int
    return np.array(np.rint(bline), dtype=start.dtype)


"""
Bresenham's algorithm in 3-D space.

@param start: Starting location
@param end: Ending location
@param max_iter: Maximum allowed iterations
@returns: Point of line in 3-D spac
"""


def bresenhamline(start, end, max_iter=5):
    # Return the points as a single array
    return _bresenhamlines(start, end, max_iter).reshape(-1, start.shape[-1])


"""
Creates smooth 3D mesh which is fitted to image gradients.

@param Idata: Intensity data
@param data: Data for mesh definition
@param threshold: Intensity threshold for mesh definition
@param resolution: data resolution in mm [x,y,z]
@param name: Name for debug use, in local temporary directory
@returns:
    verts: Mesh vertices
    faces: Mesh faces
"""


def create_mesh_smooth_fit_to_gradient(Idata, data, threshold, resolution, name):
    print(resolution)
    verts, faces, normals, values = measure.marching_cubes(data, level=threshold, spacing=(resolution[0], resolution[1], resolution[2]))
    if len(verts) == 0:
        return verts, faces

    min_loc = [data.shape[0], data.shape[1], data.shape[2]]
    max_loc = [0, 0, 0]
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

    tempname = next(tempfile._get_candidate_names())
    plyfilename = 'temp_raw_' + tempname + '.ply'
    write_plyfile(verts, faces, Idata, resolution, plyfilename)
    plyfilenamep = 'temp_' + tempname + '_preprocessed.ply'

    plyfilename = 'C:/temp/' + name + 'temp_raw_' + tempname + '.ply'
    write_plyfile(verts, faces, Idata, resolution, plyfilename)
    plyfilenamep = 'C:/temp/' + name + 'temp_' + tempname + '_preprocessed.ply'
    # Smooth with meshlab
    Meshlab_smooth(plyfilename, plyfilenamep)
    plydata = PlyData.read(plyfilenamep)
    verts = []
    for vert in plydata['vertex'].data:
        verts.append([vert[0], vert[1], vert[2]])
    verts = np.asarray(verts)
    faces = []
    for face in plydata['face'].data['vertex_indices']:
        faces.append(np.asarray(face))
    faces = np.asarray(faces)
    if len(verts) == 0:
        return verts, faces

    # Move vertices to closest gradine
    # print(Idata.shape)
    Gdata_x, Gdata_y, Gdata_z = np.gradient(Idata)
    # print((verts, faces))
    tm = trimesh.base.Trimesh(vertices=verts, faces=faces)
    vertex_normals = tm.vertex_normals
    resolution_max = np.max([resolution[0], resolution[1], resolution[2]])
    for v_i in range(len(verts)):
        v = verts[v_i]
        n = vertex_normals[v_i]
        nl = np.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2])
        n_outwards = np.multiply([n[0] / nl, n[1] / nl, n[2] / nl], resolution_max * 30)
        n_inwards = np.multiply([-n[0] / nl, -n[1] / nl, -n[2] / nl], resolution_max * 30)
        n_outwards /= resolution[0]
        n_outwards /= resolution[1]
        n_outwards /= resolution[2]
        n_inwards /= resolution[0]
        n_inwards /= resolution[1]
        n_inwards /= resolution[2]

        # resolve best gradient aling vertex normal
        # print((v, np.array([n_inwards]), np.array([n_outwards])))
        points = bresenhamline(np.array([n_inwards]), np.array([n_outwards]), max_iter=-1)
        # print(points)
        Gvalues = []
        for p_i in range(len(points)):
            # print((int(points[p_i][0]), int(points[p_i][1]), int(points[p_i][2])))
            vx = Gdata_x[int(points[p_i][0]), int(points[p_i][1]), int(points[p_i][2])]
            vy = Gdata_y[int(points[p_i][0]), int(points[p_i][1]), int(points[p_i][2])]
            vz = Gdata_z[int(points[p_i][0]), int(points[p_i][1]), int(points[p_i][2])]
            Gvalues.append(np.mean([abs(vx), abs(vy), abs(vz)]))
        Gvalues = [x / np.sum(Gvalues) for x in Gvalues]
        # print(Gvalues)
        wpoint = [0, 0, 0]
        for p_i in range(len(points)):
            wpoint[0] += points[p_i][0] * Gvalues[p_i] * resolution[0]
            wpoint[1] += points[p_i][1] * Gvalues[p_i] * resolution[1]
            wpoint[2] += points[p_i][2] * Gvalues[p_i] * resolution[2]
        # print((verts[v_i], wpoint, v+wpoint))
        verts[v_i][0] = (0.7 * verts[v_i][0] + 0.3 * (verts[v_i][0] + wpoint[0]))
        verts[v_i][1] = (0.7 * verts[v_i][1] + 0.3 * (verts[v_i][1] + wpoint[1]))
        verts[v_i][2] = (0.7 * verts[v_i][2] + 0.3 * (verts[v_i][2] + wpoint[2]))

    plyfilename = 'C:/temp/' + name + 'temp_' + tempname + '_preprocessed_opt.ply'
    write_plyfile(verts, faces, Idata, resolution, plyfilename)

    plyfilename = 'C:/temp/' + name + 'temp_' + tempname + '_preprocessed_opt.ply'
    plyfilenamep = 'C:/temp/' + name + 'temp_' + tempname + '.ply'
    Meshlab_smooth(plyfilename, plyfilenamep)
    plydata = PlyData.read(plyfilenamep)
    verts = []
    for vert in plydata['vertex'].data:
        verts.append([vert[0], vert[1], vert[2]])
    verts = np.asarray(verts)
    faces = []
    for face in plydata['face'].data['vertex_indices']:
        faces.append(np.asarray(face))
    faces = np.asarray(faces)
    plyfilename = 'C:/temp/' + name + 'temp_' + tempname + '_preprocessed_opt_sm.ply'
    write_plyfile(verts, faces, Idata, resolution, plyfilename)

    return verts, faces


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
'distance_mean', 'distance_median', 'CSM_mean_curvature', 'CSM_Gaus_mean_curvature', 'BGdistROI_median', 'BGdistROI_SD',
'BGdistROI_skewness', 'BGdistROI_kurtosis')

"""
Colletion of 3D shape features. 

@param LESIONDATAr: Intensity data
@param LESIONr: Lesion mask
@param BGr: Background region mask
@param resolution: data resolution in mm [x,y,z]
@returns: Please see casefun_3D_shape_names
"""


def casefun_3D_shape(LESIONDATAr, LESIONr, BGr, resolution):
    ROIdata = LESIONDATAr[LESIONr[0] > 0]
    BGdata = LESIONDATAr[BGr > 0]
    print((len(ROIdata), np.min(ROIdata), np.max(ROIdata)))
    print((len(BGdata), np.min(BGdata), np.max(BGdata)))

    # try:
    verts, faces, normals, values = create_mesh(LESIONr[0], 0.5, resolution)
    # except:
    #    print('failed to create')
    #    return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    sarea = skimage.measure.mesh_surface_area(verts, faces)
    # try:
    vertsw, facesw, normalsw, valuesw = create_mesh(BGr, 0.5, resolution)
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
    CSM_Gaus_mean_curvature = trimesh.base.curvature.discrete_gaussian_curvature_measure(tm, [CoM], np.max(distances))
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
Colletion of 3D shape features for background shape only

@param LESIONDATAr: Intensity data
@param LESIONr: Lesion mask
@param BGr: Background region mask
@param resolution: data resolution in mm [x,y,z]
@returns: Please see casefun_3D_shape_names_BG
"""


def casefun_3D_shape_BG(LESIONDATAr, LESIONr, BGr, resolution):
    BGdata = LESIONDATAr[BGr > 0]
    print((len(BGdata), np.min(BGdata), np.max(BGdata)))

    vertsw, facesw, normalsw, valuesw = create_mesh(BGr, 0.5, resolution)
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
    CSM_Gaus_mean_curvature = trimesh.base.curvature.discrete_gaussian_curvature_measure(tmw, [CoM], np.max(distances))
    return sareaw, np.median(tmw.area_faces), distance_mean, distance_median, CSM_mean_curvature[0], CSM_Gaus_mean_curvature[0]


"""
Colletion of 3D shape feature names for smoothed shape. 
Some features use Trimesh package (https://trimsh.org/trimesh.html).

    sarea3Dsm: 3D surface area, smoothed
    relsarea3Dsm: relative 3D surface area, smoothed
    tm_area_facessm: trimesh package faces area, smoothed
    tm_relarea_facessm: trimesh relative faces area, smoothed
    mean_anglessm: mean surface curvature angles, smoothed
    median_anglessm: medina surface curvature angles, smoothed
    SD_anglessm: standard deviation of suurface survature angles, smoothed
    distance_meansm: mean distance to the Center of Mass, smoothed
    distance_mediansm: median distance to the Center of Mass, smoothed
    CSM_mean_curvaturesm: Cohen-Steiner and Morvan mean curvature, smoothed
    CSM_Gaus_mean_curvaturesm: Gaussian curvature mean angle, smoothed
    BG_mediansm: Median distance to background surface, smoothed
    BG_SDsm: Standard deviation fo distances to background surface, smoothed
    BG_skewnesssm: Skewness of distances to background surface, smoothed
    BG_kurtosissm: Kurtosity of distances to background surface, smoothed
"""
casefun_3D_shape2_names = (
'sarea3Dsm', 'relsarea3Dsm', 'tm_area_facessm', 'tm_relarea_facessm', 'mean_anglessm', 'median_anglessm', 'SD_anglessm',
'distance_meansm', 'distance_mediansm', 'CSM_mean_curvaturesm', 'CSM_Gaus_mean_curvaturesm', 'BG_mediansm', 'BG_SDsm',
'BG_skewnesssm', 'BG_kurtosissm')

"""
Colletion of 3D shape features. 

@param LESIONDATAr: Intensity data
@param LESIONr: Lesion mask
@param BGr: Background region mask
@param resolution: data resolution in mm [x,y,z]
@returns: Please see casefun_3D_shape2_names
"""


def casefun_3D_shape2(LESIONDATAr, LESIONr, BGr, resolution):
    ROIdata = LESIONDATAr[LESIONr[0] > 0]

    # Whole Gland
    vertsw, facesw = create_mesh_smooth(LESIONDATAr, BGr, 0.5, resolution, 'BG')
    if len(vertsw) == 0:
        print('failed to create BG')
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    tmw = trimesh.base.Trimesh(vertices=vertsw, faces=facesw)

    # Lesion1, then other lesions, if found
    verts_all, faces_all = create_mesh_smooth(LESIONDATAr, LESIONr[0], 0.5, resolution, 'L1')
    if len(verts_all) == 0:
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    sarea = skimage.measure.mesh_surface_area(verts_all, faces_all)
    tm = trimesh.base.Trimesh(vertices=verts_all, faces=faces_all)
    angles_all = trimesh.base.curvature.face_angles_sparse(tm)
    angles_all = angles_all.data
    CoM = tm.center_mass
    distances_all = []
    for v in tm.vertices:
        distances_all.append(
            np.sqrt(np.power(v[0] - CoM[0], 2.0) + np.power(v[1] - CoM[1], 2.0) + np.power(v[2] - CoM[2], 2.0)))
    CSM_mean_curvature = trimesh.base.curvature.discrete_mean_curvature_measure(tm, [CoM], np.max(distances_all))
    CSM_Gaus_mean_curvature = trimesh.base.curvature.discrete_gaussian_curvature_measure(tm, [CoM], np.max(distances_all))
    closest, distancew_all, triangle_id = trimesh.base.proximity.closest_point(tm, tmw.vertices)
    for LESION_i in range(2, len(LESIONr)):
        try:
            verts, faces = create_mesh_smooth(LESIONDATAr, LESIONr[LESION_i], 1.0, resolution, 'L' + str(LESION_i - 1))
        except:
            continue
        if len(verts) == 0:
            continue
        print((verts_all.shape, verts.shape))
        verts_all = np.concatenate((verts_all, verts))
        faces_all = np.concatenate((faces_all, faces))
        sarea = sarea + skimage.measure.mesh_surface_area(verts, faces)
        tm = trimesh.base.Trimesh(vertices=verts, faces=faces)
        angles = trimesh.base.curvature.face_angles_sparse(tm)
        angles = angles.data
        angles_all = np.concatenate((angles_all, angles))
        CoM = tm.center_mass
        distances = []
        for v in tm.vertices:
            distances.append(
                np.sqrt(np.power(v[0] - CoM[0], 2.0) + np.power(v[1] - CoM[1], 2.0) + np.power(v[2] - CoM[2], 2.0)))
        CSM_mean_curvature = CSM_mean_curvature + trimesh.base.curvature.discrete_mean_curvature_measure(tm, [CoM], np.max(distances))
        CSM_Gaus_mean_curvature = CSM_Gaus_mean_curvature + trimesh.base.curvature.discrete_gaussian_curvature_measure(tm, [CoM], np.max(distances))
        distances_all = np.concatenate((distances_all, distances))
        closest, distancew, triangle_id = trimesh.base.proximity.closest_point(tm, tmw.vertices)
        distancew_all = np.concatenate((distancew_all, distancew))
    verts = verts_all
    faces = faces_all
    sarea = sarea / (len(LESIONr) - 1)
    angles = angles_all
    distances = distances_all
    CSM_Gaus_mean_curvature = CSM_Gaus_mean_curvature / (len(LESIONr) - 1)
    CSM_mean_curvature = CSM_mean_curvature / (len(LESIONr) - 1)
    distancew = distancew_all

    sareaw = skimage.measure.mesh_surface_area(vertsw, facesw)
    mean_angles = np.mean(angles)
    median_angles = np.median(angles)
    SD_angles = np.std(angles)
    distance_mean = np.mean(distances)
    distance_median = np.median(distances)

    # Distance to whole gland
    w1 = np.median(distancew)
    w2 = np.std(distancew)
    w3 = scipy.stats.skew(distancew)
    w4 = scipy.stats.kurtosis(distancew)

    return sarea, sarea / sareaw, np.median(tm.area_faces), np.median(tm.area_faces) / len(
        ROIdata), mean_angles, median_angles, SD_angles, distance_mean, distance_median, CSM_mean_curvature[0], CSM_Gaus_mean_curvature[0], w1, w2, w3, w4


"""
Colletion of 3D shape feature names for smoothed shape. 
Some features use Trimesh package (https://trimsh.org/trimesh.html).

    sarea3Dsm: 3D surface area, smoothed
    relsarea3Dsm: relative 3D surface area, smoothed
    tm_area_facessm: trimesh package faces area, smoothed
    tm_relarea_facessm: trimesh relative faces area, smoothed
    mean_anglessm: mean surface curvature angles, smoothed
    median_anglessm: medina surface curvature angles, smoothed
    SD_anglessm: standard deviation of suurface survature angles, smoothed
    distance_meansm: mean distance to the Center of Mass, smoothed
    distance_mediansm: median distance to the Center of Mass, smoothed
    CSM_mean_curvaturesm: Cohen-Steiner and Morvan mean curvature, smoothed
    CSM_Gaus_mean_curvaturesm: Gaussian curvature mean angle, smoothed
    BG_mediansm: Median distance to background surface, smoothed
    BG_SDsm: Standard deviation fo distances to background surface, smoothed
    BG_skewnesssm: Skewness of distances to background surface, smoothed
    BG_kurtosissm: Kurtosity of distances to background surface, smoothed
"""
casefun_3D_surface_textures_names = (
'surf_mean', 'surf_median', 'surf_25percentile', 'surf_75percentile', 'surf_skewness', 'surf_kurtosis', 'surf_SD',
'surf_range', 'surf_volume', 'surf_CV')

"""
Colletion of 3D shape features. 

@param LESIONDATAr: Intensity data
@param LESIONr: Lesion mask
@param BGr: Background region mask
@param resolution: data resolution in mm [x,y,z]
@returns: Please see casefun_3D_surface_textures_names
"""


def casefun_3D_surface_textures(LESIONDATAr, LESIONr, BGr, resolution):
    ROIdata = LESIONDATAr[LESIONr[0] > 0]
    print(len(ROIdata))
    # try:
    verts_all, faces_all = create_mesh_smooth(LESIONDATAr, LESIONr[0], 0.5, resolution, 'L1')
    # except:
    #    return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    if len(verts_all) == 0:
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    Surface_data = get_mesh_surface_samples(verts_all, faces_all, LESIONDATAr, resolution)
    for LESION_i in range(2, len(LESIONr)):
        # try:
        verts, faces = create_mesh_smooth(LESIONDATAr, LESIONr[LESION_i], 0.5, resolution, 'L' + str(LESION_i - 1))
        c = get_mesh_surface_samples(verts, faces, LESIONDATAr, resolution)
        # except:
        #    continue
        Surface_data = np.concatenate((Surface_data, c))

    mean = np.mean(Surface_data)
    median = np.median(Surface_data)
    p25 = np.percentile(Surface_data, 25)
    p75 = np.percentile(Surface_data, 75)
    skewness = scipy.stats.skew(Surface_data)
    kurtosis = scipy.stats.kurtosis(Surface_data)
    SD = np.std(Surface_data)
    rng = np.max(Surface_data) - np.min(Surface_data)
    volume = len(Surface_data)
    if not mean == 0:
        CV = SD / mean
    else:
        CV = 0.0
    return mean, median, p25, p75, skewness, kurtosis, SD, rng, volume, CV


"""
Names for 3D surface textures of background region

    BGsurf_mean: Surface mean intensity of background region
    BGsurf_median: Surface median intensity of background region
    BGsurf_25percentile: Surface intensity 25% intensity of background region
    BGsurf_75percentile: Surface intensity 75% percentile of background region
    BGsurf_skewness: Surface intensity skewness of background region
    BGsurf_kurtosis: Surface intensity kurtosis of background region
    BGsurf_SD: Surface intensity standard deviation of background region
    BGsurf_range: Surface intensity range of background region
    BGsurf_volume: Number of surface samples of background region
    BGsurf_CV: Surface intensity Coefficient of Variation of background region
"""
casefun_3D_surface_textures_names_BG = (
'BGsurf_mean', 'BGsurf_median', 'BGsurf_25percentile', 'BGsurf_75percentile', 'BGsurf_skewness', 'BGsurf_kurtosis',
'BGsurf_SD', 'BGsurf_range', 'BGsurf_volume', 'BGsurf_CV')

"""
3D surface textures of background region

@param LESIONDATAr: Intensity data
@param LESIONr: Lesion mask
@param BGr: Background region mask
@param resolution: data resolution in mm [x,y,z]
@returns: Please see casefun_3D_surface_textures_names_BG
"""


def casefun_3D_surface_textures_BG(LESIONDATAr, LESIONr, BGr, resolution):
    ROIdata = LESIONDATAr[BGr > 0]
    verts_all, faces_all = create_mesh_smooth(LESIONDATAr, BGr, 0.5, resolution, 'BG')
    if len(verts_all) == 0:
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    Surface_data = get_mesh_surface_samples(verts_all, faces_all, LESIONDATAr, resolution)
    for LESION_i in range(2, len(LESIONr)):
        verts, faces = create_mesh_smooth(LESIONDATAr, LESIONr[LESION_i], 0.5, resolution, 'L' + str(LESION_i - 1))
        c = get_mesh_surface_samples(verts, faces, LESIONDATAr, resolution)
        Surface_data = np.concatenate((Surface_data, c))
    Surface_data_v = []
    for v in Surface_data:
        if np.isnan(v):
            continue
        Surface_data_v.append(v)
    Surface_data = Surface_data_v
    mean = np.mean(Surface_data)
    median = np.median(Surface_data)
    p25 = np.percentile(Surface_data, 25)
    p75 = np.percentile(Surface_data, 75)
    skewness = scipy.stats.skew(np.ndarray(Surface_data))
    kurtosis = scipy.stats.kurtosis(Surface_data)
    SD = np.std(Surface_data)
    rng = np.max(Surface_data) - np.min(Surface_data)
    volume = len(Surface_data)
    if not mean == 0:
        CV = SD / mean
    else:
        CV = 0.0
    return mean, median, p25, p75, skewness, kurtosis, SD, rng, volume, CV


"""
Otsu's thersholding of intensity distributions

@param gray: Distribution of intensity values
@returns: Otsu's threshold value
"""


def otsu(gray):
    pixel_number = len(gray)
    if pixel_number == 0:
        return 0
    mean_weigth = 1.0 / pixel_number
    his, bins = np.histogram(gray, bins=25)
    final_thresh = -1
    final_value = -1
    intensity_arr = np.arange(25)
    for t in range(1, 25):
        pcb = np.sum(his[:t])
        pcf = np.sum(his[t:])
        Wb = pcb * mean_weigth
        Wf = pcf * mean_weigth
        mub = np.sum(intensity_arr[:t] * his[:t]) / float(pcb)
        muf = np.sum(intensity_arr[t:] * his[t:]) / float(pcf)
        value = Wb * Wf * (mub - muf) ** 2
        if value > final_value:
            final_thresh = t
            final_value = value
    return bins[final_thresh]


"""
3D sliding window help function

@param image: Intensity image
@param mask: Masking window
@param stepSize: step size in voxels
@param windowSize: window size in voxels
@returns: yield for sliding window processing
"""


def sliding_window3D(image, mask, stepSize, windowSize):
    # slide a window across the image
    for z in range(0, image.shape[2], stepSize):
        if np.max(mask[:, :, z]) == 0:
            continue
        for y in range(0, image.shape[1], stepSize):
            if np.max(mask[:, y, z]) == 0:
                continue
            for x in range(0, image.shape[0], stepSize):
                # yield the current window
                yield x, y, z, image[x:x + windowSize[0], y:y + windowSize[1], z:z + windowSize[2]]
        print(('%d/%d Laws' % (z, image.shape[2])))


"""
3D analogue of laws texture features.

Suzuki, M.T. and Yaginuma, Y., 2007, January.
A solid texture analysis based on three-dimensional convolution kernels. In Videometrics IX (Vol. 6491, p. 64910W). International Society for Optics and Photonics.
Suzuki, M.T., Yaginuma, Y., Yamada, T. and Shimizu, Y., 2006, December. A shape feature extraction method based on 3D convolution masks.
In Multimedia, 2006. ISM'06. Eighth IEEE International Symposium on (pp. 837-844). IEEE.

@param filename: txt file to be used for kernel definitions
@returns: 3D kernel objects
"""


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

"""
Names for 3D Laws features. Contains basic first order statistics to be takesn for each feature map.
"""
casefun_3D_Laws_names = ['mean_ROI', 'median_ROI', 'SD_ROI', 'IQR_ROI', 'skewnessROI', 'kurtosisROI', 'p25ROI', 'p75ROI', 'rel']

# Initialize array formats for use in numba
laws3D_names = []
length_Laws3Dkernel = len(Laws3Dkernel)
Laws3Dkernel_array = np.zeros([length_Laws3Dkernel, 5, 5, 5])
for kernel_i in range(0, length_Laws3Dkernel):
    Laws3Dkernel_array[kernel_i, :, :, :] = Laws3Dkernel[kernel_i]['data']
for kernel_i in range(1, length_Laws3Dkernel):
    laws3D_names.append(Laws3Dkernel[kernel_i]['name'])

"""
Names generator function for 3D Laws

@param params: Feature set parameters
@returns: Feature names
"""


def casefun_3D_Laws_names_generator(params):
    names = []
    for laws3D_names_i in range(len(laws3D_names)):
        for name in casefun_3D_Laws_names:
            names.append('UTU3DLaws%s_%s_f%2.1f' % (laws3D_names[laws3D_names_i], name, params[0]))
    return names


"""
Names generator function for 3D Laws in background region

@param params: Feature set parameters
@returns: Feature names
"""


def casefun_3D_Laws_names_generator_BG(params):
    names = []
    for laws3D_names_i in range(len(laws3D_names)):
        for name_i in range(len(casefun_3D_Laws_names) - 1):
            name = casefun_3D_Laws_names[name_i]
            names.append('BGUTU3DLaws%s_%s_f%2.1f' % (laws3D_names[laws3D_names_i], name, params[0]))
    return names


"""
Appends Laws feature values to the end of exiting list

@param outdata: Intensity data
@param LESIONr: Lesion mask
@param BGr: Background region mask
@param ret: List of results
@returns: List of results, with new data appended
"""


def append_Laws_results(outdata, LESIONrs, BGrs, ret):
    ROIdata = outdata[LESIONrs > 0]
    BGdata = outdata[BGrs > 0]
    if len(ROIdata) == 0:
        for x in casefun_3D_Laws_names:
            ret.append(float('nan'))
        return ret
    mean1 = np.mean(ROIdata)
    ret.append(np.mean(ROIdata))
    ret.append(np.median(ROIdata))
    std1 = np.std(ROIdata)
    ret.append(std1)
    ret.append(iqr(ROIdata))
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
    ret.append(iqr(ROIdata))
    ret.append(scipy.stats.skew(ROIdata))
    ret.append(scipy.stats.kurtosis(ROIdata))
    ret.append(np.percentile(ROIdata, 25))
    ret.append(np.percentile(ROIdata, 75))
    return ret


if numba_found:
    """
    Vectorized 3D cross-colleration for 3D Laws feature extraction. 
    This implementation is useing only 5x5x5 window for now. 
    
    @param img1: Mask window
    @param img2: Lesion mask
    @param res: Results image
    """


    @numba.guvectorize(['(float64[:,:,:], float64[:,:,:], float64[:,:,:])'], '(n,n,n),(n,n,n)->(n,n,n)', nopython=True)
    def numba_vectorized_correlate(img1, img2, ret):
        m1 = 0
        m2 = 0
        s = 2
        N = 5
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    m1 += img1[x, y, z]
                    m2 += img2[x, y, z]
                    ret[x, y, z] = 0
        m1 /= 5 * 5 * 5
        m2 /= 5 * 5 * 5
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    for ii in range(-s, s):
                        for jj in range(-s, s):
                            for kk in range(-s, s):
                                ret[i, j, k] = (img2[ii + s, jj + s, kk + s] - m2) * (img1[i + ii, j + jj, k + kk] - m1)


    """
    3D Laws with numba GPU acceleration
    
    @param LESIONDATAr: Intensity data
    @param LESIONr: Lesion mask
    @param BGr: Background region mask
    @param resolution: data resolution in mm [x,y,z]
    @param params: param[0]: isotropic resampling in mm
    @returns:
        Please see corresponding result name list generation function
    """


    def casefun_3D_Laws_numba(LESIONDATAr, LESIONr, BGr, resolution, params):
        # resolution factor affecting laws feature sampling ratio
        # 1: original resolution
        # <1: upsampling
        # >1: downsampling
        res_f = params[0]

        x_lo, x_hi, y_lo, y_hi, z_lo, z_hi = find_bounded_subregion3D(BGr)
        LESIONDATArs = LESIONDATAr[x_lo:x_hi, y_lo:y_hi, :]
        LESIONrs_temp = LESIONr[0][x_lo:x_hi, y_lo:y_hi, :]
        BGrs_temp = BGr[x_lo:x_hi, y_lo:y_hi, :]

        # Create masks and output data to desired resolution, intensity data is resliced later for non-zero only
        min_res = np.max(resolution)
        new_res = [min_res * res_f, min_res * res_f, min_res * res_f]
        LESIONrs_temp, affineLESIONrs_temp = reslice_array(LESIONrs_temp, resolution, new_res, 0)
        BGrs_temp, affineBGrs_temp = reslice_array(BGrs_temp, resolution, new_res, 0)
        LESIONDATArs, affineLESIONDATArs = reslice_array(LESIONDATArs, resolution, new_res, 1)

        outdatas = []
        for kernel_i in range(1, len(Laws3Dkernel)):
            outdatas.append(np.zeros_like(LESIONDATArs))

        s = 5
        sys.stderr = io.StringIO()
        ret = np.zeros((5, 5, 5))
        for (x, y, z, window) in sliding_window3D(LESIONDATArs, LESIONrs_temp, 1, (s, s, s)):
            window = np.subtract(window, np.mean(window))
            w_std = np.std(window)
            if w_std > 0:
                window = np.divide(window, np.std(window))
            xmid = x + 2
            ymid = y + 2
            zmid = z + 2
            if window.shape[0] < s:
                continue
            if window.shape[1] < s:
                continue
            if window.shape[2] < s:
                continue
            for kernel_i in range(1, len(Laws3Dkernel)):
                c = numba_vectorized_correlate(window, Laws3Dkernel[kernel_i]['data'], ret)
                outdatas[kernel_i - 1][xmid, ymid, zmid] = np.sum(c)
        sys.stderr = sys.__stderr__
        ret = []
        for kernel_i in range(1, len(Laws3Dkernel)):
            ret = append_Laws_results(outdatas[kernel_i - 1], LESIONrs_temp, BGrs_temp, ret)
        return ret


    """
    3D Laws with numba GPU acceleration using background region
    
    @param LESIONDATAr: Intensity data
    @param LESIONr: not in use
    @param BGr: Background region mask
    @param resolution: data resolution in mm [x,y,z]
    @param params: param[0]: isotropic resampling in mm
    @returns: Please see corresponding result name list generation function
    """


    def casefun_3D_Laws_BG_numba(LESIONDATAr, LESIONr, BGr, resolution, params):
        # resolution factor affecting laws feature sampling ratio
        # 1: original resolution
        # <1: upsampling
        # >1: downsampling
        res_f = params[0]

        x_lo, x_hi, y_lo, y_hi, z_lo, z_hi = find_bounded_subregion3D(LESIONDATAr)
        x_lo -= 2
        x_hi += 2
        y_lo -= 2
        y_hi += 2
        if x_lo < 0:
            x_lo = 0
        if y_lo < 0:
            y_lo = 0
        if x_hi >= BGr.shape[0]:
            x_hi = BGr.shape[0] - 1
        if y_hi >= BGr.shape[1]:
            y_hi = BGr.shape[1] - 1

        LESIONDATArs = LESIONDATAr[x_lo:x_hi, y_lo:y_hi, :]
        BGrs_temp = BGr[x_lo:x_hi, y_lo:y_hi, :]

        # Create masks and output data to desired resolution, intensity data is resliced later for non-zero only
        min_res = np.max(resolution)
        new_res = [min_res * res_f, min_res * res_f, min_res * res_f]
        BGrs_temp, affineBGrs_temp = reslice_array(BGrs_temp, resolution, new_res, 0)
        LESIONDATArs, affineLESIONDATArs = reslice_array(LESIONDATArs, resolution, new_res, 1)

        outdatas = []
        for kernel_i in range(1, len(Laws3Dkernel)):
            outdatas.append(np.zeros_like(LESIONDATArs))

        s = 5
        sys.stderr = io.StringIO()
        ret = np.zeros((5, 5, 5))
        for (x, y, z, window) in sliding_window3D(LESIONDATArs, BGrs_temp, 1, (s, s, s)):
            window = np.subtract(window, np.mean(window))
            w_std = np.std(window)
            if w_std > 0:
                window = np.divide(window, np.std(window))
            xmid = x + 2
            ymid = y + 2
            zmid = z + 2
            if window.shape[0] < s:
                continue
            if window.shape[1] < s:
                continue
            if window.shape[2] < s:
                continue
            for kernel_i in range(1, len(Laws3Dkernel)):
                c = numba_vectorized_correlate(window, Laws3Dkernel[kernel_i]['data'], ret)
                outdatas[kernel_i - 1][xmid, ymid, zmid] = np.sum(c)
        sys.stderr = sys.__stderr__
        ret = []
        for kernel_i in range(1, len(Laws3Dkernel)):
            ret = append_Laws_results_BG(outdatas[kernel_i - 1], BGrs_temp, ret)
        return ret


"""
3D Laws

@param LESIONDATAr: Intensity data
@param LESIONr: Lesion mask
@param BGr: Background region mask
@param resolution: params[0]: isotropic data resampling in mm [x,y,z]
@returns: Please see corresponding featue list variable
"""


def casefun_3D_Laws(LESIONDATAr, LESIONr, BGr, resolution, params):
    # resolution factor affecting laws feature sampling ratio
    # 1: original resolution
    # <1: upsampling
    # >1: downsampling
    res_f = params[0]

    x_lo, x_hi, y_lo, y_hi, z_lo, z_hi = find_bounded_subregion3D(BGr)
    LESIONDATArs = LESIONDATAr[x_lo:x_hi, y_lo:y_hi, :]
    LESIONrs_temp = LESIONr[0][x_lo:x_hi, y_lo:y_hi, :]
    BGrs_temp = BGr[x_lo:x_hi, y_lo:y_hi, :]

    # Create masks and output data to desired resolution, intensity data is resliced later for non-zero only
    min_res = np.max(resolution)
    new_res = [min_res * res_f, min_res * res_f, min_res * res_f]
    LESIONrs_temp, affineLESIONrs_temp = reslice_array(LESIONrs_temp, resolution, new_res, 0)
    BGrs_temp, affineBGrs_temp = reslice_array(BGrs_temp, resolution, new_res, 0)
    LESIONDATArs, affineLESIONDATArs = reslice_array(LESIONDATArs, resolution, new_res, 1)

    outdatas = []
    for kernel_i in range(1, len(Laws3Dkernel)):
        outdatas.append(np.zeros_like(LESIONDATArs))

    s = 5
    sys.stderr = io.StringIO()
    for (x, y, z, window) in sliding_window3D(LESIONDATArs, LESIONrs_temp, 1, (s, s, s)):
        window = np.subtract(window, np.mean(window))
        w_std = np.std(window)
        if w_std > 0:
            window = np.divide(window, np.std(window))
        xmid = 2 + x
        ymid = 2 + y
        zmid = 2 + z
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
        ret = append_Laws_results(outdatas[kernel_i - 1], LESIONrs_temp, BGrs_temp, ret)
    return ret


"""
3D Laws for backrgound region

@param LESIONDATAr: Intensity data
@param LESIONr: Lesion mask
@param BGr: Background region mask
@param resolution: params[0]: isotropic data resampling in mm [x,y,z]
@returns: Please see corresponding featue list variable
"""


def casefun_3D_Laws_BG(LESIONDATAr, LESIONr, BGr, resolution, params):
    # resolution factor affecting laws feature sampling ratio
    # 1: original resolution
    # <1: upsampling
    # >1: downsampling
    res_f = params[0]

    s = 5
    x_lo, x_hi, y_lo, y_hi, z_lo, z_hi = find_bounded_subregion3D(LESIONDATAr)
    x_lo -= 2
    x_hi += 2
    y_lo -= 2
    y_hi += 2
    if x_lo < 0:
        x_lo = 0
    if y_lo < 0:
        y_lo = 0
    if x_hi >= BGr.shape[0]:
        x_hi = BGr.shape[0] - 1
    if y_hi >= BGr.shape[1]:
        y_hi = BGr.shape[1] - 1

    LESIONDATArs = LESIONDATAr[x_lo:x_hi, y_lo:y_hi, :]
    BGrs_temp = BGr[x_lo:x_hi, y_lo:y_hi, :]

    # Create masks and output data to desired resolution, intensity data is resliced later for non-zero only
    min_res = np.max(resolution)
    new_res = [min_res * res_f, min_res * res_f, min_res * res_f]
    BGrs_temp, affineBGrs_temp = reslice_array(BGrs_temp, resolution, new_res, 0)
    LESIONDATArs, affineLESIONDATArs = reslice_array(LESIONDATArs, resolution, new_res, 1)

    outdatas = []
    for kernel_i in range(1, len(Laws3Dkernel)):
        outdatas.append(np.zeros_like(LESIONDATArs))

    sys.stderr = io.StringIO()
    for (x, y, z, window) in sliding_window3D(LESIONDATArs, BGrs_temp, 1, (s, s, s)):
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
        ret = append_Laws_results_BG(outdatas[kernel_i - 1], BGrs_temp, ret)
    return ret


"""
First order statistics for SNR measurements
"""
casefun_SNR_names = ('mean', 'median', '25percentile', '75percentile', 'skewness', 'kurtosis', 'SD', 'IQR')

"""
Names generator for signal-to-noise ration measurements features

@param LESIONDATAr: Intensity data
@param LESIONr: Lesion mask
@param resolution: data resolution in mm [x,y,z]
@returns: List of SNR feature names
"""


def casefun_SNR_name_generator(params):
    names = []
    for Lname in ['', 'BG', 'rel']:
        for name in casefun_SNR_names:
            names.append('UTUSNRMoments_%s_%s' % (Lname, name))
    names.append('UTUSNRMoments_CNR')
    names.append('UTUSNRMoments_npCNR')
    return names


"""
Helper function to calculate relative feature values, robust for zeros. 
"""


def relvalue(val1, val2):
    if val1 == 0 and val2 == 0:
        return 1
    else:
        return abs(val1) / ((val1 + val2) / 2.0)


"""
Signal to noise ratio measurement features.

@param LESIONDATAr: Intensity data
@param LESIONr: Lesion mask
@param BGr: Background region mask
@param resolution: data resolution in mm [x,y,z]
@returns: Please see corresponding names variable
"""


def casefun_SNR(LESIONDATAr, LESIONr, BGr, resolution, params):
    ROIdata = LESIONDATAr[LESIONr[0] > 0]
    BGdata = LESIONDATAr[BGr > 0]

    mean = np.mean(ROIdata)
    median = np.median(ROIdata)
    p25 = np.percentile(ROIdata, 25)
    p75 = np.percentile(ROIdata, 75)
    skewness = scipy.stats.skew(ROIdata)
    kurtosis = scipy.stats.kurtosis(ROIdata)
    SD = np.std(ROIdata)
    IQrange = iqr(ROIdata)

    wmean = np.mean(BGdata)
    wmedian = np.median(BGdata)
    wp25 = np.percentile(BGdata, 25)
    wp75 = np.percentile(BGdata, 75)
    wskewness = scipy.stats.skew(BGdata)
    wkurtosis = scipy.stats.kurtosis(BGdata)
    wSD = np.std(BGdata)
    wIQrange = iqr(BGdata)

    results = []
    results.append(mean)
    results.append(median)
    results.append(p25)
    results.append(p75)
    results.append(skewness)
    results.append(kurtosis)
    results.append(SD)
    results.append(IQrange)
    results.append(wmean)
    results.append(wmedian)
    results.append(wp25)
    results.append(wp75)
    results.append(wskewness)
    results.append(wkurtosis)
    results.append(wSD)
    results.append(wIQrange)

    results.append(relvalue(mean, wmean))
    results.append(relvalue(median, wmedian))
    results.append(relvalue(p25, wp25))
    results.append(relvalue(p75, wp75))
    results.append(relvalue(skewness, wskewness))
    results.append(relvalue(kurtosis, wkurtosis))
    results.append(relvalue(SD, wSD))
    results.append(relvalue(IQrange, wIQrange))

    CNR = abs(mean - wmean) / ((SD + wSD) / 2)
    npCNR = abs(median - wmedian) / ((IQrange + wIQrange) / 2)
    results.append(CNR)
    results.append(npCNR)
    print(results)
    return results


"""
Experimental method where thresholded region afte kernel density estimation 
is considered

@param LESIONDATAr: Intensity data
@param LESIONr: Lesion mask
@param BGr: background mask
@param resolution: data resolution in mm [x,y,z]
@param amount: coefficient for thresholding of KDE distribution
@returns:
  ROIdata: ROI data
  BGdata: background data
"""


def Moment2_fun_KDE(LESIONDATAr, LESIONr, BGr, resolution, amount):
    KDEregion_all = np.zeros_like(LESIONr[0])
    for z in range(LESIONr[0].shape[2]):
        eroded = LESIONr[0][:, :, z]
        if len(eroded[eroded > 0]) == 0:
            continue
        KDEregion = np.zeros_like(eroded)
        centroid_x = []
        centroid_y = []
        centroid_z = []
        ROIdata_z = LESIONDATAr[:, :, z][eroded > 0]
        ROIdata_zmax = np.max(ROIdata_z)
        ROIdata_zmin = np.min(ROIdata_z)
        for x in range(eroded.shape[0]):
            for y in range(eroded.shape[1]):
                if eroded[x, y] > 0:
                    KDEregion_all[x, y, z] = 1
                    val = 1 - LESIONDATAr[x, y, z] / ROIdata_zmax
                    for v in range(int(val * 1000)):
                        centroid_x.append(x)
                        centroid_y.append(y)
        vs = np.vstack([centroid_x, centroid_y])
        if len(np.unique(vs[0])) < 2:
            continue
        if len(np.unique(vs[1])) < 2:
            continue
        kernel = stats.gaussian_kde(vs)
        centroid_x = []
        centroid_y = []
        for x in range(eroded.shape[0]):
            for y in range(eroded.shape[1]):
                if eroded[x, y] > 0:
                    centroid_x.append(x)
                    centroid_y.append(y)
        centroid_xyz = np.vstack([centroid_x, centroid_y])
        values = kernel.evaluate(centroid_xyz)
        th = np.max(values) * amount
        for c_i in range(len(centroid_xyz[0])):
            if values[c_i] > th:
                KDEregion[centroid_xyz[0][c_i], centroid_xyz[1][c_i]] = 1
        KDEregion_all[:, :, z] = KDEregion
    ROIdata = LESIONDATAr[KDEregion_all > 0]
    BGr[KDEregion_all > 0] = 0
    BGdata = LESIONDATAr[BGr > 0]
    return ROIdata, BGdata


"""
Statistical descriptors for background region

@param LESIONDATAr: Intensity data
@param LESIONr: Lesion mask
@param BGr: Background region mask
@param resolution: data resolution in mm [x,y,z]
@returns: Please see casefun_01_moments2_name_generator, 
"""


def casefun_01_Moments2(LESIONDATAr, LESIONr, BGr, resolution, params):
    # eroded, u1, u2 = operations3D.fun_voxelwise_erodeLesion_voxels_3Dmesh(LESIONr[0], BGr, LESIONDATAr, resolution, amount)
    # LESIONr[0][BGr == 0] = 0
    # resolve slice with largest volume
    if params[1] == 'largest_slice':
        ROIdata, BGdata = Moment2_fun_largest_slice(LESIONDATAr, LESIONr, BGr, resolution, params[0])
    elif params[1] == 'largest_sliceKDE':
        ROIdata, BGdata = Moment2_fun_largest_sliceKDE(LESIONDATAr, LESIONr, BGr, resolution, params[0])
    elif params[1] == 'Moment2_fun_KDE':
        ROIdata, BGdata = Moment2_fun_KDE(LESIONDATAr, LESIONr, BGr, resolution, params[0])
    else:
        raise Exception('Moments 2 measurement method was not found:' + params[1])

    mean = np.mean(ROIdata)
    median = np.median(ROIdata)
    print(ROIdata)
    print(params[1])
    p25 = np.percentile(ROIdata, 25)
    p75 = np.percentile(ROIdata, 75)
    skewness = scipy.stats.skew(ROIdata)
    kurtosis = scipy.stats.kurtosis(ROIdata)
    SD = np.std(ROIdata)
    IQR = iqr(ROIdata)
    wmean = np.mean(BGdata)
    wmedian = np.median(BGdata)
    if len(BGdata) == 0:
        BGdata = [0]
    wp25 = np.percentile(BGdata, 25)
    print(BGdata)
    wp75 = np.percentile(BGdata, 75)
    wskewness = scipy.stats.skew(BGdata)
    wkurtosis = scipy.stats.kurtosis(BGdata)
    wSD = np.std(BGdata)
    wIQrange = iqr(BGdata)
    if mean == 0:
        relBGmean = 0
    else:
        relBGmean = mean / (mean + np.mean(BGdata))
    if median == 0:
        relBGmedian = 0
    else:
        relBGmedian = median / (median + np.median(BGdata))
    if p25 == 0:
        relBG25percentile = 0
    else:
        relBG25percentile = p25 / (p25 + np.percentile(BGdata, 25))
    if p75 == 0:
        relBG75percentile = 0
    else:
        relBG75percentile = p75 / (p75 + np.percentile(BGdata, 75))
    if skewness == 0:
        relBGskewness = 0
    else:
        relBGskewness = skewness / (skewness + scipy.stats.skew(BGdata))
    if kurtosis == 0:
        relBGkurtosis = 0
    else:
        relBGkurtosis = kurtosis / (kurtosis + scipy.stats.kurtosis(BGdata))
    if SD == 0:
        relBGSD = 0
    else:
        relBGSD = SD / (SD + np.std(BGdata))
    if IQR == 0:
        BGIQR = 0
    else:
        BGIQR = IQR / (IQR + iqr(BGdata))

    return mean, median, p25, p75, skewness, kurtosis, SD, IQR, relBGmean, relBGmedian, relBG25percentile, relBG75percentile, relBGskewness, relBGkurtosis, relBGSD, BGIQR
