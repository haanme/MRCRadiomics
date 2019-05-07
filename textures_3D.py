#!/usr/bin/env python

import os
import cv2
import tempfile
import numpy as np
import nibabel as nib
from glob import glob
import sklearn.metrics
import DicomIO_G as dcm
from dipy.align.reslice import reslice
from sklearn.metrics import confusion_matrix
from scipy.signal import correlate
import csv
from GleasonScore import GS
from scipy import ndimage
from scipy.stats import iqr
import scipy.ndimage
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pymesh
import skimage
import trimesh
import sys
import os
import subprocess
from plyfile import PlyData, PlyElement
import cv2
from skimage import measure
from skimage.segmentation import active_contour
from scipy.interpolate import UnivariateSpline
import numpy as np
from scipy import ndimage
from skimage.morphology import convex_hull_image
from skimage import feature
from scipy import ndimage
import step6_calculate_AUCs_utilities as step6utils
import operations3D
import copy
from scipy import stats

# Directory where result data are located
experiment_dir = ''
stat_funs = []
    

# Resolve non-zero subregion in the image
def find_bounded_subregion3D(img):
    x_lo = 0
    for x in range(img.shape[0]):
        if np.max(img[x, :, :]) > 0:
           x_lo = x
           break
    x_hi = 0
    for x in range(img.shape[0]-1,-1,-1):
        if np.max(img[x, :, :]) > 0:
           x_hi = x
           break
    y_lo = 0
    for y in range(img.shape[1]):
        if np.max(img[:, y, :]) > 0:
           y_lo = y
           break
    y_hi = 0
    for y in range(img.shape[1]-1,-1,-1):
        if np.max(img[:, y, :]) > 0:
           y_hi = y
           break
    z_lo = 0
    for z in range(img.shape[2]):
        if np.max(img[:, :, z]) > 0:
           z_lo = z
           break
    z_hi = 0
    for z in range(img.shape[2]-1,-1,-1):
        if np.max(img[:, :, z]) > 0:
           z_hi = z
           break
    return x_lo, x_hi, y_lo, y_hi, z_lo, z_hi


def reslice_array(data, orig_resolution, new_resolution, int_order):
    zooms = orig_resolution
    new_zooms = (new_resolution[0], new_resolution[1], new_resolution[2])
    affine = np.eye(4)
    affine[0,0] = orig_resolution[0]
    affine[1,1] = orig_resolution[1]
    affine[2,2] = orig_resolution[2]
    print(data.shape)
    print(zooms)
    print(new_zooms)
    data2, affine2 = reslice(data, affine, zooms, new_zooms, order=int_order)
    data3 = np.zeros((data2.shape[1], data2.shape[0], data2.shape[2]))
    for zi in range(data3.shape[2]):
        data3[:, :, zi] = np.rot90(data2[:, :, zi], k=3)
    return data3, affine2


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
    curvature = (xd* ydd - yd* xdd) / np.power(xd** 2 + yd** 2, 3 / 2)
    return curvature


casefun_levelset_names = ('Levelset_median', 'Levelset_std')
def casefun_levelset(LESIONDATAr, LESIONr, WGr, resolution):
    ROIdata = LESIONDATAr[LESIONr[0] > 0]

    #print(LESIONr[0].shape)
    Curvatures2 = []
    for z in range(LESIONr[0].shape[2]):
        rimg = np.squeeze(LESIONr[0][:, :, z])
        img = np.squeeze(LESIONDATAr[:, :, z])
        img[rimg == 0] = 0
        if np.max(rimg) == 0:
            continue
        no_values = len(rimg[rimg > 0])
        #print('positive values:' + str(no_values))
        if no_values < 5:
           continue
        init = np.where(rimg > 0)
        CoM = (np.mean(init[0]), np.mean(init[1]))
        Dists = []
        for init_i in range(len(init[0])):
            coord = (init[0][init_i],init[1][init_i])
            Dists.append(np.sqrt(np.power(coord[0]-CoM[0], 2)+np.power(coord[1]-CoM[1], 2)))
        R = np.max(Dists)
        init = np.squeeze(np.array(init)).T
        s = np.linspace(0, 2*np.pi, 100)
        x = CoM[0] + R*np.cos(s)
        y = CoM[1] + R*np.sin(s)
        init = np.array([x, y]).T

        snake = active_contour(img, init, alpha=0.015, beta=20, gamma=0.001)
        c = curvature_splines(snake[:, 0], snake[:, 1])
        for v in c:
           Curvatures2.append(v)

    return np.median(Curvatures2), np.std(Curvatures2)
  

def get_3D_GLCM(LESIONDATAr, LESIONr, resolution):
    Ng = []

#    Ng.append((-1, -1, -1))
#    Ng.append((-1, -1,  0))
#    Ng.append((-1, -1,  1))
#    Ng.append((-1,  0, -1))
#    Ng.append((-1,  0,  0))
#    Ng.append((-1,  0,  1))
#    Ng.append((-1,  1, -1))
#    Ng.append((-1,  1,  0))
#    Ng.append((-1,  1,  1))
#    Ng.append(( 0, -1, -1))
#    Ng.append(( 0, -1,  0))
#    Ng.append(( 0, -1,  1))
#    Ng.append(( 0,  0, -1))
    Ng.append(( 0,  0,  1))
#    Ng.append(( 0,  1, -1))
    Ng.append(( 0,  1,  0))
    Ng.append(( 0,  1,  1))
#    Ng.append(( 1, -1, -1))
#    Ng.append(( 1, -1,  0))
#    Ng.append(( 1, -1,  1))
#    Ng.append(( 1,  0, -1))
    Ng.append(( 1,  0,  0))
    Ng.append(( 1,  0,  1))
#    Ng.append(( 1,  1, -1))
    Ng.append(( 1,  1,  0))
    Ng.append(( 1,  1,  1))

    c_all = LESIONDATAr[LESIONr > 0]
    c_min = np.min(c_all)
    c_all = np.subtract(LESIONDATAr, c_min)
    c_max = np.max(c_all)
    c_all = np.divide(c_all, c_max)
    c_all = np.round(np.multiply(c_all, 127))
    #print((np.min(c_all), np.mean(c_all), np.max(c_all)))
    #print(c_all.shape)
    G = np.zeros([128, 128, 1, len(Ng)])
    # Calculate Haralick across surface
    comparisons = 0
    ROI_coords = np.nonzero(LESIONr > 0)
    ROI_coord = [0,0,0]
    for ROI_coord_i in range(len(ROI_coords)):
        ROI_coord[0] = ROI_coords[0][ROI_coord_i]
        ROI_coord[1] = ROI_coords[1][ROI_coord_i]
        ROI_coord[2] = ROI_coords[2][ROI_coord_i]
        # Divide into
        for Ng_coord_i in range(len(Ng)):
            Ng_coord = Ng[Ng_coord_i]
            v = [ROI_coord[0]+Ng_coord[0], ROI_coord[1]+Ng_coord[1], ROI_coord[2]+Ng_coord[2]]
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
#    print(('contrast', skimage.feature.texture.greycoprops(P, prop='contrast')))
    contrast = np.mean(skimage.feature.texture.greycoprops(P, prop='contrast'))
#    print(('dissimilarity', skimage.feature.texture.greycoprops(P, prop='dissimilarity')))
    dissimilarity = np.mean(skimage.feature.texture.greycoprops(P, prop='dissimilarity'))
#    print(('homogeneity', skimage.feature.texture.greycoprops(P, prop='homogeneity')))
    homogeneity = np.mean(skimage.feature.texture.greycoprops(P, prop='homogeneity'))
    ASM = np.mean(skimage.feature.texture.greycoprops(P, prop='ASM'))
    energy = np.mean(skimage.feature.texture.greycoprops(P, prop='energy'))
#    print(('correlation', skimage.feature.texture.greycoprops(P, prop='correlation')))
    correlation = np.mean(skimage.feature.texture.greycoprops(P, prop='correlation'))
    return contrast, dissimilarity, homogeneity, ASM, energy, correlation


def get_mesh_surface_features(tmw, verts, faces, Idata, resolution):
    CoM = tmw.center_mass
    c_all = []
    polar_coordinates = []
    for v_i in range(len(verts)):
        vert = verts[v_i]
        avg_loc_x = int(round(vert[0]/resolution[0]))
        avg_loc_y = int(round(vert[1]/resolution[1]))
        avg_loc_z = int(round(vert[2]/resolution[2]))
        if avg_loc_x < 0 or avg_loc_y < 0 or avg_loc_z < 0:
           c = float('NaN')
        elif avg_loc_x >= Idata.shape[0] or avg_loc_y >= Idata.shape[1] or avg_loc_z >= Idata.shape[2]:
           c = float('NaN')
        else:
           c = Idata[avg_loc_x, avg_loc_y, avg_loc_z]
        c_all.append(c)
        x = (CoM[0]-vert[0])
        y = (CoM[1]-vert[1])
        z = (CoM[2]-vert[2])
        r = np.sqrt(np.power(x, 2.0) + np.power(y, 2.0) + np.power(z, 2.0))
        t = np.arccos(z/r)
        s = np.arctan(y/x)
        polar_coordinates.append((r,t,s))
    c_all = np.array(c_all)
    c_all = c_all[~np.isnan(c_all)]
    if len(c_all) == 0:
       return 0, 0, 0, 0, 0, 0
    c_min = np.min(c_all)
    c_max = np.max(c_all)-c_min
    c_all = [int(x) for x in np.round(np.multiply(np.divide(np.subtract(c_all, c_min), c_max), 127))]
    G = np.zeros([128, 128, 1, 10])
    # Calculate Haralick across surface
    comparisons = 0
    for v_i in range(len(verts)):
        Ng_verts = tmw.vertex_neighbors[v_i]
        v_t = polar_coordinates[v_i][1]
        v_s = polar_coordinates[v_i][2]
        # Divide into
        for Ng_i in range(len(Ng_verts)):
            Ng = Ng_verts[Ng_i]
            i = c_all[Ng]
            j = c_all[v_i]
            t = polar_coordinates[v_i][1]
            s = polar_coordinates[v_i][2]
            angle = int(round((np.arctan2(v_t-t, v_s-s)+np.pi)/(2*np.pi)*9))
            G[i, j, 0, angle] += 1
            comparisons += 1
    P = np.divide(G, comparisons)
    contrast = np.mean(skimage.feature.texture.greycoprops(P, prop='contrast'))
    dissimilarity = np.mean(skimage.feature.texture.greycoprops(P, prop='dissimilarity'))
    homogeneity = np.mean(skimage.feature.texture.greycoprops(P, prop='homogeneity'))
    ASM = np.mean(skimage.feature.texture.greycoprops(P, prop='ASM'))
    energy = np.mean(skimage.feature.texture.greycoprops(P, prop='energy'))
    correlation = np.mean(skimage.feature.texture.greycoprops(P, prop='correlation'))
#    print((contrast, dissimilarity, homogeneity, ASM, energy, correlation))

    return contrast, dissimilarity, homogeneity, ASM, energy, correlation

casefun_3D_GLCM_names = ('median_contrastS', 'median_dissimilarityS', 'median_homogeneityS', 'median_ASMS', 'median_energyS', 'median_correlationS', 'median_contrastV', 'median_dissimilarityV', 'median_homogeneityV', 'median_ASMV', 'median_energyV', 'median_correlationV')
def casefun_3D_GLCM(LESIONDATAr, LESIONr, WGr, resolution):
    ROIdata = LESIONDATAr[LESIONr[0] > 0]
    #try:
    verts_all, faces_all = create_mesh_smooth(LESIONDATAr, LESIONr[1], 0.5, resolution, 'L1')
    #except:
    #    return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    if len(verts_all) == 0:
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    tm = trimesh.base.Trimesh(vertices=verts_all, faces=faces_all)
    contrastS_all, dissimilarityS_all, homogeneityS_all, ASMS_all, energyS_all, correlationS_all = get_mesh_surface_features(tm, verts_all, faces_all, LESIONDATAr, resolution)
    contrastV_all, dissimilarityV_all, homogeneityV_all, ASMV_all, energyV_all, correlationV_all = get_3D_GLCM(LESIONDATAr, LESIONr[1], resolution)
    for LESION_i in range(2, len(LESIONr)):
        verts, faces = create_mesh_smooth(LESIONDATAr, LESIONr[LESION_i], 1.0, resolution, 'L' + str(LESION_i-1))
        tm = trimesh.base.Trimesh(vertices=verts, faces=faces)
        contrastS, dissimilarityS, homogeneityS, ASMS, energyS, correlationS = get_mesh_surface_features(tm, verts, faces, LESIONDATAr, resolution)
        contrastV, dissimilarityV, homogeneityV, ASMV, energyV, correlationV = get_3D_GLCM(LESIONDATAr, LESIONr[LESION_i], resolution)
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


# Features for each
casefun_01_moments_names = ('mean', 'median', '25percentile', '75percentile', 'skewness', 'kurtosis', 'SD', 'range', 'volume', 'CV')
def casefun_01_moments(LESIONDATAr, LESIONr, WGr, resolution):
    
    ROIdata = LESIONDATAr[LESIONr[0] > 0]
    mean = np.mean(ROIdata)
    median = np.median(ROIdata)
    p25 = np.percentile(ROIdata, 25)
    p75 = np.percentile(ROIdata, 75)
    skewness = scipy.stats.skew(ROIdata)
    kurtosis = scipy.stats.kurtosis(ROIdata)
    SD = np.std(ROIdata)
    rng = np.max(ROIdata)-np.min(ROIdata)
    volume = len(ROIdata)
    if not mean == 0:
        CV = SD/mean
    else:
        CV = 0.0
    return mean, median, p25, p75, skewness, kurtosis, SD, rng, volume, CV


# Features for each
casefun_01_moments_WG_names = ('WGmean', 'WGmedian', 'WG25percentile', 'WG75percentile', 'WGskewness', 'WGkurtosis', 'WGSD', 'WGIQR',
                                'relWGmean', 'relWGmedian', 'relWG25percentile', 'relWG75percentile', 'relWGskewness', 'relWGkurtosis', 'relWGSD', 'WGIQR')
def casefun_01_moments_WG(LESIONDATAr, LESIONr, WGr, resolution):
    ROIdata = LESIONDATAr[LESIONr[0] > 0]
    WGdata = LESIONDATAr[WGr > 0]
    mean = np.mean(ROIdata)
    median = np.median(ROIdata)
    print(ROIdata.shape)
    p25 = np.percentile(ROIdata, 25)
    p75 = np.percentile(ROIdata, 75)
    skewness = scipy.stats.skew(ROIdata)
    kurtosis = scipy.stats.kurtosis(ROIdata)
    SD = np.std(ROIdata)
    IQrange = iqr(ROIdata)
    wmean = np.mean(WGdata)
    wmedian = np.median(WGdata)
    wp25 = np.percentile(WGdata, 25)
    wp75 = np.percentile(WGdata, 75)
    wskewness = scipy.stats.skew(WGdata)
    wkurtosis = scipy.stats.kurtosis(WGdata)
    wSD = np.std(WGdata)
    wIQrange = iqr(WGdata)
    if(mean == 0):
      WGmean = 0
    else:
      WGmean = mean/(mean+np.mean(WGdata))
    if(median == 0):
      WGmedian = 0
    else:
      WGmedian = median/(median+np.median(WGdata))
    if(p25 == 0):
      WGp25 = 0
    else:
      WGp25 = p25/(p25+np.percentile(WGdata, 25))
    if(p75 == 0):
      WGp75 = 0
    else:
      WGp75 = p75/(p75+np.percentile(WGdata, 75))
    print(skewness)
    if(skewness == 0):
      WGskewness = 0
    else:
      WGskewness = skewness/(skewness+scipy.stats.skew(WGdata))
    if(kurtosis == 0):
      WGkurtosis = 0
    else:
      WGkurtosis = kurtosis/(kurtosis+scipy.stats.kurtosis(WGdata))
    if(SD == 0):
      WGSD = 0
    else:
      WGSD = SD/(SD+np.std(WGdata))
    if(IQrange == 0):
      WGIQrange = 0
    else:
      WGIQrange = IQrange/(IQrange+iqr(WGdata))

    return wmean, wmedian, wp25, wp75, wskewness, wkurtosis, wSD, wIQrange, WGmean, WGmedian, WGp25, WGp75, WGskewness, WGkurtosis, WGSD, WGIQrange


# largest slice statistics
def Moment2_fun_largest_slice(LESIONDATAr, LESIONr, WGr, resolution, amount):
    volumes = []
    indexes = []
    max_vol = -1
    max_idx = -1
    for z in range(LESIONr[0].shape[2]):
        LESIONslice = LESIONr[0][:, :, z]
        vol = len(LESIONslice[LESIONslice>0])
        if vol > 0:
            volumes.append(vol)
            indexes.append(z)
            if vol > max_vol:
                max_vol = vol
                max_idx = z
    eroded = copy.deepcopy(LESIONr[0][:, :, max_idx])
    ROIdata = LESIONDATAr[:, :, max_idx][eroded > 0]
    WGr = WGr[:, :, max_idx]
    WGr[eroded > 0] = 0
    WGdata = LESIONDATAr[:, :, max_idx][WGr > 0]
    return ROIdata, WGdata


# 5x5 from largest slice statistics
def Moment2_fun_largest_slice5x5(LESIONDATAr, LESIONr, WGr, resolution, amount):
    volumes = []
    indexes = []
    max_vol = -1
    max_idx = -1
    print(np.max(LESIONr[0]))
    for z in range(LESIONr[0].shape[2]):
        LESIONslice = LESIONr[0][:, :, z]
        vol = len(LESIONslice[LESIONslice>0])
        print((vol,z))
        if vol > 0:
            volumes.append(vol)
            indexes.append(z)
            if vol > max_vol:
                max_vol = vol
                max_idx = z
    eroded = copy.deepcopy(LESIONr[0][:, :, max_idx])

    # resolve 5x5 region around minimum or maximum
    NG = []
    for x in [-1, 0, 1]:
        for y in [-1, 0, 1]:
            NG.append((x,y))

    ROIdata = LESIONDATAr[:, :, max_idx][eroded > 0]
    WGdata = LESIONDATAr[:, :, max_idx][WGr[:, :, max_idx] > 0]

    min_val = np.max(ROIdata)
    min_coords = [-1, -1]
    for x in range(eroded.shape[0]):
        for y in range(eroded.shape[1]):
            if eroded[x,y] == 0:
                continue
            NGvals = []
            for NGi in NG:
                NGvals.append(LESIONDATAr[x+NGi[0], y+NGi[1], max_idx])
            NGval_mean = np.mean(NGvals)
            if NGval_mean < min_val:
                min_val = NGval_mean
                min_coords = [x, y]
    eroded = np.zeros_like(eroded)
    for x in [-1, 0, 1]:
        for y in [-1, 0, 1]:
            eroded[min_coords[0]+x, min_coords[0]+y] = 1

    ROIdata = LESIONDATAr[:, :, max_idx][eroded > 0]
    WGr = WGr[:, :, max_idx]
    WGr[eroded > 0] = 0
    WGdata = LESIONDATAr[:, :, max_idx][WGr > 0]
    return ROIdata, WGdata


# 5x5 from largest slice statistics
def Moment2_fun_largest_sliceKDE(LESIONDATAr, LESIONr, WGr, resolution, amount):

    volumes = []
    indexes = []
    max_vol = -1
    max_idx = -1
    for z in range(LESIONr[0].shape[2]):
        LESIONslice = LESIONr[0][:, :, z]
        vol = len(LESIONslice[LESIONslice>0])
        if vol > 0:
            volumes.append(vol)
            indexes.append(z)
            if vol > max_vol:
                max_vol = vol
                max_idx = z
    eroded = copy.deepcopy(LESIONr[0][:, :, max_idx])
    eroded = ndimage.morphology.binary_dilation(eroded, iterations=1).astype("uint8")
    orig_vol = len(eroded[eroded>0])

    # resolve 5x5 region around minimum or maximum
    centroid_x = []
    centroid_y = []
    ROIdata_z = LESIONDATAr[:, :, max_idx][eroded > 0]
    ROIdata_zmax = np.max(ROIdata_z)
    print(ROIdata_zmax)
    ROIdata_zmin = np.min(ROIdata_z)
    for x in range(eroded.shape[0]):
        for y in range(eroded.shape[1]):
            if eroded[x,y] > 0:
                val = 1 - LESIONDATAr[x, y, max_idx]/ROIdata_zmax
                for v in range(int(val*1000)):
                    centroid_x.append(x)
                    centroid_y.append(y)
    kernel = stats.gaussian_kde(np.vstack([centroid_x, centroid_y]))
    KDEregion = np.zeros_like(eroded)
    centroid_x = []
    centroid_y = []
    for x in range(eroded.shape[0]):
        for y in range(eroded.shape[1]):
            if eroded[x,y] > 0:
                centroid_x.append(x)
                centroid_y.append(y)
    centroid_xy = np.vstack([centroid_x, centroid_y])
    values = kernel.evaluate(centroid_xy)
    th = np.max(values)*amount
    for c_i in range(len(centroid_xy[0])):
        if values[c_i] > th:
            KDEregion[centroid_xy[0][c_i], centroid_xy[1][c_i]] = 1
    ROIdata = LESIONDATAr[:, :, max_idx][KDEregion > 0]
    WGr_idx = WGr[:, :, max_idx]
    WGr_idx[KDEregion > 0] = 0
    WGdata = LESIONDATAr[:, :, max_idx][WGr_idx > 0]
    return ROIdata, WGdata


def Moment2_fun_KDE(LESIONDATAr, LESIONr, WGr, resolution, amount):

    KDEregion_all = np.zeros_like(LESIONr[0])
    for z in range(LESIONr[0].shape[2]):
        eroded = LESIONr[0][:,:,z]
        if len(eroded[eroded>0]) == 0:
            continue
        KDEregion = np.zeros_like(eroded)
        centroid_x = []
        centroid_y = []
        centroid_z = []
        ROIdata_z = LESIONDATAr[:,:,z][eroded > 0]
        ROIdata_zmax = np.max(ROIdata_z)
        ROIdata_zmin = np.min(ROIdata_z)
        for x in range(eroded.shape[0]):
            for y in range(eroded.shape[1]):
                if eroded[x, y] > 0:
                    KDEregion_all[x,y,z] = 1
                    val = 1 - LESIONDATAr[x, y, z]/ROIdata_zmax
                    for v in range(int(val*1000)):
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
                if eroded[x,y] > 0:
                    centroid_x.append(x)
                    centroid_y.append(y)
        centroid_xyz = np.vstack([centroid_x, centroid_y])
        values = kernel.evaluate(centroid_xyz)
        th = np.max(values)*amount
        for c_i in range(len(centroid_xyz[0])):
            if values[c_i] > th:
                KDEregion[centroid_xyz[0][c_i], centroid_xyz[1][c_i]] = 1
        KDEregion_all[:,:,z] = KDEregion
    ROIdata = LESIONDATAr[KDEregion_all > 0]
    WGr[KDEregion_all > 0] = 0
    WGdata = LESIONDATAr[WGr > 0]
    return ROIdata, WGdata


casefun_01_moments2_WG_names = ('mean', 'median', '25percentile', '75percentile', 'skewness', 'kurtosis', 'SD', 'IQR',
                                'relWGmean', 'relWGmedian', 'relWG25percentile', 'relWG75percentile', 'relWGskewness', 'relWGkurtosis', 'relWGSD', 'WGIQR')
def casefun_01_moments2_name_generator(params):
    names = []
    for name in casefun_01_moments2_WG_names:
        names.append('UTUMoments2_%2.1f_%s_%s' % (params[0], params[1], name))
    return names


def casefun_01_Moments2(LESIONDATAr, LESIONr, WGr, resolution, params):

    #eroded, u1, u2 = operations3D.fun_voxelwise_erodeLesion_voxels_3Dmesh(LESIONr[0], WGr, LESIONDATAr, resolution, amount)
    #LESIONr[0][WGr == 0] = 0
    #resolve slice with largest volume
    if params[1] == 'largest_slice':
        ROIdata, WGdata = Moment2_fun_largest_slice(LESIONDATAr, LESIONr, WGr, resolution, params[0])
    elif params[1] == 'largest_slice5x5':
        ROIdata, WGdata = Moment2_fun_largest_slice5x5(LESIONDATAr, LESIONr, WGr, resolution, params[0])
    elif params[1] == 'largest_sliceCCRG':
        ROIdata, WGdata = Moment2_fun_largest_sliceCCRG(LESIONDATAr, LESIONr, WGr, resolution, params[0])
    elif params[1] == 'largest_sliceKDE':
        ROIdata, WGdata = Moment2_fun_largest_sliceKDE(LESIONDATAr, LESIONr, WGr, resolution, params[0])
    elif params[1] == 'Moment2_fun_KDE':
        ROIdata, WGdata = Moment2_fun_KDE(LESIONDATAr, LESIONr, WGr, resolution, params[0])
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
    wmean = np.mean(WGdata)
    wmedian = np.median(WGdata)
    wp25 = np.percentile(WGdata, 25)
    print(WGdata)
    wp75 = np.percentile(WGdata, 75)
    wskewness = scipy.stats.skew(WGdata)
    wkurtosis = scipy.stats.kurtosis(WGdata)
    wSD = np.std(WGdata)
    wIQrange = iqr(WGdata)
    if(mean == 0):
      relWGmean = 0
    else:
      relWGmean = mean/(mean+np.mean(WGdata))
    if(median == 0):
      relWGmedian = 0
    else:
      relWGmedian = median/(median+np.median(WGdata))
    if(p25 == 0):
      relWG25percentile = 0
    else:
      relWG25percentile = p25/(p25+np.percentile(WGdata, 25))
    if(p75 == 0):
      relWG75percentile = 0
    else:
      relWG75percentile = p75/(p75+np.percentile(WGdata, 75))
    if(skewness == 0):
      relWGskewness = 0
    else:
      relWGskewness = skewness/(skewness+scipy.stats.skew(WGdata))
    if(kurtosis == 0):
      relWGkurtosis = 0
    else:
      relWGkurtosis = kurtosis/(kurtosis+scipy.stats.kurtosis(WGdata))
    if(SD == 0):
      relWGSD = 0
    else:
      relWGSD = SD/(SD+np.std(WGdata))
    if(IQR == 0):
      WGIQR = 0
    else:
      WGIQR = IQR/(IQR+iqr(WGdata))

    return mean, median, p25, p75, skewness, kurtosis, SD, IQR, relWGmean, relWGmedian, relWG25percentile, relWG75percentile, relWGskewness, relWGkurtosis, relWGSD, WGIQR


def Meshlab_hausdorf(in_file, in_file2, out_file):
    # Add input mesh
    command = "C:/Program Files/VCG/Meshlab/meshlabserver -i " + in_file
    command += " -i " + in_file2
    # Add the filter script
    command += " -s D:/PRODIF_Jussi/scripts_whole_PRODIF/meshlab_smooth.mlx"
    # Add the output filename and output flags
    if os.path.exists(out_file):
        os.remove(out_file)
    command += " -o " + out_file + " -m vn fn"
    # Execute command
    print("Going to execute: " + command)
    try:
        output = subprocess.check_output(command, shell=False)
    except:
        print('Exception')


def Meshlab_smooth(in_file, out_file):
    # Add input mesh
    command = "C:/Program Files/VCG/Meshlab/meshlabserver -i " + in_file
    # Add the filter script
    command += " -s D:/PRODIF_Jussi/scripts_whole_PRODIF/meshlab_smooth.mlx"
    # Add the output filename and output flags
    if os.path.exists(out_file):
        os.remove(out_file)
    command += " -o " + out_file + " -m vn fn"
    # Execute command
    print("Going to execute: " + command)
    try:
        output = subprocess.check_output(command, shell=False)
    except:
        print('Exception')


def Meshlab_smooth2(in_file, out_file):
    # Add input mesh
    command = "C:/Program Files/VCG/Meshlab/meshlabserver -i " + in_file
    # Add the filter script
    command += " -s D:/PRODIF_Jussi/scripts_whole_PRODIF/meshlab_smooth2.mlx"
    # Add the output filename and output flags
    if os.path.exists(out_file):
        os.remove(out_file)
    command += " -o " + out_file + " -m vn fn"
    # Execute command
    print("Going to execute: " + command)
    try:
        output = subprocess.check_output(command, shell=False)
    except:
        print('Exception')


def get_mesh_surface_samples(verts, faces, Idata, resolution):
    c_all = []
    for face in faces:
        avg_loc_x = int(round(np.mean([verts[face[0]][0], verts[face[1]][0], verts[face[2]][0]])/resolution[0]))
        avg_loc_y = int(round(np.mean([verts[face[0]][1], verts[face[1]][1], verts[face[2]][1]])/resolution[1]))
        avg_loc_z = int(round(np.mean([verts[face[0]][2], verts[face[1]][2], verts[face[2]][2]])/resolution[2]))
        if avg_loc_x < 0 or avg_loc_y < 0 or avg_loc_z < 0:
           c = float('NaN')
        elif avg_loc_x >= Idata.shape[0] or avg_loc_y >= Idata.shape[1] or avg_loc_z >= Idata.shape[2]:
           c = float('NaN')
        else:
           c = Idata[avg_loc_x, avg_loc_y, avg_loc_z]
        c_all.append(c)
    return c_all


def write_plyfile(verts, faces, Idata, resolution, plyfilename):
    verts_for_ply = []
    faces_for_ply = []
    for vert in verts:
        verts_for_ply.append((vert[0], vert[1], vert[2]))
    max_I = np.amax(Idata)
    c_all = get_mesh_surface_samples(verts, faces, Idata, resolution)
    for face_i in range(len(faces)):
        face = faces[face_i]
        c = c_all[face_i]/max_I*255
        if not np.isnan(c):
           faces_for_ply.append(([face[0], face[1], face[2]], c, c, c))
        else:
           faces_for_ply.append(([face[0], face[1], face[2]], 0, 0, 0))
    elv = PlyElement.describe(np.array(verts_for_ply, dtype=[('x','f4'), ('y','f4'), ('z', 'f4')]), 'vertex')
    elf = PlyElement.describe(np.array(faces_for_ply, dtype=[('vertex_indices', 'i4', (3,)), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]), 'face')
    PlyData([elv, elf], text=True).write(plyfilename)


def create_mesh(data, l, resolution):
    verts, faces, normals, values = measure.marching_cubes_lewiner(data, level=l, spacing=(resolution[0], resolution[1], resolution[2]))
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

    elv = PlyElement.describe(np.array(verts_for_ply, dtype=[('x','f4'), ('y','f4'), ('z', 'f4')]), 'vertex')
    elf = PlyElement.describe(np.array(faces_for_ply, dtype=[('vertex_indices', 'i4', (3,)), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]), 'face')
    plyfilename = 'C:/temp/temp_raw_PRODIF.ply'
    PlyData([elv, elf], text=True).write(plyfilename)

    return verts, faces, normals, values


def closest_point_on_line(a, b, p):
    ap = p-a
    ab = b-a
    result = a + np.dot(ap,ab)/np.dot(ab,ab) * ab
    return result


def create_mesh_smooth(Idata, data, l, resolution, name):
    print(resolution)
    verts, faces, normals, values = measure.marching_cubes_lewiner(data, level=l, spacing=(resolution[0], resolution[1], resolution[2]))
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
            
            
    f = tempfile.NamedTemporaryFile(delete=True)
    tempname = os.path.basename(f.name)
    f.close()

    plyfilename = 'C:/temp/' + name + 'temp_raw_' + tempname + '.ply'
    write_plyfile(verts, faces, Idata, resolution, plyfilename)
    plyfilenamep = 'C:/temp/' + name + 'temp_' + tempname + '.ply'
    
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
    write_plyfile(verts, faces, Idata, resolution, 'C:/temp/' + name + 'temp_' + tempname + '_preprocessed.ply')
    if os.path.exists(plyfilename):
        os.remove(plyfilename)
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
N-D Bresenham line algo
"""
import numpy as np
def _bresenhamline_nslope(slope):

    scale = np.amax(np.abs(slope), axis=1).reshape(-1, 1)
    zeroslope = (scale == 0).all(1)
    scale[zeroslope] = np.ones(1)
    normalizedslope = np.array(slope, dtype=np.double) / scale
    normalizedslope[zeroslope] = np.zeros(slope[0].shape)
    return normalizedslope

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


def bresenhamline(start, end, max_iter=5):

    # Return the points as a single array
    return _bresenhamlines(start, end, max_iter).reshape(-1, start.shape[-1])


def create_mesh_smooth_fit_to_gradient(Idata, data, l, resolution, name):
    print(resolution)
    verts, faces, normals, values = measure.marching_cubes_lewiner(data, level=l, spacing=(resolution[0], resolution[1], resolution[2]))
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
        
    f = tempfile.NamedTemporaryFile(delete=True)
    tempname = os.path.basename(f.name)
    f.close()

    plyfilename = 'C:/temp/' + name + 'temp_raw_' + tempname + '.ply'
    write_plyfile(verts, faces, Idata, resolution, plyfilename)
    plyfilenamep = 'C:/temp/' + name + 'temp_' + tempname + '_preprocessed.ply'
    # Smooth with meshlab
    Meshlab_smooth2(plyfilename, plyfilenamep)
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
    #print(Idata.shape)
    Gdata_x, Gdata_y, Gdata_z = np.gradient(Idata)
    #print((verts, faces))
    tm = trimesh.base.Trimesh(vertices=verts, faces=faces)
    vertex_normals = tm.vertex_normals
    resolution_max = np.max([resolution[0], resolution[1], resolution[2]])
    for v_i in range(len(verts)):
        v = verts[v_i]
        n = vertex_normals[v_i]
        nl = np.sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2])
        n_outwards = np.multiply([n[0]/nl, n[1]/nl, n[2]/nl], resolution_max*30)
        n_inwards = np.multiply([-n[0]/nl, -n[1]/nl, -n[2]/nl], resolution_max*30)
        n_outwards /= resolution[0]
        n_outwards /= resolution[1]
        n_outwards /= resolution[2]
        n_inwards /= resolution[0]
        n_inwards /= resolution[1]
        n_inwards /= resolution[2]

        # resolve best gradient aling vertex normal
        #print((v, np.array([n_inwards]), np.array([n_outwards])))
        points = bresenhamline(np.array([n_inwards]), np.array([n_outwards]), max_iter=-1)
        #print(points)
        Gvalues = []
        for p_i in range(len(points)):
            #print((int(points[p_i][0]), int(points[p_i][1]), int(points[p_i][2])))
            vx = Gdata_x[int(points[p_i][0]), int(points[p_i][1]), int(points[p_i][2])]
            vy = Gdata_y[int(points[p_i][0]), int(points[p_i][1]), int(points[p_i][2])]
            vz = Gdata_z[int(points[p_i][0]), int(points[p_i][1]), int(points[p_i][2])]
            Gvalues.append(np.mean([abs(vx), abs(vy), abs(vz)]))
        Gvalues = Gvalues/np.sum(Gvalues)
        #print(Gvalues)
        wpoint = [0,0,0]
        for p_i in range(len(points)):
            wpoint[0] += points[p_i][0]*Gvalues[p_i]*resolution[0]
            wpoint[1] += points[p_i][1]*Gvalues[p_i]*resolution[1]
            wpoint[2] += points[p_i][2]*Gvalues[p_i]*resolution[2]
        #print((verts[v_i], wpoint, v+wpoint))
        verts[v_i][0] = (0.7*verts[v_i][0]+0.3*(verts[v_i][0]+wpoint[0]))
        verts[v_i][1] = (0.7*verts[v_i][1]+0.3*(verts[v_i][1]+wpoint[1]))
        verts[v_i][2] = (0.7*verts[v_i][2]+0.3*(verts[v_i][2]+wpoint[2]))

    plyfilename = 'C:/temp/' + name + 'temp_' + tempname + '_preprocessed_opt.ply'
    write_plyfile(verts, faces, Idata, resolution, plyfilename)

    plyfilename = 'C:/temp/' + name + 'temp_' + tempname + '_preprocessed_opt.ply'
    plyfilenamep = 'C:/temp/' + name + 'temp_' + tempname + '.ply'
    Meshlab_smooth2(plyfilename, plyfilenamep)
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


# Features for each
casefun_3D_shape_names = ('sarea3D','relsarea3D', 'tm_area_faces', 'tm_relarea_faces', 'mean_angles', 'median_angles', 'SD_angles', 'distance_mean', 'distance_median', 'CSM_mean_curvature', 'CSM_Gaus_mean_curvature', 'WG_median', 'WG_SD', 'WG_skewness', 'WG_kurtosis')
def casefun_3D_shape(LESIONDATAr, LESIONr, WGr, resolution):
    ROIdata = LESIONDATAr[LESIONr[0] > 0]
    WGdata = LESIONDATAr[WGr > 0]
    print((len(ROIdata), np.min(ROIdata), np.max(ROIdata)))
    print((len(WGdata), np.min(WGdata), np.max(WGdata)))

    #try:
    verts, faces, normals, values = create_mesh(LESIONr[0], 0.5, resolution)
    #except:
    #    print('failed to create')
    #    return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    sarea = skimage.measure.mesh_surface_area(verts, faces)
    #try:
    vertsw, facesw, normalsw, valuesw = create_mesh(WGr, 0.5, resolution)
    #except:
    #    print('failed to create WG')
    #    return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    sareaw = skimage.measure.mesh_surface_area(vertsw, facesw)
    tm = trimesh.base.Trimesh(vertices=verts, faces=faces, face_normals=normals)
    tmw = trimesh.base.Trimesh(vertices=vertsw, faces=facesw, face_normals=normalsw)
    angles = trimesh.curvature.face_angles(tm)
    mean_angles = np.mean(angles)
    median_angles = np.median(angles)
    SD_angles = np.std(angles)
    CoM = tm.center_mass
    distances = []
    for v in tm.vertices:
        distances.append(np.sqrt(np.power(v[0]-CoM[0], 2.0)+np.power(v[1]-CoM[1], 2.0)+np.power(v[2]-CoM[2], 2.0)))
    distance_mean = np.mean(distances)
    distance_median = np.median(distances)
    CSM_mean_curvature = trimesh.curvature.discrete_mean_curvature_measure(tm, [CoM], np.max(distances))
    CSM_Gaus_mean_curvature = trimesh.curvature.discrete_gaussian_curvature_measure(tm, [CoM], np.max(distances))
    # Distance to whole gland
    closest, distancew, triangle_id = trimesh.proximity.closest_point(tm, tmw.vertices)
    w1 = np.median(distancew)
    w2 = np.std(distancew)
    w3 = scipy.stats.skew(distancew)
    w4 = scipy.stats.kurtosis(distancew)

    return sarea, sarea/sareaw, np.median(tm.area_faces), np.median(tm.area_faces)/len(ROIdata), mean_angles, median_angles, SD_angles, distance_mean, distance_median, CSM_mean_curvature[0], CSM_Gaus_mean_curvature[0], w1, w2, w3, w4


casefun_3D_shape2_names = ('sarea3Dsm','relsarea3Dsm', 'tm_area_facessm', 'tm_relarea_facessm', 'mean_anglessm', 'median_anglessm', 'SD_anglessm', 'distance_meansm', 'distance_mediansm', 'CSM_mean_curvaturesm', 'CSM_Gaus_mean_curvaturesm', 'WG_mediansm', 'WG_SDsm', 'WG_skewnesssm', 'WG_kurtosissm')
def casefun_3D_shape2(LESIONDATAr, LESIONr, WGr, resolution):
    ROIdata = LESIONDATAr[LESIONr[0] > 0]
    #print((len(ROIdata), np.min(ROIdata), np.max(ROIdata)))

    # Whole Gland
    #try:
    vertsw, facesw = create_mesh_smooth(LESIONDATAr, WGr, 0.5, resolution, 'WG')
    #except:
    #    print('failed to create smooth WG')
    #    return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    if len(vertsw) == 0:
        print('failed to create WG')
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    tmw = trimesh.base.Trimesh(vertices=vertsw, faces=facesw)

    # Lesion1, then other lesions, if found
    #try:
    verts_all, faces_all = create_mesh_smooth(LESIONDATAr, LESIONr[0], 0.5, resolution, 'L1')
    #except:
    #    print('failed to smooth')
    #    return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    if len(verts_all) == 0:
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    #print((len(verts_all), len(faces_all)))
    sarea = skimage.measure.mesh_surface_area(verts_all, faces_all)
    tm = trimesh.base.Trimesh(vertices=verts_all, faces=faces_all)
    angles_all = trimesh.curvature.face_angles(tm)
    CoM = tm.center_mass
    distances_all = []
    for v in tm.vertices:
        distances_all.append(np.sqrt(np.power(v[0]-CoM[0], 2.0)+np.power(v[1]-CoM[1], 2.0)+np.power(v[2]-CoM[2], 2.0)))
    CSM_mean_curvature = trimesh.curvature.discrete_mean_curvature_measure(tm, [CoM], np.max(distances_all))
    CSM_Gaus_mean_curvature = trimesh.curvature.discrete_gaussian_curvature_measure(tm, [CoM], np.max(distances_all))
    closest, distancew_all, triangle_id = trimesh.proximity.closest_point(tm, tmw.vertices)
    for LESION_i in range(2, len(LESIONr)):
        try:
            verts, faces = create_mesh_smooth(LESIONDATAr, LESIONr[LESION_i], 1.0, resolution, 'L' + str(LESION_i-1))
        except:
            continue
        if len(verts) == 0:
            continue
        print((verts_all.shape, verts.shape))
        verts_all = np.concatenate((verts_all, verts))
        faces_all = np.concatenate((faces_all, faces))
        sarea = sarea + skimage.measure.mesh_surface_area(verts, faces)
        tm = trimesh.base.Trimesh(vertices=verts, faces=faces)
        angles = trimesh.curvature.face_angles(tm)
        angles_all = np.concatenate((angles_all, angles))
        CoM = tm.center_mass
        distances = []
        for v in tm.vertices:
            distances.append(np.sqrt(np.power(v[0]-CoM[0], 2.0)+np.power(v[1]-CoM[1], 2.0)+np.power(v[2]-CoM[2], 2.0)))
        CSM_mean_curvature = CSM_mean_curvature + trimesh.curvature.discrete_mean_curvature_measure(tm, [CoM], np.max(distances))
        CSM_Gaus_mean_curvature = CSM_Gaus_mean_curvature + trimesh.curvature.discrete_gaussian_curvature_measure(tm, [CoM], np.max(distances))
        distances_all = np.concatenate((distances_all, distances))
        closest, distancew, triangle_id = trimesh.proximity.closest_point(tm, tmw.vertices)
        distancew_all = np.concatenate((distancew_all, distancew))
    verts = verts_all
    faces = faces_all
    sarea = sarea / (len(LESIONr)-1)
    angles = angles_all
    distances = distances_all
    CSM_Gaus_mean_curvature = CSM_Gaus_mean_curvature / (len(LESIONr)-1)
    CSM_mean_curvature = CSM_mean_curvature / (len(LESIONr)-1)
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

    return sarea, sarea/sareaw, np.median(tm.area_faces), np.median(tm.area_faces)/len(ROIdata), mean_angles, median_angles, SD_angles, distance_mean, distance_median, CSM_mean_curvature[0], CSM_Gaus_mean_curvature[0], w1, w2, w3, w4


casefun_3D_shape3_names = ('sarea3Dsm3','relsarea3Dsm3', 'tm_area_facessm3', 'tm_relarea_facessm3', 'mean_angless3m', 'median_anglessm3', 'SD_anglessm3', 'distance_meansm3', 'distance_mediansm3', 'CSM_mean_curvaturesm3', 'CSM_Gaus_mean_curvaturesm3', 'WG_mediansm3', 'WG_SDsm3', 'WG_skewnesssm3', 'WG_kurtosissm3')
def casefun_3D_shape3(LESIONDATAr, LESIONr, WGr, resolution):
    ROIdata = LESIONDATAr[LESIONr[0] > 0]

    # Whole Gland
    #try:
    vertsw, facesw = create_mesh_smooth(LESIONDATAr, WGr, 1.0, resolution, 'WG')
    #except:
    #    return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    tmw = trimesh.base.Trimesh(vertices=vertsw, faces=facesw)

    # Lesion1, then other lesions, if found
    #try:
    verts_all, faces_all = create_mesh_smooth_fit_to_gradient(LESIONDATAr, LESIONr[1], 1.0, resolution, 'L1')
    if len(verts_all) == 0:
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    #except:
    #    return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    sarea = skimage.measure.mesh_surface_area(verts_all, faces_all)
    tm = trimesh.base.Trimesh(vertices=verts_all, faces=faces_all)
    angles_all = trimesh.curvature.face_angles(tm)
    CoM = tm.center_mass
    distances_all = []
    for v in tm.vertices:
        distances_all.append(np.sqrt(np.power(v[0]-CoM[0], 2.0)+np.power(v[1]-CoM[1], 2.0)+np.power(v[2]-CoM[2], 2.0)))
    CSM_mean_curvature = trimesh.curvature.discrete_mean_curvature_measure(tm, [CoM], np.max(distances_all))
    CSM_Gaus_mean_curvature = trimesh.curvature.discrete_gaussian_curvature_measure(tm, [CoM], np.max(distances_all))
    closest, distancew_all, triangle_id = trimesh.proximity.closest_point(tm, tmw.vertices)
    for LESION_i in range(2, len(LESIONr)):
        try:
            verts, faces = create_mesh_smooth_fit_to_gradient(LESIONDATAr, LESIONr[LESION_i], 1.0, resolution, 'L' + str(LESION_i-1))
        except:
            continue
        if len(verts) == 0:
            continue
        verts_all = np.concatenate((verts_all, verts))
        faces_all = np.concatenate((faces_all, faces))
        sarea = sarea + skimage.measure.mesh_surface_area(verts, faces)
        tm = trimesh.base.Trimesh(vertices=verts, faces=faces)
        angles = trimesh.curvature.face_angles(tm)
        angles_all = np.concatenate((angles_all, angles))
        CoM = tm.center_mass
        distances = []
        for v in tm.vertices:
            distances.append(np.sqrt(np.power(v[0]-CoM[0], 2.0)+np.power(v[1]-CoM[1], 2.0)+np.power(v[2]-CoM[2], 2.0)))
        CSM_mean_curvature = CSM_mean_curvature + trimesh.curvature.discrete_mean_curvature_measure(tm, [CoM], np.max(distances))
        CSM_Gaus_mean_curvature = CSM_Gaus_mean_curvature + trimesh.curvature.discrete_gaussian_curvature_measure(tm, [CoM], np.max(distances))
        distances_all = np.concatenate((distances_all, distances))
        closest, distancew, triangle_id = trimesh.proximity.closest_point(tm, tmw.vertices)
        distancew_all = np.concatenate((distancew_all, distancew))
    verts = verts_all
    faces = faces_all
    sarea = sarea / (len(LESIONr)-1)
    angles = angles_all
    distances = distances_all
    CSM_Gaus_mean_curvature = CSM_Gaus_mean_curvature / (len(LESIONr)-1)
    CSM_mean_curvature = CSM_mean_curvature / (len(LESIONr)-1)
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

    return sarea, sarea/sareaw, np.median(tm.area_faces), np.median(tm.area_faces)/len(ROIdata), mean_angles, median_angles, SD_angles, distance_mean, distance_median, CSM_mean_curvature[0], CSM_Gaus_mean_curvature[0], w1, w2, w3, w4


casefun_3D_surface_textures_names = ('surf_mean', 'surf_median', 'surf_25percentile', 'surf_75percentile', 'surf_skewness', 'surf_kurtosis', 'surf_SD', 'surf_range', 'surf_volume', 'surf_CV')
def casefun_3D_surface_textures(LESIONDATAr, LESIONr, WGr, resolution):

    ROIdata = LESIONDATAr[LESIONr[1] > 0]
    print(len(ROIdata))
    #try:
    verts_all, faces_all = create_mesh_smooth(LESIONDATAr, LESIONr[1], 0.5, resolution, 'L1')
    #except:
    #    return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    if len(verts_all) == 0:
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    Surface_data = get_mesh_surface_samples(verts_all, faces_all, LESIONDATAr, resolution)
    for LESION_i in range(2, len(LESIONr)):
        #try:
        verts, faces = create_mesh_smooth(LESIONDATAr, LESIONr[LESION_i], 0.5, resolution, 'L' + str(LESION_i-1))
        c = get_mesh_surface_samples(verts, faces, LESIONDATAr, resolution)
        #except:
        #    continue
        Surface_data = np.concatenate((Surface_data, c))

    mean = np.mean(Surface_data)
    median = np.median(Surface_data)
    p25 = np.percentile(Surface_data, 25)
    p75 = np.percentile(Surface_data, 75)
    skewness = scipy.stats.skew(Surface_data)
    kurtosis = scipy.stats.kurtosis(Surface_data)
    SD = np.std(Surface_data)
    rng = np.max(Surface_data)-np.min(Surface_data)
    volume = len(Surface_data)
    if not mean == 0:
        CV = SD/mean
    else:
        CV = 0.0
    return mean, median, p25, p75, skewness, kurtosis, SD, rng, volume, CV


def otsu(gray):
    pixel_number = len(gray)
    if pixel_number == 0:
        return 0
    mean_weigth = 1.0/pixel_number
    his, bins = np.histogram(gray, bins=25)
    final_thresh = -1
    final_value = -1
    intensity_arr = np.arange(25)
    for t in range(1,25):
        pcb = np.sum(his[:t])
        pcf = np.sum(his[t:])
        Wb = pcb * mean_weigth
        Wf = pcf * mean_weigth
        mub = np.sum(intensity_arr[:t]*his[:t]) / float(pcb)
        muf = np.sum(intensity_arr[t:]*his[t:]) / float(pcf)
        value = Wb * Wf * (mub - muf) ** 2
        if value > final_value:
            final_thresh = t
            final_value = value
    #print 'otsu th:' + str(bins[final_thresh])
    return bins[final_thresh]


def sliding_window3D(image, mask, stepSize, windowSize):
    # slide a window across the image
    for z in range(0, image.shape[2], stepSize):
        if np.max(mask[:,:,z]) == 0:
            continue
        for y in range(0, image.shape[1], stepSize):
            if np.max(mask[:,y,z]) == 0:
                continue
            for x in range(0, image.shape[0], stepSize):
                # yield the current window
                yield (x, y, z, image[x:x + windowSize[0], y:y + windowSize[1], z:z + windowSize[2]])
        print(('%d/%d Laws' % (z, image.shape[2])))


"""
3D analogue of laws texture features
Suzuki, M.T. and Yaginuma, Y., 2007, January.
A solid texture analysis based on three-dimensional convolution kernels. In Videometrics IX (Vol. 6491, p. 64910W). International Society for Optics and Photonics.
Suzuki, M.T., Yaginuma, Y., Yamada, T. and Shimizu, Y., 2006, December. A shape feature extraction method based on 3D convolution masks. 
In Multimedia, 2006. ISM'06. Eighth IEEE International Symposium on (pp. 837-844). IEEE.
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
            #print(str(line_no) + ':' + str(row))
            if(not row[0][0] == '-' and not row[0].isdigit()):
                #print(str(line_no) + '[HEADER]:' + str(row))
                kernels.append({'name':kernelcode,'data':data})
                kernelcode = row[0].strip()
                data = np.empty([5, 5, 5])
                data_read = 0
                x = 0
                y = 0
                z = 0
                continue
            #else:
            #    print(str(line_no) + ':' + str(row))
            for x in range(len(row)):
                data[x,y,z] = float(row[x])
            y += 1
            if y == 5:
                y = 0
                z += 1
    kernels.append({'name':kernelcode,'data':data})

    # Find redundant kernels
    kernels_pruned = []
    for i in range(len(kernels)):
        already_included = False
        for ii in range(len(kernels_pruned)):
            if np.max(abs(np.subtract(kernels[i]['data'],kernels_pruned[ii]['data']))) == 0:
                print('Excluding redundant kernel ' + kernels[i]['name'] + ' ' + kernels_pruned[ii]['name'])
                already_included = True
                break
        if not already_included:
            kernels_pruned.append(kernels[i])
    #print(str(len(kernels_pruned)) + ' pruned 3D Laws kernels out of ' + str(len(kernels)))
    return kernels_pruned

Laws3Dkernel = read_3DLaws_kernel('mask-3d-5.txt')

casefun_3D_Laws_names = ['mean_ROI', 'median_ROI', 'SD_ROI', 'IQR_ROI', 'skewnessROI', 'kurtosisROI', 'p25ROI', 'p75ROI', 'rel']
laws3D_names = []
for kernel_i in range(1, len(Laws3Dkernel)):
    laws3D_names.append(Laws3Dkernel[kernel_i]['name'])

def casefun_3D_Laws_names_generator(params):
    names = []
    for l in range(len(laws3D_names)):
        for name in casefun_3D_Laws_names:
            names.append('UTU3DLaws%s_%s_f%2.1f' % (laws3D_names[l], name, params[0]))
    return names


def append_Laws_results(outdata, LESIONrs, WGrs, ret):
    ROIdata = outdata[LESIONrs > 0]
    WGdata = outdata[WGrs > 0]
    if(len(ROIdata) == 0):
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
    if(mean1 == 0):
       ret.append(0)
    else:
       ret.append(mean1/(mean1+np.mean(WGdata)))
    return ret


def casefun_3D_Laws(LESIONDATAr, LESIONr, WGr, resolution, params):

    # resolution factor affecting laws feature sampling ratio
    # 1: original resolution
    # <1: upsampling
    # >1: downsampling
    res_f = params[0]

    x_lo, x_hi, y_lo, y_hi, z_lo, z_hi = find_bounded_subregion3D(LESIONDATAr)
    LESIONDATArs = LESIONDATAr[x_lo:x_hi, y_lo:y_hi, :]
    LESIONrs_temp = LESIONr[0][x_lo:x_hi, y_lo:y_hi, :]
    WGrs_temp = WGr[x_lo:x_hi, y_lo:y_hi, :]

    # Create masks and output data to desired resolution, intensity data is resliced later for non-zero only
    min_res = np.max(resolution)
    new_res = [min_res*res_f, min_res*res_f, min_res*res_f]
    LESIONrs_temp, affineLESIONrs_temp = reslice_array(LESIONrs_temp, resolution, new_res, 0)
    WGrs_temp, affineWGrs_temp = reslice_array(WGrs_temp, resolution, new_res, 0)
    LESIONDATArs, affineLESIONDATArs = reslice_array(LESIONDATArs, resolution, new_res, 1)                

    outdatas = []
    for kernel_i in range(1, len(Laws3Dkernel)):
        outdatas.append(np.zeros_like(LESIONDATArs))

    s = 5
    mid = int(np.floor(s/2.0))
    for (x, y, z, window) in sliding_window3D(LESIONDATArs, LESIONrs_temp, 1, (s, s, s)):
        window = np.subtract(window, np.mean(window))
        w_std = np.std(window)
        if w_std > 0:
            window = np.divide(window, np.std(window))
        xmid = x + mid
        ymid = y + mid
        zmid = z + mid
        if xmid >= LESIONDATArs.shape[0]:
            continue
        if ymid >= LESIONDATArs.shape[1]:
            continue
        if zmid >= LESIONDATArs.shape[2]:
            continue

        correlates = []
        for kernel_i in range(1, len(Laws3Dkernel)):
            c = correlate(window, Laws3Dkernel[kernel_i]['data'])
            correlates.append(c[2:7,2:7,2:7])
        for c_i in range(len(correlates)):
            outdatas[c_i][xmid, ymid, zmid] = np.sum(correlates[c_i])                

    ret = []
    for kernel_i in range(1, len(Laws3Dkernel)):
        ret = append_Laws_results(outdatas[kernel_i-1], LESIONrs_temp, WGrs_temp, ret)
    return ret


casefun_SNR_names = ('mean', 'median', '25percentile', '75percentile', 'skewness', 'kurtosis', 'SD', 'IQR')
def casefun_SNR_name_generator(params):
    names = []
    for Lname in ['', 'WG', 'rel']:
        for name in casefun_SNR_names:
            names.append('UTUSNRMoments_%s_%s' % (Lname, name))
    names.append('UTUSNRMoments_CNR')
    names.append('UTUSNRMoments_npCNR')
    return names


def relvalue(val1, val2):
    if(val1 == 0 and val2 == 0):
        return 1
    else:
        return abs(val1)/((val1+val2)/2.0)

def casefun_SNR(LESIONDATAr, LESIONr, WGr, resolution, params):
    ROIdata = LESIONDATAr[LESIONr[0] > 0]
    WGdata = LESIONDATAr[WGr > 0]

    mean = np.mean(ROIdata)
    median = np.median(ROIdata)
    p25 = np.percentile(ROIdata, 25)
    p75 = np.percentile(ROIdata, 75)
    skewness = scipy.stats.skew(ROIdata)
    kurtosis = scipy.stats.kurtosis(ROIdata)
    SD = np.std(ROIdata)
    IQrange = iqr(ROIdata)

    wmean = np.mean(WGdata)
    wmedian = np.median(WGdata)
    wp25 = np.percentile(WGdata, 25)
    wp75 = np.percentile(WGdata, 75)
    wskewness = scipy.stats.skew(WGdata)
    wkurtosis = scipy.stats.kurtosis(WGdata)
    wSD = np.std(WGdata)
    wIQrange = iqr(WGdata)

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

    CNR = abs(mean-wmean)/((SD+wSD)/2)
    npCNR = abs(median-wmedian)/((IQrange+wIQrange)/2)
    results.append(CNR)
    results.append(npCNR)
    print(results)
    return results