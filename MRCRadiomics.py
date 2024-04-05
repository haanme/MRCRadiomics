#!/usr/bin/env python
"""
Radiomics for Medical Imaging

Copyright (C) 2019-2022 Harri Merisaari haanme@MRC.fi

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

__version__ = "1.1.2"

# $Source$

import os
import numpy as np
import sys
import textures_2D as textures_2D
import textures_3D
from utilities import load_nifti
from glob import glob
from argparse import ArgumentParser
import copy



"""
Printout depending on verbosity setting

@param entry: text otbe printed
@param verbose: verbose status
"""
verbose = False
def print_verbose(entry, verbose):
    if verbose:
        print(entry)



"""
Adds definitions of 2D Laws features

@param datafuns: current feature settings
@param prefix: input data basename
@returns: updated feature settings list
"""


def add_Laws(method, datafuns, prefix):
    if method is None:
        datafuns.append('Laws')
    if method == 'Laws':
        datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_Laws,
                         textures_2D.casefun_3D_2D_Laws_names_generator, True, True, [0.5]))
        datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_Laws,
                         textures_2D.casefun_3D_2D_Laws_names_generator, True, True, [1.0]))
        datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_Laws,
                         textures_2D.casefun_3D_2D_Laws_names_generator, True, True, [2.0]))
        datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_Laws,
                         textures_2D.casefun_3D_2D_Laws_names_generator, True, True, [4.0]))
    return datafuns


"""
Adds definitions of 3D Laws features

@param datafuns: current feature settings
@param prefix: input data basename
@returns: updated feature settings list
"""


def add_Laws3D(method, datafuns, prefix):
    if method is None:
        datafuns.append('Laws3D')
    if method == 'Laws3D':
        datafuns.append((prefix + '.nii', prefix, 2.0, textures_3D.casefun_3D_Laws,
                         textures_3D.casefun_3D_Laws_names_generator, True, True, [0.5]))
        datafuns.append((prefix + '.nii', prefix, 2.0, textures_3D.casefun_3D_Laws,
                         textures_3D.casefun_3D_Laws_names_generator, True, True, [1.0]))
        datafuns.append((prefix + '.nii', prefix, 2.0, textures_3D.casefun_3D_Laws,
                         textures_3D.casefun_3D_Laws_names_generator, True, True, [2.0]))
        datafuns.append((prefix + '.nii', prefix, 2.0, textures_3D.casefun_3D_Laws,
                         textures_3D.casefun_3D_Laws_names_generator, True, True, [4.0]))
    return datafuns


"""
Adds definitions of 3D Laws features, for whole organ

@param datafuns: current feature settings
@param prefix: input data basename
@returns: updated feature settings list
"""


def add_Laws3D_BG(method, datafuns, prefix):
    if method is None:
        datafuns.append('BGLaws3D')
    if method == 'BGLaws3D':
        datafuns.append((prefix + '.nii', prefix, 2.0, textures_3D.casefun_3D_Laws_BG,
                         textures_3D.casefun_3D_Laws_names_generator_BG, False, True, [0.5]))
        datafuns.append((prefix + '.nii', prefix, 2.0, textures_3D.casefun_3D_Laws_BG,
                         textures_3D.casefun_3D_Laws_names_generator_BG, False, True, [1.0]))
        datafuns.append((prefix + '.nii', prefix, 2.0, textures_3D.casefun_3D_Laws_BG,
                         textures_3D.casefun_3D_Laws_names_generator_BG, False, True, [2.0]))
    return datafuns


"""
Adds definitions of 3D Laws features, with Apparent Diffusion Coefficient specific settings
tailored for prostate cancer detection.

@param datafuns: current feature settings
@param prefix: input data basename
@returns: updated feature settings list
"""


def add_Laws3D_ADC(method, datafuns, prefix):
    if method is None:
        datafuns.append('Laws3D_ADC')
    if method == 'Laws3D_ADC':
        datafuns.append((prefix + '.nii', prefix, 2.0, textures_3D.casefun_3D_Laws,
                         textures_3D.casefun_3D_Laws_names_generator, True, True, [0.5]))
        datafuns.append((prefix + '.nii', prefix, 2.0, textures_3D.casefun_3D_Laws,
                         textures_3D.casefun_3D_Laws_names_generator, True, True, [2.0]))
    return datafuns


"""
Adds definitions of Local Binary Patterns.

@param datafuns: current feature settings
@param prefix: input data basename
@returns: updated feature settings list
"""


def add_LBP(method, datafuns, prefix):
    if method is None:
        datafuns.append('LBP')
    if method == 'LBP':
        datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_local_binary_pattern_41,
                         textures_2D.casefun_3D_2D_local_binary_pattern_41_names, True, True, []))
        datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_local_binary_pattern_81,
                         textures_2D.casefun_3D_2D_local_binary_pattern_81_names, True, True, []))
        datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_local_binary_pattern_42,
                         textures_2D.casefun_3D_2D_local_binary_pattern_42_names, True, True, []))
        datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_local_binary_pattern_82,
                         textures_2D.casefun_3D_2D_local_binary_pattern_82_names, True, True, []))
        datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_local_binary_pattern_43,
                         textures_2D.casefun_3D_2D_local_binary_pattern_43_names, True, True, []))
        datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_local_binary_pattern_83,
                         textures_2D.casefun_3D_2D_local_binary_pattern_83_names, True, True, []))
    return datafuns


"""
Adds definitions of Hu invariant moments.

@param datafuns: current feature settings
@param prefix: input data basename
@returns: updated feature settings list
"""


def add_Hu(method, datafuns, prefix):
    if method is None:
        datafuns.append('HU')
    if method == 'HU':
        datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_Hu_moments_rawintensity,
                         textures_2D.casefun_3D_2D_Hu_moments_rawintensity_names, True, True, []))
        datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_Hu_moments_bins,
                         textures_2D.casefun_3D_2D_Hu_moments_2bins_names, True, True, [2]))
        datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_Hu_moments_bins,
                         textures_2D.casefun_3D_2D_Hu_moments_3bins_names, True, True, [3]))
        datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_Hu_moments_bins,
                         textures_2D.casefun_3D_2D_Hu_moments_4bins_names, True, True, [4]))
    return datafuns


"""
Adds definitions of Zernike features

@param datafuns: current feature settings
@param prefix: input data basename
@returns: updated feature settings list
"""


def add_Zernike(method, datafuns, prefix):
    if method is None:
        datafuns.append('Zernike')
    if method == 'Zernike':
        datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_Zernike_9_8_8,
                         textures_2D.casefun_3D_2D_Zernike_9_8_8_names, True, True))
        datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_Zernike_15_5_5,
                         textures_2D.casefun_3D_2D_Zernike_15_5_5_names, True, True))
        datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_Zernike_15_6_6,
                         textures_2D.casefun_3D_2D_Zernike_15_6_6_names, True, True))
        datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_Zernike_17_6_6,
                         textures_2D.casefun_3D_2D_Zernike_17_6_6_names, True, True))
        datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_Zernike_19_6_6,
                         textures_2D.casefun_3D_2D_Zernike_19_6_6_names, True, True))
        datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_Zernike_21_8_8,
                         textures_2D.casefun_3D_2D_Zernike_21_8_8_names, True, True))
        datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_Zernike_25_12_12,
                         textures_2D.casefun_3D_2D_Zernike_25_12_12_names, True, True))
    return datafuns


"""
Adds definitions of Wavelet features

@param datafuns: current feature settings
@param prefix: input data basename
@returns: updated feature settings list
"""


def add_Wavelet(method, datafuns, prefix):
    if method is None:
        datafuns.append('Wavelet')
    if method == 'Wavelet':
        datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_Wavelet,
                         textures_2D.casefun_3D_2D_Wavelet_names_generator, True, True, ['Haar', 1.0]))
    return datafuns


"""
Adds definitions of Gabor filter features

@param datafuns: current feature settings
@param prefix: input data basename
@returns: updated feature settings list
"""


def add_Gabor(method, datafuns, prefix):
    if method is None:
        datafuns.append('Gabor')
    if method == 'Gabor':
        for directions in [2]:
            for kernelsize in [2]:
                for frequency in [1.0]:
                    if frequency >= kernelsize:
                        continue
                    params = [frequency, directions, kernelsize]
                    datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_gabor_filter,
                                     textures_2D.casefun_3D_2D_gabor_filter_name_generator, True, True, params, []))
    return datafuns


"""
Adds definitions of 2D corner edge detector features

@param datafuns: current feature settings
@param prefix: input data basename
@returns: updated feature settings list
"""


def add_edges_corners2D3D(method, datafuns, prefix):
    if method is None:
        datafuns.append('EdgesCorners2D3D')
    if method == 'EdgesCorners2D3D':
        # blockSize (mm) - Neighborhood size (see the details on cornerEigenValsAndVecs()) will be truncated so closest effective voxels
        # ksize - Aperture parameter for the Sobel() operator. Size of the extended Sobel kernel; it must be 1, 3, 5, or 7.
        # k - Harris-Stephens detector free parameter.
        for blockSize in [2, 3, 4]:
            for ksize in [1, 3, 7]:
                for k in [0.01, 0.05, 0.5]:
                    datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_Harris,
                                     textures_2D.casefun_3D_2D_Harris_name_generator, True, True,
                                     [blockSize, ksize, k]))
        # maxCorners (int) Maximum number of corners to return. If there are more corners than are found, the strongest of them is returned.
        # qualityLevel (%) Parameter characterizing the minimal accepted quality of image corners. The parameter value is multiplied by the best corner quality measure, which is the minimal eigenvalue (see cornerMinEigenVal() ) or the Harris function response (see cornerHarris() ). The corners with the quality measure less than the product are rejected. For example, if the best corner has the quality measure = 1500, and the qualityLevel=0.01 , then all the corners with the quality measure less than 15 are rejected.
        # minDistance (mm) Minimum possible Euclidean distance between the returned corners.
        datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_ShiTomasi,
                         textures_2D.casefun_3D_2D_ShiTomasi_name_generator, True, True, [1000, 0.001, 2.0]))
        datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_ShiTomasi,
                         textures_2D.casefun_3D_2D_ShiTomasi_name_generator, True, True, [1000, 0.05, 2.0]))
        datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_ShiTomasi,
                         textures_2D.casefun_3D_2D_ShiTomasi_name_generator, True, True, [1000, 0.1, 2.0]))
        datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_Frangi_objectprops,
                         textures_2D.casefun_3D_2D_Frangi_objectprops_names, True, True, []))
        datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_Hessian_objectprops,
                         textures_2D.casefun_3D_2D_Hessian_objectprops_name_generator, True, True, [0.005, 15]))
        datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_Hessian_objectprops,
                         textures_2D.casefun_3D_2D_Hessian_objectprops_name_generator, True, True, [0.025, 15]))
        datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_Scharr_objectprops,
                         textures_2D.casefun_3D_2D_Scharr_objectprops_names, True, True, []))
    return datafuns


"""
Adds definitions of 2D corner edge detector features, for whole organ

@param datafuns: current feature settings
@param prefix: input data basename
@returns: updated feature settings list
"""


def add_bg_edges_corners2D3D(method, datafuns, prefix):
    if method is None:
        datafuns.append('BGEdgesCorners2D3D')
    if method == 'BGEdgesCorners2D3D':
        """
        blockSize (mm) - Neighborhood size (see the details on cornerEigenValsAndVecs()) 
                         will be truncated so closest effective voxels
        ksize          - Aperture parameter for the Sobel() operator. 
                         Size of the extended Sobel kernel; it must be 1, 3, 5, or 7
        k              - Harris-Stephens detector free parameter.
        """
        datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_Harris_BG,
                         textures_2D.casefun_3D_2D_Harris_name_generator_BG, False, True, [2, 1, 0.01]))
        # maxCorners (int) Maximum number of corners to return. If there are more corners than are found, the strongest of them is returned.
        # qualityLevel (%) Parameter characterizing the minimal accepted quality of image corners. The parameter value is multiplied by the best corner quality measure, which is the minimal eigenvalue (see cornerMinEigenVal() ) or the Harris function response (see cornerHarris() ). The corners with the quality measure less than the product are rejected. For example, if the best corner has the quality measure = 1500, and the qualityLevel=0.01 , then all the corners with the quality measure less than 15 are rejected.
        # minDistance (mm) Minimum possible Euclidean distance between the returned corners.
        datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_Hessian_objectprops_BG,
                         textures_2D.casefun_3D_2D_Hessian_objectprops_name_generator_BG, False, True, [0.025, 15]))
    return datafuns


"""
Adds definitions of 1st order statistics

@param datafuns: current feature settings
@param prefix: input data basename
@returns: updated feature settings list
"""


def add_moments(method, datafuns, prefix):
    if method is None:
        datafuns.append('Moments')
    if method == 'Moments':
        datafuns.append((prefix + '.nii', prefix, 2.0, textures_3D.casefun_01_moments,
                         textures_3D.casefun_01_moments_names, True, False))
    return datafuns


"""
Adds definitions of shape, topology, and surface intensity features

@param datafuns: current feature settings
@param prefix: input data basename
@returns: updated feature settings list
"""


def add_shapes(method, datafuns, prefix):
    if method is None:
        datafuns.append('Shapes')
    if method == 'Shapes':
        datafuns.append((prefix + '.nii', prefix, 2.0, textures_3D.casefun_3D_shape, textures_3D.casefun_3D_shape_names,
                         True, True, []))
        datafuns.append((prefix + '.nii', prefix, 2.0, textures_3D.casefun_3D_shape2,
                         textures_3D.casefun_3D_shape2_names, True, True, []))
        datafuns.append((prefix + '.nii', prefix, 2.0, textures_3D.casefun_3D_surface_textures,
                         textures_3D.casefun_3D_surface_textures_names, True, True, []))
        datafuns.append((prefix + '.nii', prefix, 2.0, textures_3D.casefun_levelset, textures_3D.casefun_levelset_names,
                         True, True, []))
        datafuns.append((prefix + '.nii', prefix, 2.0, textures_3D.casefun_3D_GLCM, textures_3D.casefun_3D_GLCM_names,
                         True, True, []))
    return datafuns


"""
Adds definitions of shape, topology, and surface intensity features, for whole organ

@param datafuns: current feature settings
@param prefix: input data basename
@returns: updated feature settings list
"""


def add_BGShapes(method, datafuns, prefix):
    if method is None:
        datafuns.append('BGShapes')
    if method == 'BGShapes':
        datafuns.append((prefix + '.nii', prefix, 2.0, textures_3D.casefun_3D_shape_BG,
                         textures_3D.casefun_3D_shape_names_BG, False, True, []))
        datafuns.append((prefix + '.nii', prefix, 2.0, textures_3D.casefun_3D_surface_textures_BG,
                         textures_3D.casefun_3D_surface_textures_names_BG, False, True, []))
        datafuns.append((prefix + '.nii', prefix, 2.0, textures_3D.casefun_3D_GLCM_BG,
                         textures_3D.casefun_3D_GLCM_names_BG, False, True, []))
    return datafuns


"""
Adds definitions of 1st order statistics, for whole organ

@param datafuns: current feature settings
@param prefix: input data basename
@returns: updated feature settings list
"""


def add_BGMoments(method, datafuns, prefix):
    if method is None:
        datafuns.append('BGMoments')
    if method == 'BGMoments':
        datafuns.append((prefix + '.nii', prefix, 2.0, textures_3D.casefun_01_moments_BG,
                         textures_3D.casefun_01_moments_BG_names, False, True, []))
    return datafuns


"""
Adds definitions of 1st order statistics, for whole organ / lesion relative values

@param datafuns: current feature settings
@param prefix: input data basename
@returns: updated feature settings list
"""


def add_relativeBGMoments(method, datafuns, prefix):
    if method is None:
        datafuns.append('relativeBGMoments')
    if method == 'relativeBGMoments':
        datafuns.append((prefix + '.nii', prefix, 2.0, textures_3D.casefun_01_moments_relativeBG,
                         textures_3D.casefun_01_moments_relativeBG_names, True, True, []))
    return datafuns


"""
Adds definitions of 1st order statistics, with ROI refinement operations

@param datafuns: current feature settings
@param prefix: input data basename
@returns: updated feature settings list
"""


def add_Moments2(method, datafuns, prefix):
    if method is None:
        datafuns.append('Moments2')
    if method == 'Moments2':
        datafuns.append((prefix + '.nii', prefix, 2.0, textures_3D.casefun_01_Moments2,
                         textures_3D.casefun_01_moments2_name_generator, True, True, [1, 'largest_slice']))
        datafuns.append((prefix + '.nii', prefix, 2.0, textures_3D.casefun_01_Moments2,
                         textures_3D.casefun_01_moments2_name_generator, True, True, [1, 'largest_slice5x5']))
        datafuns.append((prefix + '.nii', prefix, 2.0, textures_3D.casefun_01_Moments2,
                         textures_3D.casefun_01_moments2_name_generator, True, True, [0.75, 'largest_sliceKDE']))
        datafuns.append((prefix + '.nii', prefix, 2.0, textures_3D.casefun_01_Moments2,
                         textures_3D.casefun_01_moments2_name_generator, True, True, [0.9, 'Moment2_fun_KDE']))
    return datafuns


"""
Adds definitions of SNR measurements.

@param list_datafun: list of feature settings
@param prefix: input data basename
@returns: updated feature settings list
"""


def add_SignalToNoiseRatios(method, list_datafun, prefix):
    if method is None:
        list_datafun.append('SNR')
    if method == 'SNR':
        list_datafun.append((prefix + '.nii', prefix, 2.0, textures_3D.casefun_SNR, textures_3D.casefun_SNR_name_generator,
                             True, True, []))
    return list_datafun


"""
Adds features with Fast Fourier Transform (FFT) based features. 
This method is generally used as reference method for comparison 
with other features.

@param datafuns: current feature settings
@param prefix: input data basename
@returns: updated feature settings
"""


def add_FFT2D(method, datafuns, prefix):
    if method is None:
        datafuns.append('FFT2D')
    if method == 'FFT2D':
        datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_FFT2D,
                         textures_2D.casefun_3D_2D_FFT2D_names_generator, True, True, [1.0, 1.0, 5.0, 5.0]))
    return datafuns


"""
Adds features with Fast Fourier Transform (FFT) based features, for whole organ. 
This method is generally used as reference method for comparison 
with other features.

@param datafuns: current feature settings
@param prefix: input data basename
@returns: updated feature settings
"""


def add_FFT2DBG(method, datafuns, prefix):
    if method is None:
        datafuns.append('BGFFT2D')
    if method == 'BGFFT2D':
        datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_FFT2D_BG,
                         textures_2D.casefun_3D_2D_FFT2D_names_generator_BG, False, True, [1.0, 1.0, 5.0, 5.0]))
    return datafuns


"""
Test image dimensions for consistency

@param data: input image data
@param bg_mask: background binary mask
@param lesionmask_list: lesion binary mask
@param fid_logfile: open file stream for writing results
@param dim_i: dimension to be tested
@returns: True if mismatch between data and binary masks was found
"""


def test_dimensions(data, bg_mask, lesionmask_list, fid_logfile, dim_i):
    mismatch = False
    if (bg_mask is not None) and (not data.shape[dim_i] == bg_mask.shape[dim_i]):
        print('Data %d and BG mask %d x dimension mismatch\n' % (data.shape[dim_i], bg_mask.shape[dim_i]))
        fid_logfile.write('Data %d and BG mask %d x dimension mismatch\n' % (data.shape[dim_i], bg_mask.shape[dim_i]))
        mismatch = True
    if (lesionmask_list[0] is not None) and (not data.shape[dim_i] == lesionmask_list[0].shape[dim_i]):
        print('Data %d and BG mask %d x dimension mismatch\n' % (data.shape[dim_i], lesionmask_list[0].shape[dim_i]))
        fid_logfile.write(
            'Data %d and BG mask %d x dimension mismatch\n' % (data.shape[dim_i], lesionmask_list[0].shape[dim_i]))
        mismatch = True
    return mismatch


"""
Creates settings array for feature extraction

@param method: settings group name as string, or None for resolving supported feature group names
@param modality: modality name (basename of file to be run)
@param boilerplate: collect boilerplate list of strings
@returns: list of settings
"""


def resolve_datafuns(method, modality, boilerplate):
    print_verbose('Resolving radiomic data functions to be used', verbose)
    datafuns = []
    datafuns = add_FFT2D(method, datafuns, modality)
    datafuns = add_FFT2DBG(method, datafuns, modality)
    datafuns = add_Laws(method, datafuns, modality)
    datafuns = add_Laws3D_ADC(method, datafuns, modality)
    datafuns = add_Laws3D(method, datafuns, modality)
    datafuns = add_Laws3D_BG(method, datafuns, modality)
    datafuns = add_edges_corners2D3D(method, datafuns, modality)
    datafuns = add_bg_edges_corners2D3D(method, datafuns, modality)
    datafuns = add_Gabor(method, datafuns, modality)
    datafuns = add_LBP(method, datafuns, modality)
    datafuns = add_Hu(method, datafuns, modality)
    datafuns = add_moments(method, datafuns, modality)
    datafuns = add_shapes(method, datafuns, modality)
    datafuns = add_BGShapes(method, datafuns, modality)
    datafuns = add_relativeBGMoments(method, datafuns, modality)
    datafuns = add_Moments2(method, datafuns, modality)
    datafuns = add_BGMoments(method, datafuns, modality)
    datafuns = add_Zernike(method, datafuns, modality)
    datafuns = add_Wavelet(method, datafuns, modality)
    datafuns = add_SignalToNoiseRatios(method, datafuns, modality)
    return datafuns


"""
Reads output file for already existing entries.

@param destination_path: destination path
@param modality: modality name (basename of file to be run)
@param method: settings group name as string
@returns: [output txt filename, case numbers found, True if header found, True if end of line found]
"""


def resolve_found_cases(destination_path, modality, method):
    featurefile_header_found = False
    featurefile_cases_found = []
    print(modality)
    featurefile = destination_path + os.sep + 'MRCRadiomics_features_' + modality + '_' + method + '.txt'
    featurefile_EOL_found = False
    if os.path.exists(featurefile):
        print('Reading feature file ' + featurefile)
        f = open(featurefile, 'r')
        for line in f.readlines():
            if 'case' in line.strip() and 'ROI' in line.strip() and 'background_ROI' in line.strip():
                featurefile_header_found = True
            else:
                case_id = line.split('\t')[0]
                ROI_id = line.split('\t')[1]
                BG_id = line.split('\t')[2]
                featurefile_cases_found.append((case_id, ROI_id, BG_id))
            if line[-1] == '\n':
                featurefile_EOL_found = True
            else:
                featurefile_EOL_found = False
        f.close()
    else:
        print('Feature file ' + featurefile + ' was not found')
    return featurefile, featurefile_cases_found, featurefile_header_found, featurefile_EOL_found


###############
# MAIN SCRIPT #
###############
"""
Data is expected to be organized in Nifti format, as:

SUBJECTS <-('--input' command line argument)
 +-1_L1 <- (optional '--case' command line argument, '1' case number, 'L1' lesion number)
 |  +-MODALITY.nii <-('--modality' command line argument)
 |  +-LS.nii
 |  +-BG.nii
 .     
 +-N_L1
    +-MODALITY.nii
    +-LS.nii
    +-BG.nii
"""
if __name__ == "__main__":
    # Parse input arguments into args structure
    supported_methods = resolve_datafuns(None, None, False)
    parser = ArgumentParser()
    parser.add_argument("--version", dest="version", help="prints version number", required=False)
    parser.add_argument("--modality", dest="modality", help="modality (basename of input Nifti)", required=True)
    parser.add_argument("--method", dest="method", help="One of: " + str(supported_methods), required=True)
    parser.add_argument("--input", dest="input", help="input base folder", required=True)
    parser.add_argument("--output", dest="output", help="output base folder", required=True)
    parser.add_argument("--case", dest="case", help="case number", required=False, default='')
    parser.add_argument("--voxelsize", dest="voxelsize", help="voxelsize in: '[x,y,z]'", required=False, default='')
    parser.add_argument("--create_visualization", dest="create_visualization", help="1 for extra visualization/debug data if feature is supporting it", required=False, default='')
    parser.add_argument("--BGname", dest="BGname", help="Background mask Nifti filename", required=False, default='N/A')
    parser.add_argument("--LSname", dest="LSname", help="Lesion (i e foreground) mask Nifti filename", required=False, default='N/A')
    parser.add_argument("--verbose", dest="verbose", help="Print verbose output Yes/No[default]", required=False, default='No')
    parser.add_argument("--boilerplate", dest="boilerplate", help="Write boilerplate.txt with citation(s) and descriptions for the used radiomics Yes/No[default]", required=False,
                        default='No')
    args = parser.parse_args()
    modalityname = args.modality
    methodname = args.method
    inputpath = args.input
    outputpath = args.output
    required_case = args.case
    BGname = args.BGname
    LSname = args.LSname
    verbose = args.verbose == 'Yes'
    if(BGname == 'N/A' and LSname == 'N/A'):
        print('Either of LSname or BGname must be given')
        sys.exit(1)
    boilerplate = (args.boilerplate == 'Yes')
    print_verbose('Writing boilerplate:' + str(boilerplate), verbose)

    create_visualization = args.create_visualization
    if len(create_visualization) > 0:
        create_visualization = True
    else:
        create_visualization = False
    if len(args.voxelsize) > 0:
        voxelsize = [float(x) for x in args.voxelsize.strip('[').strip(']').split(',')]
    else:
        voxelsize = []

    # Print version
    if hasattr(args, 'version'):
        print('version %s' % __version__)

    # Create output paths if not existing
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    if create_visualization:
        if not os.path.exists(outputpath + os.sep + 'visualizations'):
            os.makedirs(outputpath + os.sep + 'visualizations')

    # Resolve settings from command line arguments
    print_verbose('Method name:' + methodname, verbose)
    print_verbose('Modality name:' + modalityname, verbose)
    datafun_names = resolve_datafuns(methodname, modalityname, boilerplate)
    if len(datafun_names) == 0:
        print('No data functions to processs')
        sys.exit(1)

    # Open log file
    print_verbose('Opening log file:' + outputpath + os.sep + 'MRCRadiomics_features_' + modalityname + '_' + methodname + '_log.txt', verbose)
    Nlog = open(outputpath + os.sep + 'MRCRadiomics_features_' + modalityname + '_' + methodname + '_log.txt', 'w')

    # Resolve existing output values in the output file
    N_filename, cases_found, header_found, EOL_found = resolve_found_cases(outputpath, modalityname, methodname)
    print_verbose('Feature value file:' + N_filename, verbose)
    print_verbose('Cases found:' + str(cases_found), verbose)
    print_verbose('Header found:' + str(header_found), verbose)
    print_verbose('End of line found:' + str(EOL_found), verbose)
    if header_found:
        print('Header found in output file')

    # Write header if it was not found from the file yet
    if not header_found:
        print_verbose('Creating new file', verbose)
        fout = open(N_filename, 'w')
        fout.write('case\tROI\tbackground_ROI')
    else:
        print_verbose('Appending to existing file', verbose)
        fout = open(N_filename, 'a')
    for datafun in datafun_names:
        if type(datafun[4]) is list:
            print_verbose('Data function type: list', verbose)
            feature_names = datafun[4]
        elif type(datafun[4]) is tuple:
            print_verbose('Data function type: tuple', verbose)
            feature_names = datafun[4]
        elif len(datafun) > 7 and callable(datafun[4]):
            print_verbose('Data function type: callable', verbose)
            feature_names = datafun[4](datafun[7])
        else:
            print_verbose('Data function type: function without parameters', verbose)
            feature_names = datafun[4]()
        if not header_found:
            for name in feature_names:
                fout.write('\t%s' % (datafun[1] + '_' + name))
    print_verbose('Resolved feature names:' + str(feature_names), verbose)
    if not header_found:
        fout.write('\n')
        EOL_found = True

    # Process cases in loop
    logdata = []
    runs = 0
    LS_missing = []
    PM_missing = []
    print_verbose('Reading folder:' + inputpath, verbose)
    folders = glob(inputpath + os.sep + '*')
    print_verbose('Resolved ' + str(len(folders)) + ' subfolders', verbose)
    found_non_dir = 0
    found_case_mismatch = 0
    found_already_in_results = 0
    if len(required_case) > 0:
        print_verbose('Required case [' + required_case + ']', verbose)
    for folder_i in range(len(folders)):
        folder = folders[folder_i]
        case = os.path.basename(folder)

        # Skip case folder if:
        # - not selected from command line arguments
        # - not having numerical subject name (subject folder)
        # - already found in the results file
        if not os.path.isdir(folder):
            found_non_dir += 1
            print_verbose('Folder [' + folder + '] not a directory', verbose)
            continue
        if len(required_case) > 0 and not case == required_case and not case.split('_')[0] == required_case:
            found_case_mismatch += 1
            print_verbose('Folder [' + folder + '] not matching required subfolder ' + required_case, verbose)
            continue
        print(cases_found)
        if (case.strip(), LSname.strip('.nii'), BGname.strip('.nii')) in cases_found:
            found_already_in_results += 1
            print_verbose('Folder [' + folder + '] already found in results', verbose)
            continue
        runs = runs + 1
        print_verbose('Folder [' + folder + '] RUN', verbose)

        # Read ROIs
        if os.path.exists(folder + os.sep + BGname):
            PM_data, PM_affine, PM_voxelsize = load_nifti(case + ' ' + BGname, folder + os.sep + BGname)
        else:
            PM_data, PM_affine, PM_voxelsize = [None, None, None]
        if os.path.exists(folder + os.sep + LSname):
            LS_data, LS_affine, LS_voxelsize = load_nifti(case + ' ' + LSname, folder + os.sep + LSname)
        else:
            LS_data, LS_affine, LS_voxelsize = [None, None, None]
        LESIONmasks = [LS_data, LS_data]
        BGr = PM_data

        # Process all feature extraction function in settings
        write_missing = True
        write_case = True
        write_EOL = False
        for datafun_i in range(len(datafun_names)):
            # Verify ROI existence against feature requirements
            if datafun_names[datafun_i][5]:
                if not os.path.exists(folder + os.sep + LSname):
                    print('%s is missing\n' % (folder + os.sep + LSname))
                    if not os.path.basename(folder) in LS_missing:
                        Nlog.write('%s is missing\n' % (folder + os.sep + LSname))
                        LS_missing.append(not os.path.basename(folder))
                    continue
            if datafun_names[datafun_i][6]:
                if not os.path.exists(folder + os.sep + BGname):
                    print('%s is missing\n' % (folder + os.sep + BGname))
                    if not os.path.basename(folder) in PM_missing:
                        Nlog.write('%s is missing\n' % (folder + os.sep + BGname))
                        PM_missing.append(not os.path.basename(folder))
                    continue

            # Verify data existence
            if not os.path.exists(folder + os.sep + datafun_names[datafun_i][0]):
                print('%s is missing\n' % (folder + os.sep + datafun_names[datafun_i][0]))
                if write_missing:
                    Nlog.write('%s is missing\n' % (folder + os.sep + datafun_names[datafun_i][0]))
                    write_missing = False
                continue
            DATA1_data, DATA1_affine, DATA1_voxelsize = load_nifti(case + ' DATA', folder + os.sep + datafun_names[datafun_i][0])
            if len(DATA1_data.shape) > 3:
                Nlog.write('%s\n' % str(DATA1_data.shape))
                continue
            LESIONDATAr = DATA1_data

            # Verify ROI vs data dimensions match
            if test_dimensions(DATA1_data, BGr, LESIONmasks, Nlog, 0) or test_dimensions(DATA1_data, BGr, LESIONmasks,
                                                                                         Nlog, 1) or test_dimensions(
                    DATA1_data, BGr, LESIONmasks, Nlog, 2):
                Nlog.write('Dimension errors found for case ' + case + '\n')
                continue

            # Apply voxelsize in mm from data or from command line
            if len(voxelsize) > 0:
                resolution = voxelsize
            else:
                resolution = DATA1_voxelsize

            # Apply feature
            datafun_name = datafun_names[datafun_i][1]
            datafun_scale = datafun_names[datafun_i][2]
            casefun = datafun_names[datafun_i][3]
            print_verbose('Function to run:' + str(casefun), verbose)
            # Special handling for feature having specific input parameters
            if len(datafun_names[datafun_i]) > 7:
                if type(datafun_names[datafun_i][4]) is list or type(datafun_names[datafun_i][4]) is tuple:
                    casefun_names = datafun_names[datafun_i][4]
                else:
                    casefun_names = datafun_names[datafun_i][4](datafun_names[datafun_i][7])
                if np.max(LESIONDATAr) == 0 or np.max(LESIONmasks[0]) == 0:
                    print('NULL DATA' + str((np.max(LESIONDATAr) == 0, np.max(LESIONmasks[0]) == 0)))
                    casefun_vals = [float('nan') for x in casefun_names]
                else:
                    if create_visualization:
                        datafun_names[datafun_i][7].append(
                            {'write_visualization': outputpath + os.sep + 'visualizations', 'name': case})
                        datafun_params = datafun_names[datafun_i][7]
                        casefun_vals = casefun(LESIONDATAr, copy.deepcopy(LESIONmasks), BGr, resolution, datafun_params)
                    else:
                        casefun_vals = casefun(LESIONDATAr, copy.deepcopy(LESIONmasks), BGr, resolution, datafun_names[datafun_i][7])
            # Handling for features not having specific parameters
            else:
                casefun_names = datafun_names[datafun_i][4]
                if np.max(LESIONDATAr) == 0 or np.max(LESIONmasks[0]) == 0:
                    print('NULL DATA' + str((np.max(LESIONDATAr) == 0, np.max(LESIONmasks[0]) == 0)))
                    casefun_vals = [float('nan') for x in casefun_names]
                else:
                    casefun_vals = casefun(LESIONDATAr, copy.deepcopy(LESIONmasks), BGr, resolution)

            # Write output numbers to file if they were produced by the feature extraction function
            if casefun_vals is None:
                raise Exception('Return names values is None')
            if casefun_names is None:
                raise Exception('Return names is None')
            if not len(casefun_vals) == len(casefun_names):
                raise Exception('Return names ' + str(len(casefun_names)) + ' and values ' + str(
                    len(casefun_vals)) + ' number do not match')
            if (not create_visualization) and (len(casefun_vals) > 0 and write_case):
                if not EOL_found:
                    fout.write('\n')
                fout.write('%s\t%s\t%s' % (case.strip(), LSname.strip('.nii'), BGname.strip('.nii')))
                write_case = False
            for val in casefun_vals:
                fout.write('\t%10.9f' % val)
                write_EOL = True
        if write_EOL:
            fout.write('\n')
    # Closing operations
    fout.close()
    Nlog.close()
    if runs == 0:
        print('No cases executed')
        print_verbose('Reasons for not running subfolders:', verbose)
        print_verbose('Not dir:' + str(found_non_dir), verbose)
        print_verbose('Not matching required case:' + str(found_case_mismatch), verbose)
        print_verbose('Found already in result file:' + str(found_already_in_results), verbose)
    sys.exit(0)
