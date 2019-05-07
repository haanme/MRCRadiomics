#!/usr/bin/env python

import os
import numpy as np
import sys
import textures_2D as textures_2D
import textures_3D
import step6_calculate_AUCs_utilities as step6utils
from glob import glob
from argparse import ArgumentParser
import copy


def add_Laws(datafuns, prefix):
    datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_Laws, textures_2D.casefun_3D_2D_Laws_names_generator, [0.5]))
    datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_Laws, textures_2D.casefun_3D_2D_Laws_names_generator, [1.0]))
    datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_Laws, textures_2D.casefun_3D_2D_Laws_names_generator, [2.0]))
    datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_Laws, textures_2D.casefun_3D_2D_Laws_names_generator, [4.0]))
    return datafuns


def add_Laws3D(datafuns, prefix):
    datafuns.append((prefix + '.nii', prefix, 2.0, textures_3D.casefun_3D_Laws, textures_3D.casefun_3D_Laws_names_generator, [0.5]))
    datafuns.append((prefix + '.nii', prefix, 2.0, textures_3D.casefun_3D_Laws, textures_3D.casefun_3D_Laws_names_generator, [1.0]))
    datafuns.append((prefix + '.nii', prefix, 2.0, textures_3D.casefun_3D_Laws, textures_3D.casefun_3D_Laws_names_generator, [2.0]))
    datafuns.append((prefix + '.nii', prefix, 2.0, textures_3D.casefun_3D_Laws, textures_3D.casefun_3D_Laws_names_generator, [4.0]))
    return datafuns


def add_Laws3D_ADC(datafuns, prefix):
    datafuns.append((prefix + '.nii', prefix, 2.0, textures_3D.casefun_3D_Laws, textures_3D.casefun_3D_Laws_names_generator, [0.5]))
    datafuns.append((prefix + '.nii', prefix, 2.0, textures_3D.casefun_3D_Laws, textures_3D.casefun_3D_Laws_names_generator, [2.0]))
    return datafuns


def add_LBP(datafuns, prefix):
    datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_local_binary_pattern_41, textures_2D.casefun_3D_2D_local_binary_pattern_41_names))
    datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_local_binary_pattern_81, textures_2D.casefun_3D_2D_local_binary_pattern_81_names))
    datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_local_binary_pattern_42, textures_2D.casefun_3D_2D_local_binary_pattern_42_names))
    datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_local_binary_pattern_82, textures_2D.casefun_3D_2D_local_binary_pattern_82_names))
    datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_local_binary_pattern_43, textures_2D.casefun_3D_2D_local_binary_pattern_43_names))
    datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_local_binary_pattern_83, textures_2D.casefun_3D_2D_local_binary_pattern_83_names))
    return datafuns


def add_Hu(datafuns, prefix):
    datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_Hu_moments_rawintensity, textures_2D.casefun_3D_2D_Hu_moments_rawintensity_names))
    datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_Hu_moments_2bins, textures_2D.casefun_3D_2D_Hu_moments_2bins_names))
    datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_Hu_moments_2bins, textures_2D.casefun_3D_2D_Hu_moments_2bins_names))
    datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_Hu_moments_3bins, textures_2D.casefun_3D_2D_Hu_moments_3bins_names))
    datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_Hu_moments_4bins, textures_2D.casefun_3D_2D_Hu_moments_4bins_names))
    return datafuns


def add_Hu_Nyul(datafuns, prefix):
    import textures_normalized
    datafuns.append((prefix + '.nii', prefix, 2.0, textures_normalized.casefun_3D_2D_Hu_moments_rawintensity_Nyul, textures_normalized.casefun_3D_2D_Hu_moments_rawintensity_names_Nyul))
    return datafuns


def add_Zernike(datafuns, prefix):
    datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_Zernike_9_8_8, textures_2D.casefun_3D_2D_Zernike_9_8_8_names))
    datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_Zernike_15_5_5, textures_2D.casefun_3D_2D_Zernike_15_5_5_names))
    datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_Zernike_15_6_6, textures_2D.casefun_3D_2D_Zernike_15_6_6_names))
    datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_Zernike_17_6_6, textures_2D.casefun_3D_2D_Zernike_17_6_6_names))
    datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_Zernike_19_6_6, textures_2D.casefun_3D_2D_Zernike_19_6_6_names))
    datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_Zernike_21_8_8, textures_2D.casefun_3D_2D_Zernike_21_8_8_names))
    datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_Zernike_25_12_12, textures_2D.casefun_3D_2D_Zernike_25_12_12_names))
    return datafuns


def add_Wavelet(datafuns, prefix):
    #'haar', 'db', 'sym', 'coif', 'bior', 'rbio', 'dmey', 'gaus',
    #'mexh', 'morl', 'cgau', 'shan', 'fbsp', 'cmor'
    datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_Wavelet, textures_2D.casefun_3D_2D_Wavelet_names_generator,['Haar', 1.0]))
    return datafuns


def add_Gabor(datafuns, prefix):
    for directions in [2,4,8]:
       for kernelsize in [1,2,4,8,16]:
           for frequency in [1.0,1.5,2.0,3.0,5.0,8.0,12.0]:
               if frequency >= kernelsize:
                    continue
               params = [frequency,directions,kernelsize]
               datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_gabor_filter, textures_2D.casefun_3D_2D_gabor_filter_name_generator, params))
    return datafuns


def add_EdgesCorners2D3D(datafuns, prefix):
    # blockSize (mm) - Neighborhood size (see the details on cornerEigenValsAndVecs()) will be truncated so closest effective voxels
    # ksize - Aperture parameter for the Sobel() operator. Size of the extended Sobel kernel; it must be 1, 3, 5, or 7.
    # k - Harris detector free parameter.
    for blockSize in [2,3,4]:
       for ksize in [1,3,7]:
           for k in [0.01, 0.05, 0.5]:
               datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_Harris, textures_2D.casefun_3D_2D_Harris_name_generator, [blockSize, ksize, k]))
    # maxCorners (int) Maximum number of corners to return. If there are more corners than are found, the strongest of them is returned.
    # qualityLevel (%) Parameter characterizing the minimal accepted quality of image corners. The parameter value is multiplied by the best corner quality measure, which is the minimal eigenvalue (see cornerMinEigenVal() ) or the Harris function response (see cornerHarris() ). The corners with the quality measure less than the product are rejected. For example, if the best corner has the quality measure = 1500, and the qualityLevel=0.01 , then all the corners with the quality measure less than 15 are rejected.
    # minDistance (mm) Minimum possible Euclidean distance between the returned corners.
    datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_ShiTomasi, textures_2D.casefun_3D_2D_ShiTomasi_name_generator, [1000,0.001,2.0]))
    datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_ShiTomasi, textures_2D.casefun_3D_2D_ShiTomasi_name_generator, [1000,0.05,2.0]))
    datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_ShiTomasi, textures_2D.casefun_3D_2D_ShiTomasi_name_generator, [1000,0.1,2.0]))
    datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_Frangi_objectprops, textures_2D.casefun_3D_2D_Frangi_objectprops_names))
    datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_Hessian_objectprops, textures_2D.casefun_3D_2D_Hessian_objectprops_name_generator, [0.005, 15]))
    datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_Hessian_objectprops, textures_2D.casefun_3D_2D_Hessian_objectprops_name_generator, [0.025, 15]))
    datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_Scharr_objectprops, textures_2D.casefun_3D_2D_Scharr_objectprops_names))
    return datafuns


def add_MomentsShapes(datafuns, prefix):
    datafuns.append((prefix + '.nii', prefix, 2.0, textures_3D.casefun_01_moments, textures_3D.casefun_01_moments_names))
    datafuns.append((prefix + '.nii', prefix, 2.0, textures_3D.casefun_3D_shape, textures_3D.casefun_3D_shape_names))
    datafuns.append((prefix + '.nii', prefix, 2.0, textures_3D.casefun_3D_shape2, textures_3D.casefun_3D_shape2_names))
    datafuns.append((prefix + '.nii', prefix, 2.0, textures_3D.casefun_3D_surface_textures, textures_3D.casefun_3D_surface_textures_names))
    datafuns.append((prefix + '.nii', prefix, 2.0, textures_3D.casefun_levelset, textures_3D.casefun_levelset_names))
    datafuns.append((prefix + '.nii', prefix, 2.0, textures_3D.casefun_3D_GLCM, textures_3D.casefun_3D_GLCM_names))
    return datafuns


def add_WGMoments(datafuns, prefix):
    datafuns.append((prefix + '.nii', prefix, 2.0, textures_3D.casefun_01_moments_WG, textures_3D.casefun_01_moments_WG_names))
    return datafuns


def add_Moments2(datafuns, prefix):
    datafuns.append((prefix + '.nii', prefix, 2.0, textures_3D.casefun_01_Moments2, textures_3D.casefun_01_moments2_name_generator, [1, 'largest_slice']))
    datafuns.append((prefix + '.nii', prefix, 2.0, textures_3D.casefun_01_Moments2, textures_3D.casefun_01_moments2_name_generator, [1, 'largest_slice5x5']))
    #datafuns.append((prefix + '.nii', prefix, 2.0, textures_3D.casefun_01_Moments2, textures_3D.casefun_01_moments2_name_generator, [1, 'largest_sliceCCRG']))
    datafuns.append((prefix + '.nii', prefix, 2.0, textures_3D.casefun_01_Moments2, textures_3D.casefun_01_moments2_name_generator, [0.75, 'largest_sliceKDE']))
    datafuns.append((prefix + '.nii', prefix, 2.0, textures_3D.casefun_01_Moments2, textures_3D.casefun_01_moments2_name_generator, [0.9, 'Moment2_fun_KDE']))
    return datafuns


def add_WGMoments_ADC(datafuns, prefix):
    if sys.version_info[0] < 3:
        raise Exception("Must be using Python 3 with WGMoments_ADC")
    import textures_normalized
    datafuns.append((prefix + '.nii', prefix, 2.0, textures_normalized.casefun_ADCKwak_01_moments_WG, textures_normalized.casefun_ADCKwak_01_moments_names))
    return datafuns


def add_WGMoments_ADCN(datafuns, prefix):
    if sys.version_info[0] < 3:
        raise Exception("Must be using Python 3 with WGMoments_ADCN (normalized ADC)")
    import textures_normalized
    datafuns.append((prefix + '.nii', prefix, 2.0, textures_normalized.casefun_ADCNyul_01_moments_WG, textures_normalized.casefun_ADCNyul_01_moments_names))
    return datafuns


def add_SignalToNoiseRatios(datafuns, prefix):
    datafuns.append((prefix + '.nii', prefix, 2.0, textures_3D.casefun_SNR, textures_3D.casefun_SNR_name_generator,[]))
    return datafuns


def test_dimensions(DATA1_data, WGr, LESIONmasks, fout, dim_i):
    mismatch = False
    if not DATA1_data.shape[dim_i] == WGr.shape[dim_i]:
        print('Data %d and WG mask %d x dimension mismatch\n' % (DATA1_data.shape[dim_i], WGr.shape[dim_i]))
        fout.write('Data %d and WG mask %d x dimension mismatch\n' % (DATA1_data.shape[dim_i], WGr.shape[dim_i]))
        mismatch = True
    if not DATA1_data.shape[dim_i] == LESIONmasks[0].shape[dim_i]:
        print('Data %d and WG mask %d x dimension mismatch\n' % (DATA1_data.shape[dim_i], LESIONmasks[0].shape[dim_i]))
        fout.write('Data %d and WG mask %d x dimension mismatch\n' % (DATA1_data.shape[dim_i], LESIONmasks[0].shape[dim_i]))
        mismatch = True
    return mismatch


def resolve_datafuns(method, modality):
    datafuns = []
    if method == 'FFT2D':
       prefix = modality
       datafuns.append((prefix + '.nii', prefix, 2.0, textures_2D.casefun_3D_2D_FFT2D, textures_2D.casefun_3D_2D_FFT2D_names_generator, [1.0, 1.0, 5.0, 5.0]))
    if method == 'Laws':
       datafuns = add_Laws(datafuns, modality)
    if method == 'Laws3D_ADC':
        datafuns = add_Laws3D_ADC(datafuns, modality)
    if method == 'Laws3D':
        datafuns = add_Laws3D(datafuns, modality)
    if method == 'EdgesCorners2D3D':
       datafuns = add_EdgesCorners2D3D(datafuns, modality)
    if method == 'Gabor':
        datafuns = add_Gabor(datafuns, modality)
    if method == 'LBP':
        datafuns = add_LBP(datafuns, modality)
    if method == 'HU':
        datafuns = add_Hu(datafuns, modality)
    if method == 'HUN':
        datafuns = add_Hu_Nyul(datafuns, modality)
    if method == 'MomentsShapes':
        datafuns = add_MomentsShapes(datafuns, modality)
    if method == 'Moments2':
        datafuns = add_Moments2(datafuns, modality)
    if method == 'WGMoments':
        datafuns = add_WGMoments(datafuns, modality)
    if method == 'WGMoments_ADC':
        datafuns = add_WGMoments_ADC(datafuns, modality)
    if method == 'WGMoments_ADCN':
        datafuns = add_WGMoments_ADCN(datafuns, modality)
    if method == 'Zernike':
        datafuns = add_Zernike(datafuns, modality)
    if method == 'Wavelet':
        datafuns = add_Wavelet(datafuns, modality)
    elif method == 'SNR':
        datafuns = add_SignalToNoiseRatios(datafuns, modality)        
    return datafuns


def resolve_found_cases(outputpath, modality, method):
    header_found = False
    cases_found = []
    print(modality)
    if modality == 'DWI':
        N_filename = outputpath + os.sep + 'UTU_features4D_' + modality + '_' + method + '.txt'
    else:
        N_filename = outputpath + os.sep + 'UTU_features_' + modality + '_' + method + '.txt'
    EOL_found = False
    if os.path.exists(N_filename):
        print('Reading feature file ' + N_filename)
        f = open(N_filename, 'r')
        for line in f.readlines():
            if 'case' in line.strip():
                header_found = True
            else:
                case = line.split('\t')[0]
                cases_found.append(case)                
            if line[-1] == '\n':
                EOL_found = True
            else:
                EOL_found = False
        f.close()    
    else:
        print('Feature file ' + N_filename + ' was not found')
    return N_filename, cases_found, header_found, EOL_found

###############
# MAIN SCRIPT #
###############
if __name__ == "__main__":
    # Parse input arguments into args structure
    parser = ArgumentParser()
    parser.add_argument("--modality", dest="modality", help="modality", required=True)
    parser.add_argument("--method", dest="method", help="One of: FFT2D, Laws, EdgesCorners2D3D, Gabor, LBP, HU, MomentsShapes, WGMoments, Zernike", required=True)
    parser.add_argument("--input", dest="input", help="input", required=True)
    parser.add_argument("--output", dest="output", help="output", required=True)
    parser.add_argument("--case", dest="case", help="case", required=False, default='')
    parser.add_argument("--voxelsize", dest="voxelsize", help="voxelsize", required=False, default='')
    args = parser.parse_args()
    modality = args.modality
    method = args.method
    inputpath = args.input
    outputpath = args.output
    required_case = args.case
    if len(args.voxelsize) > 0:
        voxelsize = [float(x) for x in args.voxelsize.strip('[').strip(']').split(',')]
    else:
        voxelsize = []
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)

    datafuns = resolve_datafuns(method, modality)

    if len(datafuns) == 0:
       print('No data functions to processs')
       sys.exit(1)

    # Open log file
    Nlog = open(outputpath + os.sep + 'UTU_features_' + modality + '_' + method + '_log.txt', 'w')

    folders = glob(inputpath + os.sep + '*')

    # resolve existing data in output file
    N_filename, cases_found, header_found, EOL_found = resolve_found_cases(outputpath, modality, method)
    if(header_found):
        print('Header found in output file')

    # Write header if it was not found from the file yet
    if not header_found:
        fout = open(N_filename, 'w')
        fout.write('case')
    else:
        fout = open(N_filename, 'a')
    for datafun in datafuns:
        if len(datafun) > 5:
            feature_names = datafun[4](datafun[5])
        else:
            feature_names = datafun[4]
        if not header_found:
            for name in feature_names:
                fout.write('\t%s' % (datafun[1] + '_' + name))            
    if not header_found:
        fout.write('\n')
        EOL_found = True

    logdata = []
    runs = 0
    for folder_i in range(len(folders)):
        folder = folders[folder_i]
        case = os.path.basename(folder)
        if not os.path.isdir(folder):
           continue
        if not case.split('_')[0].isdigit():
           continue
        if len(required_case) > 0 and not case == required_case and not case.split('_')[0] == required_case:
           continue
        if case in cases_found:
           print('SKIP: case ' + case + ' already found')
           continue
        runs = runs + 1

        if not os.path.exists(folder + os.sep + 'LS.nii'):
            print('%s is missing\n' % (folder + os.sep + 'LS.nii'))
            Nlog.write('%s is missing\n' % (folder + os.sep + 'LS.nii'))
        if not os.path.exists(folder + os.sep + 'PM.nii'):
            print('%s is missing\n' % (folder + os.sep + 'PM.nii'))
            Nlog.write('%s is missing\n' % (folder + os.sep + 'PM.nii'))
        if (not os.path.exists(folder + os.sep + 'LS.nii')) or (not os.path.exists(folder + os.sep + 'PM.nii')):
            continue
        LS_data, LS_affine, LS_voxelsize = step6utils.load_nifti(case + ' LS', folder + os.sep + 'LS.nii')
        PM_data, PM_affine, PM_voxelsize = step6utils.load_nifti(case + ' PM', folder + os.sep + 'PM.nii')

        LESIONmasks = [LS_data, LS_data]
        WGr = PM_data
        write_missing = True
        write_case = True
        write_EOL = False
        for datafun_i in range(len(datafuns)):
            if not os.path.exists(folder + os.sep + datafuns[datafun_i][0]):
                print('%s is missing\n' % (folder + os.sep + datafuns[datafun_i][0]))
                if write_missing:
                    Nlog.write('%s is missing\n' % (folder + os.sep + datafuns[datafun_i][0]))
                    write_missing = False
                continue
            DATA1_data, DATA1_affine, DATA1_voxelsize = step6utils.load_nifti(case + ' DATA', folder + os.sep + datafuns[datafun_i][0])
            if len(DATA1_data.shape) > 3:
                Nlog.write('%s\n' % str(DATA1_data.shape))
                continue
            LESIONDATAr = DATA1_data

            if test_dimensions(DATA1_data, WGr, LESIONmasks, Nlog, 0) or test_dimensions(DATA1_data, WGr, LESIONmasks, Nlog, 1) or test_dimensions(DATA1_data, WGr, LESIONmasks, Nlog, 2):
                Nlog.write('Dimension errors found for case ' + case + '\n')
                continue

            if len(voxelsize) > 0:
                resolution = voxelsize
            else:
                resolution = DATA1_voxelsize
            datafun_name = datafuns[datafun_i][1]
            datafun_scale = datafuns[datafun_i][2]
            casefun = datafuns[datafun_i][3]
            if len(datafuns[datafun_i]) > 5:
                casefun_names = datafuns[datafun_i][4](datafuns[datafun_i][5])
                if np.max(LESIONDATAr) == 0 or np.max(LESIONmasks[0]) == 0:
                    print('NULL DATA' + str((np.max(LESIONDATAr) == 0, np.max(LESIONmasks[0]) == 0)))
                    casefun_vals = [float('nan') for x in casefun_names]
                else:
                    casefun_vals = casefun(LESIONDATAr, copy.deepcopy(LESIONmasks), WGr, resolution, datafuns[datafun_i][5])
            else:
                casefun_names = datafuns[datafun_i][4]
                if np.max(LESIONDATAr) == 0 or np.max(LESIONmasks[0]) == 0:
                    print('NULL DATA' + str((np.max(LESIONDATAr) == 0, np.max(LESIONmasks[0]) == 0)))
                    casefun_vals = [float('nan') for x in casefun_names]
                else:           
                    casefun_vals = casefun(LESIONDATAr, copy.deepcopy(LESIONmasks), WGr, resolution)
            if casefun_vals is None:
                raise Exception('Return names values is None')
            if casefun_names is None:
                raise Exception('Return names is None')
            if not len(casefun_vals) == len(casefun_names):
                raise Exception('Return names ' + str(len(casefun_names)) + ' and values ' + str(len(casefun_vals)) + ' number do not match')
            if len(casefun_vals) > 0 and write_case:
                if not EOL_found:
                    fout.write('\n')    
                fout.write('%s' % case.strip())
                write_case = False
            for val in casefun_vals:
                fout.write('\t%10.9f' % val)
                write_EOL = True
        if write_EOL:
            fout.write('\n')
    fout.close()
    Nlog.close()
    if(runs==0):
        print('no cases executed')
    sys.exit(0)
