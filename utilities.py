#!/usr/bin/env python

import nibabel as nib
from glob import glob
import sklearn.metrics
import DicomIO_G as dcm
from dipy.align.reslice import reslice
from sklearn.metrics import confusion_matrix
from scipy import ndimage
import scipy.ndimage
import skimage
import os
import subprocess
from skimage import measure
import SimpleITK as sitk
import copy


# Directory where result data are located
experiment_dir = ''
stat_funs = []
dcmio = dcm.DicomIO()
seg36_base_segnames = ['2a', '2p', '13asr', '13asl', '1a', '1ap', '1p', '7a', '7ap', '7p', '8a', '8p']
seg36_mid_segnames = ['4a', '4p', '14asr', '14asl', '3a', '3ap', '3p', '9a', '9ap', '9p', '10a', '10p']
seg36_apex_segnames = ['6a', '6p', '5a', '5ap', '5p', '15asr', '15asl', '11a', '11ap', '11p', '12a', '12p']
seg36_all_segnames = seg36_base_segnames + seg36_mid_segnames + seg36_apex_segnames

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def load_nifti(name, filename, voxel_source='pixdim'):
    """
    Loads Nifti data
    :param name: data name for stdout
    :param filename: loaded filename
    :param voxel_source: source for resolving voxel size as information pixdim(default)/sform/qform
    :return: data matrix, affine transformation matrix, data voxelsize
    """
    img = nib.load(filename)
    if voxel_source == 'pixdim':
        print(img.header)
        voxelsize = [img.header['pixdim'][1], img.header['pixdim'][2], img.header['pixdim'][3]]
    img = nib.Nifti1Image(img.get_fdata(), np.eye(4))
    try:
        affine = img.affine
    except:
        affine = img.get_affine()
    try:
        data_sform = img.get_sform()
    except:
        data_sform = img.get_header().get_sform()
    try:
        data_qform = img.get_sform()
    except:
        data_qform = img.get_header().get_qform()
    data = img.get_fdata()
    if voxel_source == 'data_qform':
        voxelsize = [abs(data_qform[0, 0]), abs(data_qform[1, 1]), abs(data_qform[2, 2])]
    else:
        voxelsize = [abs(data_sform[0, 0]), abs(data_sform[1, 1]), abs(data_sform[2, 2])]
    print('Loading ' + name + ':' + filename + ' ' + str(data.shape) + ' ' + str(voxelsize))

    return np.squeeze(data), affine, voxelsize



def load_mha(name, filename):
    """
    Loads ITK data
    :param name: data name for stdout
    :param filename: loaded filename
    :return: data matrix, affine transformation matrix, data voxelsize
    """

    image = sitk.ReadImage(filename, imageIO="MetaImageIO")
    data = copy.deepcopy(sitk.GetArrayViewFromImage(image))
    voxelsize = image.GetSpacing()
    print(voxelsize)
    affine = np.eye(4)
    affine[0, 0] = voxelsize[0]
    affine[1, 1] = voxelsize[1]
    affine[2, 2] = voxelsize[2]
    print('Loading ' + name + ':' + filename + ' ' + str(data.shape) + ' ' + str(voxelsize))
    data = np.squeeze(data)
    data = data[:, :, ::-1]
    data = np.rollaxis(data, 0, 3)
    data = np.transpose(data, (1, 0, 2))
    data = np.flipud(data)
    return data, affine, voxelsize


def load_dcm(path, case, prefix):
        
    #print('LOAD:' + path + os.sep + '*.dcm')
    frame_avg = dcmio.ReadDICOM_frames(path + os.sep + '*.dcm', tname='', printout=0)
    #print(len(frame_avg), len(frame_avg[0]), len(frame_avg[0][0]))
    no_frames = len(frame_avg[0])
    no_slices = len(frame_avg[0][0])
    xdim = frame_avg[0][0][0].Columns
    ydim = frame_avg[0][0][0].Rows
    imgdata = np.zeros([xdim, ydim, no_slices, no_frames])
    voxelsize = [-1, -1, -1]
    slice_ii = 1
    for frame_i in range(no_frames):
        for slice_i in range(no_slices):
            dslice = frame_avg[0][frame_i][slice_i].pixel_array
            imgdata[:, :, slice_i, frame_i] = dslice
            slice_ii += 1
        if (0x0028, 0x0030) in frame_avg[0][frame_i][0]:
            voxelsize[0:2] = [float(x) for x in frame_avg[0][frame_i][0][0x0028, 0x0030].value]
        if (0x0018, 0x0050) in frame_avg[0][frame_i][0]:
            voxelsize[2] = float(frame_avg[0][frame_i][0][0x0018, 0x0050].value)
    #print(np.max(imgdata))
    return np.squeeze(imgdata), voxelsize


def load_dcm2(path, case, prefix):
    
    #print('Reading [' + path + ']')
    slices = dcmio.ReadDICOM_slices(path + os.sep + '*.dcm')
    no_frames = 1
    no_slices = len(slices[0])
    xdim = slices[0][0].Columns
    ydim = slices[0][0].Rows
    imgdata = np.zeros([xdim, ydim, no_slices, no_frames])
    #print(imgdata.shape)
    voxelsize = [-1, -1, -1]
    for slice_i in range(no_slices):
        dslice = slices[0][slice_i].pixel_array
        imgdata[:, :, slice_i, 0] = dslice
    if (0x0028, 0x0030) in slices[0]:
        voxelsize[0:2] = [float(x) for x in slices[0][0x0028, 0x0030].value]
    if (0x0018, 0x0050) in slices[0]:
        voxelsize[2] = float(slices[0][0x0018, 0x0050].value)
    #print(np.max(imgdata))
    return np.squeeze(imgdata), voxelsize


def remove_suffix(filename):
    return filename.rstrip('.nii').rstrip('.gz').rstrip('.mha')


def check_exists(name, filename, files_missing, printout=True):
    """
    Checks if file is missing
    :param name: name of the file for messages
    :param filename: filename
    :param files_missing: number of missing files so far
    :return: number of missing files after testing filename
    """
    if not os.path.exists(filename):
        if printout:
            print(name + ' does not exist [' + filename + ']')
        files_missing += 1
    return files_missing


def write_QC_file_mask(out_filename, data, mask_fun, roimask, name, case_no, no_frames, procfun, scale, text=False):
    scales = [scale]
    scales.append(2.0)
    #print('write_QC_file_mask------------------ ' + out_filename)
    return mask_fun(data, name + '_data', roimask, out_filename, scales, procfun, text)


def percentile(data, p):
    if len(data) == 0:
        return float('NaN')
    else:
        return np.percentile(data, p)

def procfun_noop(data, imagetitle):
    return data


def procfun_takeB0(data, imagetitle):
    print(data.shape)
    if len(data.shape) > 3:
        return np.squeeze(data[:,:,:,0])
    return data


def procfun_takeB0_flipud(data, imagetitle):
    print(data.shape)
    if len(data.shape) > 3:
        return np.flipud(np.squeeze(data[:,:,:,0]))
    return data



def get_dataset_basedir(case):
    if case > 2000:
        return '..' + os.sep + 'pipelineS34'
    elif case > 1000:
        return '..' + os.sep + 'pipeline18'
    else:
        return '..' + os.sep + 'pipeline'


def get_prefix(path, case):
    if case > 2000 or case > 1000:
        return os.path.basename(path)
    else:
        return ('%06d' % case) + "-111I"


def getfun_T2_tra_NyulStandardized(paths, case_i, case):
    if case > 2000:
        return paths[case_i] + os.sep + '_'.join(os.path.basename(paths[case_i]).split('_')[:2]) + "_T2_tra_NyulStandardized.nii.gz"
    elif case > 1000:
        return paths[case_i] + os.sep + '_'.join(os.path.basename(paths[case_i]).split('_')[:2]) + "_T2_tra_NyulStandardized.nii.gz"
    else:
        return paths[case_i] + os.sep + get_prefix(paths[case_i], case) + "_T2_tra_NyulStandardized.nii.gz"

# T2 texture feature maps
def getfun_Curvelet_T2_tra_NuylStandardized(paths, case_i, case):
    return paths[case_i] + os.sep + get_prefix(paths[case_i], case) + os.sep + "tmaps" + os.sep + get_prefix(paths[case_i], case) + "_Curvelet_T2_tra_NuylStandardized.nii.gz"

def getfun_GLCM_7x7_5_16_contrast_T2_LESIONROIS_contrst(paths, case_i, case):
    return paths[case_i] + os.sep + "tmaps" + os.sep + get_prefix(paths[case_i], case) + "_GLCM_7x7_5_16_contrast_T2_LESIONROIS_contrst.nii.gz"
def getfun_GLCM_7x7_5_16_contrast_T2_LESIONROIS_correlation(paths, case_i, case):
    return paths[case_i] + os.sep + "tmaps" + os.sep + get_prefix(paths[case_i], case) + "_GLCM_7x7_5_16_contrast_T2_LESIONROIS_correlation.nii.gz"
def getfun_GLCM_7x7_5_16_contrast_T2_LESIONROIS_dissimilarity(paths, case_i, case):
    return paths[case_i] + os.sep + "tmaps" + os.sep + get_prefix(paths[case_i], case) + "_GLCM_7x7_5_16_contrast_T2_LESIONROIS_dissimilarity.nii.gz"
def getfun_GLCM_7x7_5_16_contrast_T2_LESIONROIS_energy(paths, case_i, case):
    return paths[case_i] + os.sep + "tmaps" + os.sep + get_prefix(paths[case_i], case) + "_GLCM_7x7_5_16_contrast_T2_LESIONROIS_energy.nii.gz"
def getfun_GLCM_7x7_5_16_contrast_T2_LESIONROIS_entropy(paths, case_i, case):
    return paths[case_i] + os.sep + "tmaps" + os.sep + get_prefix(paths[case_i], case) + "_GLCM_7x7_5_16_contrast_T2_LESIONROIS_entropy.nii.gz"
def getfun_GLCM_7x7_5_16_contrast_T2_LESIONROIS_homogeneity(paths, case_i, case):
    return paths[case_i] + os.sep + "tmaps" + os.sep + get_prefix(paths[case_i], case) + "_GLCM_7x7_5_16_contrast_T2_LESIONROIS_homogeneity.nii.gz"
def getfun_GLCM_9x9_5_16_contrast_T2_LESIONROIS_contrst(paths, case_i, case):
    return paths[case_i] + os.sep + "tmaps" + os.sep + get_prefix(paths[case_i], case) + "_GLCM_9x9_5_16_contrast_T2_LESIONROIS_contrst.nii.gz"
def getfun_GLCM_9x9_5_16_contrast_T2_LESIONROIS_correlation(paths, case_i, case):
    return paths[case_i] + os.sep + "tmaps" + os.sep + get_prefix(paths[case_i], case) + "_GLCM_9x9_5_16_contrast_T2_LESIONROIS_correlation.nii.gz"
def getfun_GLCM_9x9_5_16_contrast_T2_LESIONROIS_dissimilarity(paths, case_i, case):
    return paths[case_i] + os.sep + "tmaps" + os.sep + get_prefix(paths[case_i], case) + "_GLCM_9x9_5_16_contrast_T2_LESIONROIS_dissimilarity.nii.gz"
def getfun_GLCM_9x9_5_16_contrast_T2_LESIONROIS_energy(paths, case_i, case):
    return paths[case_i] + os.sep + "tmaps" + os.sep + get_prefix(paths[case_i], case) + "_GLCM_9x9_5_16_contrast_T2_LESIONROIS_energy.nii.gz"
def getfun_GLCM_9x9_5_16_contrast_T2_LESIONROIS_entropy(paths, case_i, case):
    return paths[case_i] + os.sep + "tmaps" + os.sep + get_prefix(paths[case_i], case) + "_GLCM_9x9_5_16_contrast_T2_LESIONROIS_entropy.nii.gz"
def getfun_GLCM_9x9_5_16_contrast_T2_LESIONROIS_homogeneity(paths, case_i, case):
    return paths[case_i] + os.sep + "tmaps" + os.sep + get_prefix(paths[case_i], case) + "_GLCM_9x9_5_16_contrast_T2_LESIONROIS_homogeneity.nii.gz"
def getfun_GLCM_11x11_5_16_contrast_T2_LESIONROIS_contrst(paths, case_i, case):
    return paths[case_i] + os.sep + "tmaps" + os.sep + get_prefix(paths[case_i], case) + "_GLCM_11x11_5_16_contrast_T2_LESIONROIS_contrst.nii.gz"
def getfun_GLCM_11x11_5_16_contrast_T2_LESIONROIS_correlation(paths, case_i, case):
    return paths[case_i] + os.sep + "tmaps" + os.sep + get_prefix(paths[case_i], case) + "_GLCM_11x11_5_16_contrast_T2_LESIONROIS_correlation.nii.gz"
def getfun_GLCM_11x11_5_16_contrast_T2_LESIONROIS_dissimilarity(paths, case_i, case):
    return paths[case_i] + os.sep + "tmaps" + os.sep + get_prefix(paths[case_i], case) + "_GLCM_11x11_5_16_contrast_T2_LESIONROIS_dissimilarity.nii.gz"
def getfun_GLCM_11x11_5_16_contrast_T2_LESIONROIS_energy(paths, case_i, case):
    return paths[case_i] + os.sep + "tmaps" + os.sep + get_prefix(paths[case_i], case) + "_GLCM_11x11_5_16_contrast_T2_LESIONROIS_energy.nii.gz"
def getfun_GLCM_11x11_5_16_contrast_T2_LESIONROIS_entropy(paths, case_i, case):
    return paths[case_i] + os.sep + "tmaps" + os.sep + get_prefix(paths[case_i], case) + "_GLCM_11x11_5_16_contrast_T2_LESIONROIS_entropy.nii.gz"
def getfun_GLCM_11x11_5_16_contrast_T2_LESIONROIS_homogeneity(paths, case_i, case):
    return paths[case_i] + os.sep + "tmaps" + os.sep + get_prefix(paths[case_i], case) + "_GLCM_11x11_5_16_contrast_T2_LESIONROIS_homogeneity.nii.gz"
def getfun_GLCM_13x13_5_16_contrast_T2_LESIONROIS_contrst(paths, case_i, case):
    return paths[case_i] + os.sep + "tmaps" + os.sep + get_prefix(paths[case_i], case) + "_GLCM_13x13_5_16_contrast_T2_LESIONROIS_contrst.nii.gz"
def getfun_GLCM_13x13_5_16_contrast_T2_LESIONROIS_correlation(paths, case_i, case):
    return paths[case_i] + os.sep + "tmaps" + os.sep + get_prefix(paths[case_i], case) + "_GLCM_13x13_5_16_contrast_T2_LESIONROIS_correlation.nii.gz"
def getfun_GLCM_13x13_5_16_contrast_T2_LESIONROIS_dissimilarity(paths, case_i, case):
    return paths[case_i] + os.sep + "tmaps" + os.sep + get_prefix(paths[case_i], case) + "_GLCM_13x13_5_16_contrast_T2_LESIONROIS_dissimilarity.nii.gz"
def getfun_GLCM_13x13_5_16_contrast_T2_LESIONROIS_energy(paths, case_i, case):
    return paths[case_i] + os.sep + "tmaps" + os.sep + get_prefix(paths[case_i], case) + "_GLCM_13x13_5_16_contrast_T2_LESIONROIS_energy.nii.gz"
def getfun_GLCM_13x13_5_16_contrast_T2_LESIONROIS_entropy(paths, case_i, case):
    return paths[case_i] + os.sep + "tmaps" + os.sep + get_prefix(paths[case_i], case) + "_GLCM_13x13_5_16_contrast_T2_LESIONROIS_entropy.nii.gz"
def getfun_GLCM_13x13_5_16_contrast_T2_LESIONROIS_homogeneity(paths, case_i, case):
    return paths[case_i] + os.sep + "tmaps" + os.sep + get_prefix(paths[case_i], case) + "_GLCM_13x13_5_16_contrast_T2_LESIONROIS_homogeneity.nii.gz"

def getfun_Zernike_s_9_0_n_8_0_std_8_0_T2(paths, case_i, case):
    return paths[case_i] + os.sep + "tmaps" + os.sep + get_prefix(paths[case_i], case) + "_Zernike_s_9_0_n_8_0_std_8_0_T2.nii.gz"
def getfun_Zernike_s_15_0_n_5_0_std_5_0_T2(paths, case_i, case):
    return paths[case_i] + os.sep + "tmaps" + os.sep + get_prefix(paths[case_i], case) + "_Zernike_s_15_0_n_5_0_std_5_0_T2.nii.gz"
def getfun_Zernike_s_15_0_n_6_0_std_6_0_T2(paths, case_i, case):
    return paths[case_i] + os.sep + "tmaps" + os.sep + get_prefix(paths[case_i], case) + "_Zernike_s_15_0_n_6_0_std_6_0_T2.nii.gz"
def getfun_Zernike_s_17_0_n_6_0_std_6_0_T2(paths, case_i, case):
    return paths[case_i] + os.sep + "tmaps" + os.sep + get_prefix(paths[case_i], case) + "_Zernike_s_17_0_n_6_0_std_6_0_T2.nii.gz"
def getfun_Zernike_s_19_0_n_6_0_std_6_0_T2(paths, case_i, case):
    return paths[case_i] + os.sep + "tmaps" + os.sep + get_prefix(paths[case_i], case) + "_Zernike_s_19_0_n_6_0_std_6_0_T2.nii.gz"
def getfun_Zernike_s_21_0_n_8_0_std_8_0_T2(paths, case_i, case):
    return paths[case_i] + os.sep + "tmaps" + os.sep + get_prefix(paths[case_i], case) + "_Zernike_s_21_0_n_8_0_std_8_0_T2.nii.gz"
def getfun_Zernike_s_25_0_n_12_0_std_12_0_T2(paths, case_i, case):
    return paths[case_i] + os.sep + "tmaps" + os.sep + get_prefix(paths[case_i], case) + "_Zernike_s_25_0_n_12_0_std_12_0_T2.nii.gz"
def getfunGabor_freq_0_1_dist_27_0_std_3_0_T2(paths, case_i, case):
    print(('datafun:', paths[case_i], case_i, case))
    return paths[case_i] + os.sep + "tmaps" + os.sep + get_prefix(paths[case_i], case) + "_Gabor_freq_0_1_dist_27_0_std_3_0_T2.nii.gz"
def getfunGabor_freq_0_2_dist_27_0_std_3_0_T2(paths, case_i, case):
    print((paths[case_i], case_i, case))
    return paths[case_i] + os.sep + "tmaps" + os.sep + get_prefix(paths[case_i], case) + "_Gabor_freq_0_2_dist_27_0_std_3_0_T2.nii.gz"
def getfunGabor_freq_0_3_dist_27_0_std_3_0_T2(paths, case_i, case):
    print((paths[case_i], case_i, case))
    return paths[case_i] + os.sep + "tmaps" + os.sep + get_prefix(paths[case_i], case) + "_Gabor_freq_0_3_dist_27_0_std_3_0_T2.nii.gz"

# ADC maps in T2 space
def getfun_ADC_Curvelet_T2_tra_DWI_500ADC_Curvelet_T2_tra_atT2(paths, case_i, case):
    return paths[case_i] + os.sep + get_prefix(paths[case_i], case) + os.sep + "coreg" + os.sep + get_prefix(paths[case_i], case) + "_ADC_Curvelet_T2_tra_DWI_500ADC_Curvelet_T2_tra_atT2.nii.nii.gz"


def getfun_ADC_DWI_500_ADC(paths, case_i, case):
    return paths[case_i] + os.sep + get_prefix(paths[case_i], case) + "_ADC.nii.gz"


def getfun_ADC_DWI_500_ADC_atT2(paths, case_i, case):
    if os.path.isfile(paths[case_i] + os.sep + "coreg" + os.sep + get_prefix(paths[case_i], case) + "_ADC_DWI_500_ADC_atT2_ITKSNAP.nii.gz"):
        return paths[case_i] + os.sep + "coreg" + os.sep + get_prefix(paths[case_i], case) + "_ADC_DWI_500_ADC_atT2_ITKSNAP.nii.gz"
    return paths[case_i] + os.sep + "coreg" + os.sep + get_prefix(paths[case_i], case) + "_ADC_DWI_500_ADC_atT2.nii.gz"

def getfun_DWI_500_sub0DWI_500_combinedWarp_0_DWI_500_atT2(paths, case_i, case):
    return paths[case_i] + os.sep + get_prefix(paths[case_i], case) + os.sep + "coreg" + os.sep + get_prefix(paths[case_i], case) + "_DWI_500_sub0DWI_500_combinedWarp_0_DWI_500_atT2.nii.gz"
def getfun_DWI_1500_sub0DWI_1500_combinedWarp_0_DWI_1500_atT2(paths, case_i, case):
    return paths[case_i] + os.sep + get_prefix(paths[case_i], case) + os.sep + "coreg" + os.sep + get_prefix(paths[case_i], case) + "_DWI_1500_sub0DWI_1500_combinedWarp_0_DWI_1500_atT2.nii.gz"
def getfun_DWI_2000_sub0DWI_2000_combinedWarp_0_DWI_2000_atT2(paths, case_i, case):
    return paths[case_i] + os.sep + get_prefix(paths[case_i], case) + os.sep + "coreg" + os.sep + get_prefix(paths[case_i], case) + "_DWI_2000_sub0DWI_2000_combinedWarp_0_DWI_2000_atT2.nii.gz"
def getfun_Gabor_freq_0_2_dist_32_0_std_1_0_ADC_DWI_500Gabor_freq_0_2_dist_32_0_std_1_0_ADC_atT2(paths, case_i, case):
    return paths[case_i] + os.sep + get_prefix(paths[case_i], case) + os.sep + "coreg" + os.sep + get_prefix(paths[case_i], case) + "_Gabor_freq_0_2_dist_32_0_std_1_0_ADC_DWI_500Gabor_freq_0_2_dist_32_0_std_1_0_ADC_atT2.nii.gz"
def getfun_Gabor_freq_0_2_dist_32_0_std_2_0_ADC_DWI_500Gabor_freq_0_2_dist_32_0_std_2_0_ADC_atT2(paths, case_i, case):
    return paths[case_i] + os.sep + get_prefix(paths[case_i], case) + os.sep + "coreg" + os.sep + get_prefix(paths[case_i], case) + "_Gabor_freq_0_2_dist_32_0_std_2_0_ADC_DWI_500Gabor_freq_0_2_dist_32_0_std_2_0_ADC_atT2.nii.gz"
def getfun_Gabor_freq_0_3_dist_32_0_std_1_0_ADC_DWI_500Gabor_freq_0_3_dist_32_0_std_1_0_ADC_atT2(paths, case_i, case):
    return paths[case_i] + os.sep + get_prefix(paths[case_i], case) + os.sep + "coreg" + os.sep + get_prefix(paths[case_i], case) + "_Gabor_freq_0_3_dist_32_0_std_1_0_ADC_DWI_500Gabor_freq_0_3_dist_32_0_std_1_0_ADC_atT2.nii.gz"
def getfun_Gabor_freq_0_3_dist_32_0_std_2_0_ADC_DWI_500Gabor_freq_0_3_dist_32_0_std_2_0_ADC_atT2(paths, case_i, case):
    return paths[case_i] + os.sep + get_prefix(paths[case_i], case) + os.sep + "coreg" + os.sep + get_prefix(paths[case_i], case) + "_Gabor_freq_0_3_dist_32_0_std_2_0_ADC_DWI_500Gabor_freq_0_3_dist_32_0_std_2_0_ADC_atT2.nii.gz"
def getfun_Gabor_freq_0_7_ADC_nonresliced_DWI_500Gabor_freq_0_7_ADC_nonresliced_atT2(paths, case_i, case):
    return paths[case_i] + os.sep + get_prefix(paths[case_i], case) + os.sep + "coreg" + os.sep + get_prefix(paths[case_i], case) + "_Gabor_freq_0_7_ADC_nonresliced_DWI_500Gabor_freq_0_7_ADC_nonresliced_atT2.nii.gz"
def getfun_Gabor_freq_0_7_DWI_nonresliced_DWI_500Gabor_freq_0_7_DWI_nonresliced_atT2(paths, case_i, case):
    return paths[case_i] + os.sep + get_prefix(paths[case_i], case) + os.sep + "coreg" + os.sep + get_prefix(paths[case_i], case) + "_Gabor_freq_0_7_DWI_nonresliced_DWI_500Gabor_freq_0_7_DWI_nonresliced_atT2.nii.gz"
def getfun_Gabor_freq_0_8_DWI_1500_div_N_nonresliced_DWI_1500Gabor_freq_0_8_DWI_1500_div_N_nonresliced_atT2(paths, case_i, case):
    return paths[case_i] + os.sep + get_prefix(paths[case_i], case) + os.sep + "coreg" + os.sep + get_prefix(paths[case_i], case) + "_Gabor_freq_0_8_DWI_1500_div_N_nonresliced_DWI_1500Gabor_freq_0_8_DWI_1500_div_N_nonresliced_atT2.nii.gz"
def getfun_Gabor_freq_0_8_DWI_2000_div_N_nonresliced_DWI_2000Gabor_freq_0_8_DWI_2000_div_N_nonresliced_atT2(paths, case_i, case):
    return paths[case_i] + os.sep + get_prefix(paths[case_i], case) + os.sep + "coreg" + os.sep + get_prefix(paths[case_i], case) + "_Gabor_freq_0_8_DWI_2000_div_N_nonresliced_DWI_2000Gabor_freq_0_8_DWI_2000_div_N_nonresliced_atT2.nii.gz"
def getfun_Gabor_freq_0_9_DWI_nonresliced_DWI_500Gabor_freq_0_9_DWI_nonresliced_atT2(paths, case_i, case):
    return paths[case_i] + os.sep + get_prefix(paths[case_i], case) + os.sep + "coreg" + os.sep + get_prefix(paths[case_i], case) + "_Gabor_freq_0_9_DWI_nonresliced_DWI_500Gabor_freq_0_9_DWI_nonresliced_atT2.nii.gz"
def getfun_LBP_a16_r2_ADC_DWI_500LBP_a16_r2_ADC_atT2(paths, case_i, case):
    return paths[case_i] + os.sep + get_prefix(paths[case_i], case) + os.sep + "coreg" + os.sep + get_prefix(paths[case_i], case) + "_LBP_a16_r2_ADC_DWI_500LBP_a16_r2_ADC_atT2.nii.gz"
def getfun_LBP_a16_r2_DWI_DWI_500LBP_a16_r2_DWI_atT2(paths, case_i, case):
    return paths[case_i] + os.sep + get_prefix(paths[case_i], case) + os.sep + "coreg" + os.sep + get_prefix(paths[case_i], case) + "_LBP_a16_r2_DWI_DWI_500LBP_a16_r2_DWI_atT2.nii.gz"

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


def Meshlab_hausdorf(in_file, in_file2, out_file):
    # Add input mesh
    command = "C:/Program Files/VCG/Meshlab/meshlabserver -i " + in_file
    command += " -i " + in_file2
    # Add the filter script
    command += " -s J:/IMPROD_PROSTATE_TEXTURES/Scripts/meshlab_smooth.mlx"
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
    command += " -s J:/IMPROD_PROSTATE_TEXTURES/Scripts/meshlab_smooth.mlx"
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
    command += " -s J:/IMPROD_PROSTATE_TEXTURES/Scripts/meshlab_smooth2.mlx"
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
        c = Idata[avg_loc_x, avg_loc_y, avg_loc_z]
        c_all.append(c)
    return c_all


def get_mesh_surface_GLCM(verts, faces, Idata, resolution):
    c_all = get_mesh_surface_samples(verts, faces, Idata, resolution)
    # make histogram of samples
    c_hist, c_bin_edges = np.histogram(c_all, bins=20)
    
    for face_i in range(len(faces)): 
        avg_loc_x = int(round(np.mean([verts[face[0]][0], verts[face[1]][0], verts[face[2]][0]])/resolution[0]))
        avg_loc_y = int(round(np.mean([verts[face[0]][1], verts[face[1]][1], verts[face[2]][1]])/resolution[1]))
        avg_loc_z = int(round(np.mean([verts[face[0]][2], verts[face[1]][2], verts[face[2]][2]])/resolution[2]))
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
        faces_for_ply.append(([face[0], face[1], face[2]], c, c, c))
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
    plyfilename = 'C:/temp/temp_raw.ply'
    PlyData([elv, elf], text=True).write(plyfilename)

    return verts, faces, normals, values


def closest_point_on_line(a, b, p):
    ap = p-a
    ab = b-a
    result = a + np.dot(ap,ab)/np.dot(ab,ab) * ab
    return result


def create_mesh_smooth(Idata, data, l, resolution, name):
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
    
    plyfilename = 'C:/temp/' + name + 'temp_raw.ply'
    write_plyfile(verts, faces, Idata, resolution, plyfilename)
    plyfilenamep = 'C:/temp/' + name + 'temp.ply'
    
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
    write_plyfile(verts, faces, Idata, resolution, 'C:/temp/' + name + 'temp_preprocessed.ply')
    
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
        
    plyfilename = 'C:/temp/' + name + 'temp_raw.ply'
    write_plyfile(verts, faces, Idata, resolution, plyfilename)
    plyfilenamep = 'C:/temp/' + name + 'temp_preprocessed.ply'
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
    print(Idata.shape)
    Gdata_x, Gdata_y, Gdata_z = np.gradient(Idata)
    print((verts, faces))
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
    
    plyfilename = 'C:/temp/' + name + 'temp_preprocessed_opt.ply'
    write_plyfile(verts, faces, Idata, resolution, plyfilename)

    plyfilename = 'C:/temp/' + name + 'temp_preprocessed_opt.ply'
    plyfilenamep = 'C:/temp/' + name + 'temp.ply'
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
    plyfilename = 'C:/temp/' + name + 'temp_preprocessed_opt_sm.ply'
    write_plyfile(verts, faces, Idata, resolution, plyfilename)
    
    return verts, faces


# Features for each
casefun_3D_shape_names = ('sarea3D','relsarea3D', 'tm_area_faces', 'tm_relarea_faces', 'mean_angles', 'median_angles', 'SD_angles', 'distance_mean', 'distance_median', 'CSM_mean_curvature', 'CSM_Gaus_mean_curvature', 'WG_median', 'WG_SD', 'WG_skewness', 'WG_kurtosis')
def casefun_3D_shape(LESIONDATAr, LESIONr, WGr, resolution):
    ROIdata = LESIONDATAr[LESIONr[0] > 0]
    print((len(ROIdata), np.min(ROIdata), np.max(ROIdata)))
    
    try:
        verts, faces, normals, values = create_mesh(LESIONr[0], 1.0, resolution)
    except:
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    sarea = skimage.measure.mesh_surface_area(verts, faces)    
    vertsw, facesw, normalsw, valuesw = create_mesh(WGr, 1.0, resolution)
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
    print((len(ROIdata), np.min(ROIdata), np.max(ROIdata)))

    # Whole Gland
    try:
        vertsw, facesw = create_mesh_smooth(LESIONDATAr, WGr, 1.0, resolution, 'WG')
    except:
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0    
    if len(vertsw) == 0:
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0    
    tmw = trimesh.base.Trimesh(vertices=vertsw, faces=facesw)

    # Lesion1, then other lesions, if found
    try:
        verts_all, faces_all = create_mesh_smooth(LESIONDATAr, LESIONr[1], 1.0, resolution, 'L1')
    except:
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0    
    if len(verts_all) == 0:
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0    
    print((len(verts_all), len(faces_all)))
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
    print((len(ROIdata), np.min(ROIdata), np.max(ROIdata)))

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
    ROIdata = LESIONDATAr[LESIONr[0] > 0]
    print((len(ROIdata), np.min(ROIdata), np.max(ROIdata)))

    vertsw, facesw = create_mesh_smooth(LESIONDATAr, WGr, 1.0, resolution, 'WG')
    tmw = trimesh.base.Trimesh(vertices=vertsw, faces=facesw)

    verts_all, faces_all = create_mesh_smooth(LESIONDATAr, LESIONr[1], 1.0, resolution, 'L1')
    if len(verts_all) == 0:
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    Surface_data = get_mesh_surface_samples(verts_all, faces_all, LESIONDATAr, resolution)
    for LESION_i in range(2, len(LESIONr)):
        try:
            verts, faces = create_mesh_smooth(LESIONDATAr, LESIONr[LESION_i], 1.0, resolution, 'L' + str(LESION_i-1))
            c = get_mesh_surface_samples(verts_all, faces_all, LESIONDATAr, resolution)
        except:
            continue    
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
    print('otsu th:' + str(bins[final_thresh]))
    return bins[final_thresh]


##########################
# Statistical functions #
##########################

#rpackage_pROC = importr('pROC')
#def fun_pROC_AUC(y_true, y):
#    try:
#        r_y = ro.conversion.py2ri(y)
#        r_y_true = ro.conversion.py2ri(y_true)
#        r_rocobj = rpackage_pROC.roc(r_y_true, r_y)
#        r_ci = rpackage_pROC.ci(r_rocobj, of="auc", conf_level=0.95, method="delong")
#        #r_auc = rpackage_pROC.auc(r_rocobj)
#        #r_auc_bs = rpackage_pROC.auc(r_rocobj, progress=ro.r.getOption("pROCProgress"), boot_stratified=False, boot_n=2000, parallel=True)
#        ret =  np.asarray(r_ci)
#    except:
#        raise
#    return ret

def load_ASCII(path, case):
    #print("Loading " + path)
    f = open(path, 'r')
    lines = f.readlines()
    data = []
    for line_i in range(1,len(lines)):
        if len(lines[line_i].strip()) == 0:
            continue
        val = float(lines[line_i].strip())
        data.append(val)
    return data


def reslice2Nifti_T(data, outputpath, orig_resolution, new_resolution):
    zooms = orig_resolution
    new_zooms = (new_resolution[0], new_resolution[1], new_resolution[2])
    affine = np.eye(4)
    affine[0,0] = orig_resolution[0]
    affine[1,1] = orig_resolution[1]
    affine[2,2] = orig_resolution[2]
    print(data.shape)
    print(zooms)
    print(new_zooms)
    data2, affine2 = reslice(data, affine, zooms, new_zooms, order=0)
    data3 = np.zeros_like(data2)
    for zi in range(data3.shape[2]):
        data3[:, :, zi] = np.rot90(data2[:, :, zi], k=3)
    return outputpath, data3, affine2


def reslice2Nifti(data, outputpath, orig_resolution, new_resolution):
    zooms = orig_resolution
    new_zooms = (new_resolution[0], new_resolution[1], new_resolution[2])
    affine = np.eye(4)
    affine[0,0] = orig_resolution[0]
    affine[1,1] = orig_resolution[1]
    affine[2,2] = orig_resolution[2]
    #print(str(affine))
    img = nib.Nifti1Image(data, affine)
    #nib.save(img, outputpath)
    data2, affine2 = reslice(data, affine, zooms, new_zooms, order=0)
    return outputpath, data2, affine2


##########################
# LESION PROCESSING FUNS #
##########################



def fun_lesionproc_Otsu_fg(RALPr, LESIONr, LESIONmasks, WGr, DATAr, casename, prefix, GT_data, case):
    th = otsu(DATAr[LESIONr > 0])
    LESIONfg = np.where(DATAr > th, LESIONr, 0)
    print('FG:' + str(len(LESIONfg[LESIONfg > 0])))
    for LESIONmask_i in range(len(LESIONmasks)):
        LESIONmasks[LESIONmask_i] = np.where(DATAr > th, LESIONmasks[LESIONmask_i], 0)
    return LESIONfg, LESIONmasks, -1


def fun_lesionproc_Otsu_bg(RALPr, LESIONr, LESIONmasks, WGr, DATAr, casename, prefix, GT_data, case):
    th = otsu(DATAr[LESIONr > 0])
    LESIONbg = np.where(DATAr <= th, LESIONr, 0)
    print('BG:' + str(len(LESIONbg[LESIONbg > 0])))
    for LESIONmask_i in range(len(LESIONmasks)):
        LESIONmasks[LESIONmask_i] = np.where(DATAr <= th, LESIONmasks[LESIONmask_i], 0)
    return LESIONbg, LESIONmasks, -1


def fun_lesionproc_noop(RALPr, LESIONr, LESIONmasks, WGr, DATAr, casename, prefix, GT_data, case):
    return LESIONr, LESIONmasks, -1


def subfun_dilation(mask, iters):
    # find edge voxels
    outmask = np.zeros_like(mask)
    blocs = []
    queue = []
    for zi in range(1, mask.shape[2]-1):
        if np.max(mask[:,:,zi]) > 0:
            blocs.append({'z':zi, 'blocs':[], 'center':(-1,-1,-1)})
    for bi in range(len(blocs)):
        zi = blocs[bi]['z']
        for yi in range(1, mask.shape[1]-1):
            for xi in range(1, mask.shape[0]-1):
                if mask[xi,yi,zi] > 0:
                    outmask[xi,yi,zi] = 1
                    if np.sum(mask[xi-1:xi+1, yi-1:yi+1, zi-1:zi+1]) < 27:
                        blocs[bi]['z'].append((xi, yi, zi))
                        queue.append((xi, yi, zi))

    for zi in range(mask.shape[2]):
        z_blocs = blocs[zi]['blocs']
        center = (0,0,0)
        for li in range(len(z_blocs)):
            center[0] = center[0] + z_blocs[li][0]
            center[1] = center[1] + z_blocs[li][1]
            center[2] = center[2] + z_blocs[li][2]
        center[0] = center[0]/float(len(z_blocs))
        center[1] = center[1]/float(len(z_blocs))
        center[2] = center[2]/float(len(z_blocs))        
        blocs[zi] = {'z':zi, 'blocs':z_blocs, 'center':center}

#    boutmask = ndimage.morphology.binary_dilation(mask, iterations=iters+1).astype("uint8")

    while(len(queue) > 0):
        print(len(queue))
        loc = queue.popleft()
        if loc[0] == 0 or loc[1] == 0 or loc[2] == 0:
            continue
        if loc[0] == mask.shape[0]-1 or loc[1] == mask.shape[1]-1 or loc[2] == mask.shape[2]-1:
            continue
        xi = loc[0]
        yi = loc[1]
        zi = loc[2]
        for xii in [-1,0,1]:
            for yii in [-1,0,1]:
                for zii in [-1,0,1]:
                    ng = (xi+xii, yi+yii, zi+zii)
                    if(mask[ng[0], ng[1], ng[2]] > 0):
                        continue
                    closest_dist = 1000
                    z_bloc_i = -1
                    for li in range(len(z_blocs)):
                        center = z_blocs[li]['center']
                        dist = np.sqrt(np.power(center[0]-ng[0], 2.0)+np.power(center[1]-ng[1], 2.0)+np.power(center[2]-ng[2], 2.0))
                        if dist < closest_dist:
                            closest_dist = dist
                            z_bloc_i = li
                    z_bloc_locs = z_blocs[z_bloc_i]['blocs']
                    closest_dist = 1000
                    for li in range(len(z_bloc_locs)):
                        z_bloc = z_bloc_locs[li]
                        dist = np.sqrt(np.power(z_bloc[0]-ng[0], 2.0)+np.power(z_bloc[1]-ng[1], 2.0)+np.power(z_bloc[2]-ng[2], 2.0))
                        if dist < closest_dist:
                            closest_dist = dist
                    if closest_dist > iters:
                        continue
                    # Add new point                      
                    outmask[ng[0], ng[1], ng[2]] = 1
                    queue.append((ng[0], ng[1], ng[2]))

    return outmask


dilateLesionMax_iterations = 25

def fun_lesionproc_voxelwise_estimate_dilateLesion099(RALPr, LESIONr, WGr, DATAr, casename, prefix, GT_data, case):
    
    valuesRALP = RALPr[WGr > 0]
    valuesRALP = np.where(valuesRALP > 0, 1, 0)
    y_true = valuesRALP

    required_iterations = 0
    for iters in range(1,dilateLesionMax_iterations):
#        LESIONr_dil = subfun_dilation(LESIONr, iters)
        LESIONr_dil = ndimage.morphology.binary_dilation(LESIONr, iterations=iters).astype("uint8")
        valuesLESION = LESIONr_dil[WGr > 0]
        valuesLESION = np.where(valuesLESION > 0, 1, 0)
        y = valuesLESION
        Ntot, Npos, Nneg, eNpos, eNneg, TN, FP, FN, TP, AUC, acc, sens, spec = calculate_classification_stats(y_true, y)
        #print((iters, Ntot, Npos, Nneg, eNpos, eNneg, TN, FP, FN, TP, AUC, acc, sens, spec))
        if spec == 1.0 and sens == 0.0:
            required_iterations = dilateLesionMax_iterations
            break        
        if sens >= 0.99:
            if spec == 0.0:
                required_iterations = dilateLesionMax_iterations
                break
            required_iterations = iters
            break
    if required_iterations == 0:
        return LESIONr, 0
    return LESIONr_dil, required_iterations


def fun_lesionproc_voxelwise_estimate_dilateLesion100(RALPr, LESIONr, WGr, DATAr, casename, prefix, GT_data, case):
    
    valuesRALP = RALPr[WGr > 0]
    valuesRALP = np.where(valuesRALP > 0, 1, 0)
    y_true = valuesRALP

    required_iterations = 0
    for iters in range(1,dilateLesionMax_iterations):
        LESIONr_dil = ndimage.morphology.binary_dilation(LESIONr, iterations=iters).astype("uint8")
        valuesLESION = LESIONr_dil[WGr > 0]
        valuesLESION = np.where(valuesLESION > 0, 1, 0)
        y = valuesLESION
        Ntot, Npos, Nneg, eNpos, eNneg, TN, FP, FN, TP, AUC, acc, sens, spec = calculate_classification_stats(y_true, y)
        #print((iters, Ntot, Npos, Nneg, eNpos, eNneg, TN, FP, FN, TP, AUC, acc, sens, spec))
        if spec == 1.0 and sens == 0.0:
            required_iterations = dilateLesionMax_iterations
            break        
        if sens >= 1.00:
            if spec == 0.0:
                required_iterations = dilateLesionMax_iterations
                break
            required_iterations = iters
            break
    if required_iterations == 0:
        return LESIONr, 0
    return LESIONr_dil, required_iterations


def fun_lesionproc_voxelwise_estimate_dilateLesion095(RALPr, LESIONr, WGr, DATAr, casename, prefix, GT_data, case):
    
    valuesRALP = RALPr[WGr > 0]
    valuesRALP = np.where(valuesRALP > 0, 1, 0)
    y_true = valuesRALP

    required_iterations = 0
    for iters in range(1,dilateLesionMax_iterations):
        LESIONr_dil = ndimage.morphology.binary_dilation(LESIONr, iterations=iters).astype("uint8")
        valuesLESION = LESIONr_dil[WGr > 0]
        valuesLESION = np.where(valuesLESION > 0, 1, 0)
        y = valuesLESION
        Ntot, Npos, Nneg, eNpos, eNneg, TN, FP, FN, TP, AUC, acc, sens, spec = calculate_classification_stats(y_true, y)
        print((iters, Ntot, Npos, Nneg, eNpos, eNneg, TN, FP, FN, TP, AUC, acc, sens, spec))
        if spec == 1.0 and sens == 0.0:
            required_iterations = dilateLesionMax_iterations
            break        
        if sens >= 0.95:
            if spec == 0.0:
                required_iterations = dilateLesionMax_iterations
                break
            required_iterations = iters
            break
    if required_iterations == 0:
        return LESIONr, 0
    return LESIONr_dil, required_iterations


def sphere(shape, radius, position):
    # assume shape and position are both a 3-tuple of int or float
    # the units are pixels / voxels (px for short)
    # radius is a int or float in px
    semisizes = (radius,) * 3

    # genereate the grid for the support points
    # centered at the position indicated by position
    grid = [slice(-x0, dim - x0) for x0, dim in zip(position, shape)]
    position = np.ogrid[grid]
    # calculate the distance of all points from `position` center
    # scaled by the radius
    arr = np.zeros(shape, dtype=float)
    for x_i, semisize in zip(position, semisizes):
        arr += (np.abs(x_i / semisize) ** 2)
    # the inner part of the sphere will have distance below 1
    return arr <= 1.0


def fun_lesionproc_voxelwise_estimate_dilateLesion095_Sphere(RALPr, LESIONr, WGr, DATAr, casename, prefix, GT_data, case):

    valuesRALP = RALPr[WGr > 0]
    valuesRALP = np.where(valuesRALP > 0, 1, 0)
    y_true = valuesRALP

    required_iterations = 0    
    kernel_shape = (2*dilateLesionMax_iterations+1, 2*dilateLesionMax_iterations+1, 2*dilateLesionMax_iterations+1)
    sphere_shape = (dilateLesionMax_iterations,dilateLesionMax_iterations,dilateLesionMax_iterations)
    for iters in range(1,dilateLesionMax_iterations):
        kernel = sphere(kernel_shape, iters, sphere_shape)
        LESIONr_dil = scipy.ndimage.binary_dilation(LESIONr, structure=kernel).astype("uint8")
        valuesLESION = LESIONr_dil[WGr > 0]
        valuesLESION = np.where(valuesLESION > 0, 1, 0)
        y = valuesLESION
        Ntot, Npos, Nneg, eNpos, eNneg, TN, FP, FN, TP, AUC, acc, sens, spec = calculate_classification_stats(y_true, y)
        print((iters, Ntot, Npos, Nneg, eNpos, eNneg, TN, FP, FN, TP, AUC, acc, sens, spec))
        if spec == 1.0 and sens == 0.0:
            required_iterations = dilateLesionMax_iterations
            break        
        if sens >= 0.95:
            if spec == 0.0:
                required_iterations = dilateLesionMax_iterations
                break
            required_iterations = iters
            break
    if required_iterations == 0:
        return LESIONr, 0
    return LESIONr_dil, required_iterations


def fun_lesionproc_voxelwise_estimate_dilateLesion099_Sphere(RALPr, LESIONr, WGr, DATAr, casename, prefix, GT_data, case):
    
    valuesRALP = RALPr[WGr > 0]
    valuesRALP = np.where(valuesRALP > 0, 1, 0)
    y_true = valuesRALP

    required_iterations = 0
    kernel_shape = (2*dilateLesionMax_iterations+1, 2*dilateLesionMax_iterations+1, 2*dilateLesionMax_iterations+1)
    sphere_shape = (dilateLesionMax_iterations,dilateLesionMax_iterations,dilateLesionMax_iterations)
    for iters in range(1,dilateLesionMax_iterations):
        kernel = sphere(kernel_shape, iters, sphere_shape)
        LESIONr_dil = scipy.ndimage.binary_dilation(LESIONr, structure=kernel).astype("uint8")
        valuesLESION = LESIONr_dil[WGr > 0]
        valuesLESION = np.where(valuesLESION > 0, 1, 0)
        y = valuesLESION
        Ntot, Npos, Nneg, eNpos, eNneg, TN, FP, FN, TP, AUC, acc, sens, spec = calculate_classification_stats(y_true, y)
        print((iters, Ntot, Npos, Nneg, eNpos, eNneg, TN, FP, FN, TP, AUC, acc, sens, spec))
        if spec == 1.0 and sens == 0.0:
            required_iterations = 25
            break        
        if sens >= 0.99:
            if spec == 0.0:
                required_iterations = 25
                break
            required_iterations = iters
            break
    if required_iterations == 0:
        return LESIONr, 0
    return LESIONr_dil, required_iterations


def fun_lesionproc_voxelwise_estimate_dilateLesion100_Sphere(RALPr, LESIONr, WGr, DATAr, casename, prefix, GT_data, case):

    valuesRALP = RALPr[WGr > 0]
    valuesRALP = np.where(valuesRALP > 0, 1, 0)
    y_true = valuesRALP

    required_iterations = 0
    kernel_shape = (2*dilateLesionMax_iterations+1, 2*dilateLesionMax_iterations+1, 2*dilateLesionMax_iterations+1)
    sphere_shape = (dilateLesionMax_iterations,dilateLesionMax_iterations,dilateLesionMax_iterations)
    for iters in range(1,dilateLesionMax_iterations):
        kernel = sphere(kernel_shape, iters, sphere_shape)
        LESIONr_dil = scipy.ndimage.binary_dilation(LESIONr, structure=kernel).astype("uint8")
        valuesLESION = LESIONr_dil[WGr > 0]
        valuesLESION = np.where(valuesLESION > 0, 1, 0)
        y = valuesLESION
        Ntot, Npos, Nneg, eNpos, eNneg, TN, FP, FN, TP, AUC, acc, sens, spec = calculate_classification_stats(y_true, y)
        print((iters, Ntot, Npos, Nneg, eNpos, eNneg, TN, FP, FN, TP, AUC, acc, sens, spec))
        if spec == 1.0 and sens == 0.0:
            required_iterations = 25
            break        
        if sens >= 1.00:
            if spec == 0.0:
                required_iterations = 25
                break
            required_iterations = iters
            break
    if required_iterations == 0:
        return LESIONr, 0
    return LESIONr_dil, required_iterations



def fun_lesionproc_voxelwise_dilateLesion_1voxel(RALPr, LESIONr, WGr, DATAr, casename, prefix, GT_data, case):
    
    valuesRALP = RALPr[WGr > 0]
    valuesRALP = np.where(valuesRALP > 0, 1, 0)
    LESIONr = ndimage.morphology.binary_dilation(LESIONr).astype("uint8")
    return LESIONr, 0


def fun_lesionproc_voxelwise_dilateLesion_2voxels(RALPr, LESIONr, WGr, DATAr, casename, prefix, GT_data, case):
    
    valuesRALP = RALPr[WGr > 0]
    valuesRALP = np.where(valuesRALP > 0, 1, 0)
    LESIONr = ndimage.morphology.binary_dilation(LESIONr, iterations=2).astype("uint8")
    return LESIONr, 0


def fun_lesionproc_voxelwise_dilateLesion_3voxels(RALPr, LESIONr, WGr, DATAr, casename, prefix, GT_data, case):
    
    valuesRALP = RALPr[WGr > 0]
    valuesRALP = np.where(valuesRALP > 0, 1, 0)
    LESIONr = ndimage.morphology.binary_dilation(LESIONr, iterations=3).astype("uint8")
    return LESIONr, 0


def fun_lesionproc_voxelwise_dilateLesion_4voxels(RALPr, LESIONr, WGr, DATAr, casename, prefix, GT_data, case):
    
    valuesRALP = RALPr[WGr > 0]
    valuesRALP = np.where(valuesRALP > 0, 1, 0)
    LESIONr = ndimage.morphology.binary_dilation(LESIONr, iterations=4).astype("uint8")
    return LESIONr, 0


def fun_lesionproc_voxelwise_dilateLesion_5voxels(RALPr, LESIONr, WGr, DATAr, casename, prefix, GT_data, case):
    
    valuesRALP = RALPr[WGr > 0]
    valuesRALP = np.where(valuesRALP > 0, 1, 0)
    LESIONr = ndimage.morphology.binary_dilation(LESIONr, iterations=5).astype("uint8")
    return LESIONr, 0


def fun_lesionproc_voxelwise_dilateLesion_10voxels(RALPr, LESIONr, WGr, DATAr, casename, prefix, GT_data, case):
    
    valuesRALP = RALPr[WGr > 0]
    valuesRALP = np.where(valuesRALP > 0, 1, 0)
    LESIONr = ndimage.morphology.binary_dilation(LESIONr, iterations=10).astype("uint8")
    return LESIONr, 0


# Value pairs for intensity values and RALP segmentation
def fun_Ivoxelwise(RALPr, LESIONr, WGr, DATAr, casename, prefix, GT_data, case):

    valuesRALP = RALPr[WGr > 0]
    valuesRALP = np.where(valuesRALP > 0, 1, 0)
    valuesLESION = LESIONr[WGr > 0]
    valuesLESION = np.where(valuesLESION > 0, 1, 0)
    valuesDATA = DATAr[WGr > 0]
    
    y = valuesDATA
    y_true = valuesRALP
    return y, y_true, ''


def fun_ROIcomp_voxelwise(RALPr, LESIONr, WGr, DATAr, casename, prefix, GT_data, case):
    
    print('---fun_ROIcomp_voxelwise---')
    print(RALPr.shape)
    print(LESIONr.shape)
    print(WGr.shape)
    print(DATAr.shape)
    print('---------------------------')
    valuesRALP = RALPr[WGr > 0]
    valuesRALP = np.where(valuesRALP > 0, 1, 0)
    valuesLESION = LESIONr[WGr > 0]
    valuesLESION = np.where(valuesLESION > 0, 1, 0)

    y = valuesLESION
    y_true = valuesRALP
    return y, y_true, ''


def fun_36seg_binarize(field):
    binarized = [0]*36
#    if field == 'x':
#        return None
    if field == '0' or field == 'x':
        return np.array(binarized)
    segstrs = field.split(',')
    for segstr in segstrs:
        if len(segstr.strip()) == 0:
            continue
        recognized = 0
        for segname_i in range(len(seg36_all_segnames)):
            if segstr.strip() == seg36_all_segnames[segname_i]:
                binarized[segname_i] = 1
                recognized = 1
                break
        if recognized == 0:
            raise Exception("String " + segstr + " was not recognized from GT file")
    #print(field)
    #print(binarized)
    return np.array(binarized)


def fun_36seg_33_vs_rest(RALPr, LESIONr, WGr, DATAr, casename, prefix, GT_data, case):
    # special handling for combination of L1-L4
    if prefix == 'L1L2L3L4':
        y, y_true, add_str = fun_36seg(RALPr, LESIONr, WGr, DATAr, casename, 'L1', GT_data, case)
        for prefix_addition in ['L2', 'L3', 'L4']:
            y_add, y_true_add, add_str = fun_36seg(RALPr, LESIONr, WGr, DATAr, casename, prefix_addition, GT_data, case)
            y = np.where(y_add > 0, 1, y)
            y_true = np.where(y_true_add > 0, 1, y_true)
    else:
        # resolve row and columns
        case_row = -1
        for row_i in range(len(GT_data[0]['data'])):
            if GT_data[0]['data'][row_i] == case:
                case_row = row_i
                break
        col_R = -1
        for col_i in range(len(GT_data)):
            col = GT_data[col_i]
            if prefix == 'L1' and col['name'] == 'GT1_l':
                col_R = col_i
                break
            if prefix == 'L2' and col['name'] == 'GT2_l':
                col_R = col_i
                break
            if prefix == 'L3' and col['name'] == 'GT3_l':
                col_R = col_i
                break
            if prefix == 'L4' and col['name'] == 'GT4_l':
                col_R = col_i
                break
        col_L = -1
        for col_i in range(len(GT_data)):
            col = GT_data[col_i]
            if prefix == 'L1' and col['name'] == 'MRI_L1':
                col_L = col_i
                break
            if prefix == 'L2' and col['name'] == 'MRI_L2':
                col_L = col_i
                break
            if prefix == 'L3' and col['name'] == 'MRI_L3':
                col_L = col_i
                break
            if prefix == 'L4' and col['name'] == 'MRI_L4':
                col_L = col_i
                break
        y = fun_36seg_binarize(GT_data[col_L]['data'][case_row])
        y_true = fun_36seg_binarize(GT_data[col_R]['data'][case_row])
        
    return y, y_true, ''


def fun_36seg(RALPr, LESIONr, WGr, DATAr, casename, prefix, GT_data, case):
    # special handling for combination of L1-L4
    if prefix == 'L1L2L3L4':
        y, y_true, add_str = fun_36seg(RALPr, LESIONr, WGr, DATAr, casename, 'L1', GT_data, case)
        for prefix_addition in ['L2', 'L3', 'L4']:
            y_add, y_true_add, add_str = fun_36seg(RALPr, LESIONr, WGr, DATAr, casename, prefix_addition, GT_data, case)
            y = np.where(y_add > 0, 1, y)
            y_true = np.where(y_true_add > 0, 1, y_true)
    else:
        # resolve row and columns
        case_row = -1
        for row_i in range(len(GT_data[0]['data'])):
            if GT_data[0]['data'][row_i] == case:
                case_row = row_i
                break    
        col_R = -1
        for col_i in range(len(GT_data)):
            col = GT_data[col_i]
            if prefix == 'L1' and col['name'] == 'GT1_l':
                col_R = col_i
                break
            if prefix == 'L2' and col['name'] == 'GT2_l':
                col_R = col_i
                break
            if prefix == 'L3' and col['name'] == 'GT3_l':
                col_R = col_i
                break
            if prefix == 'L4' and col['name'] == 'GT4_l':
                col_R = col_i
                break
        col_L = -1
        for col_i in range(len(GT_data)):
            col = GT_data[col_i]
            if prefix == 'L1' and col['name'] == 'MRI_L1':
                col_L = col_i
                break
            if prefix == 'L2' and col['name'] == 'MRI_L2':
                col_L = col_i
                break
            if prefix == 'L3' and col['name'] == 'MRI_L3':
                col_L = col_i
                break
            if prefix == 'L4' and col['name'] == 'MRI_L4':
                col_L = col_i
                break
        y = fun_36seg_binarize(GT_data[col_L]['data'][case_row])
        y_true = fun_36seg_binarize(GT_data[col_R]['data'][case_row])
        
    return y, y_true, ''


def calculate_classification_stats_continuous_variable(y_true, y):
    Ntot = len(np.array(y_true))
    
    if Ntot == 0:
        return 0, 0, 0, 0, 0, 0.5
    
    if np.max(y_true) == 0:
        Npos = 0
        Nneg = len(np.array(y_true))
    elif np.min(y_true) == 1:
        Npos = len(y_true)
        Nneg = 0
    else:
        Nneg = len(np.array(y_true[y_true == 0]))
        Npos = len(np.array(y_true[y_true > 0]))
    if np.max(y) == 0:
        eNpos = 0
        eNneg = len(np.array(y))
    elif np.min(y) == 1:
        eNpos = len(y)
        eNneg = 0
    else:
        eNneg = len(np.array(y[y == 0]))
        eNpos = len(np.array(y[y > 0]))

    if len(np.unique(np.array(y_true))) == 1 or len(np.array(y_true)) == 0:
        AUC = 0.5
    else:
        try:
            AUC = sklearn.metrics.roc_auc_score(np.array(y_true), np.array(y))
        except:
            print(np.unique(np.array(y_true)))
            raise
    if AUC < 0.5:
        try:
            AUC = sklearn.metrics.roc_auc_score(np.array(y_true), -np.array(y))
        except:
            print(np.unique(np.array(y_true)))
            raise    
    return Ntot, Npos, Nneg, eNpos, eNneg, AUC


def calculate_classification_stats(y_true, y):
    Ntot = len(np.array(y_true))
    
    if Ntot == 0:
        if len(np.array(y_true)) > 0:
            sens = 1.0
            spec = 0.0
            acc = 0.0
            AUC = 0.5
        else:
            sens = 0.0
            spec = 0.0
            acc = 0.0
            AUC = 0.5            
        # Ntot, Npos, Nneg, eNpos, eNneg, TN, FP, FN, TP, AUC, acc, sens, spec
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, AUC, acc, sens, spec
    
    if np.max(y_true) == 0:
        Npos = 0
        Nneg = len(np.array(y_true))
    elif np.min(y_true) == 1:
        Npos = len(y_true)
        Nneg = 0
    else:
        Nneg = len(np.array(y_true[y_true == 0]))
        Npos = len(np.array(y_true[y_true > 0]))
    if np.max(y) == 0:
        eNpos = 0
        eNneg = len(np.array(y))
    elif np.min(y) == 1:
        eNpos = len(y)
        eNneg = 0
    else:
        eNneg = len(np.array(y[y == 0]))
        eNpos = len(np.array(y[y > 0]))

    if len(np.unique(np.array(y_true))) == 1 or len(np.array(y_true)) == 0:
        AUC = 0.5
        if np.unique(np.array(y_true)) == 0:
            TP = 0
            TN = eNneg
            FN = 0
            FP = eNpos
            sens = 0.0
            spec = np.float(TN)/np.float(TN+FP)
        else:
            TP = eNpos
            TN = 0
            FN = eNneg
            FP = 0
            sens = np.float(TP)/np.float(TP+FN)
            spec = 0.0
        acc=(np.float(TP)+np.float(TN))/np.float(TP+FP+FN+TN)
    else:
        CM = confusion_matrix(np.array(y_true), np.array(y))
        TN = CM[0,0]
        FP = CM[0,1]
        FN = CM[1,0]
        TP = CM[1,1]
        if np.float(TP+FP+FN+TN) > 0:
            acc=(np.float(TP)+np.float(TN))/np.float(TP+FP+FN+TN)
        else:
            acc=-1.0
        if (np.float(TP)+np.float(FN)) > 0:
            sens = np.float(TP)/(np.float(TP)+np.float(FN))
        else:
            sens = 0.0
        if (np.float(TN)+np.float(FP)) > 0:
            spec = np.float(TN)/(np.float(TN)+np.float(FP))
        else:
            spec = 0.0
        try:
            AUC = sklearn.metrics.roc_auc_score(np.array(y_true), np.array(y))
        except:
            print(np.unique(np.array(y_true)))
            raise
    if AUC < 0.5:
        try:
            AUC = sklearn.metrics.roc_auc_score(np.array(y_true), -np.array(y))
        except:
            print(np.unique(np.array(y_true)))
            raise    
    return Ntot, Npos, Nneg, eNpos, eNneg, TN, FP, FN, TP, AUC, acc, sens, spec


def resolve_cases_no_RALP(statname, filtername, prefix, forced_inclusions_lesion, outputpath, output_ASCII, included_cases, datafun):

    if prefix == 'L1L2L3L4':
        casefilterprefix = 'L1'
    else:
        casefilterprefix = prefix
    
    RALP_ROI_path = 'D:\Desktop\IMPROD_PROSTATE_TEXTURES\Robotic_assisted_laparoscopic_prostatectomy_ROIs' + os.sep + casefilterprefix + '_RALP_lesion'
    LESION_ROI_path = 'D:\Desktop\IMPROD_PROSTATE_TEXTURES\Human_drawn_T2_based_ROIs' + os.sep + casefilterprefix + '_lesions'
    WG_ROI_path = 'D:\Desktop\IMPROD_PROSTATE_TEXTURES\Whole_Gland_prostate_masks\WG_prostate'
    VISUAL_T2_path = 'D:\Desktop\IMPROD_PROSTATE_TEXTURES\pipeline'
    
    # Resolve human drawn lesions
    LESIONfoldernames = glob(LESION_ROI_path +os.sep + 'DICOMmasks' + os.sep + '*')
    lesion_cases = []
    print("Lesion folder names[" + LESION_ROI_path + os.sep + "DICOMmasks" + os.sep + "*]" + str(len(LESIONfoldernames)))
    for LESIONfolder in LESIONfoldernames:
        lesion_cases.append(int(float(os.path.basename(LESIONfolder).split('-')[0])))

    # Resolve intensity data cases
    LESIONdatanames = glob(VISUAL_T2_path + os.sep + '*')
    lesion_cases_data = []
    for LESIONDATAfolder in LESIONdatanames:
        LESIONDATAbasename = os.path.basename(LESIONDATAfolder)
        #print((LESIONDATAfolder, LESIONDATAbasename))
        if not (LESIONDATAbasename[0] == "0"):
            continue
        lesion_cases_data.append(int(float(LESIONDATAbasename.split('-')[0])))
    
    # Resolve Robotic assisted laparoscopic prostatectomy ROIs             
    RALPfoldernames = glob(RALP_ROI_path + os.sep + 'DICOMmasks' + os.sep + '*')
    ralp_cases = []
    for RALPfoldername in RALPfoldernames:
        ralp_cases.append(int(float(os.path.basename(RALPfoldername).split('-')[0])))

    # Resolve whole gland ROIs                
    WGfoldernames = glob(WG_ROI_path + os.sep + 'DICOMmasks' + os.sep + '*')
    WG_cases = []
    used_cases = []
    for WGfoldername in WGfoldernames:
        WG_cases.append(int(float(os.path.basename(WGfoldername).split('-')[0])))
        used_cases.append(int(float(os.path.basename(WGfoldername).split('-')[0])))

    # Make additions of cases where there is no lesion ROI and considered as completely missed lesion
    for inclusion in forced_inclusions_lesion:
        if inclusion in used_cases:
            continue
        used_cases.append(inclusion)
    for inclusion in ralp_cases:
        if inclusion in used_cases:
            continue
        used_cases.append(inclusion)
        
    print('lesion cases:[' + LESION_ROI_path + os.sep + 'DICOMmasks' + os.sep + '*' + ']' + str(len(lesion_cases)))
    print('lesion data cases:[' + VISUAL_T2_path + os.sep + '*]' + str(len(lesion_cases_data)))
    print('RALP cases[' + RALP_ROI_path + os.sep + 'DICOMmasks' + os.sep + '*' + ']:' + str(len(ralp_cases)))
    print('WG cases:' + str(len(WG_cases)))

    all_found_cases = []
    all_found_RALPpaths = []
    all_found_LESIONpaths = []
    all_found_LESIONDATApaths = []
    all_found_WGpaths = []
    missing_ralp_cases = []
    missing_lesion_cases = []
    missing_WG_cases = []
    missing_DATA_cases = []
    RALP_but_no_human = []
    human_but_no_RALP = []
    for lesion_i in range(len(used_cases)):
        lesion = used_cases[lesion_i]
        if not lesion in included_cases:
            continue
        
        WGfound = 1
        if lesion in WG_cases:
            WG_index = WG_cases.index(lesion)
            #print(WGfoldernames[WG_index])
            all_found_WGpaths.append(WGfoldernames[WG_index])
        elif lesion in forced_inclusions_lesion:
            #print('')
            all_found_WGpaths.append('')
            missing_WG_cases.append(lesion)
            WGfound = 0
        else:
            WGfound = 0

        # Case found in all modalities
        RALPfound = 1
        if lesion in ralp_cases:
            ralp_index = ralp_cases.index(lesion)
            #print(RALPfoldernames[ralp_index])
            all_found_RALPpaths.append(RALPfoldernames[ralp_index])
        else:
            all_found_RALPpaths.append('')
            missing_ralp_cases.append(lesion)
            RALPfound = 0
            
        LESIONfound = 1
        if lesion in lesion_cases:
            lesion_index = lesion_cases.index(lesion)
            all_found_LESIONpaths.append(LESIONfoldernames[lesion_index])
        else:
            all_found_LESIONpaths.append('')
            missing_lesion_cases.append(lesion)
            LESIONfound = 0

        LESIONDATAfound = 1
        if lesion in WG_cases:
            lesion_data_i = lesion_cases_data.index(lesion)
            #print('>>' + datafun(LESIONdatanames, lesion_data_i, lesion))
            if not os.path.exists(datafun(LESIONdatanames, lesion_data_i, lesion)):
                all_found_LESIONDATApaths.append('')
                missing_DATA_cases.append(lesion)
                LESIONDATAfound = 0
            else:
                #print('adding ' + LESIONdatanames[lesion_data_i])
                all_found_LESIONDATApaths.append(LESIONdatanames[lesion_data_i])
        else:
            all_found_LESIONDATApaths.append('')
            missing_DATA_cases.append(lesion)
            LESIONDATAfound = 0

        # Sanity check case in RALP but already included by forece
        if (RALPfound == 1) and (lesion in forced_inclusions_lesion):
            print(all_found_RALPpaths)
            print(ralp_cases)
            raise Exception(prefix + ":Forced inclusion " + str(lesion) +  " already present in RALP cases")
                       
        all_found_cases.append(lesion)
        #print(lesion, RALPfound, LESIONfound, WGfound, LESIONDATAfound)
        
        
    print('RALP_but_no_human')
    print(RALP_but_no_human)
    print('human_but_no_RALP')
    print(human_but_no_RALP)
            
    print('missing lesion:' + str(missing_lesion_cases))
    f = open(output_ASCII + os.sep + filtername + '_' + statname + '_' + prefix + 'missing_lesion_cases.txt', 'w')
    for subjname in missing_lesion_cases:       
        f.write('%s\n' % (subjname))
    f.close()

    print('missing ralp:' + str(len(missing_ralp_cases)))
    f = open(output_ASCII + os.sep + filtername + '_' + statname + '_' + prefix + 'missing_ralp_cases.txt', 'w')
    for subjname in missing_ralp_cases:       
        f.write('%s\n' % (subjname))
    f.close()

    print('missing WG:' + str(len(missing_WG_cases)))
    f = open(output_ASCII + os.sep + filtername + '_' + statname + '_' + prefix + 'missing_WG_cases.txt', 'w')
    for subjname in missing_WG_cases:
        f.write('%s\n' % (subjname))
    f.close()

    print('missing data:' + str(len(missing_DATA_cases)))
    f = open(output_ASCII + os.sep + filtername + '_' + statname + '_' + prefix + 'missing_DATA_cases.txt', 'w')
    for subjname in missing_DATA_cases:       
        f.write('%s\n' % (subjname))
    f.close()

    return all_found_cases, all_found_WGpaths, all_found_RALPpaths, all_found_LESIONDATApaths, all_found_LESIONpaths, casefilterprefix


def extract_numbers_IMPROD(casefilterprefix):
    
    RALP_ROI_path = '..\Robotic_assisted_laparoscopic_prostatectomy_ROIs' + os.sep + casefilterprefix + '_RALP_lesion'
    LESION_ROI_path = '..\Human_drawn_T2_based_ROIs' + os.sep + casefilterprefix + '_lesions'
    WG_ROI_path = '..\Whole_Gland_prostate_masks\WG_prostate'
    VISUAL_T2_path = '..\pipeline' 
    # Resolve human drawn lesions
    LESIONfoldernames = glob(LESION_ROI_path +os.sep + 'DICOMmasks' + os.sep + '*')
    lesion_cases = []
    used_cases = []
    print("Lesion folder names[" + LESION_ROI_path + os.sep + "DICOMmasks" + os.sep + "*]" + str(len(LESIONfoldernames)))
    for LESIONfolder in LESIONfoldernames:
        lesion_cases.append(int(float(os.path.basename(LESIONfolder).split('-')[0])))
        used_cases.append(int(float(os.path.basename(LESIONfolder).split('-')[0])))

    # Resolve intensity data cases
    LESIONdatanames_glob = glob(VISUAL_T2_path + os.sep + '*')
    LESIONdatanames = []
    lesion_cases_data = []
    for LESIONDATAfolder in LESIONdatanames_glob:
        LESIONDATAbasename = os.path.basename(LESIONDATAfolder)
        if not (LESIONDATAbasename[0] == "0"):
            continue
        #print((LESIONDATAfolder, LESIONDATAbasename, int(float(LESIONDATAbasename.split('-')[0]))))
        lesion_cases_data.append(int(float(LESIONDATAbasename.split('-')[0])))
        LESIONdatanames.append(LESIONDATAfolder)
    
    # Resolve Robotic assisted laparoscopic prostatectomy ROIs
    RALPfoldernames = glob(RALP_ROI_path + os.sep + 'DICOMmasks' + os.sep + '*')
    ralp_cases = []
    for RALPfoldername in RALPfoldernames:
        ralp_cases.append(int(float(os.path.basename(RALPfoldername).split('-')[0])))

    # Resolve whole gland ROIs                
    WGfoldernames = glob(WG_ROI_path + os.sep + 'DICOMmasks' + os.sep + '*')
    WG_cases = []
    for WGfoldername in WGfoldernames:
        WG_cases.append(int(float(os.path.basename(WGfoldername).split('-')[0])))
    return used_cases, ralp_cases, WG_cases, lesion_cases, lesion_cases_data, RALPfoldernames, LESIONfoldernames, LESIONdatanames, WGfoldernames


def extract_numbers_IMPROD_DWI(casefilterprefix):
    
    RALP_ROI_path = '..\Robotic_assisted_laparoscopic_prostatectomy_ROIs' + os.sep + casefilterprefix + '_RALP_lesion'
    LESION_ROI_path = '..\Human_drawn_T2_based_ROIs' + os.sep + 'ADCm0_500_human_' + casefilterprefix
    WG_ROI_path = '..\Human_drawn_T2_based_ROIs' + os.sep + 'ADCm0_500_WG'
    VISUAL_T2_path = '..\pipeline' 
    # Resolve human drawn lesions
    LESIONfoldernames = glob(LESION_ROI_path +os.sep + 'DICOMmasks' + os.sep + '*')
    lesion_cases = []
    used_cases = []
    print("Lesion folder names[" + LESION_ROI_path + os.sep + "DICOMmasks" + os.sep + "*]" + str(len(LESIONfoldernames)))
    for LESIONfolder in LESIONfoldernames:
        lesion_cases.append(int(float(os.path.basename(LESIONfolder).split('-')[0])))
        used_cases.append(int(float(os.path.basename(LESIONfolder).split('-')[0])))

    # Resolve intensity data cases
    LESIONdatanames_glob = glob(VISUAL_T2_path + os.sep + '*')
    LESIONdatanames = []
    lesion_cases_data = []
    for LESIONDATAfolder in LESIONdatanames_glob:
        LESIONDATAbasename = os.path.basename(LESIONDATAfolder)
        if not (LESIONDATAbasename[0] == "0"):
            continue
        #print((LESIONDATAfolder, LESIONDATAbasename, int(float(LESIONDATAbasename.split('-')[0]))))
        lesion_cases_data.append(int(float(LESIONDATAbasename.split('-')[0])))
        LESIONdatanames.append(LESIONDATAfolder)
    
    # Resolve Robotic assisted laparoscopic prostatectomy ROIs             
    RALPfoldernames = glob(RALP_ROI_path + os.sep + 'DICOMmasks' + os.sep + '*')
    ralp_cases = []
    for RALPfoldername in RALPfoldernames:
        ralp_cases.append(int(float(os.path.basename(RALPfoldername).split('-')[0])))

    # Resolve whole gland ROIs                
    WGfoldernames = glob(WG_ROI_path + os.sep + 'DICOMmasks' + os.sep + '*')    
    WG_cases = []
    for WGfoldername in WGfoldernames:
        WG_cases.append(int(float(os.path.basename(WGfoldername).split('-')[0])))
    return used_cases, ralp_cases, WG_cases, lesion_cases, lesion_cases_data, RALPfoldernames, LESIONfoldernames, LESIONdatanames, WGfoldernames


def extract_numbers_SUPP_PRO3(casefilterprefix):
    # Resolve human drawn lesions
    lesion_cases = []
    used_cases = []
    LESIONfoldernames1 = glob('..\MASKS_PRO3_1to18_VISUAL' + os.sep + '*_' + casefilterprefix + '_*')
    print("Lesion folder names[MASKS_PRO3_1to18_VISUAL]" + str(len(LESIONfoldernames1)))
    for LESIONfolder in LESIONfoldernames1:
        if not casefilterprefix in LESIONfolder:
            continue
        lesion_cases.append(1000+int(float(os.path.basename(LESIONfolder).split('_')[0])))
        used_cases.append(1000+int(float(os.path.basename(LESIONfolder).split('_')[0])))
    LESIONfoldernames2 = glob('..\MASKS_SUPP_S1_S34_VISUAL' + os.sep + '*_' + casefilterprefix + '_*')
    print("Lesion folder names[MASKS_SUPP_S1_S34_VISUAL]" + str(len(LESIONfoldernames2)))
    for LESIONfolder in LESIONfoldernames2:
        if not casefilterprefix in LESIONfolder:
            continue
        lesion_cases.append(2000+int(float(os.path.basename(LESIONfolder).split('_')[0][1:])))
        used_cases.append(2000+int(float(os.path.basename(LESIONfolder).split('_')[0][1:])))
    LESIONfoldernames = LESIONfoldernames1 + LESIONfoldernames2

    # Resolve intensity data cases
    LESIONdatanames = []
    lesion_cases_data = []
    LESIONdatanames_glob = glob('..\pipeline18' + os.sep + '*')
    for LESIONDATAfolder in LESIONdatanames_glob:
        base_folder = os.path.basename(LESIONDATAfolder)
        if not base_folder[0].isdigit() or not '_' in base_folder:
            continue
        LESIONDATAbasename = base_folder
        #print((LESIONDATAfolder, LESIONDATAbasename, int(float(LESIONDATAbasename.split('-')[0]))))
        lesion_cases_data.append(1000+int(float(LESIONDATAbasename.split('_')[0])))
        LESIONdatanames.append(LESIONDATAfolder)
    LESIONdatanames_glob = glob('..\pipelineS34' + os.sep + '*')
    for LESIONDATAfolder in LESIONdatanames_glob:
        base_folder = os.path.basename(LESIONDATAfolder)
        if not (base_folder[0] == 'S') or not base_folder[1].isdigit() or not '_' in base_folder:
            continue
        LESIONDATAbasename = base_folder
        #print((LESIONDATAfolder, LESIONDATAbasename, int(float(LESIONDATAbasename.split('-')[0]))))
        lesion_cases_data.append(2000+int(float(LESIONDATAbasename.split('_')[0][1:])))
        LESIONdatanames.append(LESIONDATAfolder)
    
    # Resolve Robotic assisted laparoscopic prostatectomy ROIs             
    ralp_cases = []
    RALPfoldernames1 = glob('..\MASKS_PRO3_1to18_RALP' + os.sep + '*_' + casefilterprefix + '_*')
    for RALPfoldername in RALPfoldernames1:
        ralp_cases.append(1000+int(float(os.path.basename(RALPfoldername).split('_')[0])))
    RALPfoldernames2 = glob('..\MASKS_SUPP_S1_S34_RALP' + os.sep + '*_' + casefilterprefix + '_*')
    for RALPfoldername in RALPfoldernames2:
        ralp_cases.append(2000+int(float(os.path.basename(RALPfoldername).split('_')[0][1:])))
    RALPfoldernames = RALPfoldernames1 + RALPfoldernames2

    # Resolve whole gland ROIs
    WG_cases = []
    WGfoldernames1 = glob('..\MASKS_PRO3_1to18_VISUAL' + os.sep + '*')
    print("Lesion folder names[MASKS_PRO3_1to18_VISUAL]" + str(len(WGfoldernames1)))
    WGfoldernames = []
    for WGfoldername in WGfoldernames1:
        base_folder = os.path.basename(WGfoldername)
        if not base_folder[0].isdigit() or not '_' in base_folder or not 'WG' in base_folder:
            continue
        LESIONDATAbasename = base_folder
#        subjname = '_'.join(LESIONDATAbasename.split('_')[:2])
#        print(WGfoldername + os.sep + subjname + '_WG_1st_onT2tra.nii.gz')
#        if not os.path.exists(WGfoldername + os.sep + subjname + '_WG_1st_onT2tra.nii.gz'):
#            continue
        if not os.path.exists(WGfoldername):
            continue
        WG_cases.append(1000+int(float(LESIONDATAbasename.split('_')[0])))
        WGfoldernames.append(WGfoldername)
    WGfoldernames2 = glob('..\MASKS_SUPP_S1_S34_VISUAL' + os.sep + '*')
    print("Lesion folder names[MASKS_SUPP_S1_S34_VISUAL]" + str(len(WGfoldernames2)))    
    for WGfoldername in WGfoldernames2:
        base_folder = os.path.basename(WGfoldername)
        if not (base_folder[0] == 'S') or not base_folder[1].isdigit() or not '_' in base_folder or not 'WG' in base_folder:
            continue
        LESIONDATAbasename = base_folder
#        subjname = '_'.join(LESIONDATAbasename.split('_')[:2])
#        print(WGfoldername + os.sep + subjname + '_WG_1st_onT2tra.nii.gz')
#        if not os.path.exists(WGfoldername + os.sep + subjname + '_WG_1st_onT2tra.nii.gz'):
#            continue
        if not os.path.exists(WGfoldername):
            continue
        WG_cases.append(2000+int(float(LESIONDATAbasename.split('_')[0][1:])))
        WGfoldernames.append(WGfoldername)
        
    return used_cases, ralp_cases, WG_cases, lesion_cases, lesion_cases_data, RALPfoldernames, LESIONfoldernames, LESIONdatanames, WGfoldernames


def resolve_cases(statname, filtername, prefix, forced_inclusions_lesion, outputpath, output_ASCII, included_cases, datafun, dirfun):

    if prefix == 'L1L2L3L4':
        casefilterprefix = 'L1'
    else:
        casefilterprefix = prefix
      
    used_cases, ralp_cases, WG_cases, lesion_cases, lesion_cases_data, RALPfoldernames, LESIONfoldernames, LESIONdatanames, WGfoldernames = dirfun(casefilterprefix)

    # Make additions of cases where there is no lesion ROI and considered as completely missed lesion
    for inclusion in forced_inclusions_lesion:
        if inclusion in used_cases:
            continue
        used_cases.append(inclusion)
    for inclusion in ralp_cases:
        if inclusion in used_cases:
            continue
        used_cases.append(inclusion)

    print('WG cases:' + str(len(WG_cases)))

    all_found_cases = []
    all_found_RALPpaths = []
    all_found_LESIONpaths = []
    all_found_LESIONDATApaths = []
    all_found_WGpaths = []
    missing_ralp_cases = []
    missing_lesion_cases = []
    missing_WG_cases = []
    missing_DATA_cases = []
    RALP_but_no_human = []
    human_but_no_RALP = []
    for lesion_i in range(len(used_cases)):
        lesion = used_cases[lesion_i]
        if not lesion in included_cases:
            continue
        # Case found in all modalities
        RALPfound = 1 
        if lesion in ralp_cases:
            ralp_index = ralp_cases.index(lesion)
            #print(RALPfoldernames[ralp_index])
            all_found_RALPpaths.append(RALPfoldernames[ralp_index])
        elif lesion in lesion_cases:
            #print('')
            all_found_RALPpaths.append('')
            missing_ralp_cases.append(lesion)
            RALPfound = 0
            human_but_no_RALP.append(lesion)
        elif lesion in forced_inclusions_lesion:
            #print('')
            all_found_RALPpaths.append('')
            missing_ralp_cases.append(lesion)
            RALPfound = 0
        else:
            RALPfound = 0
            missing_ralp_cases.append(lesion)
            continue
            
        LESIONfound = 1
        if lesion in lesion_cases:
            lesion_index = lesion_cases.index(lesion)
            #print(LESIONfoldernames[lesion_index])
            all_found_LESIONpaths.append(LESIONfoldernames[lesion_index])
        elif lesion in ralp_cases:
            #print('')
            all_found_LESIONpaths.append('')
            missing_lesion_cases.append(lesion)
            LESIONfound = 0
            RALP_but_no_human.append(lesion)
        elif lesion in forced_inclusions_lesion:
            #print('')
            all_found_LESIONpaths.append('')
            missing_lesion_cases.append(lesion)
            LESIONfound = 0
        else:
            LESIONfound = 0

        WGfound = 1
        if lesion in WG_cases:
            WG_index = WG_cases.index(lesion)
            #print(WGfoldernames[WG_index])
            all_found_WGpaths.append(WGfoldernames[WG_index])
        else:
            #all_found_WGpaths.append('')
            missing_WG_cases.append(lesion)
            WGfound = 0

        LESIONDATAfound = 1
        if lesion in WG_cases:
            lesion_data_i = lesion_cases_data.index(lesion)
            if not os.path.exists(datafun(LESIONdatanames, lesion_data_i, lesion)):
                all_found_LESIONDATApaths.append('')
                missing_DATA_cases.append(lesion)
                LESIONDATAfound = 0
                print('>>' + datafun(LESIONdatanames, lesion_data_i, lesion) + ' MISSING')
            else:
                #print('adding ' + LESIONdatanames[lesion_data_i])
                all_found_LESIONDATApaths.append(LESIONdatanames[lesion_data_i])
                print('>>' + datafun(LESIONdatanames, lesion_data_i, lesion) + ' FOUND')
        elif lesion in forced_inclusions_lesion:
            all_found_LESIONDATApaths.append('')
            missing_DATA_cases.append(lesion)
            LESIONDATAfound = 0
        else:
            LESIONDATAfound = 0

        # Sanity check case in RALP but already included by forece
        if (RALPfound == 1) and (lesion in forced_inclusions_lesion):
            print(all_found_RALPpaths)
            print(ralp_cases)
            raise Exception(prefix + ":Forced inclusion " + str(lesion) +  " already present in RALP cases")

        print(lesion, RALPfound, LESIONfound, WGfound, LESIONDATAfound)
        if WGfound == 1:
            all_found_cases.append(lesion)


    print('RALP_but_no_human')
    print(RALP_but_no_human)
    print('human_but_no_RALP')
    print(human_but_no_RALP)

    print('missing lesion:' + str(missing_lesion_cases))
    f = open(output_ASCII + os.sep + filtername + '_' + statname + '_' + prefix + 'missing_lesion_cases.txt', 'w')
    for subjname in missing_lesion_cases:
        f.write('%s\n' % (subjname))
    f.close()

    print('missing ralp:' + str(len(missing_ralp_cases)))
    f = open(output_ASCII + os.sep + filtername + '_' + statname + '_' + prefix + 'missing_ralp_cases.txt', 'w')
    for subjname in missing_ralp_cases:
        f.write('%s\n' % (subjname))
    f.close()

    print('missing WG:' + str(len(missing_WG_cases)))
    f = open(output_ASCII + os.sep + filtername + '_' + statname + '_' + prefix + 'missing_WG_cases.txt', 'w')
    for subjname in missing_WG_cases:
        f.write('%s\n' % (subjname))
    f.close()

    print('missing data:' + str(len(missing_DATA_cases)))
    f = open(output_ASCII + os.sep + filtername + '_' + statname + '_' + prefix + 'missing_DATA_cases.txt', 'w')
    for subjname in missing_DATA_cases:
        f.write('%s\n' % (subjname))
    f.close()

    if not len(all_found_cases) == len(all_found_WGpaths):
        raise Exception('len(all_found_cases) == len(all_found_WGpaths) mismatch:' + str(len(all_found_cases)) + ' ' + str(len(all_found_WGpaths)))
    if not len(all_found_cases) == len(all_found_RALPpaths):
        print(all_found_cases)
        print(all_found_RALPpaths)
        raise Exception('len(all_found_cases) == len(all_found_RALPpaths) mismatch:' + str(len(all_found_cases)) + ' ' + str(len(all_found_RALPpaths)))
    if not len(all_found_cases) == len(all_found_LESIONDATApaths):
        raise Exception('len(all_found_cases) == len(all_found_LESIONDATApaths) mismatch:' + str(len(all_found_cases)) + ' ' + str(len(all_found_LESIONDATApaths)))
    if not len(all_found_cases) == len(all_found_LESIONpaths):
        raise Exception('len(all_found_cases) == len(all_found_LESIONpaths) mismatch:' + str(len(all_found_cases)) + ' ' + str(len(all_found_LESIONpaths)))
    return all_found_cases, all_found_WGpaths, all_found_RALPpaths, all_found_LESIONDATApaths, all_found_LESIONpaths, casefilterprefix
