#!/usr/bin/env python

import cv2
import numpy as np
try:
    from skimage import filters
except ImportError:
    from skimage import filter as filters
try:
    xrange
except NameError:
    xrange = range

# Directory where result data are located
experiment_dir = ''
stat_funs = []
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

def write_slice2D(data, filename):
    data = np.flipud(data)
    height = data.shape[0]
    width = data.shape[1]
    blank_image = np.zeros((height,width,3), np.uint8)
    blank_image[:,:,0]=data
    blank_image[:,:,1]=data
    blank_image[:,:,2]=data
    print((filename, blank_image.shape))
    cv2.imwrite(filename, blank_image)

def write_slice2D_color(data, filename, RGB):
    data = np.flipud(data)
    height = data.shape[0]
    width = data.shape[1]
    blank_image = np.zeros((height,width,3), np.uint8)
    if RGB[0] == 1:
        blank_image[:,:,0]=data
    elif RGB[0] == -1:
        blank_image[:,:,0]=np.max(data)-data
    else:
        blank_image[:,:,0]=0

    if RGB[1] == 1:
        blank_image[:,:,1]=data
    elif RGB[1] == -1:
        blank_image[:,:,1]=np.max(data)-data
    else:
        blank_image[:,:,1]=0

    if RGB[2] == 1:
        blank_image[:,:,2]=data
    elif RGB[2] == -1:
        blank_image[:,:,2]=np.max(data)-data
    else:
        blank_image[:,:,2]=0
    
    blank_image[:,:,2]=data*RGB[2]
    print((filename, blank_image.shape))
    cv2.imwrite(filename, blank_image)

def write_slice2D_ROI_color(data, ROIdata, cdata, filename, alpha):
    data = np.flipud(data)
    ROIdata = np.flipud(ROIdata)
    height = data.shape[0]
    width = data.shape[1]
    blank_image = np.zeros((height,width,3), np.uint8)

    cdata_max = np.max(cdata)
    ROIdataR = np.zeros_like(ROIdata)
    ROIdataB = np.zeros_like(ROIdata)
    for y in range(blank_image.shape[1]):
        for x in range(blank_image.shape[0]):
            if(cdata[x, y]/cdata_max < 0.5):
                ROIdataR[x, y] = (cdata[x, y]/cdata_max)*2.0
            else:
                ROIdataR[x, y] = 1.0
            if(cdata[x, y]/cdata_max > 0.25):
                ROIdataB[x, y] = (1-cdata[x, y]/cdata_max)*4.0
            else:
                ROIdataB[x, y] = 1.0

    R = np.maximum(np.multiply(cdata,(1-alpha)),np.multiply(cdata,alpha))
    G = np.maximum(np.multiply(cdata,(1-alpha)),ROIdataB)
    B = np.maximum(np.multiply(cdata,(1-alpha)),ROIdataR)
    max_all = np.max([np.max(R),np.max(B),np.max(B)])

    blank_image[:,:,0] = data
    blank_image[:,:,1] = data
    blank_image[:,:,2] = data
    for y in range(blank_image.shape[1]):
        for x in range(blank_image.shape[0]):
            if(ROIdata[x, y] > 0):
                blank_image[x, y, 0] = R[x, y]/max_all*255.0
                blank_image[x, y, 1] = G[x, y]/max_all*255.0
                blank_image[x, y, 2] = B[x, y]/max_all*255.0

    cv2.imwrite(filename, blank_image)

def write_slice2D_ROI(data, ROIdata, filename, alpha):
    data = np.flipud(data)
    ROIdata = np.flipud(ROIdata)
    height = data.shape[0]
    width = data.shape[1]
    blank_image = np.zeros((height,width,3), np.uint8)
    R = np.maximum(np.multiply(data,(1-alpha)),np.multiply(ROIdata,alpha))
    G = np.maximum(np.multiply(data,(1-alpha)),np.multiply(data,alpha))
    B = np.maximum(np.multiply(data,(1-alpha)),np.multiply(data,alpha))
    max_all = np.max([np.max(R),np.max(B),np.max(B)])
    blank_image[:,:,0]=R/max_all*255.0
    blank_image[:,:,1]=G/max_all*255.0
    blank_image[:,:,2]=B/max_all*255.0
    cv2.imwrite(filename, blank_image)

def write_slice2D_ROI(data, ROIdata, filename, alpha):
    data = np.flipud(data)
    ROIdata = np.flipud(ROIdata)
    height = data.shape[0]
    width = data.shape[1]
    blank_image = np.zeros((height,width,3), np.uint8)
    R = np.maximum(np.multiply(data,(1-alpha)),np.multiply(ROIdata,alpha))
    G = np.maximum(np.multiply(data,(1-alpha)),np.multiply(data,alpha))
    B = np.maximum(np.multiply(data,(1-alpha)),np.multiply(data,alpha))
    max_all = np.max([np.max(R),np.max(B),np.max(B)])
    blank_image[:,:,0]=R/max_all*255.0
    blank_image[:,:,1]=G/max_all*255.0
    blank_image[:,:,2]=B/max_all*255.0
    cv2.imwrite(filename, blank_image)

def write_slice2D_polygon(data, points, filename):
    data = np.flipud(data)
    height = data.shape[0]
    width = data.shape[1]
    blank_contour = np.zeros((height,width,3), np.uint8)
    for p_i in range(1,len(points)):
        p1 = points[p_i-1]
        p2 = points[p_i]
        cv2.line(blank_contour,(p1[0],height-p1[1]-1),(p2[0],height-p2[1]-1),(255,255,255),1)
    p1 = points[len(points)-1]
    p2 = points[0]
    cv2.line(blank_contour,(p1[0],height-p1[1]-1),(p2[0],height-p2[1]-1),(255,255,255),1)

    blank_image = np.zeros((height,width,3), np.uint8)
    alpha = 0.6
    R = np.maximum(np.multiply(data,(1-alpha)),np.multiply(blank_contour[:,:,0],alpha))
    G = np.maximum(np.multiply(data,(1-alpha)),np.multiply(blank_contour[:,:,0],alpha))
    B = np.maximum(np.multiply(data,(1-alpha)),np.multiply(blank_contour[:,:,0],alpha))
    max_all = np.max([np.max(R),np.max(B),np.max(B)])
    blank_image[:,:,0]=R/max_all*255.0
    blank_image[:,:,1]=G/max_all*255.0
    blank_image[:,:,2]=B/max_all*255.0
    cv2.imwrite(filename, blank_image)


def write_slice2D_locations(data, filename):
    cv2.imwrite(filename, data)
