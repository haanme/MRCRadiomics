# -*- coding: utf-8 -*-
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
import pydicom as dicom
import os
import numpy as np
import glob

class DICOMError(Exception):
     def __init__(self, value):
         self.value = value
     def __str__(self):
         return repr(self.value)

class DICOMReadError(DICOMError):
     def __str__(self):
         return repr('Read error ' + self.value)


class DICOMWriteError(DICOMError):
     def __str__(self):
         return repr('Write error ' + self.value)
   
class DicomIO:
    """I/O functions for DICOM stacks"""  
    def ReadDICOM_slices(self, path):
        root, tail = os.path.split(path)
        if not(os.access(root, os.F_OK)):
            raise DICOMReadError('Path ' + path + ' does not exist' )
        if not(os.access(root, os.R_OK)):
            raise DICOMReadError('Path ' + path + ' does not have read permission' )
        #resolve filenames
        filenames = glob.glob(path)
        if len(filenames) == 0:
            raise DICOMReadError('Path ' + path + ' does not have files' )                
        #read slices into array
        slice_list = []
        error_list = []
        print_step = np.floor(len(filenames)/10)
        if print_step == 0:
            print_step = 1
        for filename_i in range(len(filenames)): 
            print((path, filenames[filename_i]))
            #filename = os.path.join(path, filenames[filename_i])
            filename = filenames[filename_i]
            if np.mod(filename_i, print_step) == 0 or filename_i == (len(filenames)-1):
                print('Reading ' + filename)
            #Skip files that could not be read
            if not(os.access(filename, os.R_OK)):
                print('File ' + filename + ' does not have read permission')
                continue            
            try:
                ds = dicom.read_file(filename)
                slice_list.append([ds.ImagePositionPatient[2],ds])
            except Exception as e:
                print(e.message)
                error_list.append(DICOMReadError(e.message))
        slice_list.sort(key=lambda tup: tup[0])                
        slice_list = [tup[1] for tup in slice_list]
        return slice_list, error_list

    def resolve_tdim(self, ds):
        framereferencetime = -1
        try:
            sname = ds.SequenceName.strip(" ")
            if sname.startswith('*ep_b'):
                sname = sname[5:]
            if sname.endswith('t'):
                sname = sname[:-2]
            framereferencetime = int(sname)*10
        except:
            pass
        if framereferencetime == -1:
            if 'FrameReferenceTime' in ds.trait_names():
                framereferencetime = ds.FrameReferenceTime
            elif 'DiffusionBValue' in ds.trait_names():
                framereferencetime = ds.DiffusionBValue
            elif 'DiffusionBValue' in ds.trait_names():
                framereferencetime = ds.DiffusionBValue
            elif 'EchoTime' in ds.trait_names():
                framereferencetime = ds.EchoTime
            else:
                framereferencetime = 0
        return framereferencetime

    def resolve_tdim_tname(self, ds, tname):
        framereferencetime = -1
        if tname == 'SequenceName':
            try:
                sname = ds.SequenceName.strip(" ")
                if sname.startswith('*ep_b'):
                    sname = sname[5:]
                if sname.endswith('t'):
                    sname = sname[:-2]
                framereferencetime = int(sname)*10
            except:
                pass
            return framereferencetime
        if tname == 'FrameReferenceTime':
            framereferencetime = ds.FrameReferenceTime
        elif tname == 'DiffusionBValue':
            framereferencetime = ds.DiffusionBValue
        elif tname == 'DiffusionBValue':
            framereferencetime = ds.DiffusionBValue
        elif tname == 'EchoTime':
            framereferencetime = ds.EchoTime
        return framereferencetime

    def ReadDICOM_frames(self, path, tname='', printout = 1):
        root, tail = os.path.split(path)
        if not(os.access(root, os.F_OK)):
            raise DICOMReadError('Path ' + path + ' does not exist' )
        if not(os.access(root, os.R_OK)):
            raise DICOMReadError('Path ' + path + ' does not have read permission' )
        #resolve filenames
        filenames = glob.glob(path)
        if len(filenames) == 0:
            raise DICOMReadError('Path ' + path + ' does not have files' )
        #read slices into array
        slice_list = []
        ds_list = []
        print_step = np.floor(len(filenames)/10)
        if print_step == 0:
            print_step = 1
        for filename_i in range(len(filenames)):
            #filename = os.path.join(path, filenames[filename_i])
            filename = filenames[filename_i]
            if printout == 1 and (np.mod(filename_i, print_step) == 0 or filename_i == (len(filenames)-1)):
                print('Reading ' + filename)
            #Skip files that could not be read
            if not(os.access(filename, os.R_OK)):
                print('File ' + filename + ' does not have read permission')
                continue
            try:
                ds = dicom.read_file(filename)
            except Exception as e:
                print(e)
                print("Failed to read file " + filename + ":" + e.message)
                raise DICOMReadError("Failed to read file " + filename + ":" + e.message)
            if len(tname) == 0:
                framereferencetime = self.resolve_tdim(ds)
            else:
                framereferencetime = self.resolve_tdim_tname(ds, tname)
            if framereferencetime == -1:
                print("Failed to find field FrameReferenceTime")
                continue
            slice_list.append(ds)
            if 'AcquisitionTime' in ds.trait_names():
                ds_list.append((ds.AcquisitionTime, framereferencetime, ds.ImagePositionPatient[2], ds))
            else:
                ds_list.append((0.0, framereferencetime, ds.ImagePositionPatient[2], ds))
        ds_list = sorted(ds_list, key=lambda x: (x[0], x[1], x[2]))
        #collect data for frames in order
        AcquisitionTimes = []
        for ds_i in range(len(ds_list)):
            if not ds_list[ds_i][0] in AcquisitionTimes:
                AcquisitionTimes.append(ds_list[ds_i][0])
        AcquisitionTimes.sort()
        series_list = []
        for AcquisitionTime in AcquisitionTimes:
            FrameTimes = []
            for ds_i in range(len(ds_list)):
                if not ds_list[ds_i][0] == AcquisitionTime:
                    continue
                if not ds_list[ds_i][1] in FrameTimes:
                    FrameTimes.append(ds_list[ds_i][1])
            FrameTimes.sort()
            frame_list = []
            for FrameTime in FrameTimes:
                frame = []
                for ds_i in range(len(ds_list)):
                    if not ds_list[ds_i][0] == AcquisitionTime:
                        continue
                    if not ds_list[ds_i][1] == FrameTime:
                        continue
                    frame.append(ds_list[ds_i][3])
                frame_list.append(frame)
            series_list.append(frame_list)

        return series_list

    def WriteDICOM_frames(self, path, frame_list, prefix):
        file_i = 0
        filenames = []
        # go through all frames
        for frame_i in range(len(frame_list)):
            # go through all slices
            for slice_i in range(len(frame_list[frame_i])):
                filename = path + '/' + prefix + ("%06d" % file_i)
                frame_list[frame_i][slice_i].save_as(filename)
                file_i = file_i + 1
                filenames.append(filename)
        return filenames

    #
    # Write DICOM from intensity values
    # 
    # pixel_array    - 2D numpy.ndarray
    # filename       - filename where data is written
    # itemnumber     - item number, default == 0, determines also (0020,0013),(0020,1041)
    # PhotometricInterpretation - 'MONOCHROME2', 'RGB'
    #
    def WriteDICOM_slice(self, pixel_array,filename, itemnumber=0, PhotometricInterpretation="MONOCHROME2"):
        from dicom.dataset import Dataset, FileDataset
        import numpy as np
        import datetime, time
        """
        INPUTS:
        pixel_array: 2D numpy ndarray.  If pixel_array is larger than 2D, errors.
        filename: string name for the output file.
        """
        ## This code block was taken from the output of a MATLAB secondary
        ## capture.  I do not know what the long dotted UIDs mean, but
        ## this code works.
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = 'Secondary Capture Image Storage'
        file_meta.MediaStorageSOPInstanceUID = '1.3.6.1.4.1.9590.100.1.1.111165684411017669021768385720736873780'
        file_meta.ImplementationClassUID = '1.3.6.1.4.1.9590.100.1.0.100.4.0'
        ds = FileDataset(filename, {},file_meta = file_meta,preamble="\0"*128)
        ds.Modality = 'WSD'
        ds.ContentDate = str(datetime.date.today()).replace('-','')
        ds.ContentTime = str(time.time()) #milliseconds since the epoch
        ds.StudyInstanceUID =  '1.3.6.1.4.1.9590.100.1.1.124313977412360175234271287472804872093'
        ds.SeriesInstanceUID = '1.3.6.1.4.1.9590.100.1.1.369231118011061003403421859172643143649'
        ds.SOPInstanceUID =    '1.3.6.1.4.1.9590.100.1.1.111165684411017669021768385720736873780'
        ds.SOPClassUID = 'Secondary Capture Image Storage'
        ds.SecondaryCaptureDeviceManufctur = 'Python 2.7.3'
        ## These are the necessary imaging components of the FileDataset object.
        ds.SamplesPerPixel = 1
        if PhotometricInterpretation=="MONOCHROME2":
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.PixelRepresentation = 0
            ds.HighBit = 15
            ds.BitsStored = 16
            ds.BitsAllocated = 16
            ds.SmallestImagePixelValue = '\\x00\\x00'
            ds.LargestImagePixelValue = '\\xff\\xff'
        elif PhotometricInterpretation=="RGB":
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.PixelRepresentation = 0
            ds.HighBit = 15
            ds.BitsStored = 16
            ds.BitsAllocated = 16
            ds.SmallestImagePixelValue = '\\x00\\x00'
            ds.LargestImagePixelValue = '\\xff\\xff'    
            pixel_array = pixel_array[0]
            print(pixel_array.shape)
        ds.Columns = pixel_array.shape[0]
        ds.ItemNumber = str(itemnumber)
        ds.InstanceNumber = str(itemnumber)
        ds.SliceLocation = str(itemnumber)
        ds.Rows = pixel_array.shape[1]
        if pixel_array.dtype != np.uint16:
            pixel_array = pixel_array.astype(np.uint16)
            ds.PixelData = pixel_array.tostring()
        ds.save_as(filename)

        return filename

    #
    # Write DICOM from intensity values
    # 
    # pixel_arrays   - 3D numpy.ndarray
    # plans          - existing DICOM data for all planes where pixel_array is inserted
    # path           - path where data is to be written
    # prefix         - filename prefix where data is written
    #
    def WriteDICOM_slices(self, pixel_arrays, plans, path, prefix):
        import numpy as np        

        if pixel_arrays.shape[2] != len(plans):
            raise Exception('Number of slices in array and pland do not match')
            
        if not os.path.exists(path):
            os.makedirs(path)            
        
        filenames = []
        for slice_i in range(len(plans)):
            ds = plans[slice_i]
            if pixel_arrays[slice_i].dtype != np.uint16:
                pixel_array = pixel_arrays[slice_i].astype(np.uint16)
            ds.PixelData = pixel_array.tostring()
            filename = path + os.pathsep + prefix + ('%06d' % slice_i)
            ds.save_as(filename)
            filenames.append(filename)
        return filenames

    #
    # Write DICOM from intensity values
    # 
    # pixel_array               - 3D numpy.ndarray
    # prefix                    - filename prefix where data is written
    # PhotometricInterpretation - 'MONOCHROME2', 'RGB'
    #
    def WriteDICOM_slices_noplan(self, pixel_array, prefix, PhotometricInterpretation="MONOCHROME2"):
        filenames = []
        for slice_i in range(pixel_array.shape[2]):
            filename = prefix + ('%06d' % slice_i)
            print("Writing " + filename)
            filenames.append(self.WriteDICOM_slice(pixel_array[:,:,slice_i], filename, slice_i, PhotometricInterpretation))
        return filenames
