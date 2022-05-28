#!/bin/sh
# Radiomics for Medical Imaging
#
# Copyright (C) 2019-2022 Harri Merisaari haanme@MRC.fi
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# version 1.0.0 by: Harri Merisaari [05/2022]

# Folder for source DICOM data
basedir='../../..'
# Set batch definition file containing one case name per line
selectionfile=$1
modality=$2
# Folder where results are saved
experiment_dir=$3
method=$4
voxelsize=$5
if [[ ! -f "$selectionfile" ]]
then
    echo "Namelist file does not exist, creating it"
    ls $experiment_dir >> $selectionfile
fi
# Create Quality Control (QC) folder
mkdir -p $(echo $experiment_dir)'/QC'
# Set log file for batch execution
logfile="failures.txt"
# Resolve number of lines in the file i. e. number of cases to process
no_folders=(`wc -l $selectionfile`)
no_folder=${no_folders[0]}
echo $no_folders " folders to be processed"
echo "Failed executions" > $logfile
# Go trough lines in the namelist file
for (( round=1; round<=$no_folders; round++ ))
do
    subjectname=$(sed -n "$round"p $selectionfile | awk '{print $1}')
    #process patient with python script
    xtermcmd=$(echo python MRCRadiomics.py --modality $modality --method $method --input $experiment_dir --output '../../features_'$modality --case $subjectname --voxelsize $voxelsize)
    echo $(date +"%d/%m/%Y %H:%M:%S")':'$(whoami)':'$0':'$xtermcmd >> $experiment_dir'/QC/'$subjectname'.log'
    echo $xtermcmd
    ret=$(eval $xtermcmd)
    # Write exit status to log file
    if [ "$?" -eq "0" ]
    then
        echo "SUCCESS"
        echo $(date +"%d/%m/%Y %H:%M:%S")':'$(whoami)':'$0':SUCCESS' >> $experiment_dir'/QC/'$subjectname'.log'
    else
        echo "FAILURE"
        echo "Failure in " + $xtermcmd >> $logfile
        echo $(date +"%d/%m/%Y %H:%M:%S")':'$(whoami)':'$0':FAILURE' >> $experiment_dir'/QC/'$subjectname'.log'
    fi
    break
done
