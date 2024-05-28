echo docker run -it --rm -v %CD%:/data -w /data --entrypoint /bin/bash haanme/mrcradiomics:1.3.0
#docker run -it --rm -v %CD%:/data -w /data --entrypoint /usr/bin/bash haanme/mrcradiomics:1.3.0
#docker run --rm -v %CD%:/data -w /data haanme/mrcradiomics:1.3.0
#docker run --rm -v F:\PROSTATE:/data --entrypoint /bin/bash haanme/mrcradiomics:1.3.0
#docker run --rm -v F:\PROSTATE:/data haanme/mrcradiomics:1.3.0 --verbose Yes --method Moments --input /data/PI_CAI --modality T2W --intensityfile 10005_1000005_t2w.nii.gz --output /data/PI_CAI/features --case 10005 --voxelsize [2.0,2.0,3.0] --LSname /data/PI_CAI/picai_labels-main/picai_labels-main/csPCa_lesion_delineations/human_expert/resampled/10005_1000005.nii.gz
docker run --rm -v F:\PROSTATE:/data haanme/mrcradiomics:1.3.0 --verbose Yes --method FFT2D --input /data/PI_CAI --modality T2W --intensityfile 10005_1000005_t2w.nii.gz --output /data/PI_CAI/features --case 10005 --voxelsize [2.0,2.0,3.0] --LSname /data/PI_CAI/picai_labels-main/picai_labels-main/csPCa_lesion_delineations/human_expert/resampled/10005_1000005.nii.gz
