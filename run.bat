echo docker run -it --rm -v %CD%:/data -w /data --entrypoint /bin/bash haanme/mrcradiomics:1.1.0
docker run --rm -v %CD%:/data -w /data haanme/mrcradiomics:1.1.0
