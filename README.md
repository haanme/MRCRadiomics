# ProstateFeatures
Feature extraction algorithms for prostate. If you use this work in publication, please cite:

Merisaari, H, Taimen, P, Shiradkar, R, et al. Repeatability of radiomics and machine learning for DWI: Short‐term repeatability study of 112 patients with prostate cancer. Magn Reson Med. 2019; 00: 1– 17. https://doi.org/10.1002/mrm.28058

For further information about the project, please see related ISMRM 2019 abstract:

H Merisaari, R Shiradkar, Ji Toivonen, A Hiremath, M Khorrami, IM Perez, T Pahikkala, P Taimen, J Verho, PJ Boström, H Aronen, A Madabhushi, I Jambor, Repeatability of radiomics features for prostate cancer diffusion weighted imaging obtained using b-values up to 2000 s/mm2, 27th Annual Meeting & Exhibition ISMRM, May 11-16 2019, Montréal, QC, Canada, #7461

Required packages with installation insructions:
- pyzernikemoment: pip install pyzernikemoment
- numba (optional for GPU speed-up): conda install numba
- skimage: conda install skimage
- cv2: pip install opencv
For Pyradiomics wrapper:
- pyradiomics: pip install pyradiomics
- SimpleITK: conda install -c simpleitk simpleitk


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
    
    (c) Harri Merisaari 2018-2021
