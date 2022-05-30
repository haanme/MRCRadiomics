# MRCTurku Radiomic Feature Extraction
Feature extraction algorithms for prostate created at MRCTurku http://mrc.utu.fi/. 
<br><img src="http://mrc.utu.fi/mrc/static/resources/html/images/mrc/logo_mrc_dark_cyan.PNG" alt="MRCTurku logo"><br>

If you use this work in publication, please cite:


Merisaari, H, Taimen, P, Shiradkar, R, et al. Repeatability of radiomics and machine learning for DWI: Short‐term repeatability study of 112 patients with prostate cancer. Magn Reson Med. 2019; 00: 1– 17. https://doi.org/10.1002/mrm.28058

For further information about the project, please see related ISMRM 2019 abstract:

H Merisaari, R Shiradkar, J Toivonen, A Hiremath, M Khorrami, IM Perez, T Pahikkala, P Taimen, J Verho, PJ Boström, H Aronen, A Madabhushi, I Jambor, Repeatability of radiomics features for prostate cancer diffusion weighted imaging obtained using b-values up to 2000 s/mm2, 27th Annual Meeting & Exhibition ISMRM, May 11-16 2019, Montréal, QC, Canada, #7461
![Merisaari_4472_Teaser](https://user-images.githubusercontent.com/8802462/170784539-047493e0-ece7-4490-8522-7b4191ea983a.jpg)

<b>Installation instructions</b>:
- scipy image processing tools: 
```bash
install -c anaconda scipy 
```
- pyzernikemoment texture features:
```bash
pip install pyzernikemoment
```
- numba (optional for GPU speed-up): 
```bash
conda install numba
```
- skimage image processing tools: 
```bash
conda install skimage
```
- cv2 iamge processing tools: 
```bash
pip install opencv
```

For Pyradiomics wrapper:
- pyradiomics:
```bash
pip install pyradiomics
```
- SimpleITK: 
```bash
conda install -c simpleitk simpleitk
```

<b>MRCRadiomics</b> package:
1) Download and extract the repository
2) Install with setup.py
```bash
python setup.py install
```

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

(c) Harri Merisaari 2018-2022
