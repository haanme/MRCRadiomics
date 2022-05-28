from setuptools import setup

setup(
    name='MRCRadiomics',
    version='1.1.0',
    packages=[''],
    url='https://github.com/haanme/ProstateFeatures',
    license='LGPL 3.0',
    author='Harri Merisaari',
    author_email='haanme@utu.fi',
    description='MRC Radiomic feature extraction package',
    install_requires=[
        'dipy',
        'csv',
        'scipy',
        'skimage',
        'trimesh',
        'plyfile',
        'cv2',
        'nibabel',
        'sklearn',
    ]
)
