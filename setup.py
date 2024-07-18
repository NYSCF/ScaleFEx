from setuptools import setup, find_packages

setup(
    name='ScaleFEx',
    version='0.1.0',
    packages=find_packages(),
    description='Full pipeline for large screen single cell feature extraction',
    long_description=open('README.md').read(),
    author='Bianca Migliori',
    author_email='bmigliori@nyscf.org',
    url='https://github.com/NYSCF/ScaleFEx',
    install_requires=[

        'numpy','pytest','opencv-python-headless','matplotlib','scipy','tifffile',
        'pandas','scikit-image==0.20','utils',
        'Quality_control_HCI==0.1.0', 
        'data_query==0.1', 
        'Nuclei_segmentation==0.1.0', 
    ],

)



