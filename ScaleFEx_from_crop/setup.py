from setuptools import setup, find_packages

setup(
    name='ScaleFEx__from_crop', 
    version='0.1.0', 
    packages=find_packages(),  
    install_requires=['scikit-image==0.22.0','numpy','scipy','pytest','opencv-python-headless','pandas','numpy','pytest','opencv-python-headless',
                      'scipy','matplotlib','mahotas'],
    author='Bianca Migliori',
    author_email='bmigliori@nyscf.org',
    description= 'Functions to extract fixed features from crops of single cells',
    long_description=open('README.md').read(),
    url='https://github.com/NYSCF/ScaleFEx_from_crop',

)
