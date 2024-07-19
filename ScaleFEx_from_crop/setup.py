from setuptools import setup, find_packages

setup(
    name='ScaleFEx__from_crop',  # Replace with your module's name
    version='0.1.0',  # Version of your package
    packages=find_packages(),  # Automatically find all packages in this directory
    install_requires=['scikit-image==2.0','numpy','scipy','pytest','opencv-python-headless','pandas','numpy','pytest','opencv-python-headless',
                      'scipy','matplotlib','mahotas'],
    author='Bianca Migliori',
    author_email='bmigliori@nyscf.org',
    description= 'Functions to extract fixed features from crops of single cells',
    long_description=open('README.md').read(),
    url='https://github.com/NYSCF/ScaleFEx_from_crop',
    # Additional metadata can be provided here
)
