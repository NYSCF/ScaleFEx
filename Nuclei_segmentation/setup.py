from setuptools import setup, find_packages

setup(
    name='Nuclei_segmentation',  # Replace with your module's name
    version='0.1.0',  # Version of your package
    packages=find_packages(),  # Automatically find all packages in this directory
    install_requires=['scikit-image==0.22.0','numpy','scipy','pytest','opencv-python-headless'],
    author='Bianca Migliori',
    author_email='bmigliori@nyscf.org',
    description= 'Functions for retrieving nuclei coordinates',
    long_description=open('README.md').read(),
    url='https://github.com/NYSCF/Nuclei_segmentation',
    # Additional metadata can be provided here
)
