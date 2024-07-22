from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess
import os

class CustomInstall(install):
    def run(self):
        # Ensure the main installation happens first
        install.run(self)
        # Install each local package via pip
        def install_local_package(path):
            subprocess.call([sys.executable, '-m', 'pip', 'install', path])
        
        base_path = os.path.dirname(os.path.realpath(__file__))
        install_local_package(os.path.join(base_path, 'Quality_control_HCI'))
        install_local_package(os.path.join(base_path, 'data_query'))
        install_local_package(os.path.join(base_path, 'Nuclei_segmentation'))
        install_local_package(os.path.join(base_path, 'ScaleFEx_from_crop'))

setup(
    name='ScaleFEx',
    version='0.1.0',
    packages=find_packages(),
    description='Full pipeline for large screen single cell feature extraction',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Bianca Migliori',
    author_email='bmigliori@nyscf.org',
    url='https://github.com/NYSCF/ScaleFEx',
    cmdclass={
        'install': CustomInstall
    },
    install_requires=[
        'numpy', 'pytest', 'opencv-python-headless', 'matplotlib', 'scipy', 'tifffile',
        'pandas', 'scikit-image==0.20', 'utils'
    ]
)






