from setuptools import setup, find_packages

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
    install_requires=[
        'numpy==1.26.4', 'pytest==8.2.0', 'opencv-python-headless==4.9.0.80', 'matplotlib==3.8.4', 'scipy<=1.14.0', 'tifffile',
        'pandas==2.2.2', 'scikit-image<=0.22.0','PyYAML==6.0.1', 'utils','mahotas==1.4.15','Pillow==10.4.0','prettytable==3.10.2',
        'setuptools==68.2.2'
    ]
)





