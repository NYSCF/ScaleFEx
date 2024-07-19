from setuptools import setup, find_packages

setup(
    name='Quality_control_HCI',  # Replace with your module's name
    version='0.1.0',  # Version of your package
    packages=find_packages(),  # Automatically find all packages in this directory
    install_requires=['scikit-image','numpy','scipy','pytest','opencv-python-headless','pandas'],
    author='Bianca Migliori',
    author_email='bmigliori@nyscf.org',
    description= 'Functions for basic quality checks for a whole image',
    long_description=open('README.md').read(),
    url='https://github.com/NYSCF/Quality_control_HCI',
    # Additional metadata can be provided here
)
