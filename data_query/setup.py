from setuptools import setup, find_packages

setup(
    name='data-query',  # Replace with your module's name
    version='0.1.0',  # Version of your package
    packages=find_packages(),  # Automatically find all packages in this directory
    install_requires=['boto3','matplotlib','opencv-python-headless','pandas','pytest','requests','ec2-metadata==2.13.0','pyarrow==15.0.0'],
    author='Bianca Migliori',
    author_email='bmigliori@nyscf.org',
    description= 'Functions for querying large imaging datasets',
    long_description=open('README.md').read(),
    url='https://github.com/NYSCF/data_query',
    # Additional metadata can be provided here
)
