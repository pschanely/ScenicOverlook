from distutils.core import setup

setup(
    name='ScenicOverlook',
    version='0.1.0',
    author='Phillip Schanely',
    author_email='pschanely+vE7F@gmail.com',
    packages=['scenicoverlook'],
    scripts=[],
    url='https://github.com/pschanely/ScenicOverlook',
    license='LICENSE.txt', # (3 clause BSD)
    description='A library for incremental, in-memory map-reduces',
    long_description=open('README.txt').read(),
)
