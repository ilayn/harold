"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path
import io

here = path.abspath(path.dirname(__file__))

# Get the long description from the relevant file


def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)
    
long_description = read('README.rst', 'CHANGES.txt')

setup(
    name='harold',
    author='Ilhan Polat',
    author_email='harold.of.python@gmail.com',
    url='https://github.com/ilayn/harold',
    version='0.1.1a8',
    description='A control systems library for Python3',
    long_description=long_description,
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: Unix',
        'Operating System :: MacOS'
    ],
    packages=['harold'],
    package_dir={'harold': 'harold'},
    install_requires=['numpy','scipy','tabulate'],
    tests_require=['numpy','nose'],
    test_suite = 'nose.collector',
    keywords='control-theory PID controller design industrial automation',
    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    extras_require={
        'dev': ['check-manifest'],
        'test': ['coverage'],
    },
)
