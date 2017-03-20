
from setuptools import setup, find_packages
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
    version='0.1.1b4',
    description='A control systems library for Python3',
    long_description=long_description,
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
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
    install_requires=['numpy','scipy','matplotlib','tabulate'],
    tests_require=['numpy','nose'],
    test_suite = 'nose.collector',
    keywords='control-theory PID controller design industrial automation',
    extras_require={
        'dev': ['check-manifest'],
        'test': ['coverage'],
    },
)
