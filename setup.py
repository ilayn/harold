from setuptools import setup
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
    version='1.0.0',
    description='A control systems library for Python3',
    long_description=long_description,
    license='MIT',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
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
    install_requires=['numpy', 'scipy', 'matplotlib', 'tabulate'],
    setup_requires=['pytest-runner'],
    tests_require=['numpy', 'pytest'],
    keywords='control-theory PID controller design industrial automation',
    extras_require={
        'dev': ['check-manifest'],
        'test': ['coverage'],
    },
)
