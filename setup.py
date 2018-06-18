import sys
import os
from os.path import abspath, dirname
import io
import subprocess
from setuptools import setup


# Borrowing some .git machinery from SciPy's setup.py
if sys.version_info[:2] < (3, 6):
    raise RuntimeError("Python version >= 3.6 required.")

MAJOR = 1
MINOR = 0
MICRO = 1
ISRELEASED = True
VERSION = '{}.{}.{}'.format(MAJOR, MINOR, MICRO)


# Return the git revision as a string
def git_version():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                               env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except OSError:
        GIT_REVISION = "Unknown"

    return GIT_REVISION


def get_version_info():
    FULLVERSION = VERSION
    if os.path.exists('.git'):
        GIT_REVISION = git_version()
    elif os.path.exists('harold/_version.py'):
        # must be a source distribution, use existing version file
        # load it as a separate module to not load scipy/__init__.py
        import imp
        version = imp.load_source('harold._version', 'harold/_version.py')
        GIT_REVISION = version.git_revision
    else:
        GIT_REVISION = "Unknown"

    if not ISRELEASED:
        FULLVERSION += '.dev0+' + GIT_REVISION[:7]

    return FULLVERSION, GIT_REVISION


def write_version_py(filename='harold/_version.py'):
    FULLVERSION, GIT_REVISION = get_version_info()
    s = (f'''# THIS FILE IS AUTO-GENERATED FROM SETUP.PY\n'''
         f'''short_version = "{VERSION}"\n'''
         f'''version = "{VERSION}"\n'''
         f'''full_version = "{FULLVERSION}"\n'''
         f'''git_revision = "{GIT_REVISION}"\n'''
         f'''release = {ISRELEASED}\n'''
         f'''if not release:\n'''
         f'''    version = full_version\n''')
    a = open(filename, 'w')
    try:
        a.write(s)
    finally:
        a.close()

# =============================
# Back to setup.py declarations
# =============================


here = abspath(dirname(__file__))


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

CLASSIFIERS = [
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
               'Operating System :: POSIX',
               'Operating System :: Unix',
               'Operating System :: MacOS'
]


def setup_package():
    write_version_py()
    setup(
        name='harold',
        author='Ilhan Polat',
        author_email='harold.of.python@gmail.com',
        url='https://github.com/ilayn/harold',
        description='A control systems library for Python3',
        long_description=long_description,
        license='MIT',
        classifiers=CLASSIFIERS,
        packages=['harold'],
        package_dir={'harold': 'harold'},
        python_requires='>=3.6',
        install_requires=['numpy', 'scipy', 'matplotlib', 'tabulate'],
        setup_requires=['pytest-runner'],
        tests_require=['numpy', 'pytest'],
        keywords='control-theory PID controller design industrial automation',
        extras_require={
            'dev': ['check-manifest'],
            'test': ['coverage'],
        },
        version=get_version_info()[0]
    )


if __name__ == '__main__':
    setup_package()
