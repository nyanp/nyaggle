from setuptools import setup
from codecs import open
from os import path


def get_long_description():
    here = path.abspath(path.dirname(__file__))

    with open(path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
    return long_description


def get_version():
    version_filepath = path.join(path.dirname(__file__), 'nyaggle', 'version.py')
    with open(version_filepath) as f:
        for line in f:
            if line.startswith('__version__'):
                return line.strip().split()[-1][1:-1]


setup(
    name='nyaggle',
    packages=['nyaggle'],

    version=get_version(),

    license='BSD 3-Clause',

    install_requires=[
        'catboost>=0.17',
        'category_encoders',
        'lightgbm',
        'matplotlib',
        'more-itertools',
        'numpy',
        'optuna>=1.0.0',
        'pandas',
        'pyarrow',
        'seaborn',
        'sklearn',
        'tqdm',
        'transformers>=2.3.0'
    ],
    author='nyanp',
    author_email='Noumi.Taiga@gmail.com',
    url='https://github.com/nyanp/nyaggle',
    description='Code for Kaggle and Offline Competitions.',
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    keywords='nyaggle kaggle',
    classifiers=[
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'
    ]
)
