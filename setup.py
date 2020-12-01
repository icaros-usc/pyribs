#!/usr/bin/env python
"""The setup script."""

from setuptools import find_packages, setup

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md') as history_file:
    history = history_file.read()

install_requires = [
    'numpy>=1.17.0',  # >=1.17.0 mainly because of default_rng
    'numba>=0.45.1',  # Has support for numpy 1.17
    'pandas>=1.0.0',
    'scipy>=1.0.0',
    'toml>=0.10.0',
]

extras_require = {
    'all': ['matplotlib>=3.0.0',],
    # Dependencies for examples (NOT tutorials -- tutorial notebooks should
    # install deps with cell magic and only depend on ribs and ribs[all]).
    'examples': [
        'matplotlib>=3.0.0',
        'seaborn>=0.11.0',
        'gym~=0.17.0',  # Strict since different gym may give different results.
        'Box2D~=2.3.10',  # For envs such as Lunar Lander.
        'fire>=0.3.0',

        # Dask
        'dask>=2.0.0',
        'distributed>=2.0.0',
        'bokeh>=2.0.0',  # Dask dashboard.
    ],
    'dev': [
        'pip>=20.3',
        'pylint',
        'yapf',

        # Testing
        'tox==3.14.0',
        'pytest==6.1.2',
        'pytest-cov==2.10.1',
        'pytest-benchmark==3.2.3',
        'pytest-xdist==2.1.0',

        # Documentation
        'Sphinx==3.2.1',
        'sphinx-material==0.0.32',
        'sphinx-autobuild==2020.9.1',
        'sphinx-copybutton==0.3.1',
        'autodocsumm==0.2.1',
        'myst-nb==0.10.1',

        # Distribution
        'bump2version==0.5.11',
        'wheel==0.33.6',
        'twine==1.14.0',
    ]
}

setup(
    author="ICAROS Lab",
    author_email='tjanaka@usc.edu',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="A bare-bones quality diversity optimization library.",
    install_requires=install_requires,
    extras_require=extras_require,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='ribs',
    name='ribs',
    packages=find_packages(include=['ribs', 'ribs.*']),
    python_requires=">=3.6.0",
    test_suite='tests',
    url='https://github.com/icaros-usc/ribs',
    version='0.0.0',
    zip_safe=False,
)
