#!/usr/bin/env python
"""The setup script."""

from setuptools import find_packages, setup

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

install_requires = [
    'numpy>=1.17.0',  # >=1.17.0 mainly because of default_rng.
    'pandas>=1.0.0',
    'scipy>=1.0.0',
]

extras_require = {
    'all': ['matplotlib>=3.0.0',],
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
    test_suite='tests',
    url='https://github.com/icaros-usc/ribs',
    version='0.0.0',
    zip_safe=False,
)
