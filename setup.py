#!/usr/bin/env python
"""The setup script."""

from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as readme_file:
    readme = readme_file.read()

with open("HISTORY.md", encoding="utf-8") as history_file:
    history = history_file.read()

# NOTE: Update pinned_reqs whenever install_requires or extras_require changes.
install_requires = [
    # numpy>=1.17.0 is when default_rng becomes available;
    # scikit-learn 1.1.0 requires numpy 1.17.3+
    "numpy>=1.17.3",
    "numpy_groupies>=0.9.16",  # Supports Python 3.7 and up.
    "numba>=0.51.0",
    "pandas>=1.0.0",
    "sortedcontainers>=2.0.0",  # Primarily used in SlidingBoundariesArchive.
    "scikit-learn>=1.1.0",  # Primarily used in CVTArchive.
    "scipy>=1.4.0",  # Primarily used in CVTArchive.
    "threadpoolctl>=3.0.0",
]

extras_require = {
    "visualize": [
        "matplotlib>=3.0.0",
        "shapely>=2.0.0",
    ],
    # All dependencies except for dev. Don't worry if there are duplicate
    # dependencies, since setuptools automatically handles duplicates.
    "all": [
        ### visualize ###
        "matplotlib>=3.0.0",
        "shapely>=2.0.0",
    ],
    "dev": [
        "pip>=20.3",
        "pylint",
        "yapf",
        "pre-commit",

        # Testing
        "pytest==7.0.1",
        "pytest-cov==3.0.0",
        "pytest-benchmark==3.4.1",
        "pytest-xdist==2.5.0",

        # Documentation
        "Sphinx==4.5.0",
        "sphinx-material==0.0.32",
        "sphinx-autobuild==2021.3.14",
        "sphinx-copybutton==0.3.1",
        "myst-nb==0.17.1",
        "sphinx-toolbox==3.1.0",
        "sphinx-autodoc-typehints==1.18.2",
        "sphinx-codeautolink==0.12.1",

        # Distribution
        "bump2version==1.0.1",
        "wheel==0.40.0",
        "twine==4.0.2",
        "check-wheel-contents==0.4.0",
    ],
}

setup(
    author="ICAROS Lab pyribs Team",
    author_email="team@pyribs.org",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    description=
    "A bare-bones Python library for quality diversity optimization.",
    install_requires=install_requires,
    extras_require=extras_require,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="ribs",
    name="ribs",
    packages=find_packages(include=["ribs", "ribs.*"]),
    python_requires=">=3.8.0",
    test_suite="tests",
    url="https://github.com/icaros-usc/pyribs",
    version="0.6.1",
    zip_safe=False,
)
