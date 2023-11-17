"""Utilities for all visualization tests.

See README.md for instructions on writing tests.
"""
import matplotlib.pyplot as plt
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def clean_matplotlib():
    """Cleans up matplotlib figures before and after each test."""
    # Before the test.
    plt.close("all")

    yield

    # After the test.
    plt.close("all")


def add_uniform_sphere_1d(archive, x_range):
    """Adds points from the negative sphere function in a 1D grid w/ 100 elites.

    The solutions are the same as the measures

    x_range is a tuple of (lower_bound, upper_bound).
    """
    x = np.linspace(x_range[0], x_range[1], 100)
    archive.add(
        solution=x[:, None],
        objective=-x**2,
        measures=x[:, None],
    )


def add_uniform_sphere_2d(archive, x_range, y_range):
    """Adds points from the negative sphere function in a 100x100 grid.

    The solutions are the same as the measures (the (x,y) coordinates).

    x_range and y_range are tuples of (lower_bound, upper_bound).
    """
    xxs, yys = np.meshgrid(
        np.linspace(x_range[0], x_range[1], 100),
        np.linspace(y_range[0], y_range[1], 100),
    )
    xxs, yys = xxs.ravel(), yys.ravel()
    coords = np.stack((xxs, yys), axis=1)
    sphere = xxs**2 + yys**2
    archive.add(
        solution=coords,
        objective=-sphere,  # Negative sphere.
        measures=coords,
    )


def add_uniform_sphere_3d(archive, x_range, y_range, z_range):
    """Adds points from the negative sphere function in a 100x100x100 grid.

    The solutions are the same as the measures (the (x,y,z) coordinates).

    x_range, y_range, and z_range are tuples of (lower_bound, upper_bound).
    """
    xxs, yys, zzs = np.meshgrid(
        np.linspace(x_range[0], x_range[1], 40),
        np.linspace(y_range[0], y_range[1], 40),
        np.linspace(z_range[0], z_range[1], 40),
    )
    xxs, yys, zzs = xxs.ravel(), yys.ravel(), zzs.ravel()
    coords = np.stack((xxs, yys, zzs), axis=1)
    sphere = xxs**2 + yys**2 + zzs**2
    archive.add(
        solution=coords,
        objective=-sphere,  # Negative sphere.
        measures=coords,
    )
