import os
from tempfile import TemporaryDirectory
import shutil

import pytest

import numpy as np

import pystencils.plot as plt


def example_scalar_field(t=0):
    x, y = np.meshgrid(np.linspace(0, 2 * np.pi, 100), np.linspace(0, 2 * np.pi, 100))
    z = np.cos(x + 0.1 * t) * np.sin(y + 0.1 * t) + 0.1 * x * y
    return z


def example_vector_field(t=0, shape=(40, 40)):
    result = np.empty(shape + (2,))
    x, y = np.meshgrid(np.linspace(0, 2 * np.pi, shape[0]), np.linspace(0, 2 * np.pi, shape[1]))
    result[..., 0] = np.cos(x + 0.1 * t) * np.sin(y + 0.1 * t) + 0.01 * x * y
    result[..., 1] = np.cos(0.001 * y)
    return result


@pytest.mark.skipif(shutil.which('ffmpeg') is None, reason="ffmpeg not available")
def test_animation():
    t = 0

    def run_scalar():
        nonlocal t
        t += 1
        return example_scalar_field(t)

    def run_vec():
        nonlocal t
        t += 1
        return example_vector_field(t)

    plt.clf()
    plt.cla()

    with TemporaryDirectory() as tmp_dir:
        ani = plt.vector_field_magnitude_animation(run_vec, interval=1, frames=2)
        ani.save(os.path.join(tmp_dir, "animation1.avi"))

        ani = plt.vector_field_animation(run_vec, interval=1, frames=2, rescale=True)
        ani.save(os.path.join(tmp_dir, "animation2.avi"))

        ani = plt.vector_field_animation(run_vec, interval=1, frames=2, rescale=False)
        ani.save(os.path.join(tmp_dir, "animation3.avi"))

        ani = plt.scalar_field_animation(run_scalar, interval=1, frames=2, rescale=True)
        ani.save(os.path.join(tmp_dir, "animation4.avi"))

        ani = plt.scalar_field_animation(run_scalar, interval=1, frames=2, rescale=False)
        ani.save(os.path.join(tmp_dir, "animation5.avi"))

        ani = plt.surface_plot_animation(run_scalar, frames=2)
        ani.save(os.path.join(tmp_dir, "animation6.avi"))
