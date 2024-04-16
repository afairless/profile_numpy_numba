#! /usr/bin/env python3


import time
import tracemalloc
import numpy as np
from numba import jit
from skimage import io
from pathlib import Path
from typing import Any, Callable


def convert_numpy(image: np.ndarray) -> np.ndarray:
    """
    Convert color image to black-and-white image

    adapted from:
        https://pythonspeed.com/articles/slow-numba/
        The wrong way to speed up your code with Numba
        by Itamar Turner-Trauring
        Last updated 21 Mar 2024, originally created 21 Mar 2024
    """

    bw_image = np.round(
        0.2989 * image[:, :, 0] +
        0.5870 * image[:, :, 1] +
        0.1140 * image[:, :, 2])

    return bw_image.astype(np.uint8)


@jit
def convert_numba(image: np.ndarray) -> np.ndarray:
    """
    Convert color image to black-and-white image

    adapted from:
        https://pythonspeed.com/articles/slow-numba/
        The wrong way to speed up your code with Numba
        by Itamar Turner-Trauring
        Last updated 21 Mar 2024, originally created 21 Mar 2024
    """

    bw_image = np.round(
        0.2989 * image[:, :, 0] +
        0.5870 * image[:, :, 1] +
        0.1140 * image[:, :, 2])

    return bw_image.astype(np.uint8)


@jit
def convert_numba_loop(image: np.ndarray) -> np.ndarray:
    """
    Convert color image to black-and-white image

    adapted from:
        https://pythonspeed.com/articles/slow-numba/
        The wrong way to speed up your code with Numba
        by Itamar Turner-Trauring
        Last updated 21 Mar 2024, originally created 21 Mar 2024
    """

    bw_image = np.empty(image.shape[:2], dtype=np.uint8)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            r, g, b = image[y, x, :]
            bw_image[y, x] = np.round(0.2989 * r + 0.5870 * g + 0.1140 * b)

    return bw_image.astype(np.uint8)


def profile_function(func: Callable, *args, **kwargs) -> tuple[Any, float, int]:
    """
    Run the function 'func' with provided parameters/arguments and record its
        execution time and maximum memory usage
    Return the function output, elapsed time, and maximum memory usage
    """

    tracemalloc.start()
    start_time = time.time()
    output = func(*args, **kwargs)
    end_time = time.time()
    max_memory = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    elapsed_time = end_time - start_time

    return output, elapsed_time, max_memory


def main():
    """
    Compares execution time and maximum memory usage for different methods of 
        converting a color image to black-and-white 
    The conversion methods are:
        1) using Numpy
        2) using Numba to 'jit' the Numpy method
        3) using a for-loop in Numba

    This project is derived from and replicates the results from:
        https://pythonspeed.com/articles/slow-numba/
        The wrong way to speed up your code with Numba
        by Itamar Turner-Trauring
        Last updated 21 Mar 2024, originally created 21 Mar 2024
    """

    input_path = Path.cwd() / 'input'
    input_filepaths = list(input_path.glob('*.JPG'))

    output_path = Path.cwd() / 'output'
    output_path.mkdir(exist_ok=True, parents=True)

    # the 'jitted' functions must run once to optimize their execution speed; 
    #   the first run is slow, while subsequent runs are much faster
    # a second image is used for subsequent, jit-optimized runs to measure 
    #   timing; while it is not a concern here, it is a good default practice 
    #   in case the results for the first image are cached by an overlooked 
    #   process
    for e in input_filepaths:

        image = io.imread(e)

        print('\n')

        bw_image_a, e_time, memory = profile_function(convert_numpy, image)
        print(f'numpy:      {e_time:.5f} seconds, {memory} bytes')

        bw_image_b, e_time, memory = profile_function(convert_numba, image)
        print(f'numba:      {e_time:.5f} seconds, {memory} bytes')
        assert np.array_equal(bw_image_a, bw_image_b)

        bw_image_c, e_time, memory = profile_function(convert_numba_loop, image)
        print(f'numba_loop: {e_time:.5f} seconds, {memory} bytes')
        assert np.array_equal(bw_image_a, bw_image_c)

        filename = e.stem + '_bw.jpg'
        output_filepath = output_path / filename
        io.imsave(output_filepath, bw_image_a)


if __name__ == '__main__':
    main()
