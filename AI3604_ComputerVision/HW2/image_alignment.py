#!/usr/bin/env python3
import cv2
import archotech
import numpy as np
import scipy.ndimage
from typing import List, Tuple
import matplotlib.pyplot as plt


def detect_blobs(image: np.ndarray):
    """Laplacian blob detector.

    Args:
    - image (2D float64 array): A grayscale image.

    Returns:
    - corners (list of 2-tuples): A list of 2-tuples representing the locations
        of detected blobs. Each tuple contains the (x, y) coordinates of a
        pixel, which can be indexed by image[y, x].
    - scales (list of floats): A list of floats representing the scales of
        detected blobs. Has the same length as `corners`.
    - orientations (list of floats): A list of floats representing the dominant
        orientation of the blobs.
    """
    BLOB_THRESHOLD = 0.225
    sigma_0 = 5
    sigmas = []
    s = 1.25  # scaling factor
    local_maximums = []
    MAX_K = 8
    for k in range(MAX_K):
        sigmas.append(sigma_0 * (s ** k))
    for k in range(MAX_K):
        sigma = sigmas[k]
        norm_factor = sigma**2
        filtered = scipy.ndimage.gaussian_filter(image, sigma)
        filtered = np.abs(scipy.ndimage.laplace(filtered))
  
        normalized = norm_factor * filtered
        local_maximum = scipy.ndimage.maximum_filter(
            normalized,
            size=int(sigma))
        local_maximums.append(local_maximum)

        # plt.imshow(local_maximum)
        # plt.show()

    blobs = np.max(local_maximums, axis=0)
    sizes = np.argmax(local_maximums, axis=0)
    blob_bin: np.ndarray = np.where(blobs >= BLOB_THRESHOLD, 255, 0)

    labeled_blobs = archotech.ImageDivider(blob_bin).object_segmentation()
    attr_list = archotech.AttributeCounter(labeled_blobs).get_attr_list()

    # For debugging and visualization purposes
    # annot_blobs = archotech.annotate_attributes(image, attr_list)
    # plt.imshow(annot_blobs)
    corners: List[Tuple] = []
    scales: List = []
    orientations: List = []

    for attr in attr_list:
        x, y = attr['position']['x'], attr['position']['y']
        corners.append((x, y))
        scales.append(sigmas[sizes[int(y), int(x)]])
        orientations.append(attr['orientation'])

    # print(corners)
    # print(scales)
    # print(orientations)

    return corners, scales, orientations


def compute_descriptors(image, corners, scales, orientations):
    """Compute descriptors for corners at specified scales.

    Args:
    - image (2d float64 array): A grayscale image.
    - corners (list of 2-tuples): A list of (x, y) coordinates.
    - scales (list of floats): A list of scales corresponding to the corners.
        Must have the same length as `corners`.
    - orientations (list of floats): A list of floats representing the dominant
        orientation of the blobs.

    Returns:
    - descriptors (list of 1d array): A list of desciptors for each corner.
        Each element is an 1d array of length 128.
    """
    if len(corners) != len(scales) or len(corners) != len(orientations):
        raise ValueError(
            '`corners`, `scales` and `orientations`'
            + 'must all have the same length.')

    raise NotImplementedError


def match_descriptors(descriptors1, descriptors2):
    """Match descriptors based on their L2-distance and the "ratio test".

    Args:
    - descriptors1 (list of 1d arrays):
    - descriptors2 (list of 1d arrays):

    Returns:
    - matches (list of 2-tuples): A list of 2-tuples representing the matching
        indices. Each tuple contains two integer indices. For example, tuple
        (0, 42) indicates that corners1[0] is matched to corners2[42].
    """
    raise NotImplementedError


def draw_matches(image1, image2, corners1, corners2, matches,
                 outlier_labels=None):
    """Draw matched corners between images.

    Args:
    - matches (list of 2-tuples)
    - image1 (3D uint8 array): A color image having shape (H1, W1, 3).
    - image2 (3D uint8 array): A color image having shape (H2, W2, 3).
    - corners1 (list of 2-tuples)
    - corners2 (list of 2-tuples)
    - outlier_labels (list of bool)

    Returns:
    - match_image (3D uint8 array): A color image having shape
        (max(H1, H2), W1 + W2, 3).
    """
    raise NotImplementedError


def compute_affine_xform(corners1, corners2, matches):
    """Compute affine transformation given matched feature locations.

    Args:
    - corners1 (list of 2-tuples)
    - corners1 (list of 2-tuples)
    - matches (list of 2-tuples)

    Returns:
    - xform (2D float64 array): A 3x3 matrix representing the affine
        transformation that maps coordinates in image1 to the corresponding
        coordinates in image2.
    - outlier_labels (list of bool): A list of Boolean values indicating
        whether the corresponding match in `matches` is an outlier or not.
        For example, if `matches[42]` is determined as an outlier match
        after RANSAC, then `outlier_labels[42]` should have value `True`.
    """
    raise NotImplementedError


def stitch_images(image1, image2, xform):
    """Stitch two matched images given the transformation between them.

    Args:
    - image1 (3D uint8 array): A color image.
    - image2 (3D uint8 array): A color image.
    - xform (2D float64 array): A 3x3 matrix representing the transformation
        between image1 and image2. This transformation should map coordinates
        in image1 to the corresponding coordinates in image2.

    Returns:
    - image_stitched (3D uint8 array)
    """
    raise NotImplementedError


def main():
    img_path1 = 'data/bikes1.png'
    img_path2 = 'data/bikes2.png'

    img1 = cv2.imread(img_path1, cv2.IMREAD_COLOR)
    img2 = cv2.imread(img_path2, cv2.IMREAD_COLOR)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) / 255.0
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) / 255.0

    # TODO
    detect_blobs(gray1)


if __name__ == '__main__':
    main()
