#!/usr/bin/env python3
import cv2
import archotech
import numpy as np
import scipy.ndimage
from typing import List, Tuple
# import matplotlib.pyplot as plt


BLOB_QUANTILE = 0.975
RATIO_THRES = 0.7
DIST_THRES = 12


def detect_blobs(image):
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
    sigma_0 = 5
    sigmas = []
    s = 1.5  # scaling factor
    gaussian_images = []
    MAX_K = 6
    for k in range(MAX_K):
        sigmas.append(sigma_0 * (s ** k))
    for k in range(MAX_K):
        sigma = sigmas[k]
        filtered = scipy.ndimage.gaussian_filter(image, sigma)
        gaussian_images.append(filtered)

        # plt.imshow(local_maximum)
        # plt.show()
    DoGs = np.diff(gaussian_images, n=1, axis=0)

    local_maximums = scipy.ndimage.maximum_filter(DoGs, size=(3, 3, 3))

    blobs = np.max(local_maximums, axis=0)
    thres = np.quantile(blobs, BLOB_QUANTILE)
    sizes = np.argmax(local_maximums, axis=0)
    blob_bin: np.ndarray = np.where(blobs >= thres, 255, 0)

    labeled_blobs = archotech.ImageDivider(blob_bin).object_segmentation()
    attr_list = archotech.AttributeCounter(labeled_blobs).get_attr_list()

    # For debugging and visualization purposes
    # annot_blobs = archotech.annotate_attributes(image, attr_list)
    # plt.imshow(labeled_blobs)
    # plt.show()

    corners: List[Tuple] = []
    scales: List = []
    orientations: List = []

    for blob_attr in attr_list:
        # x_, y_ = blob_attr['position']['x'], blob_attr['position']['y']
        # x, y = int(x_), int(y_)
        # scale = sigmas[sizes[y, x]]

        # g_image = gaussian_images[sizes[y, x]]

        # if not _check_window_validity(x, y, image.shape):
        #     continue

        # local: np.ndarray = g_image[y-8: y+8+1, x-8: x+8+1]
        # try:
        #     # sometimes blob size is too small to compute gradient
        #     dx, dy = np.gradient(local)
        # except ValueError:
        #     corners.append((x_, y_))
        #     scales.append(scale)
        #     orientations.append(blob_attr['orientation'])
        #     continue
        # grad_mag = np.sqrt(np.square(dx) + np.square(dy))
        # grad_ori = np.arctan2(dx, dy)

        # # add gaussian weight to magnitude
        # gaussian = cv2.getGaussianKernel(
        #     17, sigma=1.5*scale, ktype=cv2.CV_64F)
        # gaussian = np.matmul(gaussian, gaussian.T)
        # grad_mag = np.multiply(grad_mag, gaussian)

        # grad_hist = np.zeros(36)
        # for yy in range(grad_ori.shape[0]):
        #     for xx in range(grad_ori.shape[1]):
        #         o = grad_ori[yy, xx]
        #         m = grad_mag[yy, xx]
        #         while o < 0:
        #             o += 2 * np.pi

        #         # compute bins
        #         relative_o = (o / (np.pi / 18))
        #         lbin_idx = int(relative_o)
        #         bli = relative_o - lbin_idx  # interpolation factor

        #         # assign current gradient to bins
        #         grad_hist[lbin_idx % 8] += (1 - bli) * m
        #         grad_hist[(lbin_idx + 1) % 8] += bli * m

        # candidates = np.argsort(grad_hist)
        # best = candidates[-1]
        # for c in candidates[:-1]:
        #     if grad_hist[c] / grad_hist[best] >= 0.8:
        #         corners.append((x, y))
        #         scales.append(scale)
        #         orientations.append(grad_hist[c])

        # corners.append((x_, y_))
        # scales.append(scale)
        # orientations.append(grad_hist[best])
        x, y = blob_attr['position']['x'], blob_attr['position']['y']
        corners.append((x, y))
        scales.append(sigmas[sizes[int(y), int(x)]])
        orientations.append(blob_attr['orientation'])

    # print(corners)
    # print(scales)
    # print(orientations)

    return corners, scales, orientations


def _check_window_validity(
        x: int,
        y: int,
        shape: Tuple) -> bool:
    """Checks if a window lies too close to the border,
    in which case the histogram cannot be computed.

    NOTE: Assumes the window size is (17, 17)

    Args:
        - x (int): x coordinate of centroid
        - y (int): y coordinate of centroid
        - shape (Tuple): Shape of the image

    Returns:
        False if the window is too close to image border.
        True otherwise
    """
    y_max, x_max = shape
    if x + 8 + 1 >= x_max or x - 8 < 0:
        return False
    if y + 8 + 1 >= y_max or y - 8 < 0:
        return False
    return True


def compute_descriptors(image: np.ndarray, corners, scales, orientations):
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

    descriptors: List[np.ndarray] = []
    for (x, y), scale, ori in zip(corners, scales, orientations):
        x, y = int(x), int(y)
        scale = int(scale)

        if not _check_window_validity(x, y, image.shape):
            descriptors.append(None)
            continue

        local: np.ndarray = image[y-8: y+8+1, x-8: x+8+1]
        dx, dy = np.gradient(local)

        grad_mag = np.sqrt(np.square(dx) + np.square(dy))
        grad_ori = np.arctan2(dx, dy)

        # normalize gradient orientation
        grad_ori = grad_ori - ori

        # add gaussian weight to magnitude
        gaussian = cv2.getGaussianKernel(
            17, sigma=1.5*scale, ktype=cv2.CV_64F)
        gaussian = np.matmul(gaussian, gaussian.T)
        grad_mag = np.multiply(grad_mag, gaussian)

        # compute histogram
        grad_ori = grad_ori[:16, :16]
        grad_mag = grad_mag[:16, :16]
        BSZ = 4  # block size
        hs = []
        for xb in range(4):
            for yb in range(4):
                h = [0] * 8  # histogram
                cob = grad_ori[
                    xb*BSZ: (xb+1)*BSZ,
                    yb*BSZ: (yb+1)*BSZ]  # (4,4) orientation block
                cmb = grad_mag[
                    xb*BSZ: (xb+1)*BSZ,
                    yb*BSZ: (yb+1)*BSZ]  # (4,4) magnitude block
                for o, m in zip(cob.flatten(), cmb.flatten()):
                    # make orientation in (0, 2pi)
                    while o < 0:
                        o += 2 * np.pi

                    # compute bins
                    relative_o = (o / (np.pi / 4))
                    lbin_idx = int(relative_o)
                    bli = relative_o - lbin_idx  # interpolation factor

                    # assign current gradient to bins
                    h[lbin_idx % 8] += (1 - bli) * m
                    h[(lbin_idx + 1) % 8] += bli * m
                hs.append(h)
        hs: np.ndarray = np.array(hs).reshape(-1)

        # normalization according to SIFT paper
        # normalize to unit length
        hs = hs / np.linalg.norm(hs, ord=2)
        # restrict each to be no longer than 0.2
        hs = np.clip(hs, -0.2, 0.2)
        # re-normalize
        hs = hs / np.linalg.norm(hs, ord=2)
        descriptors.append(hs)

    return descriptors


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
    LARGE_NUMBER = 1e8
    matches: List[Tuple] = []
    dist_mat = np.zeros((len(descriptors1), len(descriptors2)))
    for i, d1 in enumerate(descriptors1):
        for j, d2 in enumerate(descriptors2):
            if d1 is None or d2 is None:
                dist_mat[i, j] = LARGE_NUMBER
            else:
                dist_mat[i, j] = np.linalg.norm((d1 - d2), ord=2)
        candidates = np.argsort(dist_mat[i, :])
        best, second = candidates[:2]
        if dist_mat[i, best] / dist_mat[i, second] <= RATIO_THRES:
            matches.append((i, best))

    return matches


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
    if matches is None:
        H = max(image1.shape[0], image2.shape[0])
        image1 = np.pad(image1, [(0, max(0, H-image1.shape[0])), (0, 0), (0, 0)])
        image2 = np.pad(image2, [(0, max(0, H-image2.shape[0])), (0, 0), (0, 0)])
        offset = image1.shape[1]
        match_image = np.concatenate((image1, image2), axis=1)
    if outlier_labels is None:
        colors = [(255, 0, 0)] * len(matches)
    else:
        colors = [
            (0, 0, 255) if is_outlier else (255, 0, 0)
            for is_outlier in outlier_labels]
    H = max(image1.shape[0], image2.shape[0])
    image1 = np.pad(image1, [(0, max(0, H-image1.shape[0])), (0, 0), (0, 0)])
    image2 = np.pad(image2, [(0, max(0, H-image2.shape[0])), (0, 0), (0, 0)])
    offset = image1.shape[1]
    match_image = np.concatenate((image1, image2), axis=1)
    for c, (c1idx, c2idx) in zip(colors, matches):
        x1, y1 = list(map(int, corners1[c1idx]))
        x2, y2 = list(map(int, corners2[c2idx]))
        x2 += offset
        match_image = cv2.line(
            match_image,
            (x1, y1),
            (x2, y2),
            color=c,
            thickness=3)
        match_image = cv2.circle(
            match_image,
            (x1, y1),
            radius=8,
            color=(0, 255, 0),
            thickness=-1)
        match_image = cv2.circle(
            match_image,
            (x2, y2),
            radius=8,
            color=(0, 255, 0),
            thickness=-1)

    return match_image


def _formulate_lstsq(corners1, corners2, n_matches, matches):
    A = np.zeros((2 * n_matches, 6))
    b = np.zeros(2 * n_matches)
    for i, (c1idx, c2idx) in enumerate(matches):
        x1, y1 = list(map(int, corners1[c1idx]))
        x1 += 200
        y1 += 200
        x2, y2 = list(map(int, corners2[c2idx]))
        x2 += 200
        y2 += 200
        A[2*i, :] = [x1, y1, 1, 0, 0, 0]
        A[2*i+1, :] = [0, 0, 0, x1, y1, 1]
        b[2*i] = x2
        b[2*i+1] = y2
    return A, b


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
    if len(matches) < 3:
        print(f"Not enough matches! Expected at least 3. Got {len(matches)}")
        return None, None
    N = 100
    MIN_MATCHES = 3
    N_MATCHES = len(matches)
    matches = np.array(matches)
    src, dst = _formulate_lstsq(  # (2n, 6), (2n,)
        corners1, corners2, N_MATCHES, matches)
    max_inliers = -1
    xform = np.zeros((3, 3))
    for _ in range(N):
        samples = np.random.randint(matches.shape[0], size=(3,))
        A_, b_ = _formulate_lstsq(  # (2m, 6), (2m,)
            corners1, corners2, MIN_MATCHES, matches[samples])
        t = np.linalg.lstsq(A_, b_, rcond=1e-5)[0]  # (6,)
        loss = np.matmul(src, t) - dst
        loss = loss.reshape(N_MATCHES, 2)
        distances = np.sqrt(np.sum(loss**2, axis=1))
        inliers = distances < DIST_THRES
        if np.sum(inliers) > max_inliers:
            max_inliers = np.sum(inliers)
            A, b = _formulate_lstsq(
                corners1, corners2, max_inliers, matches[inliers])
            t = np.linalg.lstsq(A, b, rcond=1e-5)[0]
            xform[0, :] = t[:3]
            xform[1, :] = t[3:]
            xform[2, 2] = 1
            outliers = ~inliers
    return xform, outliers


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
    image1 = np.pad(image1, [(200, 200), (200, 200), (0, 0)])
    image2 = np.pad(image2, [(200, 200), (200, 200), (0, 0)])
    if xform is None:
        return image1.astype(np.uint8)
    image_warped: np.ndarray = cv2.warpAffine(
        image1,
        xform[:2],
        image2.shape[:2][::-1])
    image_stitched = image_warped.astype(np.float32)\
        + image2.astype(np.float32)
    image_stitched[(image2 > 1e-3) & (image_warped > 1e-3)] /= 2.0
    return image_stitched.astype(np.uint8)


def main():
    img_name = 'wall'
    img_id1 = 1
    img_id2 = 3
    img_path1 = f'data/{img_name}{img_id1}.png'
    img_path2 = f'data/{img_name}{img_id2}.png'

    img1 = cv2.imread(img_path1, cv2.IMREAD_COLOR)
    img2 = cv2.imread(img_path2, cv2.IMREAD_COLOR)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) / 255.0
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) / 255.0

    # TODO
    print(f'Matching {img_name} {img_id1} and {img_id2}')

    c1, s1, ori1 = detect_blobs(gray1)
    c2, s2, ori2 = detect_blobs(gray2)

    print('Num of Blobs:', len(c1), len(c2))

    d1 = compute_descriptors(gray1, c1, s1, ori1)
    d2 = compute_descriptors(gray2, c2, s2, ori2)

    print('Num of Descriptors', len(d1), len(d2))

    matches = match_descriptors(d1, d2)
    print('Matches:', matches)

    xform, outliers = compute_affine_xform(c1, c2, matches)

    visualization = draw_matches(img1, img2, c1, c2, matches, outliers)
    cv2.imwrite(
        f'./data/{img_name}_{img_id1}{img_id2}_match.png',
        visualization.astype(np.uint8))

    if xform is None:
        print('Matching failed. Exiting.')

    stitched = stitch_images(img1, img2, xform)
    cv2.imwrite(
        f'./data/{img_name}_{img_id1}{img_id2}_stitch.png',
        stitched)


if __name__ == '__main__':
    main()
