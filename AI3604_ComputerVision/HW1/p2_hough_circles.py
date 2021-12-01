#!/usr/bin/env python3
import cv2
import sys
import numpy as np
from tqdm import tqdm


class SobelEdger():
    def __init__(self):
        self.sobel_kernels = {
            'x': np.array([
                [-1., 0., 1.],
                [-2., 0., 2.],
                [-1., 0., 1.]
            ]),
            'y': np.array([
                [1., 2., 1.],
                [0., 0., 0.],
                [-1., -2., -1.]
            ]),
        }

    def _scale(self, image, factor=0.25):
        return factor * image

    def _normalize(self, image):
        return image / image.max() * 255

    def get_edges(self, image):
        image = self._scale(image)
        x_kernel = self.sobel_kernels['x']
        x_edges = cv2.filter2D(
            image, cv2.CV_64F, x_kernel,
            borderType=cv2.BORDER_REPLICATE)
        y_kernel = self.sobel_kernels['y']
        y_edges = cv2.filter2D(
            image, cv2.CV_64F, y_kernel,
            borderType=cv2.BORDER_REPLICATE)

        edges = self._merge(x_edges, y_edges)

        return self._normalize(edges)

    def get_edges_by_cv2(self, image):
        return cv2.Sobel(image, cv2.CV_8U, dx=1, dy=1, ksize=3)

    def _merge(self, x_edges, y_edges):
        return np.abs(x_edges) + np.abs(y_edges)


def detect_edges(image):
    """Find edge points in a grayscale image.

    Args:
    - image (2D uint8 array): A grayscale image.

    Return:
    - edge_image (2D float array): A heat map where the intensity at each point
        is proportional to the edge magnitude.
    """
    return SobelEdger().get_edges(image)


def hough_circles(edge_image, edge_thresh, radius_values):
    """Threshold edge image and calculate the Hough transform accumulator array.

    Args:
    - edge_image (2D float array): An H x W heat map where the intensity at each
        point is proportional to the edge magnitude.
    - edge_thresh (float): A threshold on the edge magnitude values.
    - radius_values (1D int array): An array of R possible radius values.

    Return:
    - thresh_edge_image (2D bool array): Thresholded edge image indicating
        whether each pixel is an edge point or not.
    - accum_array (3D int array): Hough transform accumulator array. Should have
        shape R x H x W.
    """
    thresh_edge_image: np.ndarray = np.where(
        edge_image > edge_thresh, 255, 0)
    H, W = thresh_edge_image.shape
    R = len(radius_values)
    accum_array = np.zeros((R, H, W))  # r, y, x
    t = tqdm(range(R))
    y, x = np.nonzero(thresh_edge_image)
    for idx, r in enumerate(radius_values):
        for theta in range(360):
            b = np.floor(y - r * np.sin(np.deg2rad(theta)))
            a = np.floor(x - r * np.cos(np.deg2rad(theta)))
            for b_, a_ in zip(b, a):
                b_ = int(b_)
                a_ = int(a_)
                if b_ < 0 or a_ < 0:
                    continue
                if b_ >= H or a_ >= W:
                    continue
                accum_array[idx, b_, a_] += 1
        t.update(1)
    return thresh_edge_image, accum_array


def find_circles(image, accum_array, radius_values, hough_thresh):
    """Find circles in an image using output from Hough transform.

    Args:
    - image (3D uint8 array): An H x W x 3 BGR color image. Here we use the
        original color image instead of its grayscale version so the circles
        can be drawn in color.
    - accum_array (3D int array): Hough transform accumulator array having shape
        R x H x W.
    - radius_values (1D int array): An array of R radius values.
    - hough_thresh (int): A threshold of votes in the accumulator array.

    Return:
    - circles (list of 3-tuples): A list of circle parameters. Each element
        (r, y, x) represents the radius and the center coordinates of a circle
        found by the program.
    - circle_image (3D uint8 array): A copy of the original image with detected
        circles drawn in color.
    """
    indices = np.nonzero(accum_array >= hough_thresh)
    indicesT = np.array(indices).T
    final_candidates = []
    while indicesT.shape[0]:
        votes = accum_array[tuple(indicesT.T)]
        max_index = indicesT[np.argmax(votes)]
        rm, ym, xm = max_index
        final_candidates.append(max_index)
        tobe_deleted = []
        for idx, (r, y, x) in enumerate(indicesT):
            if np.sqrt((y-ym)**2 + (x-xm)**2) < 10:
                tobe_deleted.append(idx)

        indicesT = np.delete(indicesT, tobe_deleted, axis=0)

    for r, y, x in final_candidates:
        image = cv2.circle(image, (x, y), radius_values[r], (0, 255, 0), 2)

    return image


def main(argv):
    R = range(20, 41)
    EDGE_THRESH = 110
    CIRCLE_THRESH = 120
    img_name = argv[0]
    img = cv2.imread('data/' + img_name + '.png', cv2.IMREAD_COLOR)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edged_image = detect_edges(gray_image)
    thresh_edge_imgae, accum_array = hough_circles(
        edged_image, EDGE_THRESH, R)
    circle_image = find_circles(img, accum_array, R, CIRCLE_THRESH)

    cv2.imwrite('output/' + img_name + "_detection.png", edged_image)
    cv2.imwrite('output/' + img_name + "_edges.png", thresh_edge_imgae)
    cv2.imwrite('output/' + img_name + "_circles.png", circle_image)


if __name__ == '__main__':
    main(sys.argv[1:])
