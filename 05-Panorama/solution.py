import numpy as np
import cv2
import math
import random

def FindBestMatches(descriptors1, descriptors2, threshold):
    """
    This function takes in descriptors of image 1 and image 2,
    and find matches between them. See assignment instructions for details.
    Inputs:
        descriptors: a K-by-128 array, where each row gives a descriptor
        for one of the K keypoints.  The descriptor is a 1D array of 128
        values with unit length.
        threshold: the threshold for the ratio test of "the distance to the nearest"
                   divided by "the distance to the second nearest neighbour".
                   pseudocode-wise: dist[best_idx]/dist[second_idx] <= threshold
    Outputs:
        matched_pairs: a list in the form [(i, j)] where i and j means
                       descriptors1[i] is matched with descriptors2[j].
    """
    assert isinstance(descriptors1, np.ndarray)
    assert isinstance(descriptors2, np.ndarray)
    assert isinstance(threshold, float)
    ## START
    ## the following is just a placeholder to show you the output format

    # lambda for calculate angle between two descriptor vectors
    calc_angle = lambda desc1, desc2 : np.arccos(np.dot(desc1, desc2))
    
    matched_pairs = []
    for (i, desc1) in enumerate(descriptors1):
        # Sort descriptors2 for finding first and second nearest neighbour
        first_neighbor, second_neighbor = sorted(descriptors2, key=lambda desc2: calc_angle(desc1, desc2))[0:2]
        if (calc_angle(desc1, first_neighbor) / calc_angle(desc1, second_neighbor)) < threshold:
            # Find index of best matched descriptor from descriptor2
            j = np.where((descriptors2 == first_neighbor).all(axis=1))[0][0]
            matched_pairs.append([i, j])

    ## END
    return matched_pairs


def KeypointProjection(xy_points, h):
    """
    This function projects a list of points in the source image to the
    reference image using a homography matrix `h`.
    Inputs:
        xy_points: numpy array, (num_points, 2)
        h: numpy array, (3, 3), the homography matrix
    Output:
        xy_points_out: numpy array, (num_points, 2), input points in
        the reference frame.
    """
    assert isinstance(xy_points, np.ndarray)
    assert isinstance(h, np.ndarray)
    assert xy_points.shape[1] == 2
    assert h.shape == (3, 3)

    # START
    # Convert xy_points into homogeneous coordinates
    xy_points_homo = np.concatenate(
        (xy_points, np.ones(shape=(xy_points.shape[0], 1))),
        axis=1
    )
    xy_points_out = []
    for xy_homo in xy_points_homo:
        # Conduct projection with homogeneous coordinate and homography matrix
        xy_proj_homo = np.matmul(xy_homo, h.T)
        # Convert back to original coordinates from homogeneous coordinate
        xy_proj = xy_proj_homo / (1e-10 if xy_proj_homo[2] == 0 else xy_proj_homo[2])
        # Append original coordinate points
        xy_points_out.append(xy_proj[0:2])

    # END
    return np.array(xy_points_out)

def RANSACHomography(xy_src, xy_ref, num_iter, tol):
    """
    Given matches of keyponit xy coordinates, perform RANSAC to obtain
    the homography matrix. At each iteration, this function randomly
    choose 4 matches from xy_src and xy_ref.  Compute the homography matrix
    using the 4 matches.  Project all source "xy_src" keypoints to the
    reference image.  Check how many projected keyponits are within a `tol`
    radius to the coresponding xy_ref points (a.k.a. inliers).  During the
    iterations, you should keep track of the iteration that yields the largest
    inlier set. After the iterations, you should use the biggest inlier set to
    compute the final homography matrix.
    Inputs:
        xy_src: a numpy array of xy coordinates, (num_matches, 2)
        xy_ref: a numpy array of xy coordinates, (num_matches, 2)
        num_iter: number of RANSAC iterations.
        tol: float
    Outputs:
        h: The final homography matrix.
    """
    assert isinstance(xy_src, np.ndarray)
    assert isinstance(xy_ref, np.ndarray)
    assert xy_src.shape == xy_ref.shape
    assert xy_src.shape[1] == 2
    assert isinstance(num_iter, int)
    assert isinstance(tol, (int, float))
    tol = tol*1.0

    # START
    max_inlier_num = 0
    optimal_homography = np.zeros(shape=(3,3))

    for _ in range(num_iter):
        # Randomly choose 4 matches
        idx = np.random.choice(xy_src.shape[0], 4, replace=False)
        p_xy_src = xy_src[idx]
        p_xy_ref = xy_ref[idx]
        # Calculate homography matrix with chosen points
        A = np.array([
            [p_xy_src[0][0], p_xy_src[0][1], 1, 0, 0, 0, -p_xy_ref[0][0] * p_xy_src[0][0], -p_xy_ref[0][0] * p_xy_src[0][1], -p_xy_ref[0][0]],
            [0, 0, 0, p_xy_src[0][0], p_xy_src[0][1], 1, -p_xy_ref[0][1] * p_xy_src[0][0], -p_xy_ref[0][1] * p_xy_src[0][1], -p_xy_ref[0][1]],

            [p_xy_src[1][0], p_xy_src[1][1], 1, 0, 0, 0, -p_xy_ref[1][0] * p_xy_src[1][0], -p_xy_ref[1][0] * p_xy_src[1][1], -p_xy_ref[1][0]],
            [0, 0, 0, p_xy_src[1][0], p_xy_src[1][1], 1, -p_xy_ref[1][1] * p_xy_src[1][0], -p_xy_ref[1][1] * p_xy_src[1][1], -p_xy_ref[1][1]],

            [p_xy_src[2][0], p_xy_src[2][1], 1, 0, 0, 0, -p_xy_ref[2][0] * p_xy_src[2][0], -p_xy_ref[2][0] * p_xy_src[2][1], -p_xy_ref[2][0]],
            [0, 0, 0, p_xy_src[2][0], p_xy_src[2][1], 1, -p_xy_ref[2][1] * p_xy_src[2][0], -p_xy_ref[2][1] * p_xy_src[2][1], -p_xy_ref[2][1]],

            [p_xy_src[3][0], p_xy_src[3][1], 1, 0, 0, 0, -p_xy_ref[3][0] * p_xy_src[3][0], -p_xy_ref[3][0] * p_xy_src[3][1], -p_xy_ref[3][0]],
            [0, 0, 0, p_xy_src[3][0], p_xy_src[3][1], 1, -p_xy_ref[3][1] * p_xy_src[3][0], -p_xy_ref[3][1] * p_xy_src[3][1], -p_xy_ref[3][1]],
        ])
        # Calculate eigen values and eigen vectors
        w, v = np.linalg.eig(np.matmul(A.T, A))
        # Get eigen vector with minimum eigen value
        eig_vector = v[:, np.argmin(w)]
        # Get the homography matrix
        h = eig_vector.reshape(3, 3)
        # Project source keypoints with calculated homography
        xy_proj = KeypointProjection(xy_src, h)
        # Count the number of inliers
        num_inliers = 0
        for i in range(xy_src.shape[0]):
            if np.linalg.norm(xy_proj[i] - xy_ref[i]) < tol:
                num_inliers += 1
        # Check if the number of inliers is the largest
        if num_inliers > max_inlier_num:
            max_inlier_num = num_inliers
            optimal_homography = h

    h = optimal_homography
    # END
    assert isinstance(h, np.ndarray)
    assert h.shape == (3, 3)
    return h