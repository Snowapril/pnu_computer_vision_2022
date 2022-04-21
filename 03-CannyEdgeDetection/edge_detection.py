from PIL import Image
import math
import numpy as np

def normalize(v):
    """
    Normalize a numpy array
    @param v: a numpy array
    @return: normalized array
    """
    assert np.sum(v) != 0, "sum of array must not be zero as it cause divide-by-zero error"
    return v / np.sum(v)

def boxfilter(n):
    """
    Box filter of size n
    @param n: size of the filter
    @return: a numpy array of size n*n
    @warning: n must be an odd number
    """
    assert n % 2 != 0, "n must be an odd number"
    return np.ones((n, n)) / (n * n)

def gauss1d(sigma):
    """
    Gaussian 1d filter of size n
    @param sigma: standard deviation of the gaussian
    @return: a numpy array of size n
    """
    n = round(6 * sigma)
    if n % 2 == 0: 
        n += 1
    x = np.arange(-n // 2 + 1, n // 2 + 1)
    g = np.exp(-x ** 2 / (2 * sigma ** 2))
    return normalize(g / np.sum(g))

def gauss2d(sigma):
    """
    Gaussian 2d filter of size n
    @param sigma: standard deviation of the gaussian
    @return: a numpy array of size n*n
    """
    g1 = gauss1d(sigma)
    return normalize(np.outer(g1, g1))

def convolve2d(array, filter):
    """
    Convolve an array with a filter
    @param array: a numpy array of size n*n
    @param filter: a numpy array of size n*n
    @return: a numpy array of size n*n
    """
    assert filter.shape[0] == filter.shape[1], "filter must be a square matrix"

    flipped_filter = np.fliplr(np.flipud(filter))
    m = int((flipped_filter.shape[0] - 1) / 2)
    zero_padded_array = np.pad(array, m, mode='constant')
    result = np.zeros_like(array, dtype=np.float32)
    for i in range(zero_padded_array.shape[0] - flipped_filter.shape[0] + 1):
        for j in range(zero_padded_array.shape[1] - flipped_filter.shape[1] + 1):
            result[i][j] = np.sum(flipped_filter * zero_padded_array[i:i + flipped_filter.shape[0], j:j + flipped_filter.shape[1]])
    return result

def gaussconvolve2d(array, sigma):
    """
    Convolve an array with a gaussian filter
    @param array: a numpy array of size n*n
    @param sigma: standard deviation of the gaussian
    @return: a numpy array of size n*n
    """
    return convolve2d(array, gauss2d(sigma))

def sobel_filters(img):
    """ Returns gradient magnitude and direction of input img.
    Args:
        img: Grayscale image. Numpy array of shape (H, W).
    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W).
        theta: Direction of gradient at each pixel in img.
            Numpy array of shape (H, W).
    Hints:
        - Use np.hypot and np.arctan2 to calculate square root and arctan
    """
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    img_x = convolve2d(img, sobel_x)
    img_y = convolve2d(img, sobel_y)
    
    G = np.hypot(img_x, img_y)
    G = (G / np.max(G)) * 255

    theta = np.arctan2(img_y, img_x)
    return (G, theta)

def non_max_suppression(G, theta):
    """ Performs non-maximum suppression.
    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).
    Args:
        G: gradient magnitude image with shape of (H, W).
        theta: direction of gradients with shape of (H, W).
    Returns:
        res: non-maxima suppressed image.
    """

    theta = np.where(theta < 0, theta + np.pi, theta)

    res = np.zeros_like(G)
    for i in range(1, G.shape[0] - 1):
        for j in range(1, G.shape[1] - 1):
            if G[i][j] == 0:
                continue
            rad = theta[i][j]
            if (0 <= rad < (np.pi / 8)) or ((7 * np.pi / 8) <= rad <= np.pi):
                if G[i][j] >= G[i][j+1] and G[i][j] >= G[i][j-1]:
                    res[i][j] = G[i][j]
            if ((np.pi / 8) <= rad < (3 * np.pi / 8)):
                if G[i][j] >= G[i+1][j-1] and G[i][j] >= G[i-1][j+1]:
                    res[i][j] = G[i][j]
            if ((3 * np.pi / 8) <= rad < (5 * np.pi / 8)):
                if G[i][j] >= G[i+1][j] and G[i][j] >= G[i-1][j]:
                    res[i][j] = G[i][j]
            if ((5 * np.pi / 8) <= rad < (7 * np.pi / 8)):
                if G[i][j] >= G[i+1][j+1] and G[i][j] >= G[i-1][j-1]:
                    res[i][j] = G[i][j]
    return res

def double_thresholding(img):
    """ 
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
    Returns:
        res: double_thresholded image.
    """
    diff = np.max(img) - np.min(img)
    t_high = np.min(img) + 0.15 * diff
    t_low = np.min(img) + 0.03 * diff

    original_shape = img.shape

    weak_pixel = np.where(np.logical_and(img >= t_low, img < t_high), 80, 0)
    strong_pixel = np.where(img >= t_high, 255, 0)

    return (weak_pixel + strong_pixel).reshape(original_shape)

def hysteresis(img):
    """ Find weak edges connected to strong edges and link them.
    Iterate over each pixel in strong_edges and perform depth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
    Returns:
        res: hysteresised image.
    """
    strong_pixel_locs = np.where(img == 255)

    delta = [
        (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)
    ]

    result = np.zeros_like(img)
    for (x, y) in zip(iter(strong_pixel_locs[0]), iter(strong_pixel_locs[1])):
        visited = []
        stack = [(x, y)]

        while stack:
            (x, y) = stack.pop()
            visited.append((x, y))
            result[x][y] = 255

            for (dx, dy) in delta:
                if (x + dx, y + dy) in visited:
                    continue

                if img[x+dx][y+dy] == 80:
                    stack.append((x + dx, y + dy))

    return result

# Part 1
iguana = Image.open("iguana.bmp")
iguana_array = np.asarray(iguana.convert('L'), dtype=np.float32)

filtered_iguana = gaussconvolve2d(iguana_array, 1.6)

# Part 2
(G, theta) = sobel_filters(filtered_iguana)

# Part 3
non_max_suppression_result = non_max_suppression(G, theta)

# Part 4
double_thresholding_result = double_thresholding(non_max_suppression_result)

# Part 5
hysteresis_result = hysteresis(double_thresholding_result)

# Render All images
Image.fromarray(np.vstack((
    np.hstack((iguana_array, filtered_iguana, G)),
    np.hstack((non_max_suppression_result, double_thresholding_result, hysteresis_result))
))).convert("L").save("output.jpg")