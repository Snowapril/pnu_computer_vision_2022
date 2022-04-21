from PIL import Image
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

def merge_image(im1, im2):
    """
    Merge two image into one PIL image
    @param im1: a PIL image to be merged on the left
    @param im2: a PIL image to be merged on the right
    """
    return Image.fromarray(np.hstack((
        np.array(im1), np.array(im2)
    )))

# Part1 - 1
print(boxfilter(3))
try:
    print(boxfilter(4))
except AssertionError:
    print("AssertionError: n must be an odd number")
print(boxfilter(7))

# Part1 - 2
print(gauss1d(0.3))
print(gauss1d(0.5))
print(gauss1d(1))
print(gauss1d(2))

# Part1 - 3
print(gauss2d(0.5))
print(gauss2d(1))

# Part1 - 4
dog = Image.open("2b_dog.bmp")
dog = dog.convert('L')
dog_array = np.asarray(dog)

filtered_dog_array = gaussconvolve2d(dog_array, 3)
filtered_dog = Image.fromarray(filtered_dog_array)

merge_image(dog, filtered_dog).show()

# Part2 - 1
cat = Image.open("2a_cat.bmp")
cat_array = np.asarray(cat)

sigma = 5
low_pass_cat_array_r = gaussconvolve2d(cat_array[:,:,0].astype(np.float64), sigma)
low_pass_cat_array_g = gaussconvolve2d(cat_array[:,:,1].astype(np.float64), sigma)
low_pass_cat_array_b = gaussconvolve2d(cat_array[:,:,2].astype(np.float64), sigma)

low_pass_cat_array = np.stack((
    low_pass_cat_array_r.astype(np.uint8), 
    low_pass_cat_array_g.astype(np.uint8), 
    low_pass_cat_array_b.astype(np.uint8)
), axis=2)
low_pass_cat = Image.fromarray(low_pass_cat_array)

merge_image(cat, low_pass_cat).show()

# Part2 - 2
dog = Image.open("2b_dog.bmp")
dog_array = np.asarray(dog)

sigma = 5
low_pass_dog_array_r = gaussconvolve2d(dog_array[:,:,0].astype(np.float64), sigma)
low_pass_dog_array_g = gaussconvolve2d(dog_array[:,:,1].astype(np.float64), sigma)
low_pass_dog_array_b = gaussconvolve2d(dog_array[:,:,2].astype(np.float64), sigma)

low_pass_dog_array = np.stack((
    low_pass_dog_array_r.astype(np.uint8), 
    low_pass_dog_array_g.astype(np.uint8), 
    low_pass_dog_array_b.astype(np.uint8)
), axis=2)

high_pass_dog_array = dog_array - low_pass_dog_array

merge_image(dog, Image.fromarray(high_pass_dog_array + 128)).show()

# Part2 - 3
hybrid_image_array = low_pass_cat_array + high_pass_dog_array
# After clamping
clamped_hybrid_image_array = np.clip(hybrid_image_array, 0, 255)

merge_image(Image.fromarray(hybrid_image_array), Image.fromarray(clamped_hybrid_image_array)).save("hybrid-image.jpg")