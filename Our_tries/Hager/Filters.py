import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from PyQt5 import QtWidgets,QtGui
QPixmap = QtGui.QPixmap


def corr(img,kernel_size,kernel_type):
    """
    Apply convolution operation to the image.
    Args:
        img (numpy.ndarray): The input image.
        kernel_size (int): The size of the kernel.
        kernel_type (numpy.ndarray): The convolution kernel.
    Returns:
        filtered_img(numpy.ndarray): The final filtered image.
    """
    padded_img = np.pad(img, ((kernel_size//2, kernel_size//2), (kernel_size//2, kernel_size//2)), mode='constant')
    filtered_img = np.zeros_like(img, dtype=np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            filtered_img[i, j] = np.sum(kernel_type * padded_img[i:i+kernel_size, j:j+kernel_size])
    return filtered_img
    
# -----------------------------------------------------------------Gaussian Filter-------------------------------------------------------------------------
def gaussian_kernel(kernel_size, sigma):
    """
    Generate a Gaussian kernel.
    Args:
        kernel_size (int): The size of the kernel.
        sigma (float): The standard deviation of the Gaussian distribution.
    Returns:
        numpy.ndarray: The Gaussian kernel.
    """
    gaussian =np.zeros((kernel_size,kernel_size))
    kernel_size=kernel_size//2
    for x in range(-kernel_size,kernel_size+1):
        for y in range(-kernel_size,kernel_size+1): 
            gaussian[x+kernel_size,y+kernel_size] = (1 / (2 * np.pi * sigma)) * np.exp(-(np.square(x) + np.square(y)) / (2 * np.square(sigma)))
    return gaussian

def gaussianFilter(img, kernel_size, sigma, is_gray=True):
    """
    Apply Gaussian filter to the image.
    Args:
        img (numpy.ndarray): The input image.
        kernel_size (int): The size of the Gaussian kernel.
        sigma (float): The standard deviation of the Gaussian distribution.
        is_gray (bool, optional): Whether the image is grayscale. Defaults to True.
    Returns:
        filtered_img (numpy.ndarray): The gaussian filtered image.
    """
    if is_gray:
        filtered_img = corr(img, kernel_size, gaussian_kernel(kernel_size, sigma))
        cv2.imwrite(r".\images\GaussianFilter.png",filtered_img)
        return filtered_img
    else:
        b, g, r = cv2.split(img)
        b_filtered = corr(b, kernel_size, gaussian_kernel(kernel_size, sigma))
        g_filtered = corr(g, kernel_size, gaussian_kernel(kernel_size, sigma))
        r_filtered = corr(r, kernel_size, gaussian_kernel(kernel_size, sigma))
        filtered_img = cv2.merge((b_filtered, g_filtered, r_filtered))
        cv2.imwrite(r".\images\GaussianFilter.png",filtered_img.astype(np.uint8))
        return filtered_img.astype(np.uint8)
# -----------------------------------------------------------------Average Filter-------------------------------------------------------------------------

def AvgFilter(img, kernel_size, is_gray=True):
    """
    Apply average filter to the image.
    Args:
        img (numpy.ndarray): The input image.
        kernel_size (int): The size of the average filter kernel.
        is_gray (bool, optional): Whether the image is grayscale. Defaults to True.
    Returns:
        avg_img (numpy.ndarray): The average filtered image.
    """
    if is_gray:  
        avg_kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)
        avg_img = corr(img, kernel_size, avg_kernel)
        cv2.imwrite(r".\images\AvgFilter.png", avg_img)
        return avg_img
    else: 
        b, g, r = cv2.split(img)
        b_filtered = corr(b, kernel_size, np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size))
        g_filtered = corr(g, kernel_size, np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size))
        r_filtered = corr(r, kernel_size, np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size))
        avg_img = cv2.merge((b_filtered, g_filtered, r_filtered))
        cv2.imwrite(r".\images\AvgFilter.png", avg_img.astype(np.uint8))
        return avg_img.astype(np.uint8)
# -----------------------------------------------------------------Median Filter-------------------------------------------------------------------------
    
def medianFilter(img, kernel_size, is_gray=True):
    """
    Apply median filter to the image.
    Args:
        img (numpy.ndarray): The input image.
        kernel_size (int): The size of the median filter kernel.
        is_gray (bool, optional): Whether the image is grayscale. Defaults to True.
    Returns:
        FilteredImage (numpy.ndarray): The median filtered image.
    """
    if is_gray:  
        w, h = img.shape
        FilteredImage = np.zeros_like(img,  dtype=np.float32)  
        pad_width = kernel_size // 2
        padded_img = np.pad(img, ((pad_width, pad_width), (pad_width, pad_width)), mode='constant')
        for i in range(pad_width, w + pad_width):
            for j in range(pad_width, h + pad_width):
                mask = padded_img[i-pad_width:i+pad_width+1, j-pad_width:j+pad_width+1]
                FilteredImage[i-pad_width, j-pad_width] = np.median(mask)
        cv2.imwrite(r".\images\MedianFilter.png", FilteredImage)
        return FilteredImage
    else: 
        b, g, r = cv2.split(img)
        b_filtered = medianFilter(b, kernel_size, is_gray=True)
        g_filtered = medianFilter(g, kernel_size, is_gray=True)
        r_filtered = medianFilter(r, kernel_size, is_gray=True)
        FilteredImage = cv2.merge((b_filtered, g_filtered, r_filtered))
        cv2.imwrite(r".\images\MedianFilter.png", FilteredImage.astype(np.uint8))
        return FilteredImage.astype(np.uint8)
# -----------------------------------------------------------------Testing Filter-------------------------------------------------------------------------

#-----------------for gray img -------------------
# img = cv2.imread('./images/1.jpg')
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # filtered_img = AvgFilter(gray_img,kernel_size=3)
# filtered_img = medianFilter(gray_img,kernel_size=3)
# # filtered_img = gaussianFilter(gray_img,kernel_size=3,sigma=2)
# # filtered_img = AvgFilter(gray_img,kernel_size=5)
# filtered_img = medianFilter(gray_img,kernel_size=5)
# filtered_img = gaussianFilter(gray_img,kernel_size=3,sigma=3)
# plt.imshow(gray_img,cmap='gray')
# plt.figure()
# plt.imshow(filtered_img,cmap='gray')
# plt.show()
#-----------------for colored img -------------------
# # filtered_img = AvgFilter(img,kernel_size=3,is_gray=False)
# filtered_img = medianFilter(img,kernel_size=3,is_gray=False)
# # filtered_img = gaussianFilter(img,kernel_size=3,sigma=2,is_gray=False)
# # filtered_img = AvgFilter(img,kernel_size=5,is_gray=False)
# filtered_img = medianFilter(img,kernel_size=5,is_gray=False)
# # filtered_img = gaussianFilter(img,kernel_size=5,sigma=3,is_gray=False)
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.figure()
# plt.imshow(cv2.cvtColor(filtered_img, cv2.COLOR_BGR2RGB))
# plt.show()
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------Edges----------------------------------------------------------------------------------
    
def apply_kernel(img: np.ndarray, horizontal_kernel: np.ndarray, vertical_kernel: np.ndarray, is_gray: bool = True, return_edge: bool = False):
    """
    Apply the given horizontal and vertical kernels to the image for edge detection.
    Parameters:
        img (np.ndarray): The input image.
        horizontal_kernel (np.ndarray): The horizontal kernel for convolution.
        vertical_kernel (np.ndarray): The vertical kernel for convolution.
        is_gray (bool, optional): Indicates whether the image is grayscale. Defaults to True.
        return_edge (bool, optional): Whether to return the edges of image with magnitude of image. Defaults to False.
    Returns:
        mag (np.ndarray): The edge image after applying the kernels.
    """
    if is_gray:
        horizontal_edge = convolve2d(img, horizontal_kernel, mode='same', boundary='symm')
        vertical_edge = convolve2d(img, vertical_kernel, mode='same', boundary='symm')
    else:
        b, g, r = cv2.split(img)
        b_horizontal_edge = convolve2d(b, horizontal_kernel, mode='same', boundary='symm')
        b_vertical_edge = convolve2d(b, vertical_kernel, mode='same', boundary='symm')
        g_horizontal_edge = convolve2d(g, horizontal_kernel, mode='same', boundary='symm')
        g_vertical_edge = convolve2d(g, vertical_kernel, mode='same', boundary='symm')
        r_horizontal_edge = convolve2d(r, horizontal_kernel, mode='same', boundary='symm')
        r_vertical_edge = convolve2d(r, vertical_kernel, mode='same', boundary='symm')
        horizontal_edge = cv2.merge((b_horizontal_edge, g_horizontal_edge, r_horizontal_edge))
        vertical_edge = cv2.merge((b_vertical_edge, g_vertical_edge, r_vertical_edge))
    mag = np.sqrt(np.power(horizontal_edge, 2.0) + np.power(vertical_edge, 2.0))
    if return_edge:
        return mag, horizontal_edge, vertical_edge
    return mag

# -------------------------------------------------------------------------Prewitt Edges----------------------------------------------------------------------------------
def prewitt_edge(img: np.ndarray, is_gray: bool = False):
    """
    Apply Prewitt edge detection to the image.
    Parameters:
        img (np.ndarray): The input image.
        is_gray (bool, optional): Indicates whether the image is grayscale. Defaults to False.
    Returns:
        PrewittEdges (np.ndarray): The edge image after applying Prewitt edge detection.
    """
    vertical = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    horizontal = np.flip(vertical.T)
    PrewittEdges = apply_kernel(img, horizontal, vertical, is_gray)
    return PrewittEdges

# -------------------------------------------------------------------------Roberts Edges----------------------------------------------------------------------------------
def roberts_edge(img: np.ndarray, is_gray: bool = False):
    """
    Apply Roberts edge detection to the image.
    Parameters:
        img (np.ndarray): The input image.
        is_gray (bool, optional): Indicates whether the image is grayscale. Defaults to False.
    Returns:
        RobertsEdges (np.ndarray): The edge image after applying Roberts edge detection.
    """
    vertical = np.array([[0, 1], [-1, 0]])
    horizontal = np.flip(vertical.T)
    RobertsEdges = apply_kernel(img, horizontal, vertical, is_gray)
    return RobertsEdges

# -------------------------------------------------------------------------Sobel Edges----------------------------------------------------------------------------------
def sobel_edge(img: np.ndarray, is_gray: bool = True, get_magnitude: bool = True, get_direction: bool = False):
    """
    Apply Sobel edge detection to the image.
    Parameters:
        img (np.ndarray): The input image.
        is_gray (bool, optional): Indicates whether the image is grayscale. Defaults to True.
        get_magnitude (bool, optional): Whether to return the magnitude of gradients. Defaults to True.
        get_direction (bool, optional): Whether to return the direction of gradients. Defaults to False.
    Returns:
        Union[SobelEdges (np.ndarray), Tuple[np.ndarray, np.ndarray])]: The edge image or tuple of magnitude and direction.
    """
    horizontal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    vertical = np.flip(horizontal.T)
    SobelEdges, horizontal_edge, vertical_edge = apply_kernel(img, horizontal, vertical, is_gray, True)
    horizontal_edge = horizontal_edge[:-2, :-2]
    vertical_edge = vertical_edge[:-2, :-2]
    if not get_magnitude:
        return horizontal_edge, vertical_edge
    if get_direction:
        direction = np.arctan2(vertical_edge, horizontal_edge)
        return SobelEdges, direction
    return SobelEdges

# -------------------------------------------------------------------------Canny Edges----------------------------------------------------------------------------------
def canny_edge(img: np.ndarray,is_gray: bool = True):
    """
    Apply Canny edge detection to the image.
    Parameters:
        img (np.ndarray): The input image.
        is_gray (bool, optional): Indicates whether the image is grayscale. Defaults to True.
    Returns:
        CannyEdges (np.ndarray): The edge image after applying Canny edge detection.
    Note:
        The Canny edge detection algorithm is typically applied to grayscale
        images. If the input image is colored and 'is_gray' is set to False,
        it will be converted to grayscale internally before processing.
    """
    if not is_gray:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # First Apply Gaussian Filter
    FilteredImage = gaussianFilter(img, kernel_size=3, sigma=9)
    # Second Get Gradient Magnitude & Direction
    GradientMagnitude, GradientDirection = sobel_edge(FilteredImage, get_magnitude=True, get_direction=True)
    # GradientMagnitude *= 255.0 / GradientMagnitude.max()
    print(GradientDirection.shape)
    # Third Apply Non-Maximum Suppression
    SuppressedImage = NonMaximumSuppression(GradientMagnitude, GradientDirection)
    # Fourth Apply Double Thresholding
    ThresholdedImage = DoubleThreshold(SuppressedImage, 0.05, 0.09)
    # Fifth Apply Hysteresis
    CannyEdges = Hysteresis(ThresholdedImage, 70, 255)
    return CannyEdges


def NonMaximumSuppression(GradientMagnitude: np.ndarray, GradientDirection: np.ndarray):
    """
    Perform non-maximum suppression on the gradient magnitude image.
    Parameters:
        GradientMagnitude (np.ndarray): The gradient magnitude image.
        GradientDirection (np.ndarray): The gradient direction image(in rad).
    Returns:
        SuppressedImage (np.ndarray): The result image after non-maximum suppression.
    Note:
        It's the third step in canny edge detection algorithm
    """
    M, N = GradientMagnitude.shape
    SuppressedImage = np.zeros(GradientMagnitude.shape)
    GradientDirection = np.rad2deg(GradientDirection)
    GradientDirection += 180
    for row in range(M):
        for col in range(N):
            try:
                direction = GradientDirection[row, col]
                # 0°
                if (0 <= direction < 22.5) or (337.5 <= direction <= 360):
                    before_pixel = GradientMagnitude[row, col - 1]
                    after_pixel = GradientMagnitude[row, col + 1]
                # 45°
                elif (22.5 <= direction < 67.5) or (202.5 <= direction < 247.5):
                    before_pixel = GradientMagnitude[row + 1, col - 1]
                    after_pixel = GradientMagnitude[row - 1, col + 1]
                # 90°
                elif (67.5 <= direction < 112.5) or (247.5 <= direction < 292.5):
                    before_pixel = GradientMagnitude[row - 1, col]
                    after_pixel = GradientMagnitude[row + 1, col]
                # 135°
                else:
                    before_pixel = GradientMagnitude[row - 1, col - 1]
                    after_pixel = GradientMagnitude[row + 1, col + 1]
                if GradientMagnitude[row, col]>= max(before_pixel,after_pixel):
                    SuppressedImage[row, col] = GradientMagnitude[row, col]
            except IndexError as e:
                pass
    return SuppressedImage

def DoubleThreshold(Image, LowThreshold, HighThreshold):
    """
    Apply double thresholding to the image.
    Parameters:
        Image (np.ndarray): The input image.
        LowThreshold (int): The low threshold value.
        HighThreshold (int): The high threshold value.
    Returns:
        ThresholdedImage (np.ndarray): The result image after double thresholding.
    Note:
        It's the fourth step in canny edge detection algorithm
    """
    ThresholdedImage = np.zeros(Image.shape)
    Weak =70
    Strong = 255
    StrongRow, StrongCol = np.where(Image >= HighThreshold)
    WeakRow, WeakCol = np.where((Image <= HighThreshold) & (Image >= LowThreshold))
    ThresholdedImage[StrongRow, StrongCol] = Strong
    ThresholdedImage[WeakRow, WeakCol] = Weak
    return ThresholdedImage
    

def Hysteresis(Image, Weak=70, Strong=255):
    """
    Apply hysteresis thresholding to the image.
    Parameters:
        Image (np.ndarray): The input image.
        Weak (int, optional): The weak threshold value. Defaults to 70.
        Strong (int, optional): The strong threshold value. Defaults to 255.
    Returns:
        Image (np.ndarray): The result after hysteresis thresholding.
    Note:
        It's the final step in canny edge detection algorithm
    """
    M, N = Image.shape
    for i in range(M):
        for j in range(N):
            if Image[i, j]== Weak:
                try:
                    if ((Image[i-1,j] == Strong) or (Image[i + 1, j] == Strong) or (
                            Image[i, j - 1] == Strong)
                            or (Image[i, j + 1] == Strong) or (Image[i+1, j + 1] == Strong)
                            or (Image[i - 1, j - 1] == Strong) or (Image[i - 1, j-1] == Strong) or (
                                    Image[i - 1, j + 1] == Strong)):
                        Image[i, j] = Strong
                    else:
                        Image[i, j] = 0
                except IndexError as e:
                    pass
    return Image

#-----------------for gray img -------------------
img = cv2.imread('./images/1.jpg')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# filtered_img = sobel_edge(gray_img,is_gray=True)
# filtered_img = prewitt_edge(gray_img,is_gray=True)
# filtered_img = roberts_edge(gray_img,is_gray=True)
# filtered_img = canny_edge(gray_img,is_gray=True)
# filtered_img = cv2.Canny(gray_img,0.05,0.09)
# plt.imshow(gray_img, cmap='gray')
# plt.figure()
# plt.imshow(filtered_img, cmap='gray')
# plt.show()
#-----------------for colored img -------------------
# filtered_img = sobel_edge(img,is_gray=False)
# filtered_img = prewitt_edge(img,is_gray=False)
# filtered_img = roberts_edge(img,is_gray=False)
# filtered_img = cv2.Canny(img,0.05,0.09)
filtered_img = canny_edge(img,is_gray=False)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.figure()
plt.imshow(cv2.cvtColor(filtered_img.astype(np.uint8), cv2.COLOR_BGR2RGB))
plt.show()