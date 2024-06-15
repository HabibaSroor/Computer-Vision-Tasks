import cv2
import numpy as np


def optimal_threshold(image_path):
    # Load the image in grayscale mode
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    enhanced_image = cv2.convertScaleAbs(image, alpha=1, beta=0)  # needed to be adjusted for some images
    filtered_image = cv2.GaussianBlur(enhanced_image, (3, 3), 0)
    # Normalize the image to range [0, 1]
    # filtered_image = enhanced_image2 / np.max(enhanced_image2)
    # Get the height and width of the image
    height, width = filtered_image.shape
    # Get the indices of the four corners
    corners = [(0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)]
    # Initialize the lists for pixel values
    background = [filtered_image[y, x] for (y, x) in corners]
    objects = [filtered_image[y, x] for y in range(height) for x in range(width) if (y, x) not in corners]
    u_b = np.mean(background)
    u_o = np.mean(objects)
    # Initial threshold
    T = ((u_b + u_o) / 2)
    print(f"initial T = {T} ")
    while True:
        background = filtered_image[filtered_image < T]
        objects = filtered_image[filtered_image > T]
        if len(background) == 0:
            print(f"Empty background!")
            break
        if len(objects) == 0:
            print(f"Empty objects!")
            break
        u_b = np.mean(background)
        u_o = np.mean(objects)
        T_new = (u_b + u_o) / 2
        print(f"T_new = {T_new}")
        if T_new == T:
            break
        T = T_new
    print(f"final T = {T}")
    # Apply thresholding using numpy indexing
    binary_image = np.where(filtered_image < T, 0, 255).astype(np.uint8)
    return binary_image


def local_optimal_thresholding(image_path, kernel_size):
    # Load the image in grayscale mode
    filtered_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # enhanced_image = cv2.convertScaleAbs(image, alpha=0.9, beta=0)
    # filtered_image = cv2.GaussianBlur(enhanced_image, (3, 3), 0)
    # Get the height and width of the image
    height, width = filtered_image.shape
    # Initialize the output image
    binary_image = np.zeros_like(filtered_image)
    # kernel_size = 49
    # Calculate the amount of padding needed
    padding_height = kernel_size - (height % kernel_size)
    padding_width = kernel_size - (width % kernel_size)
    # Create a new array with padding
    padded_image = np.zeros((filtered_image.shape[0] + padding_height, filtered_image.shape[1] + padding_width))
    # Copy the original image into the padded image
    padded_image[:filtered_image.shape[0], :filtered_image.shape[1]] = filtered_image
    # Get the height and width of the padded image
    height2, width2 = padded_image.shape

    # Iterate over the image with a sliding window
    for y in range(0, height2 - kernel_size, kernel_size):
        for x in range(0, width2 - kernel_size, kernel_size):
            # Get the current patch
            patch = filtered_image[y:y + kernel_size, x:x + kernel_size]
            # Calculate the local threshold using optimal_threshold function
            binary_patch = optimal_threshold(patch)
            # Apply the local threshold to the patch
            binary_image[y:y + kernel_size, x:x + kernel_size] = binary_patch

    # Crop the binary image to the original size
    binary_image = binary_image[:height, :width]
    return binary_image


def builtin_optimal_threshold(image_path):
    # Load the image in grayscale mode
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # filtered_image = cv2.medianBlur(image, 5)
    filtered_image = cv2.GaussianBlur(image, (3, 3), 0)
    ret, th1 = cv2.threshold(filtered_image, 148, 255, cv2.THRESH_BINARY)
    return th1


def builtin_local_thresholding(image_path):
    # Load the image in grayscale mode
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # filtered_image = cv2.medianBlur(image, 3)
    th2 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 45, 2)
    return th2

# use the functions
# image_path = r'D:\SBME\SBE_3.2\CV\optimal_thre_input.jpg'
image_path = r'C:/Users/ashra/OneDrive/Documents/Filtering_and_Edge_Detection_Studio/test-images/good_global_better_local.jpg'
# image_path = r'D:\SBME\SBE_3.2\CV\Assignment1\Filtering_and_Edge_Detection_Studio\test-images\2.jpg'
# image_path = r'D:\SBME\SBE_3.2\CV\Assignment1\Filtering_and_Edge_Detection_Studio\test-images\6.jpg'
# image_path = (r'D:\SBME\SBE_3.2\CV\Assignment1\Filtering_and_Edge_Detection_Studio\test-images\bad_global_threshold.jpeg')

# Call the optimal_threshold function
binary_image = optimal_threshold(image_path)
# binary_image = local_optimal_thresholding(image_path)

# Built-in functions
thresh = builtin_optimal_threshold(r'C:/Users/ashra/OneDrive/Documents/Filtering_and_Edge_Detection_Studio/test-images/good_global_better_local.jpg')
# thresh = local_thresholding(r'D:\SBME\SBE_3.2\CV\Assignment1\Filtering_and_Edge_Detection_Studio\test-images\good_global_better_local.jpg')

# Display the thresholded image
cv2.imshow('Thresholded Image', binary_image)
cv2.imshow('Built-in Image', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
