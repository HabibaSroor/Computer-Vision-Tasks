
import numpy as np
import cv2 
import matplotlib.pyplot as plt
def linearize(x):
        mask = x <= 0.04045
        x[mask] /= 12.92
        x[~mask] = ((x[~mask] + 0.055) / 1.055) ** 2.4
        return x
   
def rgb_to_luv(image):
    """
    Convert an RGB image to the Luv color space.
    Parameters:
        image (numpy.ndarray): Input RGB image.
    Returns:
        luv_image (numpy.ndarray): Converted Luv image.
    """
    # Normalize the image and Extract R, G, B channels    
    image = image.astype(np.float32) / 255.0
    R, G, B = image[..., 0], image[..., 1], image[..., 2]
    R, G, B = map(linearize, [R, G, B])
    # Convert RGB to XYZ space 
    X = (0.412453 * R) + (0.35758 * G) + (0.180423 * B)
    Y = (0.212671 * R) + (0.715160 * G) + (0.072169 * B)
    Z = (0.019334 * R) + (0.119193 * G) + (0.950227 * B)
    # Calculate lightness component L
    L = np.zeros_like(Y)
    L[Y > 0.008856] = 116 * ((Y[Y > 0.008856])**(1 / 3)) - 16
    L[Y <= 0.008856] = (903.3) * Y[Y <= 0.008856]
    # Constants for u and v
    u_n = 0.2009
    v_n = 0.4610
    # Calculate u and v components
    denom = X + 15 * Y + 3 * Z
    u_m = 4 * X / denom
    v_m = 9 * Y / denom
    u = 13 * L * (u_m - u_n)
    v = 13 * L * (v_m - v_n)
    luv_image = np.stack((L, u, v), axis=-1)
    return luv_image


image_path = "C:/Study/CV/CV_Tasks/Filtering_and_Edge_Detection_Studio/face.jpg"  
# image_path = "C:/Study/CV/CV_Tasks/Filtering_and_Edge_Detection_Studio/seg-image.png"  
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
luv_image = rgb_to_luv(image_rgb)
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(image_rgb)
axs[0].set_title('Original RGB Image')
axs[0].axis('off')
axs[1].imshow(luv_image)
axs[1].set_title('Converted Luv Image')
axs[1].axis('off')
plt.show()
