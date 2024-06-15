# Filtering and Edge Detection Studio

This project is a comprehensive implementation of various image processing techniques. The aim is to understand and implement fundamental concepts of computer vision from scratch. Below is the detailed description of functionalities and algorithms included in this project:

## Features Implemented

* Color to Grayscale Transformation: Convert color images to grayscale and analyze histograms for individual color channels (R, G, B).

* Histogram Analysis: Computing and plotting histograms and distribution curves of images (colored or gray) to analyze pixel intensity distribution.

* Additive Noise Generation: Generate various types of noise such as Uniform, Gaussian, and salt & pepper.

* Noise Filtering: Use different low-pass filters including average, Gaussian, and median filters to reduce noise from the images.

* Frequency Domain Filters: Apply high-pass and low-pass filters in the frequency domain to manipulate image features.

* Image Equalization: Enhance image contrast and visibility of details through histogram equalization.

* Image Normalization: Normalize images to a standard scale.

* Thresholding: Apply local and global thresholding techniques for image segmentation.

* Edge Detection: Using popular edge detection techniques such as Sobel, Roberts, Prewitt, and Canny edge detectors to identify edges in the image.

* Shape Detection: Utilizing Canny edge detector to detect edges, and subsequently identify lines, circles, and ellipses in the image. Detected shapes are superimposed on the original image.

* Active Contour Model (Snake): Initialize contours for given objects and evolve the active contour model using a greedy algorithm. Output is represented as chain code, and perimeter and area inside the contours are computed.

## Important Note
* Please note that for task 3 there is an extra file to be run separately (SIFT.py).

## Installation

1- Install the Requirments

```
pip install -r requirements.txt
```

2- Run the main.py file

```
python main.py
```


