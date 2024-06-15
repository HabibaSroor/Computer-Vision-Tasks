import sys
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QHBoxLayout
import cv2
import pandas as pd
import os

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set window title and size
        self.setWindowTitle("Histogram and Distribution Curve")
        self.setGeometry(100, 100, 800, 600)

        # Create central widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.hlayout = QHBoxLayout(self.central_widget)
        self.hlayout2 = QHBoxLayout(self.central_widget)

        # Create ImageView widget
        self.image_view = pg.ImageView()
        
        self.hlayout2.addWidget(self.image_view)

        self.cumulative_plot_widget = pg.PlotWidget()
        self.hlayout2.addWidget(self.cumulative_plot_widget)

        self.layout.addLayout(self.hlayout2)
        
        # Create PlotWidget
        self.plot_widget = pg.PlotWidget()
        # self.distrib_plot_widget = pg.PlotWidget()
        self.hlayout.addWidget(self.plot_widget)
        # self.hlayout.addWidget(self.distrib_plot_widget)
        self.layout.addLayout(self.hlayout)

        # ------------------------------------------------------------------------------------------------------------------------#
        #---------------LOGIC STARTS HERE-----------------------------------------------------------------------------------------#

        img = cv2.imread("Aya/sample.png") # BGR is the default in opencv
        
        img = cv2.resize(img, (224, 224)) # (to be removed) the original size 2000x2000 was too much for the computation of the kde curve 
        # blue_ch, green_ch, red_ch = self.get_channels(img)
        # img = self.to_greyscale(red_ch=red_ch, green_ch=green_ch, blue_ch=blue_ch)
        # cv2.imwrite("greyscale.png", img)
        # res = self.local_thresholding(img, region_size=50)
        # cv2.imwrite("local_th.png", res)
        blue_ch, green_ch, red_ch = self.get_channels(img) # separate channels

        greyscale_img = self.to_greyscale(red_ch=red_ch, blue_ch=blue_ch, green_ch=green_ch)

        # get the frequencies for each pixel value for the histogram
        blue_hist_values, blue_bin_edges = self.get_frequencies(blue_ch)
        green_hist_values, green_bin_edges = self.get_frequencies(green_ch)
        red_hist_values, red_bin_edges = self.get_frequencies(red_ch)

        gs_hist_values, gs_bin_edges = self.get_frequencies(greyscale_img)

        # get cumulative frequencies
        blue_hist_values_cumulative = self.get_cumulative_frequencies(blue_hist_values)
        red_hist_values_cumulative = self.get_cumulative_frequencies(red_hist_values)
        green_hist_values_cumulative = self.get_cumulative_frequencies(green_hist_values)
        gs_hist_values_cumulative = self.get_cumulative_frequencies(gs_hist_values)

        # plot histograms
        blue_bargraph = pg.BarGraphItem(x = blue_bin_edges[:-1], height = blue_hist_values, width = 1, brush ='b') 
        green_bargraph = pg.BarGraphItem(x = green_bin_edges[:-1], height = green_hist_values, width = 1, brush ='g') 
        red_bargraph = pg.BarGraphItem(x = red_bin_edges[:-1], height = red_hist_values, width = 1, brush ='r') 

        gs_bargraph = pg.BarGraphItem(x = gs_bin_edges[:-1], height = gs_hist_values, width = 1, brush ='w') 

        # add items to plot widget
        self.plot_widget.addItem(blue_bargraph)
        self.plot_widget.addItem(green_bargraph)
        self.plot_widget.addItem(red_bargraph)
        self.plot_widget.addItem(gs_bargraph)

        # cumulative distributions
        blue_bargraph_cum = pg.BarGraphItem(x = blue_bin_edges[:-1], height = blue_hist_values_cumulative, width = 1, brush ='b') 
        green_bargraph_cum = pg.BarGraphItem(x = green_bin_edges[:-1], height = green_hist_values_cumulative, width = 1, brush ='g') 
        red_bargraph_cum = pg.BarGraphItem(x = red_bin_edges[:-1], height = red_hist_values_cumulative, width = 1, brush ='r') 

        gs_bargraph_cum = pg.BarGraphItem(x = gs_bin_edges[:-1], height = gs_hist_values_cumulative, width = 1, brush ='w') 

        self.cumulative_plot_widget.addItem(blue_bargraph_cum)
        self.cumulative_plot_widget.addItem(green_bargraph_cum)
        self.cumulative_plot_widget.addItem(red_bargraph_cum)
        # self.cumulative_plot_widget.addItem(gs_bargraph_cum)

        self.show()

    # BGR
    def get_channels(self, img):
        """"
        Gets the image as an ndarray in the format BGR and separates its channels.

        Args:
            img (ndarray): image in the format BGR of dimension (img_height, img_width, 3)

        Returns:
            A tuple of the three channels separated
        """
        return(img[:, :, 0], img[:, :, 1], img[:, :, 2])
    
    def get_frequencies(self, channel_array):
        """"
        Gets a channel of the image and returns the frequency 
        of each pixel values and the bin labels to draw its histogram.

        Args:
            channel_array (ndarray): a 2d numpy array containing pixel values

        Returns:
            A tuple of the three channels separated
        """
        # flatten 2d array to a 1d array of values
        channel_array = channel_array.flatten()
        return np.histogram(channel_array, bins=256)
    
    def get_cumulative_frequencies(self, frequencies):
        """"
        Gets a channel of the image and returns the frequency 
        of each pixel values and the bin labels to draw its histogram.

        Args:
            frequencies (ndarray): a 1d numpy array containing the frequency of each pixel value

        Returns:
            A numpy array of the cumulative frequencies for every pixel frequency and the previous pixel frequencies
        """
        cum_frequencies = [frequencies[0]]
        for index in range(1, len(frequencies)):
            cum_frequencies.append(frequencies[index] + cum_frequencies[index - 1])
        return np.array(cum_frequencies)
    
    def to_greyscale(self, red_ch, green_ch, blue_ch):
        """"
        Gets the 3 channels of a colored image and converts it to grey scale

        Args:
            red_ch (ndarray): a 2d numpy array containing pixel values of the red channel (img_width, img_height)
            green_ch (ndarray): a 2d numpy array containing pixel values of the green channel (img_width, img_height)
            blue_ch (ndarray): a 2d numpy array containing pixel values of the blue channel (img_width, img_height)

        Returns:
            grey_ch (ndarray): A 2d array of one channel of pixel values 
        """
        grey_ch = 0.2989 * red_ch + 0.5870 * green_ch + 0.1140 * blue_ch
        return grey_ch
    
    def local_thresholding(self, img, region_size=3):
        """
        Apply local_thresholding to the image, takes a greyscale image, the threshold is the mean value.
        Parameters:
            img (np.ndarray): The input image.
            region_size (int): The size of the square local region
        Returns:
            thresholded_image (np.ndarray): The result image after thresholding.
        """
        result = np.zeros_like(img)

        row_regions_count = int(img.shape[0] / region_size)
        col_regions_count = int(img.shape[1] / region_size)

        for row_index in range(row_regions_count):
            for col_index in range(col_regions_count):

                # check that the index has not gotten out of bound
                row_region_end_index = (row_index + 1) * region_size
                col_region_end_index = (col_index + 1) * region_size
                if row_region_end_index > img.shape[0] - 1:
                    row_region_end_index = img.shape[0] - 1
                if col_region_end_index > img.shape[1] - 1:
                    col_region_end_index = img.shape[1] - 1

                local_region = img[row_index * region_size : row_region_end_index][col_index * region_size : col_region_end_index].copy()
                local_threshold = local_region.flatten().mean()

                result[row_index * region_size : row_region_end_index][col_index * region_size : col_region_end_index] = self.apply_threshold(local_region, local_threshold)
        
        return result
        
    def apply_threshold(self, img, threshold, low_value = 0, high_value = 255):
        """
        Apply thresholding (binary) to the image, takes a greyscale image.
        Parameters:
            img (np.ndarray): The input image.
            threshold (int): The required threshold value.
        Returns:
            thresholded_image (np.ndarray): The result image after thresholding.
        """
        thresholded_image = np.ones(img.shape) * low_value
        higherRow, higherCol = np.where(img >= threshold)
        thresholded_image[higherRow, higherCol] = high_value
        return thresholded_image


if __name__ == '__main__':
    app = QApplication(sys.argv)  
    mainWindow = MainWindow()  
    sys.exit(app.exec()) 
