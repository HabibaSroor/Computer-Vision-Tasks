#define in main file a constant for NO_CONTOUR_POINTS = 200 for example

import numpy as np
import pyqtgraph as pg
from PyQt5.QtGui import QPixmap, QImage
import cv2
from PyQt5.QtWidgets import QGraphicsPixmapItem

def get_contour_x_y(img:np.ndarray):
    angles = np.linspace(0, 2*np.pi, 200)

    radius = 0.25 * min(img.shape)
    # print(img.siz)
    center_x, center_y = img.shape[0]/2, img.shape[1]/2
    x = center_x + np.cos(angles)
    y = center_y + np.sin(angles)
    return (x, y)

def draw_circle_contour(contour_points:np.ndarray, plot_widget:pg.PlotWidget):
    return plot_widget.plot(x=contour_points[0], y=contour_points[1])
    

