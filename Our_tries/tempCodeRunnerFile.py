import sys
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QHBoxLayout, QGridLayout
import cv2
import matplotlib.pyplot as plt
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from skimage.transform import resize
from math import sqrt
from filters_edges import * # just for the try file, will meed to be modified
# from functions.filters_edges import *
import time

k = sqrt(2)
sigma = 1.6
num_scales = 5 
num_octaves = 4
# [sigma, sqrt(2)*sigma, 2*sigma, 2sqrt(2)*sigma, 4*sigma]
sigma_values = [(k**i)*sigma for i in range(num_scales)]
# kernels = [gaussian_kernel(kernel_size=3, sigma=sigma) for sigma in sigma_values ]
octaves = []
diff_of_gaussian = []

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SIFT")
        self.setGeometry(100, 100, 800, 600)

        # Create main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.vlayout = QVBoxLayout(self.central_widget)
        self.hlayout = QHBoxLayout(self.central_widget)
        self.imgv1 = pg.ImageView()
        self.imgv1.ui.histogram.hide()
        self.imgv1.ui.roiBtn.hide()
        # Disable the default context menu
        self.imgv1.ui.menuBtn.hide()
        self.hlayout.addWidget(self.imgv1)
        self.vlayout.addLayout(self.hlayout)
        # Create a grid layout
        self.grid = QGridLayout(self.central_widget)

        # Create a 4x5 array of QLabel
        self.img_views = [[pg.ImageView() for _ in range(5)] for _ in range(4)]

        # Add the QLabel to the grid layout
        for i in range(4):
            for j in range(5):
                self.img_views[i][j].ui.histogram.hide()
                self.img_views[i][j].ui.roiBtn.hide()
                # Disable the default context menu
                self.img_views[i][j].ui.menuBtn.hide()
                self.grid.addWidget(self.img_views[i][j], i, j)

        self.vlayout.addLayout(self.grid)

        # image1 = cv2.imread("images/New folder/apples-for-high-blood-pressure.jpeg")
        # image1 = cv2.imread("images/marilyn.bmp")
        image1 = cv2.imread("test-images/SIFT/img_sift.jpg")

        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        self.imgv1.setImage(np.transpose(image1, (1, 0)))
        self.construct_scale_space(image=image1)

    def construct_scale_space(self, image):
        start_time = time.time()

        # image upsampled by a factor of 2
        base_image = np.copy(image)
        base_image = base_image.astype('float32')
        # base_image = resize( image=image, output_shape=(image.shape[0]*2, image.shape[1]*2))
        # base_image = image
        # base_image = np.clip(base_image, 0, 255).astype(np.uint8)
        for i in range(0, num_octaves):
            octaves.append([ gaussian_filter(base_image, kernel_size=7, sigma=sigma, is_gray=True) 
                    for sigma in sigma_values ])
            diff_of_gaussian.append([ scale2 - scale1 
                        for (scale1, scale2) in zip( octaves[i][:-1], octaves[i][1:])])
            base_image = resize(image=octaves[i][2], output_shape=(octaves[i][2].shape[0] // 2, octaves[i][2].shape[1]//2), anti_aliasing=True)
        # ---------------- show octaves ----------------------------
        # for octave_idx in range(num_octaves):
        #     print(octave_idx)
        #     img_octave = octaves[octave_idx]
        #     for scale_idx in range(num_scales):
        #         print(scale_idx)
        #         self.img_views[octave_idx][scale_idx].setImage(np.transpose(img_octave[scale_idx], (1, 0)))
        #         cv2.imwrite(f"oct{octave_idx}_sca{scale_idx}.png", np.transpose(img_octave[scale_idx], (1, 0)))
        # ---------------- show DoG ----------------------------
        for octave_idx in range(num_octaves):
            print(octave_idx)
            dog_octave = diff_of_gaussian[octave_idx]
            for i in range(4):
                print(i)
                self.img_views[octave_idx][i].setImage(np.transpose(dog_octave[i], (1, 0)))
                cv2.imwrite(f"oct{octave_idx}_sca{i}.jpg", np.transpose(dog_octave[i], (1, 0)))
        
        end_time = time.time()

        runtime = end_time - start_time

        print(f"The runtime of construct_scale_space is {runtime} seconds.")
        return diff_of_gaussian , octaves

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec())
