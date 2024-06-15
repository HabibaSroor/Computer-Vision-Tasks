import sys
import numpy as np
import pyqtgraph as pg
from pyqtgraph import ImageView
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QHBoxLayout, QPushButton
import cv2

seed_points = []

class CustomImageView(ImageView):
    mousePressed = pyqtSignal()
    def __init__(self, *args, **kwargs):
        super(CustomImageView, self).__init__(*args, **kwargs)

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            pos = self.view.mapSceneToView(ev.pos())
            x, y = pos.x(), pos.y()
            # print(f'x={x}, y={y}')
            seed_points.append((int(y), int(x)))
            self.mousePressed.emit()
        super().mousePressEvent(ev)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Region growing segmentation")
        self.setGeometry(100, 100, 800, 600)

        # Create main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.vlayout = QVBoxLayout(self.central_widget)
        self.hlayout = QHBoxLayout(self.central_widget)
        self.hlayout2 = QHBoxLayout(self.central_widget)
        self.apply = QPushButton("Apply")

        # Create an ImageView widget
        self.imgv1 = CustomImageView()
        self.imgv2 = pg.ImageView()
        self.hlayout.addWidget(self.imgv1)
        self.hlayout.addWidget(self.imgv2)

        self.vlayout.addLayout(self.hlayout)
        self.vlayout.addWidget(self.apply)

        self.imgv1.ui.histogram.setFixedWidth(0)
        self.imgv1.ui.histogram.region.hide()
        self.imgv1.ui.histogram.vb.hide()
        self.imgv1.ui.histogram.axis.hide()
        self.imgv1.ui.histogram.gradient.hide()
        self.imgv1.ui.roiBtn.hide()
        self.imgv1.ui.menuBtn.hide()
        self.imgv1.ui.roiPlot.hide()
        self.imgv2.ui.histogram.setFixedWidth(0)
        self.imgv2.ui.histogram.region.hide()
        self.imgv2.ui.histogram.vb.hide()
        self.imgv2.ui.histogram.axis.hide()
        self.imgv2.ui.histogram.gradient.hide()
        self.imgv2.ui.roiBtn.hide()
        self.imgv2.ui.menuBtn.hide()
        self.imgv2.ui.roiPlot.hide()

        self.image1 = cv2.imread("images/nature.jpeg") #threshold=5
        # self.image1 = cv2.imread("images/CT-scan-image-of-brain-tumor-copy.png") #threshold=4
        
        self.image1 = cv2.cvtColor(self.image1, cv2.COLOR_BGR2RGB)
        self.imgv1.setImage(np.transpose(self.image1, (1, 0, 2)))
        self.apply.clicked.connect(lambda: self.grow_region(self.image1, threshold=5))
        self.imgv1.mousePressed.connect(self.handle_mouse_press)


    def handle_mouse_press(self):
        """
        Handle mouse press events on the first ImageView.

        Draws circles on the image at the locations where the mouse was clicked.

        Returns:
            None
        """
        # Create a copy of the original image
        image = np.copy(self.image1)
        # Draw circles at the locations of the seed points
        for point in seed_points:
            point = (point[1], point[0])
            cv2.circle(image, point, 3, (255, 255, 255), -1)
        # Display the modified image on the ImageView
        self.imgv1.setImage(np.transpose(image, (1, 0, 2)))

    def get_8neighbours(self, x, y, max_x, max_y):
        """
        Get 8-neighbour coordinates of a given point (x, y) within the image of size (N=max_x, M=max_y).

        Parameters:
            x: x-coordinate of the point
            y: y-coordinate of the point
            max_x: Maximum value of x-coordinate
            max_y: Maximum value of y-coordinate

        Returns:
            List of 8-neighbour coordinates
        """
        neighbours = []
        # Check if the neighbour is within the image boundaries
        if (x + 1) < max_x:
            neighbours.append((x + 1, y))
        if (x - 1) >= 0:
            neighbours.append((x - 1, y))
        if (y + 1) < max_y:
            neighbours.append((x, y + 1))
        if (y - 1) >= 0:
            neighbours.append((x, y - 1))
        if (x + 1) < max_x and (y + 1) < max_y:
            neighbours.append((x + 1, y + 1))
        if (x - 1) >= 0 and (y - 1) >= 0:
            neighbours.append((x - 1, y - 1))
        if (x - 1) >= 0 and (y + 1) < max_y:
            neighbours.append((x - 1, y + 1))
        if (x + 1) < max_x and (y - 1) >= 0:
            neighbours.append((x + 1, y - 1))
        return neighbours
    
    def get_similarity(self, image, x_seed, y_seed, x_neighbour, y_neighbour):
        """
        Calculate similarity between a seed point and its neighbour using intensity difference.

        Args:
            image(numpy.ndarray): Input image
            x_seed: x-coordinate of the seed point
            y_seed: y-coordinate of the seed point
            x_neighbour: x-coordinate of the neighbour
            y_neighbour: y-coordinate of the neighbour

        Returns:
            Absolute difference between intensities of the seed and neighbour pixels
        """
        return abs(int(image[x_seed, y_seed]) - int(image[x_neighbour, y_neighbour]))
    
    def grow_region(self, image, threshold):
        """
        Perform region growing on the input image.

        Parameters:
            image (numpy.ndarray): Input RGB image
            threshold: Similarity threshold for region growing

        Returns: None
        """
        # gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # luv_image = self.rgb_to_luv(image)
        # Convert RGB image to Luv color space
        luv_image = cv2.cvtColor(image, cv2.COLOR_RGB2Luv)
        print(luv_image.shape)

        height, width,_ = luv_image.shape
        # Extract L channel from Luv image
        L_channel = luv_image[:, :, 0]
        # print(f"max l value is {np.max(L_channel)}")
        # Create a mask to mark seed points
        seed_marked = np.array(L_channel)
        # Create an array to keep track of visited points
        visited = np.zeros(L_channel.shape)
        it = 0
        # Loop until there are no seed points (stack is empty)
        while(len(seed_points) > 0 ):
            x_seed, y_seed = seed_points.pop()
            # Mark the seed point
            seed_marked[x_seed, y_seed] = 255
            print(f"seed------------x {x_seed}-----y {y_seed}------value{seed_marked[x_seed, y_seed]}")
            visited[x_seed, y_seed] = 1
            # Iterate through 8-neighbours of the seed point
            for x_neighbour, y_neighbour in self.get_8neighbours(x=x_seed, y=y_seed, max_x=height, max_y=width):
                # print(f"x_neighbour= {x_neighbour}, y_neighbour= {y_neighbour}")
                # Check if neighbour is already visited
                if visited[x_neighbour, y_neighbour] == 1:
                    continue
                # print(f"difference {self.get_similarity(L_channel, x_seed, y_seed, x_neighbour, y_neighbour)}")
                # Check if similarity is within threshold and neighbour is not marked as seed point
                if self.get_similarity(L_channel, x_seed, y_seed, x_neighbour, y_neighbour) <= threshold and seed_marked[x_neighbour, y_neighbour] != 255:
                    seed_marked[x_neighbour, y_neighbour] = 255
                    it +=1
                    seed_points.append((x_neighbour, y_neighbour))

                visited[x_neighbour, y_neighbour] == 1
            # it += 1
            # if it > no_iterations:
            #     break

        
        print("DONE!")
        print(it)
        # Create a new array with three channels
        output_image = np.zeros_like(luv_image)

        # Copy L channel values
        output_image[:, :, 0] = seed_marked

        # Assign U and V channel values from luv_image
        output_image[:, :, 1:] = luv_image[:, :, 1:]
        print(f"output shape {output_image.shape}")

        # Transpose the array
        output_image = np.transpose(cv2.cvtColor(output_image, cv2.COLOR_Luv2RGB), (1, 0, 2))

        self.imgv2.setImage(output_image)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec())
