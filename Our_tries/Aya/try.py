import sys
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QHBoxLayout, QPushButton
import cv2
from filters_edges import gaussian_filter, sobel_edge

global blurred_image
global edges

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Active contour")
        self.setGeometry(100, 100, 800, 600)

        # Create main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.vlayout = QVBoxLayout(self.central_widget)
        self.hlayout = QHBoxLayout(self.central_widget)
        self.hlayout2 = QHBoxLayout(self.central_widget)
        self.hlayout3 = QHBoxLayout(self.central_widget)
        # Create buttons
        # self.apply_button = QPushButton("Apply")
        self.show_button = QPushButton("Start")

        # Add buttons to layout
        # self.hlayout3.addWidget(self.apply_button)
        self.hlayout3.addWidget(self.show_button)

        # Add the layout containing buttons to the main layout
        self.vlayout.addLayout(self.hlayout3)

        # Create an ImageView widget
        self.imgv1 = pg.ImageView()
        self.hlayout.addWidget(self.imgv1)

        # self.imgv3 = pg.ImageView()
        self.hlayout2.addWidget(self.imgv3)

        self.vlayout.addLayout(self.hlayout)
        self.vlayout.addLayout(self.hlayout2)
        # define 3*3 window
        self.window = np.array([[0, 0], [0, 1], [0, -1], [1, 0],[-1, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]])
        # image = cv2.imread("images/New folder/apples-for-high-blood-pressure.jpeg")
        image = cv2.imread("Salma/images.jpeg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.imgv1.setImage(np.transpose(image, (1, 0)))
        
        self.show_button.clicked.connect(lambda: self.perform_active_contour(image, 0.001, 0.1, 1, 30, 40, 30, 100))
    
    # from https://github.com/r-CV-Tasks/Computer-Vision-Tasks/blob/main/libs/Contour.py
    def create_initial_contour(self, source):
        num_points = 80
        # Create x and y lists coordinates to initialize the contour
        t = np.arange(0, num_points / 10, 0.1)

        # Coordinates for Circles_v2.png image
        contour_x = (source.shape[1] // 2) + 100 * np.cos(t)
        contour_y = (source.shape[0] // 2) + 100 * np.sin(t)

        # # Coordinates for fish.png image
        # contour_x = (source.shape[1] // 2) + 215 * np.cos(t)
        # contour_y = (source.shape[0] // 2) + 115 * np.sin(t) - 10

        contour_x = contour_x.astype(int)
        contour_y = contour_y.astype(int)

        contour_points = np.array((contour_x, contour_y))
        contour_points = contour_points.T

        return contour_points
    # from https://github.com/r-CV-Tasks/Computer-Vision-Tasks/blob/main/libs/Contour.py
    def create_square_contour(self, source, num_xpoints=180, num_ypoints=180):
        step = 2

        # Create x points lists
        t1_x = np.arange(0, num_xpoints, step)
        t2_x = np.repeat((num_xpoints) - step, num_xpoints // step)
        t3_x = np.flip(t1_x)
        t4_x = np.repeat(0, num_xpoints // step)

        # Create y points list
        t1_y = np.repeat(0, num_ypoints // step)
        t2_y = np.arange(0, num_ypoints, step)
        t3_y = np.repeat(num_ypoints - step, num_ypoints // step)
        t4_y = np.flip(t2_y)

        # Concatenate all the lists in one array
        contour_x = np.array([t1_x, t2_x, t3_x, t4_x]).ravel()
        contour_y = np.array([t1_y, t2_y, t3_y, t4_y]).ravel()

        # Shift the shape to a specific location in the image
        # contour_x = contour_x + (source.shape[1] // 2) - 85
        contour_x = contour_x + (source.shape[1] // 2) - 80
        contour_y = contour_y + (source.shape[0] // 2) - 80

        contour_points = np.array((contour_x, contour_y))
        contour_points = contour_points.T

        return contour_points

    def internal_energy(self, contour_points, alpha, beta):
        """
        Calculates the internal energy of a contour defined by a set of points.
        Internal Energy = alpha * sum((sqrt((x_i+1 - x_i)^2 + (y_i+1 - y_i)^2)) - mean(sqrt( (x_i+1 - x_i)^2 + (y_i+1 - y_i)^2)) ))^2)
                        + beta * (sum((x_i+1 - 2*x_i + x_i-1)^2 + (y_i+1 - 2*y_i + y_i-1)^2))

        Args:
            contour_points (ndarray): Array of shape (N, 2) containing the coordinates of N points defining the contour.
            alpha (float): Weight parameter controlling the contribution of elasticity to the internal energy.
            beta (float): Weight parameter controlling the contribution of stiffness (curvature) to the internal energy.

        Returns:
            internal_energy (float): The total internal energy of the contour.
        """
        # Shift the contour points to get previous and next points
        previous_points = np.roll(contour_points, 1)
        next_points = np.roll(contour_points, -1)

        # Calculate displacement between consecutive points
        displacement = next_points - contour_points
        # Calculate Euclidean distance between consecutive points------------------------dim check
        euclidean_distance = np.sqrt(displacement[:, 0]**2 + displacement[:, 1]**2)
        # Calculate elasticity energy
        elasticity = alpha * np.sum((euclidean_distance - np.mean(euclidean_distance))**2)
        
        # Calculate curvature energy
        curvature = next_points - 2 * contour_points + previous_points
        final_curvature = np.sum(curvature[:, 0]**2 + curvature[:, 1]**2)
        stiffness = beta * final_curvature
        # Compute the total internal energy
        internal_energy = elasticity + stiffness

        return internal_energy
    
    def prepare_external_energy(self, image):
        blurred_image = gaussian_filter(image, kernel_size=5, sigma=3, is_gray=True)
        edges = sobel_edge(blurred_image, True, True, False)
        # self.imgv3.setImage(np.transpose(edges, (1, 0)))
        return blurred_image, edges

    def external_energy(self, point, intensity, gradient, w_line, w_edge, gamma):
        e_line = 0
        e_edge = 0

        e_line = intensity[point[1], point[0]]
        e_edge = gradient[point[1], point[0]]

        # print(f"line intensity is {e_line} edge is {e_edge}")

        external_energy = gamma * (w_line * e_line + w_edge * e_edge)

        return external_energy

    def perform_active_contour(self, image, alpha, beta, gamma, w_line, w_edge, threshold, iterations):
        contour_points = self.create_initial_contour(image)
        # contour_points = self.create_square_contour(image, 160, 160)

        blurred_image, gradient = self.prepare_external_energy(image)
        num_changed_points = 0
        num_iterations = 0
        for it in range(iterations):
            print(it)
            for i in range(len(contour_points)):
                min_energy = np.inf
                new_contour = np.copy(contour_points)
                # initialize point location
                new_point_location = None
                for step in self.window:
                    # move the point to a new location
                    new_contour[i] =[contour_points[i][0] + step[0], contour_points[i][1] + step[1]]

                    # print(f" current point is ----{new_contour[i]}") 

                    # calculate total energy
                    total_energy = self.internal_energy(new_contour, alpha, beta) - self.external_energy(new_contour[i], blurred_image, gradient, w_line, w_edge, gamma)
                    
                    # print(f"total energy in {i} is {total_energy}")
                    
                    if total_energy < min_energy:
                        min_energy = total_energy
                        # store the point at which min energy occurs
                        new_point_location = np.copy(new_contour[i])
                        # print(f"new point location in {i} is **** {new_point_location}")
                    
                    # print(f"min energy in {i} is {min_energy}")
                
                if not np.array_equal(new_point_location, contour_points[i]):
                    # print(f"in {num_iterations} old is {contour_points[i]} new is {new_point_location}---------------")
                    contour_points[i] = np.copy(new_point_location)

                    # increment changed points counter----------to be used as a second condition----
                    num_changed_points += 1

            # num_iterations += 1

            # Create a copy of the original image
            result_image = image.copy()

            cv2.polylines(result_image, [contour_points], isClosed=False, color=(0, 0, 255), thickness=2)
            
            # Transpose the result image to match the format expected by pyqtgraph
            result_image = np.transpose(result_image, (1, 0))

            # Update the ImageView widget with the result image
            self.imgv1.setImage(result_image)

            QApplication.processEvents()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    my_app = MainWindow()
    my_app.show()
    sys.exit(app.exec_())
