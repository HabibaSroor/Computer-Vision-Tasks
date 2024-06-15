
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import cv2

class KMeans():
	"""
	k-Means Clustering technique
	"""
	def __init__(self, n_clus,img_data):
		self.n_clus = n_clus	#Number of clusters
		# Reshape the image to a 2D array of pixels
		img = img_data.reshape((-1, 3))
		# Convert the data type to float32
		self.data = np.float32(img)
		self.centroids = None
		self.clusters = None

	def getCentroids(self):
		""" Returns position of the centroids
		:return self.centroids: Centroids
		"""
		return self.centroids

	def getClusters(self):
		""" Returns clusters with shape (1, Npts).
			E.g. a data point X[i] belongs to
			cluster self.clusters[i].
		:return self.clusters: clusters
		"""
		return self.clusters

	def fit(self, max_iter ,init_state = None):
		""" Train the K-Mean model, finds ultimate positions
		for centroids and create clusters.
		:param data: Data matrix with shape (Npts, Ndim)
		:param init_state: Initial state for centroids. If nothing is given,
		the model will generate random positions.
		:param max_iter: Maximum number of iterations
		"""
		data = self.data
		Npts, Ndim = self.data.shape
		if init_state is None:
			data_max, data_min = np.max(data), np.min(data)
			self.centroids = np.random.uniform(low = data_min, high=data_max, size = (self.n_clus,Ndim))
		else:
			self.centroids = init_state
		for _ in range(max_iter):
			self.clusters = self.predict(data)
			for i in range(self.n_clus):
				# Check if cluster i is empty
				if np.sum(self.clusters == i) == 0:
					continue  # Skip updating centroid for empty cluster
				self.centroids[i] = np.mean(data[np.where(self.clusters == i)] , axis=0)

	def predict(self, data):
		""" Predicts clusters for data points.
		:param data: Data point-s with shape (Npts, Ndim)
		:return: predicted clusters
		"""
		diff = cdist(data, self.centroids, metric="euclidean")
		return np.argmin(diff, axis=1)
	
	def vizualize(self):
		""" Plot data points and final position of centroids.
			This function need to be called after training.
		"""
		plt.style.use("ggplot")
		plt.scatter(self.data[:,0], self.data[:,1], color = '#333333')
		Nclus, _ = self.centroids.shape
		for k in range(Nclus):
			plt.scatter(self.centroids[k][0], self.centroids[k][1], marker = "*", color="#4169e1", s=130)
		plt.xlabel('x1')
		plt.ylabel('x2')
		plt.show()
		
def apply_kmean_segmentation(img, max_iter, cluster_no):
    """ Apply KMeans clustering and return segmented image.
    :param img_data: Input image data
    :param max_iter: Maximum number of iterations for KMeans
    :param cluster_no: Number of clusters
    :return: Segmented image
    """
    km = KMeans(n_clus=cluster_no, img_data=img)
    km.fit(max_iter=max_iter)
    centers = km.getCentroids()
    clusters = km.getClusters()
    segmented_image = centers[clusters]
    segmented_image = segmented_image.reshape((img.shape))
    return segmented_image
    

image_path = "C:/Study/CV/CV_Tasks/Filtering_and_Edge_Detection_Studio/face.jpg"  
# image_path = "C:/Study/CV/CV_Tasks/Filtering_and_Edge_Detection_Studio/seg-image.png" 
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
k = 6 
segmented_image = apply_kmean_segmentation(image,150,k)
plt.imshow(segmented_image.astype(np.uint8))
plt.axis("off")
plt.show() 