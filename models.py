from PIL import Image, ImageOps
import numpy as np
from numpy import argmax, nonzero
from skimage.morphology import skeletonize
import cv2

# silence tf warnings
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


def transform(image: Image):
	return ImageOps.grayscale(ImageOps.invert(image)).resize((70, 70))

def img2tensor(image: Image):
	image = tf.keras.preprocessing.image.img_to_array(image) #/ 255.0
	image[nonzero(image)] = 1.0
	image = skeletonize(image)
	image = cv2.dilate(
		image,
		cv2.getStructuringElement(cv2.MORPH_RECT, (2,2)),
		iterations=1
	)

	# kernel2 = np.ones((2, 2), np.float32)/4
	# image = cv2.filter2D(src=image, ddepth=-1, kernel=kernel2)
	
	image[nonzero(image)] = 1.0
	return tf.constant((image,))

class ModelInterface:
	def __init__(self, path):
		self.model = tf.keras.models.load_model(path, compile=False)
		self.model.compile()

	@staticmethod
	def preprocess_image(image: Image):
		return img2tensor(transform(image))

class Classifier(ModelInterface):
	def classify(self, image: Image):
		shapes = ['other', 'ellipse', 'rectangle', 'triangle']
		image_tensor = self.preprocess_image(image)
		dist = self.model(image_tensor)
		return shapes[argmax(dist)]

class Regressor(ModelInterface):
	def vertices(self, image: Image):
		image_tensor = self.preprocess_image(image)
		return self.model(image_tensor)[0]