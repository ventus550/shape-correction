# silence tf warnings
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from numpy import argmax, ndarray


class ModelInterface:
	def __init__(self, path):
		self.model = tf.keras.models.load_model(path, compile=False)
		self.model.compile()

	@staticmethod
	def img2tensor(image: ndarray):
		return tf.constant((image, ))

class Classifier(ModelInterface):
	def classify(self, image: ndarray):
		shapes = ['other', 'ellipse', 'rectangle', 'triangle']
		image_tensor = self.img2tensor(image)
		dist = self.model(image_tensor)
		return shapes[argmax(dist)]

class Regressor(ModelInterface):
	def vertices(self, image: ndarray):
		image_tensor = self.img2tensor(image)
		return self.model(image_tensor)[0]