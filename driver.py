from PIL import Image, ImageOps
from numpy import argmax, nonzero
import tensorflow as tf
from canvas import Canvas

def preprocess_image(image: Image):
	return ImageOps.grayscale(ImageOps.invert(image)).resize((70, 70))

def img2tensor(image: Image):
	image = tf.keras.preprocessing.image.img_to_array(image) #/ 255.0
	image[nonzero(image)] = 1.0
	return tf.constant((image,))

class Classifier:
	def __init__(self, path):
		self.model = tf.keras.models.load_model(path, compile=False)
		self.model.compile()
		self.shapes = ['other', 'ellipse', 'rectangle', 'triangle']

	def classify(self, image_tensor):
		dist = self.model(image_tensor)
		return self.shapes[argmax(dist)]

class DrawingCanvas(Canvas):
	def __init__(self, width=700, height=700):
		super().__init__(width, height)
		self.points = []
		self.register_mouse_press(self.on_click)
		self.register_mouse_move(self.on_move)
		self.register_mouse_release(self.on_release)
		self.classifer = Classifier("classifier")

	def on_click(self, _):
		self.points.clear()

	def on_move(self, e):
		self.reset()
		self.points.append((e.x, e.y))
		self.stroke_color = "black"
		self.curve(self.points)

	def on_release(self, _):
		img, _ = self.capture()
		img.save("capture.png")
		pimg = preprocess_image(img)
		pimg.save("preprocessed_capture.png")
		tensor = img2tensor(pimg)
		print(self.classifer.classify(tensor))



canvas = DrawingCanvas()
