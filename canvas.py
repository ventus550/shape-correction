import tkinter as tk
from tkinter import Canvas
from collections import deque
from PIL import Image, ImageDraw, ImageOps
from numpy import argmax, array
import tensorflow as tf

def add_margin(pil_img, margin, color = 3 * (255,)):
	shape = array(pil_img.size)
	shape += 2*margin
	result = Image.new(pil_img.mode, tuple(shape), color)
	result.paste(pil_img, (margin, margin))
	return result

class Classifier:
	def __init__(self, path):
		self.model = tf.keras.models.load_model(path)
		self.shapes = ['other', 'ellipse', 'rectangle', 'triangle']

	def preprocess_image(self, image):
		return ImageOps.grayscale(ImageOps.invert(image)).resize((70, 70))

	def get_distribution(self, image):
		image = tf.keras.preprocessing.image.img_to_array(image) / 255.0
		tensor = tf.constant((image, ))
		return self.model(tensor)

	def classify(self, image):
		dist = self.get_distribution(image)
		return self.shapes[argmax(dist)]

class DrawingCanvas(tk.Tk):
	def __init__(self, width = 1200, height = 1200):
		super().__init__()
		self.width = width
		self.height = height
		self.classifier = Classifier("model")

		self.canvas = Canvas(self, bg="white", width=width, height=height)
		self.canvas.pack(expand=1, fill=tk.BOTH)

		self.beziers = deque([], maxlen=4)
		self.reset()

		self.canvas.bind('<B1-Motion>', self.on_mouse_drag)
		self.canvas.bind('<ButtonRelease-1>', self.on_mouse_release)
		self.canvas.bind('<ButtonPress-1>', self.on_mouse_down)
	
	def reset(self):
		self.canvas.delete("all")
		self.image = Image.new("RGB", (self.width, self.height), (255, 255, 255))
		self.region = (self.width, self.height, 0, 0)
		self.beziers.clear()
	
	def line(self, lin, width = 10):
		self.canvas.create_line(lin, width=width, capstyle="round", joinstyle="round", smooth=True)
		ImageDraw.Draw(self.image).line(lin, (0,0,0), width=width*3)
	
	def draw_bezier_curve(self, n = 150):
		p = self.beziers
		start = p[0]

		for i in range(n):
			t = i / n
			x, y = p[0] * (1-t)**3 + p[1] * 3 * t * (1-t)**2 + p[2] * 3 * t**2 * (1-t) + p[3] * t**3
			self.line((x, y, start[0], start[1]))
			start = array((x,y))
		end = self.beziers[-1]
		self.beziers.clear()
		self.beziers.append(end)

	def get_point(self, e):
		self.beziers.append(array((e.x, e.y)))
		x, y, X, Y = self.region
		m = 10
		self.region = [
			min(x, max(e.x - m, 0)),
			min(y, max(e.y - m, 0)),
			max(X, min(e.x + m, self.width)),
			max(Y, min(e.y + m, self.height))
		]

	def get_image(self, margin = 50):
		return add_margin(self.image.crop(self.region), margin)

	def on_mouse_down(self, _):
		self.reset()

	def on_mouse_drag(self, e):
		self.get_point(e)
		if len(self.beziers) == 4:
			self.draw_bezier_curve()

	def draw_classified_shape(self, clss):
		points = self.region
		self.reset()
		if clss == "rectangle":
			self.canvas.create_rectangle(*points, width=14)
		elif clss == "ellipse":
			self.canvas.create_oval(*points, width=14)
		elif clss == "triangle":
			x, y, X, Y = points
			points = [(x + X) // 2, y, x, Y, X, Y]
			self.canvas.create_polygon(*points, fill="white", outline="black", width=14)

	def on_mouse_release(self, _):
		img = self.classifier.preprocess_image(self.get_image())
		img.save("image.png")
		clss = self.classifier.classify(img)
		print(self.region, clss)
		if clss != "other":
			self.draw_classified_shape(clss)


if __name__ == "__main__":
	DrawingCanvas().mainloop()