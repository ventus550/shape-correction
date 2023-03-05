from models import Classifier, Regressor, transform
from numpy import array, argmin, linalg, arccos, pi
from canvas import Canvas

import numpy as np
def angle(a,b):
    """ return rotation angle from vector a to vector b, in degrees.
    Args:
        a : np.array vector. format (x,y)
        b : np.array vector. format (x,y)
    Returns:
        angle [float]: degrees. 0~360
    """
    unit_vector_1 = a / np.linalg.norm(a)
    unit_vector_2 = b / np.linalg.norm(b)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    angle = angle/ np.pi * 180
    c = np.cross(b,a)
    if c>0:
        angle +=180
    
    return angle

class DrawingCanvas(Canvas):
	def __init__(self, width=700, height=700):
		super().__init__(width, height)
		self.points = []
		self.register_mouse_press(self.on_click)
		self.register_mouse_move(self.on_move)
		self.register_mouse_release(self.on_release)
		self.classifer = Classifier("models/classifier")
		self.regressor = Regressor("models/rectangle_regressor")

	def draw_vertices(self, vertices: list[float]):
		self.stroke_color = 'red'
		for v in vertices:
			self.point(*v)

	def connect(self, vertices):
		centroid = sum(vertices) / len(vertices)
		angle = lambda v: np.arctan2(*(v - centroid))
		vertices = sorted(vertices, key=angle)
		
		for v, u in zip(vertices, vertices[1:]):
			self.line(*v, *u)
		self.line(*vertices[0], *u)

	def ellipse(self, vertices):
		centroid = sum(vertices) / len(vertices)
		angle = lambda v: np.arctan2(*(v - centroid))
		vertices = sorted(vertices, key=angle)
		
		for v, u in zip(vertices, vertices[1:]):
			x = v + u - centroid
			self.curve([v, x, u])
		self.curve([vertices[0], vertices[0] + u - centroid, u])

	def reconstruct(self, vertices):
		self.reset()
		self.connect(vertices)

	def on_click(self, _):
		self.reset()
		self.points.clear()

	def on_move(self, e):
		self.reset()
		self.points.append((e.x, e.y))
		self.stroke_color = "black"
		self.curve(self.points)

	def on_release(self, _):
		img, xy = self.capture()
		img.save("capture.png")
		pimg = transform(img)
		pimg.save("transform.png")
		
		vertices = self.regressor.vertices(img)
		vertices = array(vertices).reshape((len(vertices)//2, 2))
		vertices *= img.size
		vertices += xy

		self.draw_vertices(vertices)
		self.root.after(500, lambda: self.reconstruct(vertices))
		print(self.classifer.classify(img))



canvas = DrawingCanvas()
