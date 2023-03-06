import numpy as np
from TkCanvas.canvas import Canvas
from models import Classifier, Regressor, transform


class DrawingCanvas(Canvas):
	def __init__(self, width=1000, height=1000):
		super().__init__(width, height)
		self.points = []
		self.register_mouse_press(self.on_click)
		self.register_mouse_move(self.on_move)
		self.register_mouse_release(self.on_release)
		self.classifer = Classifier("models/classifier")
		self.regressors = {
			"rectangle": Regressor("models/rectangle_regressor"),
			"ellipse": Regressor("models/ellipse_regressor"),
			"triangle": Regressor("models/triangle_regressor")
		}

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

	def reconstruct(self, vertices, shape):
		self.reset()
		if shape == "other":
			return
		elif shape == "ellipse":
			self.ellipse(vertices)
		else:
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

		shape = self.classifer.classify(img)
		print(shape)
		if shape == "other": return
		
		vertices = self.regressors[shape].vertices(img)
		vertices = np.array(vertices).reshape((len(vertices)//2, 2))
		vertices *= img.size
		vertices += xy

		self.draw_vertices(vertices)
		self.root.after(500, lambda: self.reconstruct(vertices, shape))



canvas = DrawingCanvas()
