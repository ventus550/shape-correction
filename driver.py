from models import Classifier, Regressor, transform
from numpy import array, argmin
from canvas import Canvas


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
		#TODO
		def dist(v, u): return sum((v - u)**2)
		vertices = list(vertices)
		first = v = vertices.pop()
		while vertices:
			u = vertices.pop(argmin([dist(u,v) for u in vertices]))
			self.line(*v, *u)
			v = u
		self.line(*first, *v)

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
