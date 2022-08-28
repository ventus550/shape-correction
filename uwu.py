import numpy as np
import tkinter as tk
from tkinter import Canvas
from collections import deque
from PIL import Image, ImageDraw, ImageOps
import torchvision.transforms as transforms

def add_margin(pil_img, margin, color = 3 * (255,)):
	shape = np.array(pil_img.size)
	shape += 2*margin
	result = Image.new(pil_img.mode, tuple(shape), color)
	result.paste(pil_img, (margin, margin))
	return result

class DrawingCanvas(tk.Tk):
	def __init__(self, width = 1200, height = 1200):
		super().__init__()
		self.width = width
		self.height = height
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
	
	def line(self, lin, width = 5):
		self.canvas.create_line(lin, width=width)
		ImageDraw.Draw(self.image).line(lin, (0,0,0), width=width)
	
	def draw_bezier_curve(self, n = 50):
		p = self.beziers
		start = p[0]

		for i in range(n):
			t = i / n
			x, y = p[0] * (1-t)**3 + p[1] * 3 * t * (1-t)**2 + p[2] * 3 * t**2 * (1-t) + p[3] * t**3
			self.line((x, y, start[0], start[1]))
			start = np.array((x,y))

	def get_point(self, e):
		self.beziers.append(np.array((e.x, e.y)))
		x, y, X, Y = self.region
		self.region = [
			min(x, max(e.x, 0)),
			min(y, max(e.y, 0)),
			max(X, min(e.x, self.width)),
			max(Y, min(e.y, self.height))
		]

	def on_mouse_down(self, _):
		self.reset()

	def on_mouse_drag(self, e):
		self.get_point(e)
		if len(self.beziers) == 4:
			self.draw_bezier_curve()

	def on_mouse_release(self, _):
		print(self.region)
		img = ImageOps.invert(add_margin(self.image.crop(self.region), 50).resize((70, 70)))
		img.save("uwu.png")

piechart = DrawingCanvas()
piechart.mainloop()
