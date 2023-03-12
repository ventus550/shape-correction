import os
from pathlib import Path
from tensorflow import keras
from PIL import Image, ImageOps
import numpy as np
from skimage import morphology
from contextlib import contextmanager
import cv2


image_size = 70
DESTDIR = Path(os.getcwd()) / "preprocessing"
Path.mkdir(DESTDIR, parents=True, exist_ok=True)

@contextmanager
def window(fn):
	try:
		yield
	except Exception as e:
		print(f"Error encountered in {fn}\n{e}")
		exit(1)

def store_transformation(transformation):
	def inner(x):
		assert x.shape[0] == x.shape[1] == image_size
		with window(transformation.__name__):
			tr = transformation(x)
			keras.preprocessing.image.save_img(
				str(DESTDIR / transformation.__name__) + ".png",
				tr.reshape((image_size, image_size, 1))
			)
		return tr
	return inner

def image_conversion(func):
	def converted(x):
		with window(func.__name__):
			x = keras.preprocessing.image.array_to_img(x)
			y = func(x)
			y = keras.preprocessing.image.img_to_array(y)
		return y
	converted.__name__ = func.__name__
	return converted

def input(image: Image):
	image.save(DESTDIR / "input.png")
	return keras.preprocessing.image.img_to_array(image)

@image_conversion
def resize(image, size=image_size):
	return image.resize((size, size))

@store_transformation
@image_conversion
def grayinvert(image):
	return ImageOps.invert(ImageOps.grayscale(image)).resize((70, 70))

@store_transformation
def binarize(image):
	image[image > 0] = 1.0
	return image

@store_transformation
def skeletonize(image):
	return morphology.skeletonize(image)

@store_transformation
def dilate(image):
	return cv2.dilate(
		image,
		cv2.getStructuringElement(cv2.MORPH_RECT, (2,2)),
		iterations=1
	)

@store_transformation
def padding(image):
	# bad, do not use
	margin = 10
	image = cv2.copyMakeBorder(
		np.asarray(image),
		*(4*[margin]),
		borderType=cv2.BORDER_CONSTANT,
		value=0
	)
	return resize(np.expand_dims(image, axis=2))

@store_transformation
def normalize(image):
	return image / np.max(image)

@store_transformation
def blur(image, k=3):
	# probably quite bad
	return cv2.blur(image, (k, k), cv2.BORDER_DEFAULT)

def preprocessing_pipeline(image: Image, *transformation_pipeline):
	for transformation in transformation_pipeline:
		image = transformation(image)		
	return image

def preprocess(image: Image):
	return preprocessing_pipeline(
		image,
		input,
		resize,
		grayinvert,
		normalize,
		skeletonize, 
		dilate,
		# blur,
		normalize
	)
