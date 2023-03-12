import os
from pathlib import Path
from tensorflow import keras
from PIL import Image, ImageOps
import numpy as np
from skimage import morphology
import cv2


image_size = 70
DESTDIR = Path(os.getcwd()) / "preprocessing"
Path.mkdir(DESTDIR, parents=True, exist_ok=True)

def store_transformation(transformation):
	def inner(x):
		t = transformation(x)

		keras.preprocessing.image.save_img(
			str(DESTDIR / transformation.__name__) + ".png",
			t.reshape((image_size, image_size, 1))
		)
		return t
	return inner

def input(image: Image):
	image.save(DESTDIR / "input.png")
	return keras.preprocessing.image.img_to_array(image)

@store_transformation
def gray_invert_resized(image, size=70):
	image = keras.preprocessing.image.array_to_img(image)
	image = ImageOps.invert(ImageOps.grayscale(image)).resize((size, size))
	return keras.preprocessing.image.img_to_array(image)

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

def preprocess(image: Image, *transformation_pipeline):
	for transformation in transformation_pipeline:
		image = transformation(image)
	return image

def foo(image: Image):
	return preprocess(
		image,
		input,
		gray_invert_resized,
		binarize,
		skeletonize,
		dilate,
		binarize
	)
