import os
import time
from io import BytesIO
import absl
import warnings
import src.config
import sys
import tarfile
import tempfile
from six.moves import urllib
import threading

import numpy as np
from PIL import Image
import cv2, pdb, glob, argparse
from demo import main
import tensorflow as tf

warnings.filterwarnings('ignore')

class DeepLabModel(object):
	INPUT_TENSOR_NAME = 'ImageTensor:0'
	OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
	INPUT_SIZE = 513
	FROZEN_GRAPH_NAME = 'frozen_inference_graph'

	def __init__(self, tarball_path):
		self.graph = tf.Graph()
		graph_def = None
		tar_file = tarfile.open(tarball_path)
		for tar_info in tar_file.getmembers():
			if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
				file_handle = tar_file.extractfile(tar_info)
				graph_def = tf.GraphDef.FromString(file_handle.read())
				break

		tar_file.close()

		if graph_def is None:
			raise RuntimeError('Cannot find inference graph in tar archive.')

		with self.graph.as_default():
			tf.import_graph_def(graph_def, name='')

		self.sess = tf.Session(graph=self.graph)

	def run(self, image):
		"""Runs inference on a single image.

		Args:
		  image: A PIL.Image object, raw input image.

		Returns:
		  resized_image: RGB image resized from original input image.
		  seg_map: Segmentation map of `resized_image`.
		"""
		width, height = image.size
		resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
		target_size = (int(resize_ratio * width), int(resize_ratio * height))
		resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
		batch_seg_map = self.sess.run(
			self.OUTPUT_TENSOR_NAME,
			feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
		seg_map = batch_seg_map[0]
		return resized_image, seg_map

def create_pascal_label_colormap():
	"""Creates a label colormap used in PASCAL VOC segmentation benchmark.

	Returns:
	A Colormap for visualizing segmentation results.
	"""
	colormap = np.zeros((256, 3), dtype=int)
	ind = np.arange(256, dtype=int)

	for shift in reversed(range(8)):
		for channel in range(3):
			colormap[:, channel] |= ((ind >> channel) & 1) << shift
		ind >>= 3

	return colormap

def label_to_color_image(label):
	"""Adds color defined by the dataset colormap to the label.

	Args:
	label: A 2D array with integer type, storing the segmentation label.

	Returns:
	result: A 2D array with floating type. The element of the array
	  is the color indexed by the corresponding element in the input label
	  to the PASCAL color map.

	"""
	if label.ndim != 2:
		raise ValueError('Expect 2-D input label')

	colormap = create_pascal_label_colormap()

	if np.max(label) >= len(colormap):
		raise ValueError('label value too large.')

	return colormap[label]


#path = input("Enter the path of image: ")
height =  int(input('Enter the height of person: '))
parser = argparse.ArgumentParser(description='Deeplab Segmentation')
#dir_name = path



LABEL_NAMES = np.asarray([
	'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
	'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
	'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
])

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)


MODEL_NAME = 'xception_coco_voctrainval'

_DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
_MODEL_URLS = {
	'mobilenetv2_coco_voctrainaug':
		'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz',
	'mobilenetv2_coco_voctrainval':
		'deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz',
	'xception_coco_voctrainaug':
		'deeplabv3_pascal_train_aug_2018_01_04.tar.gz',
	'xception_coco_voctrainval':
		'deeplabv3_pascal_trainval_2018_01_04.tar.gz',
}
_TARBALL_NAME = _MODEL_URLS[MODEL_NAME]

model_dir = 'deeplab_model'
if not os.path.exists(model_dir):
	tf.gfile.MakeDirs(model_dir)

download_path = os.path.join(model_dir, _TARBALL_NAME)
if not os.path.exists(download_path):
	print('downloading model to %s, this might take a while...' % download_path)
	urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + _MODEL_URLS[MODEL_NAME],
				 download_path)
	print('download completed! loading DeepLab model..')

MODEL = DeepLabModel(download_path)
print('model loaded successfully!')

img_counter = 0
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_BUFFERSIZE, 2)
FPS = 1/30
FPS_MS = int(FPS * 1000)

while True:
	ret, frame = cam.read()
	time.sleep(FPS)

	#cv2.imwrite("frame.png", frame)
	if not ret:
		print("failed to grab frame")
		break
	cv2.imshow("test", frame)
	#frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	image = Image.fromarray(frame)
	print(image)
	#image = Image.open(pil_im)
	#image_rgb = cv2.imread("opencv_python_0.png")
	#image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
	back = cv2.imread('sample_data/input/background.jpeg',cv2.IMREAD_COLOR)

	res_im,seg=MODEL.run(image)

	seg=cv2.resize(seg.astype(np.uint8),image.size)
	mask_sel=(seg==15).astype(np.float32)
	mask = 255*mask_sel.astype(np.uint8)

	img = np.array(image)
	img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

	res = cv2.bitwise_and(img,img,mask = mask)
	bg_removed = res + (255 - cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))

	main(frame,bg_removed,height,None)
	#t1 = threading.Thread(target=main, args=(frame,bg_removed,height,None))
	#t1.start()
	cv2.imshow("test", frame)
	key = cv2.waitKey(1)
	if key == ord('q'):
		print("VIDEO FEED TERMINATED")
		# cam.release()
		# cv2.destroyAllWindows()
		break

cam.release()
cv2.destroyAllWindows()

# for single image
# image = Image.open("sample_data/input/ss1.jpeg")
# print(image)
# image_rgb = cv2.imread("sample_data/input/ss1.jpeg")
# #image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
# back = cv2.imread('sample_data/input/background.jpeg',cv2.IMREAD_COLOR)

# res_im,seg=MODEL.run(image)

# seg=cv2.resize(seg.astype(np.uint8),image.size)
# mask_sel=(seg==15).astype(np.float32)
# mask = 255*mask_sel.astype(np.uint8)

# img = np.array(image)
# img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#
# res = cv2.bitwise_and(img,img,mask = mask)
# bg_removed = res + (255 - cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
#
# cv2.imwrite('input_image.png',bg_removed)
# main(image_rgb, bg_removed,height,None)


#https://github.com/the-vis-sharma/Human-Body-Measurement

