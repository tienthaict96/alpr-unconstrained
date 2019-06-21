import sys, os
import keras
import cv2
import traceback
import numpy as np
import random
import time
from src.keras_utils import load_model
from glob import glob
from os.path import splitext, basename
from src.utils import im2single
from src.keras_utils import load_model, detect_lp
from src.label import Shape, writeShapes
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# config.log_device_placement = True
sess = tf.Session(config=config)
set_session(sess)


image_paths = glob("../train-detector/*.jpg")
wpod_net = load_model('./models/my-trained-model/my-trained-model_final.h5')

while True:
	image_path = random.choice(image_paths)
	Ivehicle = cv2.imread(image_path)
	start = time.time()
	Ivehicle_cpy = Ivehicle.copy()

	ratio = float(max(Ivehicle.shape[:2]))/min(Ivehicle.shape[:2])
	side  = int(ratio*288.)
	bound_dim = min(side + (side%(2**4)),608)

	lp_threshold = .5

	Llp,LlpImgs,_ = detect_lp(wpod_net,im2single(Ivehicle),bound_dim,2**4,(240,80),lp_threshold)
	if len(LlpImgs):
		s = Shape(Llp[0].pts)

		coords = s.pts.T.copy()
		coords[:, 0] *= Ivehicle.shape[1]
		coords[:, 1] *= Ivehicle.shape[0]
		coords = coords.astype('int')
		coords = coords.reshape((-1,1,2))
		cv2.polylines(Ivehicle_cpy,[coords],True,(0,0,255), 2)

	Ivehicle_cpy = cv2.resize(Ivehicle_cpy, (600, 400))
	print(len(LlpImgs), time.time() - start)
	
	cv2.imshow("image", Ivehicle_cpy)
	key = cv2.waitKey(0)
	if key == ord('q'):
		break

cv2.destroyAllWindows()