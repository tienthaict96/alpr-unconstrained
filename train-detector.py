
import sys
import numpy as np
import cv2
import argparse
import keras

from random import choice
from os.path import isfile, isdir, basename, splitext
from os import makedirs

from src.keras_utils import save_model, load_model
from src.label import readShapes
from src.loss import loss
from src.utils import image_files_from_folder, show
from src.sampler import augment_sample, labels2output_map
from src.data_generator import DataGenerator

from pdb import set_trace as pause
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# config.log_device_placement = True
sess = tf.Session(config=config)
set_session(sess)


def load_network(modelpath, input_dim):
	# input_dim = 208
	model = load_model(modelpath)
	input_shape = (input_dim,input_dim,3)

	# Fixed input size for training
	# Số chiều đầu vào là 208*208
	inputs  = keras.layers.Input(shape=(input_dim,input_dim,3))
	outputs = model(inputs)

	output_shape = tuple([s.value for s in outputs.shape[1:]])
	output_dim   = output_shape[1]
	model_stride = input_dim / output_dim

	'''kiểm tra điều kiện, nếu sai > out, kèm exception'''
	assert input_dim % output_dim == 0, \
		'The output resolution must be divisible by the input resolution'

	assert model_stride == 2**4, \
		'Make sure your model generates a feature map with resolution ' \
		'16x smaller than the input'

	return model, model_stride, input_shape, output_shape

def process_data_item(data_item,dim,model_stride):
	# data_item[0] dữ liệu ảnh opencv
	# data_item[1].pts thông tin tọa độ đã scale về 0-1
	# dim là 208
	XX, llp, pts = augment_sample(data_item[0], data_item[1].pts, dim)
	YY = labels2output_map(llp,pts,dim,model_stride)
	return XX,YY


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	# Lưu checkpoint
	parser.add_argument('-m' 		,'--model'			,type=str   , required=True		,help='Path to previous model')
	parser.add_argument('-n' 		,'--name'			,type=str   , required=True		,help='Model name')
	parser.add_argument('-tr'		,'--train-dir'		,type=str   , required=True		,help='Input data directory for training')
	parser.add_argument('-its'		,'--iterations'		,type=int   , default=300000	,help='Number of mini-batch iterations (default = 300.000)')
	parser.add_argument('-bs'		,'--batch-size'		,type=int   , default=32		,help='Mini-batch size (default = 32)')
	parser.add_argument('-od'		,'--output-dir'		,type=str   , default='./'		,help='Output directory (default = ./)')
	parser.add_argument('-op'		,'--optimizer'		,type=str   , default='Adam'	,help='Optmizer (default = Adam)')
	parser.add_argument('-lr'		,'--learning-rate'	,type=float , default=.01		,help='Optmizer (default = 0.01)')
	args = parser.parse_args()

	netname 	= basename(args.name)
	train_dir 	= args.train_dir
	outdir 		= args.output_dir

	iterations 	= args.iterations
	batch_size 	= args.batch_size
	dim 		= 208

	if not isdir(outdir):
		makedirs(outdir)

	model, model_stride, xshape, yshape = load_network(args.model,dim)

	opt = getattr(keras.optimizers,args.optimizer)(lr=args.learning_rate)
	model.compile(loss=loss, optimizer=opt)

	print('Checking input directory...')
	'''Trả lại danh sách path image có trong folder'''
	Files = image_files_from_folder(train_dir)

	Data = []
	# for file in Files:
	# 	# Lấy path+tên file (bỏ đi phần mở rộng)
	# 	labfile = splitext(file)[0] + '.txt'
	# 	if isfile(labfile):
	# 		# trả lại đối tượng bao gồm tọa độ các vị trí, lấy lại tọa độ của biển tại index 0
	# 		L = readShapes(labfile)
	# 		I = cv2.imread(file)
	# 		Data.append([I,L[0]])
	for file in Files:
		'''trả lại đối tượng bao gồm tọa độ các vị trí, lấy lại tọa độ của biển tại index 0'''
		L = readShapes(labfile)
		I = cv2.imread(file)
		Data.append([I,L[0]])

	print('%d images with labels found' % len(Data))

	dg = DataGenerator(	data=Data, \
						process_data_item_func=lambda x: process_data_item(x,dim,model_stride),\
						xshape=xshape, \
						yshape=(yshape[0],yshape[1],yshape[2]+1), \
						nthreads=2, \
						pool_size=1000, \
						min_nsamples=100 )
	dg.start()

	Xtrain = np.empty((batch_size,dim,dim,3),dtype='single')
	print(model_stride)
	Ytrain = np.empty((batch_size,int(dim/model_stride), int(dim/model_stride),2*4+1))

	model_path_backup = '%s/%s_backup' % (outdir,netname)
	model_path_final  = '%s/%s_final'  % (outdir,netname)

	for it in range(iterations):

		print('Iter. %d (of %d)' % (it+1,iterations))

		Xtrain,Ytrain = dg.get_batch(batch_size)
		train_loss = model.train_on_batch(Xtrain,Ytrain)

		print('\tLoss: %f' % train_loss)

		# Save model every 1000 iterations
		if (it+1) % 1000 == 0:
			print('Saving model (%s)' % model_path_backup)
			save_model(model,model_path_backup)

	print('Stopping data generator')
	dg.stop()

	print('Saving model (%s)' % model_path_final)
	save_model(model,model_path_final)
