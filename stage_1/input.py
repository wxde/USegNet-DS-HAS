import os
import nibabel as nib
import numpy as np
from skimage.transform import resize
import json
from glob import glob
with open('info.json') as f:
    ParSet = json.load(f)

def Normarlize(xs):

	xs_min = np.min(xs)
	xs_max = np.max(xs)
	xs = (xs - xs_min)/(xs_max - xs_min)
	return xs

def data_load_train(base_path=ParSet['TRAIN_PATH'], nii_index=0):

	label_path = [base_path + p for p in os.listdir(base_path) if p.endswith('Label.nii.gz')]
	image_path = [p.replace('Label', '') for p in label_path]

	xs, ys = [nib.load(p[nii_index]).get_data() for p in [image_path, label_path]]

	xs = Normarlize(xs)

	xs_shape = np.shape(xs)

	label_aux1_shape = [xs_shape[0]//2, xs_shape[1]//2, xs_shape[2]//2]

	label_aux2_shape = [xs_shape[0]//4, xs_shape[1]//4, xs_shape[2]//4]

	label_aux1 = resize(ys, label_aux1_shape, mode='reflect')

	label_aux2 = resize(ys, label_aux2_shape, mode='reflect')


	xs, ys, label_aux1, label_aux2 = [item[np.newaxis, ..., np.newaxis] for item in [xs, ys, label_aux1, label_aux2]]

	return xs, ys, label_aux1, label_aux2


def load_inference(base_path=ParSet['TEST_PATH'], nii_index=0):

	pair_list = glob('{}/*.nii.gz'.format(base_path))
	pair_list.sort()
	file_name = pair_list[nii_index]

	xs = nib.load(file_name).get_data()

	xs = Normarlize(xs)

	seq = file_name.split("/")[-1].split(".")[0]

	num_dex = int(seq)

	return xs[None, ..., None],num_dex