# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ dataloader.py ]
#   Synopsis     [ data loader for the Tacotron model ]
#   Author       [ Ting-Wei Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import numpy as np
#----------------#
import torch
from torch.utils import data
from torch.autograd import Variable
#---------------------------------#
from config import config
from utils.text import text_to_sequence
#-------------------------------------#
from nnmnkwii.datasets import FileSourceDataset, FileDataSource


####################
# TEXT DATA SOURCE #
####################
class TextDataSource(FileDataSource):
	def __init__(self, data_root, meta_text):
		self.data_root = data_root
		self.meta_text = meta_text
		#self._cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]

	def collect_files(self):
		meta = os.path.join(self.data_root, self.meta_text)
		with open(meta, 'r', encoding='utf-8') as f:
			lines = f.readlines()
		lines = list(map(lambda l: l.split("|")[-1][:-1], lines))
		return lines

	def collect_features(self, text):
		return np.asarray(text_to_sequence(text), dtype=np.int32)


###################
# NPY DATA SOURCE #
###################
class _NPYDataSource(FileDataSource):
	def __init__(self, col, data_root, meta_text):
		self.col = col
		self.data_root = data_root
		self.meta_text = meta_text

	def collect_files(self):
		meta = os.path.join(self.data_root, self.meta_text)
		with open(meta, 'r', encoding='utf-8') as f:
			lines = f.readlines()
		lines = list(map(lambda l: l.split("|")[self.col], lines))
		paths = list(map(lambda f: os.path.join(self.data_root, f), lines))
		return paths

	def collect_features(self, path):
		return np.load(path)


########################
# MEL SPEC DATA SOURCE #
########################
class MelSpecDataSource(_NPYDataSource):
	def __init__(self, data_root, meta_text):
		super(MelSpecDataSource, self).__init__(1, data_root, meta_text)


###########################
# LINEAR SPEC DATA SOURCE #
###########################
class LinearSpecDataSource(_NPYDataSource):
	def __init__(self, data_root, meta_text):
		super(LinearSpecDataSource, self).__init__(0, data_root, meta_text)


#######################
# PYTORCH DATA SOURCE #
#######################
class PyTorchDatasetWrapper(object):
	def __init__(self, X, Mel, Y):
		self.X = X
		self.Mel = Mel
		self.Y = Y

	def __getitem__(self, idx):
		return self.X[idx], self.Mel[idx], self.Y[idx]

	def __len__(self):
		return len(self.X)


##############
# COLLATE FN #
##############
"""
	Create batch
"""
def collate_fn(batch):
	def _pad(seq, max_len):
		return np.pad(seq, (0, max_len - len(seq)), mode='constant', constant_values=0)

	def _pad_2d(x, max_len):
		return np.pad(x, [(0, max_len - len(x)), (0, 0)], mode="constant", constant_values=0)
	
	r = config.outputs_per_step
	input_lengths = [len(x[0]) for x in batch]
	
	max_input_len = np.max(input_lengths)
	max_target_len = np.max([len(x[1]) for x in batch]) + 1 # Add single zeros frame at least, so plus 1
	
	if max_target_len % r != 0:
		max_target_len += r - max_target_len % r
		assert max_target_len % r == 0

	input_lengths = torch.LongTensor(input_lengths)
	sorted_lengths, indices = torch.sort(input_lengths.view(-1), dim=0, descending=True)
	sorted_lengths = sorted_lengths.long().numpy()

	x_batch = np.array([_pad(x[0], max_input_len) for x in batch], dtype=np.int)
	x_batch = torch.LongTensor(x_batch)

	mel_batch = np.array([_pad_2d(x[1], max_target_len) for x in batch], dtype=np.float32)
	mel_batch = torch.FloatTensor(mel_batch)

	y_batch = np.array([_pad_2d(x[2], max_target_len) for x in batch], dtype=np.float32)
	y_batch = torch.FloatTensor(y_batch)
	
	gate_batch = torch.FloatTensor(len(batch), max_target_len).zero_()
	for i, x in enumerate(batch): gate_batch[i, len(x[1])-1:] = 1

	x_batch, mel_batch, y_batch, gate_batch, = Variable(x_batch[indices]), Variable(mel_batch[indices]), Variable(y_batch[indices]), Variable(gate_batch[indices])
	return x_batch, mel_batch, y_batch, gate_batch, sorted_lengths


###############
# DATA LOADER #
###############
"""
	Create dataloader
"""
def Dataloader(data_root, meta_text):
	
	# Input dataset definitions
	X = FileSourceDataset(TextDataSource(data_root, meta_text))
	Mel = FileSourceDataset(MelSpecDataSource(data_root, meta_text))
	Y = FileSourceDataset(LinearSpecDataSource(data_root, meta_text))

	# Dataset and Dataloader setup
	dataset = PyTorchDatasetWrapper(X, Mel, Y)
	data_loader = data.DataLoader(dataset, 
								  batch_size=config.batch_size,
								  num_workers=config.num_workers, 
								  shuffle=True,
								  collate_fn=collate_fn, 
								  pin_memory=config.pin_memory)
	return data_loader

