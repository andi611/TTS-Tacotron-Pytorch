# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ train.py ]
#   Synopsis     [ Trainining script for Tacotron speech synthesis model ]
#   Author       [ Ting-Wei Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


"""
	Usage: train.py [options]

	Options:
		--checkpoint_dir <dir>    Directory where to save model checkpoints [default: checkpoints].
		--checkpoint_path <name>  Restore model from checkpoint path if given.
		--data_root <dir>         Directory contains preprocessed features.
		--meta_text <name>        Name of the model-ready training transcript.
		--summary_comment <str>   Comment for log summary writer.
		-h, --help                Show this help message and exit
"""


###############
# IMPORTATION #
###############
import os
import sys
import time
#-----------------------#
import numpy as np
import librosa.display
#---------------------#
from utils import audio
from utils.plot import plot_alignment, plot_spectrogram
from utils.text import text_to_sequence, symbols
#----------------------------------------------#
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils import data
#----------------------------------------#
from model.tacotron import Tacotron
from config import config, get_training_args
#------------------------------------------#
from nnmnkwii.datasets import FileSourceDataset, FileDataSource
from tensorboardX import SummaryWriter


####################
# GLOBAL VARIABLES #
####################
global_step = 0
global_epoch = 0
USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
	cudnn.benchmark = False
DATA_ROOT = None
META_TEXT = None


def _pad(seq, max_len):
	return np.pad(seq, (0, max_len - len(seq)),
				  mode='constant', constant_values=0)

def _pad_2d(x, max_len):
	x = np.pad(x, [(0, max_len - len(x)), (0, 0)],
			   mode="constant", constant_values=0)
	return x

####################
# TEXT DATA SOURCE #
####################
class TextDataSource(FileDataSource):
	def __init__(self):
		pass #self._cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]

	def collect_files(self):
		meta = os.path.join(DATA_ROOT, META_TEXT)
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
	def __init__(self, col):
		self.col = col

	def collect_files(self):
		meta = os.path.join(DATA_ROOT, META_TEXT)
		with open(meta, 'r', encoding='utf-8') as f:
			lines = f.readlines()
		lines = list(map(lambda l: l.split("|")[self.col], lines))
		paths = list(map(lambda f: os.path.join(DATA_ROOT, f), lines))
		return paths

	def collect_features(self, path):
		return np.load(path)


########################
# MEL SPEC DATA SOURCE #
########################
class MelSpecDataSource(_NPYDataSource):
	def __init__(self):
		super(MelSpecDataSource, self).__init__(1)


###########################
# LINEAR SPEC DATA SOURCE #
###########################
class LinearSpecDataSource(_NPYDataSource):
	def __init__(self):
		super(LinearSpecDataSource, self).__init__(0)


#######################
# PYTORCH DATA SOURCE #
#######################
class PyTorchDataset(object):
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
	
	r = config.outputs_per_step
	input_lengths = [len(x[0]) for x in batch]
	max_input_len = np.max(input_lengths)
	
	max_target_len = np.max([len(x[1]) for x in batch]) + 1 # Add single zeros frame at least, so plus 1
	if max_target_len % r != 0:
		max_target_len += r - max_target_len % r
		assert max_target_len % r == 0

	a = np.array([_pad(x[0], max_input_len) for x in batch], dtype=np.int)
	x_batch = torch.LongTensor(a)

	input_lengths = torch.LongTensor(input_lengths)

	b = np.array([_pad_2d(x[1], max_target_len) for x in batch], dtype=np.float32)
	mel_batch = torch.FloatTensor(b)

	c = np.array([_pad_2d(x[2], max_target_len) for x in batch], dtype=np.float32)
	y_batch = torch.FloatTensor(c)
	
	return x_batch, input_lengths, mel_batch, y_batch


#######################
# LEARNING RATE DECAY #
#######################
def _learning_rate_decay(init_lr, global_step):
	warmup_steps = 6000.0
	step = global_step + 1.
	lr = init_lr * warmup_steps**0.5 * np.minimum(step * warmup_steps**-1.5, step**-0.5)
	return lr


###############
# SAVE STATES #
###############
def save_states(global_step, mel_outputs, linear_outputs, attn, y,
				input_lengths, checkpoint_dir=None):

	
	idx = min(1, len(input_lengths) - 1) # idx = np.random.randint(0, len(input_lengths))
	input_length = input_lengths[idx]

	# Alignment
	path = os.path.join(checkpoint_dir, "step{}_alignment.png".format(
		global_step))
	alignment = attn[idx].cpu().data.numpy() # alignment = attn[idx].cpu().data.numpy()[:, :input_length]
	plot_alignment(alignment.T, path, info="tacotron, step={}".format(global_step))

	# Predicted spectrogram
	path = os.path.join(checkpoint_dir, "step{}_predicted_spectrogram.png".format(
		global_step))
	linear_output = linear_outputs[idx].cpu().data.numpy()
	plot_spectrogram(linear_output, path)

	# Predicted audio signal
	signal = audio.inv_spectrogram(linear_output.T)
	path = os.path.join(checkpoint_dir, "step{}_predicted.wav".format(
		global_step))
	audio.save_wav(signal, path)

	# Target spectrogram
	path = os.path.join(checkpoint_dir, "step{}_target_spectrogram.png".format(
		global_step))
	linear_output = y[idx].cpu().data.numpy()
	plot_spectrogram(linear_output, path)


###################
# SAVE CHECKPOINT #
###################
def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch):
	checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_step{}.pth".format(global_step))
	torch.save({"state_dict": model.state_dict(),
				"optimizer": optimizer.state_dict(),
				"global_step": step,
				"global_epoch": epoch,}, 
				checkpoint_path)


#################
# TACOTRON STEP #
#################
"""
	One step of training: Train a single batch of data on Tacotron
"""
def tacotron_step(model, optimizer, criterion,
				  x, input_lengths, mel, y,
				  init_lr, sample_rate, clip_thresh,
				  running_loss, data_len, global_step):
	
	#---decay learning rate---#
	current_lr = _learning_rate_decay(init_lr, global_step)
	for param_group in optimizer.param_groups:
		param_group['lr'] = current_lr

	optimizer.zero_grad()

	#---sort by length---#
	sorted_lengths, indices = torch.sort(input_lengths.view(-1), dim=0, descending=True)
	sorted_lengths = sorted_lengths.long().numpy()

	#---feed data---#
	x, mel, y = Variable(x[indices]), Variable(mel[indices]), Variable(y[indices])
	if USE_CUDA:
		x, mel, y = x.cuda(), mel.cuda(), y.cuda()
	mel_outputs, linear_outputs, attn = model(x, mel, input_lengths=sorted_lengths)

	#---Loss---#
	mel_loss = criterion(mel_outputs, mel)
	n_priority_freq = int(3000 / (sample_rate * 0.5) * model.linear_dim)
	linear_loss = 0.5 * criterion(linear_outputs, y) + 0.5 * criterion(linear_outputs[:, :, :n_priority_freq], y[:, :, :n_priority_freq])
	loss = mel_loss + linear_loss
	
	#---log loss---#
	total_L = loss.item()
	running_loss += loss.item()
	avg_L = running_loss / (data_len)
	mel_L = mel_loss.item()
	linear_L = linear_loss.item()

	#---update model---#
	loss.backward()
	grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_thresh)
	optimizer.step()

	#---wrap up returns---#
	Ms = { 'mel_outputs' : mel_outputs, 
		   'linear_outputs' : linear_outputs,
		   'attn' : attn,
		   'sorted_lengths' : sorted_lengths,
		   'grad_norm' : grad_norm,
		   'current_lr' : current_lr }
	Ls = { 'total_L': total_L,
		   'avg_L' : avg_L,
		   'mel_L' : mel_L,
		   'linear_L' : linear_L }


	return model, optimizer, Ms, Ls


#########
# TRAIN #
#########
def train(model, 
		  optimizer,
		  data_loader, 
		  summary_comment,
		  init_lr=0.002,
		  checkpoint_dir=None, 
		  checkpoint_interval=None, 
		  nepochs=None,
		  clip_thresh=1.0,
		  sample_rate=20000):

	if USE_CUDA: 
		model = model.cuda()
	
	model.train()
	writer = SummaryWriter() if summary_comment == None else SummaryWriter(summary_comment)

	global global_step, global_epoch
	criterion = nn.L1Loss()


	while global_epoch < nepochs:
		
		start = time.time()
		running_loss = 0.
		
		for x, input_lengths, mel, y in data_loader:
			
			model, optimizer, Ms, Rs = tacotron_step(model, optimizer, criterion,
												 x, input_lengths, mel, y,
												 init_lr, sample_rate, clip_thresh,
												 running_loss, len(data_loader), global_step)

			mel_outputs = Ms['mel_outputs']
			linear_outputs = Ms['linear_outputs']
			attn = Ms['attn']
			sorted_lengths = Ms['sorted_lengths']
			grad_norm = Ms['grad_norm']
			current_lr = Ms['current_lr']

			total_L = Rs['total_L']
			avg_L = Rs['avg_L']
			mel_L = Rs['mel_L']
			linear_L = Rs['linear_L']

			duration = time.time() - start
			if global_step > 0 and global_step % checkpoint_interval == 0:
				save_states(global_step, mel_outputs, linear_outputs, attn, y, sorted_lengths, checkpoint_dir)
				save_checkpoint(model, optimizer, global_step, checkpoint_dir, global_epoch)
				log = '[{}] total_L: {:.3f}, avg_L: {:.3f}, mel_L: {:.3f}, mag_L: {:.3f}, grad_norm: {:.3f}, lr: {:.5f}, t: {:.2f}s, saved: T'.format(global_step, total_L, avg_L, mel_L, linear_L, grad_norm, current_lr, duration)
				print(log)
			elif global_step % 5 == 0:
				log = '[{}] total_L: {:.3f}, avg_L: {:.3f}, mel_L: {:.3f}, mag_L: {:.3f}, grad_norm: {:.3f}, lr: {:.5f}, t: {:.2f}s, saved: F'.format(global_step, total_L, avg_L, mel_L, linear_L, grad_norm, current_lr, duration)
				print(log, end='\r')

			# Logs
			writer.add_scalar('total_loss', total_L, global_step)
			writer.add_scalar('averaged_loss', avg_L, global_step)
			writer.add_scalar('mel_loss', mel_L, global_step)
			writer.add_scalar('linear_loss', linear_L, global_step)
			writer.add_scalar('grad_norm', grad_norm, global_step)
			writer.add_scalar('learning_rate', current_lr, global_step)

			global_step += 1
			start = time.time()

		global_epoch += 1


#######################
# INITIALIZE TRAINING #
#######################
"""
	Setup and prepare for Tacotron training.
"""
def initialize_training(checkpoint_path):
	
	# Input dataset definitions
	X = FileSourceDataset(TextDataSource())
	Mel = FileSourceDataset(MelSpecDataSource())
	Y = FileSourceDataset(LinearSpecDataSource())

	# Dataset and Dataloader setup
	dataset = PyTorchDataset(X, Mel, Y)
	data_loader = data.DataLoader(dataset, 
								  batch_size=config.batch_size,
								  num_workers=config.num_workers, 
								  shuffle=True,
								  collate_fn=collate_fn, 
								  pin_memory=config.pin_memory)

	# Model
	model = Tacotron(n_vocab=len(symbols),
					 embedding_dim=config.embedding_dim,
					 mel_dim=config.num_mels,
					 linear_dim=config.num_freq,
					 r=config.outputs_per_step,
					 padding_idx=config.padding_idx,
					 use_memory_mask=config.use_memory_mask)

	optimizer = optim.Adam(model.parameters(),
						   lr=config.initial_learning_rate, 
						   betas=(config.adam_beta1, config.adam_beta2),
						   weight_decay=config.weight_decay)

	# Load checkpoint
	if checkpoint_path != None:
		print("Load checkpoint from: {}".format(checkpoint_path))
		checkpoint = torch.load(checkpoint_path)
		model.load_state_dict(checkpoint["state_dict"])
		optimizer.load_state_dict(checkpoint["optimizer"])
		try:
			global_step = checkpoint["global_step"]
			global_epoch = checkpoint["global_epoch"]
		except:
			print('Warning: global step and global epoch unable to restore!')
			sys.exit(0)
			
	return model, optimizer, data_loader


########
# MAIN #
########
def main():

	args = get_training_args()

	global DATA_ROOT, META_TEXT
	if args.data_root != None:
		DATA_ROOT = args.data_root
	if args.meta_text != None:
		META_TEXT = args.meta_text

	checkpoint_dir = args.checkpoint_dir
	checkpoint_path = args.checkpoint_path
	os.makedirs(checkpoint_dir, exist_ok=True)

	model, optimizer, data_loader = initialize_training(checkpoint_path)

	# Train!
	try:
		train(model, optimizer, data_loader, args.summary_comment,
			  init_lr=config.initial_learning_rate,
			  checkpoint_dir=checkpoint_dir,
			  checkpoint_interval=config.checkpoint_interval,
			  nepochs=config.nepochs,
			  clip_thresh=config.clip_thresh,
			  sample_rate=config.sample_rate)
	except KeyboardInterrupt:
		print()
		pass

	print("Finished")
	sys.exit(0)


if __name__ == "__main__":
	main()