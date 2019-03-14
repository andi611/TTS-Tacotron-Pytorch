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
		--ckpt_dir <dir>    	  Directory where to save model checkpoints [default: checkpoints].
		--model_name <name>  	  Restore model from checkpoint path if name is given.
		--data_root <dir>         Directory contains preprocessed features.
		--meta_text <name>        Name of the model-ready training transcript.
		--log_dir <str>   	  	  Directory for log summary writer to write in.
		--log_comment <str>   	  Comment for log summary writer.
		-h, --help                Show this help message and exit
"""


###############
# IMPORTATION #
###############
import os
import sys
import time
#----------------#
import numpy as np
#---------------------#
from utils import audio
from utils.plot import plot_alignment, plot_spectrogram
from utils.text import symbols
#----------------------------------------------#
import torch
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
#----------------------------------------#
from model.tacotron import Tacotron
from model.loss import TacotronLoss
from config import config, get_training_args
from dataloader import Dataloader
#------------------------------------------#
from tensorboardX import SummaryWriter


####################
# GLOBAL VARIABLES #
####################
global_step = 0
global_epoch = 0
USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
	cudnn.benchmark = False


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
def save_states(global_step, mel_outputs, linear_outputs, attn, y, checkpoint_dir=None):

	
	idx = 1 # idx = np.random.randint(0, len(mel_outputs))

	# Alignment
	path = os.path.join(checkpoint_dir, "step{}_alignment.png".format(global_step))
	alignment = attn[idx].cpu().data.numpy() # alignment = attn[idx].cpu().data.numpy()[:, :input_length]
	plot_alignment(alignment.T, path, info="tacotron, step={}".format(global_step))

	# Predicted spectrogram
	path = os.path.join(checkpoint_dir, "step{}_predicted_spectrogram.png".format(global_step))
	linear_output = linear_outputs[idx].cpu().data.numpy()
	plot_spectrogram(linear_output, path)

	# Predicted audio signal
	signal = audio.inv_spectrogram(linear_output.T)
	path = os.path.join(checkpoint_dir, "step{}_predicted.wav".format(global_step))
	audio.save_wav(signal, path)

	# Target spectrogram
	path = os.path.join(checkpoint_dir, "step{}_target_spectrogram.png".format(global_step))
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
				  x, mel, y, gate, sorted_lengths,
				  init_lr, clip_thresh, global_step):
	
	#---decay learning rate---#
	current_lr = _learning_rate_decay(init_lr, global_step)
	for param_group in optimizer.param_groups:
		param_group['lr'] = current_lr

	#---feed data---#
	if USE_CUDA:
		x, mel, y, gate, = x.cuda(), mel.cuda(), y.cuda(), gate.cuda()
	mel_outputs, linear_outputs, gate_outputs, attn = model(x, mel, input_lengths=sorted_lengths)

	losses = criterion([mel_outputs, linear_outputs, gate_outputs], [mel, y, gate])
	
	#---log loss---#
	loss, total_L = losses[0], losses[0].item()
	mel_loss, mel_L = losses[1], losses[1].item(), 
	linear_loss, linear_L = losses[2], losses[2].item()
	gate_loss, gate_L = losses[3], losses[3].item()

	#---update model---#
	optimizer.zero_grad()
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
		   'mel_L' : mel_L,
		   'linear_L' : linear_L,
		   'gate_L' : gate_L }

	return model, optimizer, Ms, Ls


#########
# TRAIN #
#########
"""
	Main training loop
"""
def train(model, 
		  optimizer,
		  dataloader, 
		  init_lr=0.002,
		  log_dir=None,
		  log_comment=None,
		  checkpoint_dir=None, 
		  checkpoint_interval=None, 
		  max_epochs=None,
		  max_steps=None,
		  clip_thresh=1.0):

	if USE_CUDA: 
		model = model.cuda()
	
	model.train()
	criterion = TacotronLoss()
	
	if log_dir != None:
		writer = SummaryWriter(log_dir)
	elif log_comment != None:
		writer = SummaryWriter(comment=log_comment)
	else:
		writer = SummaryWriter()

	global global_step, global_epoch

	while global_epoch < max_epochs and global_step < max_steps:
		
		start = time.time()
		
		for x, mel, y, gate, sorted_lengths in dataloader:
			
			model, optimizer, Ms, Rs = tacotron_step(model, optimizer, criterion,
													x, mel, y, gate, sorted_lengths,
													init_lr, clip_thresh, global_step)

			mel_outputs = Ms['mel_outputs']
			linear_outputs = Ms['linear_outputs']
			attn = Ms['attn']
			sorted_lengths = Ms['sorted_lengths']
			grad_norm = Ms['grad_norm']
			current_lr = Ms['current_lr']

			total_L = Rs['total_L']
			mel_L = Rs['mel_L']
			linear_L = Rs['linear_L']
			gate_L = Rs['gate_L']

			duration = time.time() - start
			if global_step > 0 and global_step % checkpoint_interval == 0:
				try:
					save_states(global_step, mel_outputs, linear_outputs, attn, y, checkpoint_dir)
					save_checkpoint(model, optimizer, global_step, checkpoint_dir, global_epoch)
				except:
					print()
					print('An error has occured during saving! Please attend and handle manually!')
					pass
				log = '[{}] total_L: {:.3f}, mel_L: {:.3f}, lin_L: {:.3f}, gate_L: {:.3f}, grad: {:.3f}, lr: {:.5f}, t: {:.2f}s, saved: T'.format(global_step, total_L, mel_L, linear_L, gate_L, grad_norm, current_lr, duration)
				print(log)
			elif global_step % 5 == 0:
				log = '[{}] total_L: {:.3f}, mel_L: {:.3f}, lin_L: {:.3f}, gate_L: {:.3f}, grad: {:.3f}, lr: {:.5f}, t: {:.2f}s, saved: F'.format(global_step, total_L, mel_L, linear_L, gate_L, grad_norm, current_lr, duration)
				print(log, end='\r')

			# Logs
			writer.add_scalar('total_loss', total_L, global_step)
			writer.add_scalar('mel_loss', mel_L, global_step)
			writer.add_scalar('linear_loss', linear_L, global_step)
			writer.add_scalar('gate_loss', gate_L, global_step)
			writer.add_scalar('grad_norm', grad_norm, global_step)
			writer.add_scalar('learning_rate', current_lr, global_step)

			global_step += 1
			start = time.time()

		global_epoch += 1

########################
# WARM FROM CHECKPOINT #
########################
"""
	Initialize training with a pre-trained model pth

	Args:
		checkpoint_path: ckpt/checkpoint_path200000.pth
		model: Pytorch model
		optimizer: Pytorch optimizer
"""
def warm_from_ckpt(checkpoint_dir, model_name, model, optimizer):
	checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_step{}.pth".format(model_name))
	print('[Trainer] - Warming up! Load checkpoint from: {}'.format(checkpoint_path))
	
	checkpoint = torch.load(checkpoint_path)
	model.load_state_dict(checkpoint['state_dict'])
	
	optimizer.load_state_dict(checkpoint['optimizer'])
	for state in optimizer.state.values():
		for k, v in state.items():
			if torch.is_tensor(v):
				state[k] = v.cuda()
	try:
		global global_step, global_epoch
		global_step = checkpoint['global_step']
		global_epoch = checkpoint['global_epoch']
	except:
		print('[Trainer] - Warning: global step and global epoch unable to restore!')
		sys.exit(0)

	return model, optimizer


#######################
# INITIALIZE TRAINING #
#######################
"""
	Setup and prepare for Tacotron training.
"""
def initialize_training(data_root, meta_text, checkpoint_dir=None, model_name=None):
	
	dataloader = Dataloader(data_root, meta_text)

	model = Tacotron(n_vocab=len(symbols),
					 embedding_dim=config.embedding_dim,
					 mel_dim=config.num_mels,
					 linear_dim=config.num_freq,
					 r=config.outputs_per_step,
					 padding_idx=config.padding_idx,
					 attention=config.attention,
					 use_mask=config.use_mask)

	optimizer = optim.Adam(model.parameters(),
						   lr=config.initial_learning_rate, 
						   betas=(config.adam_beta1, config.adam_beta2),
						   weight_decay=config.weight_decay)

	# Load checkpoint
	if model_name != None:
		model, optimizer = warm_from_ckpt(checkpoint_dir, model_name, model, optimizer)
			
	return model, optimizer, dataloader


########
# MAIN #
########
def main():

	args = get_training_args()

	os.makedirs(args.ckpt_dir, exist_ok=True)

	model, optimizer, dataloader = initialize_training(args.data_root, args.meta_text, args.ckpt_dir, args.model_name)

	# Train!
	try:
		train(model, optimizer, dataloader,
			  init_lr=config.initial_learning_rate,
			  log_dir=args.log_dir,
			  log_comment=args.log_comment,
			  checkpoint_dir=args.ckpt_dir,
			  checkpoint_interval=config.checkpoint_interval,
			  max_epochs=config.max_epochs,
			  max_steps=config.max_steps,
			  clip_thresh=config.clip_thresh)
	except KeyboardInterrupt:
		pass

	print()
	print('[Trainer] - Finished!')
	sys.exit(0)


if __name__ == '__main__':
	main()